from llm.chat import gpt
from utils.config import LLMConfig, SearchConfig, SubgraphMatchingConfig
from tot.search_graph import SearchGraph, load_pattern_graph
from graph.standard import node_edge2node

import itertools
import numpy as np
from functools import partial
from task.base import Task
import re
import os
import json
from typing import Any, Dict, Union

import networkx as nx

def operator2label(operator):
    return {
        '+': 1,
        '-': 2,
        '*': 3,
        '/': 4
    }.get(operator, 5)


def get_value(task: Task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    # print(f'-- value --: {value}')
    return value

# TODO 
def get_values(task: Task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task: Task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    # print(f'-- get_votes --: {vote_prompt}')
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    # print(f'-- votes --: {values}')
    return values

def get_proposals(task: Task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    # print(f'-- get_proposals --: {propose_prompt}')
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    # print(f'-- proposals --: {proposals}')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task: Task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    # print(f'-- get_samples --: {prompt}')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    # print(f'-- samples --: {samples}')
    return [y + _ for _ in samples]

# FIXME  
filter_type, order_type, engine_type = "GQL", "GQL", "LFTJ"
order_num, time_limit = 100, 60
matching_config = SubgraphMatchingConfig(filter_type, order_type, engine_type, order_num, time_limit)


def solve(search_config: SearchConfig, llm_config: LLMConfig, task: Task, idx, to_print=True):
    
    patterns = load_pattern_graph("/home/vincent/graphrule/data/frequent_pattern/800.txt")
    search_graph = SearchGraph(matching_config)
    id_check = {}
    ys_check = {}
    
    global node_num
    node_num = 0
    tree: Dict[str, Dict[str, Dict[str, Any]]] = {
        "nodes": {},
        "edges": {},
    }
    

    global gpt
    gpt = partial(gpt, model=llm_config.model, temperature=llm_config.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    # TODO add root node
    search_graph.add_node(node_num, label=1, step=0, last_formula=x, operator='null')
    id_check[x] = node_num
    ys_check[node_num] = x
    logNewState(tree, x, value='', parent='null', step=0)
    
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        new_ys = []
        # generation
        if search_config.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, search_config.n_generate_sample, prompt_sample=search_config.prompt_sample, stop=task.stops[step]) for y in ys]
        elif search_config.method_generate == 'propose':
            # new_ys = [get_proposals(task, x, y) for y in ys]
            for y in ys:
                children_from_y = get_proposals(task, x, y)
                new_ys += children_from_y
                for child in children_from_y:
                    last_formula = child.split('\n')[-2]
                    operator_list = re.findall(r'[+\-*/]', last_formula)
                    operator = operator_list[0] if operator_list else ''
                    # TODO add node
                    label = node_edge2node(step+1 + 1, operator2label(operator))
                    search_graph.add_node(node_num, label, step=step+1, last_formula=last_formula, operator=operator)
                    search_graph.add_frontier(node_num)
                    search_graph.add_edge(id_check[y if y else x], node_num)
                    id_check[child] = node_num
                    ys_check[node_num] = child
                    logNewState(tree, child, parent=y if y else x, step=step+1, last_formula=last_formula, operator=operator)

        # TODO prune the new_ys (stage1: after proposals)
        # search_graph.dump_graph()
        search_graph.prune_with_classification(patterns)
        new_ys = list(filter(lambda ys: id_check[ys] in search_graph.frontier, new_ys))
        search_graph.fix()
        
        # new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # initialize values
        values = []
        
        # evaluation
        if search_config.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, search_config.n_evaluate_sample)
        elif search_config.method_evaluate == 'value':
            values = get_values(task, x, new_ys, search_config.n_evaluate_sample)
        for i in ids:
            logNewState(tree, new_ys[i], value=values[i])

        # selection
        select_ids = []
        if search_config.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=search_config.n_select_sample, p=ps).tolist()
        elif search_config.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:search_config.n_select_sample]
        for id in select_ids:
            search_graph.add_frontier(id_check[new_ys[id]])
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # TODO prune the sorted_new_ys (stage2: after values)
        # search_graph.dump_graph()
        search_graph.prune_with_classification(patterns)
        select_new_ys = list(filter(lambda ys: id_check[ys] in search_graph.frontier, select_new_ys))
        search_graph.fix()
        
        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys


    # 存储tot树形结构到json中
    


    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(search_config: SearchConfig, llm_config: LLMConfig, task: Task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=llm_config.model, temperature=llm_config.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', search_config.n_generate_sample, search_config.prompt_sample, stop=None)
    return ys, {}


# record the tot tree
def save_tree_to_json(tree, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as json_file:
        json.dump(tree, json_file, indent=4)

def logNewState(tree, state, **kwargs):
    global node_num
    #kwargs: value, parent, step
    if state in tree['nodes']:
        pass
    else:
        tree['nodes'][state] = {}
        tree['nodes'][state]['node_id'] = f'n{node_num}'
        node_num += 1

    for key, value in kwargs.items():
        tree['nodes'][state][key] = value

node_num = 0
