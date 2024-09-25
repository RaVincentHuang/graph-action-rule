import sys
sys.path.append('/home/vincent/graphrule/src')
import graph
from graph.store import dataset_build, graph_feature_calc, test_raw, combine_task, dataset_build_truth
from graph.standard import Graph24PointI
from graph.transform import combine_nodes
from llm.tag import Tag
from utils.config import DatasetConfig
import random
import os
from tqdm import tqdm
import pandas as pd




node_random = lambda x: random.randint(x - 4, x + 4)


def combine_nodes_with_value(graph_path, save_path, name, tag: Tag):
    for file_name in tqdm(os.listdir(graph_path), desc="Combining nodes"):
        index = int(file_name.split('_')[-1].split('.')[0])
        graphI = Graph24PointI(f"{name}", index, tag)
        graphI.load_from_json(os.path.join(graph_path, file_name))
        combine_nodes(graphI)
        graphI.save_to_json(os.path.join(save_path, f"{name}_{index}.json"))

def calc_index(graph_path, save_path, name, tag: Tag):
    for file_name in tqdm(os.listdir(graph_path), desc="Calculating index"):
        index = int(file_name.split('_')[-1].split('.')[0])
        graphI = Graph24PointI(f"{name}I_{index}", index, tag)
        graphI.load_from_json(os.path.join(graph_path, file_name))
        graphI.save_to_json(os.path.join(save_path, f"{name}I_{index}.json"))

def re_index(graph_path, save_path, name, tag: Tag):
    
    index_check = {}
    task_path = "/home/vincent/graphrule/data/tasks/24.csv"
    for chunk in pd.read_csv(task_path, usecols=['Rank', 'Puzzles'], chunksize=1):
        for _, row in chunk.iterrows():
            rank, task = row['Rank'], row['Puzzles']
            index_check[task] = rank
    
    for file_name in tqdm(os.listdir(graph_path), desc="Reindexing"):
        index = int(file_name.split('_')[-1].split('.')[0])
        graphI = Graph24PointI(f"{name}I_{index}", index, tag)
        graphI.load_from_json(os.path.join(graph_path, file_name))
        task = graphI.nodes[0].value[0] #.replace(" ", "").replace(",", " ")
        index = index_check[task]
        graphI.index = index
        graphI.name = f"{name}I_{index}"
        graphI.save_to_json(os.path.join(save_path, f"{name}_{index}.json"))

def reformat(graph_path, save_path, name, tag: Tag):
    for file_name in tqdm(os.listdir(graph_path), desc="Reformatting root"):
        index = int(file_name.split('_')[-1].split('.')[0])
        graphI = Graph24PointI(f"{name}I_{index}", index, tag)
        graphI.load_from_json(os.path.join(graph_path, file_name))
        for node in graphI.nodes:
            v1 = node.value[0].replace(", ", " ")
            v2 = node.value[1].replace(", ", " ")
            node.value = (v1, v2, node.value[2])
        graphI.save_to_json(os.path.join(save_path, f"{name}_{index}.json"))


json_path = "/home/vincent/graphrule/data/graph/truth"
graph_path = "/home/vincent/graphrule/data/graph/gpt4o-mini-0.7-p1v5g5_1_re_combine"

source_path = "/home/vincent/graphrule/data/graph/gpt4o-mini-0.7-p1v5g5_1_re"
target_path = "/home/vincent/graphrule/data/subgraph"
truth_path = "/home/vincent/graphrule/data/graph/truth_combine_re_format"

name = "24point"
tag = Tag("gpt-4o-mini", "ToT", "24point")
# test_raw(json_path, "24point", Tag("gpt-4o-mini", "ToT", "24point"))
# graph_feature_calc(target_path, target_path, "24point", Tag("calc", "ToT", "24point"))
# calc_index(source_path, target_path, name, tag)
# combine_nodes_with_value(source_path, target_path, name, tag)
# reformat(source_path, target_path, name, tag)


# re_index(source_path, target_path, name, tag)
# combine_task(source_path, truth_path, graph_path)

dataset_build_truth(graph_path, target_path, "24point_3", Tag("gpt-4o-mini", "ToT", "24point"), DatasetConfig("random_walk", 16000, 8, node_random))
