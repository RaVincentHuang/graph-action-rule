import random
import time
from networkx import DiGraph
import torch
from torch_geometric.data import Data, Dataset, OnDiskDataset

from graph.sample import sample_graph
from graph.standard import Graph24PointI, Graph24PointII, Graph24PointIII, SubgraphType
from llm.tag import Tag
from utils.config import DatasetConfig

import os
import multiprocessing

from tqdm import tqdm

class SubgraphDataset(Dataset):
    def __init__(self, data_list):
        super(SubgraphDataset, self).__init__()
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def save(self, path):
        torch.save(self.data_list, path)
        
    @staticmethod
    def load(path):
        data_list = torch.load(path)
        return SubgraphDataset(data_list)

def _process_raw_file(file_path, name, tag, graph_path, index):
    graphI = Graph24PointI(f"{name}I_{index}", index, tag)
    graphI.load_from_native_json(file_path)
    graphI.calc_goal().calc_achievements()
    for node in graphI.nodes:
        node.calculate_feature()
    graphI.save_to_json(f"{graph_path}/{graphI.name}.json")
    
def _process_file(file_path, name, tag, graph_path, index):
    graphI = Graph24PointI(f"{name}I_{index}", index, tag)
    graphI.load_from_json(file_path)
    graphI.calc_goal().calc_achievements()
    for node in graphI.nodes:
        node.calculate_feature()
    graphI.save_to_json(f"{graph_path}/{graphI.name}.json")
    
def graph_feature_calc(json_path, graph_path, name, tag: Tag):
    with tqdm(total=len(os.listdir(json_path))) as pbar:
        pool = multiprocessing.Pool(processes=16)
        index = 1
        for file_name in os.listdir(json_path):
            file_path = os.path.join(json_path, file_name)
            pool.apply_async(_process_file, args=(file_path, name, tag, graph_path, index), callback=lambda _: pbar.update(1))
            index += 1
        pool.close()
        pool.join()

def test_raw(json_path, name, tag: Tag):
    graphI_list_acc: list[Graph24PointI] = []
    graphI_list_dead: list[Graph24PointI] = []
    for file_name in os.listdir(json_path):
        # print(f"Load file: {file_name}")
        file_path = os.path.join(json_path, file_name)
        # Read the file and process the data
        print(file_name)
        index = len(graphI_list_acc) + len(graphI_list_dead) + 1
        graphI = Graph24PointI(f"{name}I_{index}", index, tag)
        graphI.load_from_native_json(file_path)
        graphI.calc_goal().calc_achievements()
        # print(f"{graphI.name} get label {graphI.achievements}")
        if graphI.achievements:
            graphI_list_acc.append(graphI)
        else:
            graphI_list_dead.append(graphI)
        
    print(f"Get Acc {len(graphI_list_acc)} graphs")
    print(f"Get Dead {len(graphI_list_dead)} graphs")


def dataset_build_truth(json_path, dataset_path, name, tag: Tag, config: DatasetConfig):
    graphI_list: list[Graph24PointI] = []
    
    for filename in tqdm(os.listdir(json_path), desc='load graphs'):
        file_path = os.path.join(json_path, filename)
        graphI = Graph24PointI.from_json(file_path)
        graphI.calc_goal().calc_achievements()
        graphI_list.append(graphI)
    
    subgraph_type0_cnt = 0
    subgraph_type1_cnt = 0
    # subgraph_type0: list[Graph24PointIII] = []
    # subgraph_type1: list[Graph24PointIII] = []
    data_list = []
    
    with tqdm(total=config.total_num, desc='subgraph sample') as pbar:
        while subgraph_type0_cnt < config.total_num // 2 or subgraph_type1_cnt < config.total_num // 2:
            random.seed(time.time())
            graphI = random.choice(graphI_list)
            nx_graph: DiGraph = graphI.convert_to_nx()
            nx_subgraph = sample_graph(nx_graph, config.sampler, config.node_num, config.node_num_random)
            subgraph = Graph24PointIII(f"{name}II_{pbar.n + 1}", tag)
            subgraph.from_nx(nx_subgraph)
            subgraph.calc_type(graphI)
            if subgraph.type == SubgraphType.T0 and subgraph_type0_cnt < config.total_num // 2:
                subgraph_type0_cnt += 1
                data = subgraph.convert_to_pyg()
                data_list.append(data)
                pbar.update(1)
            elif subgraph.type == SubgraphType.T1 and subgraph_type1_cnt < config.total_num // 2:
                subgraph_type1_cnt += 1
                data = subgraph.convert_to_pyg()
                data_list.append(data)
                pbar.update(1)
    
    print(f"Get {len(data_list)} subgraphs")
    dataset = SubgraphDataset(data_list)
    dataset.save(f"{dataset_path}/{name}_{config}.pt")

def dataset_build(json_path, dataset_path, name, tag: Tag, config: DatasetConfig):
    graphI_dead: list[Graph24PointI] = []
    graphI_acc: list[Graph24PointI] = []
    
    with tqdm(total=len(os.listdir(json_path)), desc='load graphs') as pbar:
        for file_name in os.listdir(json_path):
            # print(f"Load file: {file_name}")
            file_path = os.path.join(json_path, file_name)
            # Read the file and process the data
            index = int(file_name.split('_')[-1].split('.')[0])
            graphI = Graph24PointI(f"{file_name.split('.')[0]}", index, tag)
            graphI.load_from_json(file_path)
            if graphI.achievements:
                graphI_acc.append(graphI)
            else:
                graphI_dead.append(graphI)
            pbar.update(1)
    
    print(f"Get Acc {len(graphI_acc)} graphs")
    print(f"Get Dead {len(graphI_dead)} graphs")
    
    subgraph_list: list[Graph24PointII] = []
    
    with tqdm(total=config.total_num // 2, desc='subgraph sample in dead') as pbar:
        for i in range(config.total_num // 2):
            random.seed(time.time())
            graphI = random.choice(graphI_dead)
            nx_graph: DiGraph = graphI.convert_to_nx()
            nx_subgraph = sample_graph(nx_graph, config.sampler, config.node_num, config.node_num_random)
            subgraph = Graph24PointII(f"{name}II_{i + 1 + (config.total_num // 2)}", tag)
            subgraph.from_nx(nx_subgraph)
            subgraph.calc_type(graphI)
            subgraph_list.append(subgraph)
            pbar.update(1)
            
    with tqdm(total=config.total_num // 2, desc='subgraph sample in acc') as pbar:
        for i in range(config.total_num // 2):
            random.seed(time.time())
            graphI = random.choice(graphI_acc)
            nx_graph: DiGraph = graphI.convert_to_nx()
            nx_subgraph = sample_graph(nx_graph, config.sampler, config.node_num, config.node_num_random)
            subgraph = Graph24PointII(f"{name}II_{i + 1}", tag)
            subgraph.from_nx(nx_subgraph)
            subgraph.calc_type(graphI)
            subgraph_list.append(subgraph)
            pbar.update(1)
    
    data_list = []
    with tqdm(total=len(subgraph_list), desc='convert to pyg') as pbar:
        for subgraph in subgraph_list:
            data = subgraph.convert_to_pyg()
            data_list.append(data)
            pbar.update(1)
    
    print(f"Get {len(data_list)} subgraphs")
    dataset = SubgraphDataset(data_list)
    dataset.save(f"{dataset_path}/{name}_{config}.pt")
    

def dataset_build_raw(json_path, dataset_path, name, tag: Tag, config: DatasetConfig):
    graphI_list_acc: list[Graph24PointI] = []
    graphI_list_dead: list[Graph24PointI] = []
    for file_name in os.listdir(json_path):
        # print(f"Load file: {file_name}")
        file_path = os.path.join(json_path, file_name)
        # Read the file and process the data
        index = len(graphI_list_acc) + len(graphI_list_dead) + 1
        graphI = Graph24PointI(f"{name}I_{index}", index, tag)
        graphI.load_from_native_json(file_path)
        graphI.calc_goal().calc_achievements()
        # print(f"{graphI.name} get label {graphI.achievements}")
        if graphI.achievements:
            graphI_list_acc.append(graphI)
        else:
            graphI_list_dead.append(graphI)
        
    print(f"Get Acc {len(graphI_list_acc)} graphs")
    print(f"Get Dead {len(graphI_list_dead)} graphs")
    subgraph_list: list[Graph24PointII] = []
    for i in range(config.total_num // 2):
        graphI = random.choice(graphI_list_acc)
        nx_graph: DiGraph = graphI.convert_to_nx()
        print(f"Sample subgraph {nx_graph}")
        nx_subgraph = sample_graph(nx_graph, config.sampler, config.node_num, config.node_num_random)
        subgraph = Graph24PointII(f"{name}II_{i + 1}", tag)
        subgraph.from_nx(nx_subgraph)
        subgraph.calc_type(graphI)
        subgraph_list.append(subgraph)
        
    for i in range(config.total_num // 2):
        graphI = random.choice(graphI_list_dead)
        nx_graph: DiGraph = graphI.convert_to_nx()
        print(f"Sample subgraph {nx_graph}")
        nx_subgraph = sample_graph(nx_graph, config.sampler, config.node_num, config.node_num_random)
        subgraph = Graph24PointII(f"{name}II_{i + 1 + (config.total_num // 2)}", tag)
        subgraph.from_nx(nx_subgraph)
        subgraph.calc_type(graphI)
        subgraph_list.append(subgraph)
    
    data_list = []
    for subgraph in subgraph_list:
        data = subgraph.convert_to_pyg()
        data_list.append(data)
    
    print(f"Get {len(data_list)} subgraphs")
    dataset = SubgraphDataset(data_list)
    dataset.save(f"{dataset_path}/{name}{config}.pt")


def combine_task(source_path, truth_path, target_path):
    
    truth_path_map = {}
    for file_name in tqdm(os.listdir(truth_path), desc='Load truth path'):
        file_path = os.path.join(truth_path, file_name)
        graphI = Graph24PointI.from_json(file_path)
        truth_path_map[graphI.index] = file_path
        
    for file_name in tqdm(os.listdir(source_path), desc='Combine nodes'):
        file_path = os.path.join(source_path, file_name)
        graphI = Graph24PointI.from_json(file_path)
        index = graphI.index
        graphI_truth = Graph24PointI.from_json(truth_path_map[index])
        combine_graph = graphI.combine(graphI_truth)
        combine_graph.save_to_json(os.path.join(target_path, f"{graphI.name}.json"))
    

def build_gspan_data(graph_path, target_path, node_label, edge_label):
    with open(f"{target_path}/graph.data", 'w', encoding='utf-8') as file:
        cnt = 0
        for file_name in tqdm(os.listdir(graph_path), desc='Build gspan data'):
            
            file.write(f"t # {cnt}\n")
            cnt += 1
            file_path = os.path.join(graph_path, file_name)
            graphI = Graph24PointI.from_json(file_path)
            graphI.re_index()
            for node in graphI.nodes:
                file.write(f"v {node.id} {node_label(node)}\n")
            for edge in graphI.edges:
                file.write(f"e {edge.src} {edge.dst} {edge_label(edge)}\n")