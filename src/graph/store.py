import random
import time
from networkx import DiGraph
import torch
from torch_geometric.data import Data, Dataset

from graph.sample import sample_graph
from graph.standrad import Graph24PointI, Graph24PointII
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

def _process_file(file_path, name, tag, graph_path, index):
    graphI = Graph24PointI(f"{name}I_{index}", tag)
    graphI.load_from_native_json(file_path)
    graphI.calc_goal().calc_achievements()
    for node in graphI.nodes:
        node.calculate_feature()
    graphI.save_to_json(f"{graph_path}/{graphI.name}.json")
    
def graph_future_calc(json_path, graph_path, name, tag: Tag):

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
        graphI = Graph24PointI(f"{name}I_{len(graphI_list_acc) + len(graphI_list_dead) + 1}", tag)
        graphI.load_from_native_json(file_path)
        graphI.calc_goal().calc_achievements()
        # print(f"{graphI.name} get label {graphI.achievements}")
        if graphI.achievements:
            graphI_list_acc.append(graphI)
        else:
            graphI_list_dead.append(graphI)
        
    print(f"Get Acc {len(graphI_list_acc)} graphs")
    print(f"Get Dead {len(graphI_list_dead)} graphs")


def dataset_build(json_path, dataset_path, name, tag: Tag, config: DatasetConfig):
    graphI_dead: list[Graph24PointI] = []
    graphI_acc: list[Graph24PointI] = []
    
    with tqdm(total=len(os.listdir(json_path)), desc='load graphs') as pbar:
        for file_name in os.listdir(json_path):
            # print(f"Load file: {file_name}")
            file_path = os.path.join(json_path, file_name)
            # Read the file and process the data
            graphI = Graph24PointI(f"{file_name.split('.')[0]}", tag)
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
        graphI = Graph24PointI(f"{name}I_{len(graphI_list_acc) + len(graphI_list_dead) + 1}", tag)
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
