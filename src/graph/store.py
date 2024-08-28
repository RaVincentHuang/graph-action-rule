import random
from networkx import DiGraph
import torch
from torch_geometric.data import Data, Dataset

from graph.sample import sample_graph
from graph.standrad import Graph24PointI, Graph24PointII
from llm.tag import Tag
from utils.config import DatasetConfig

import os

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
    

def dataset_build(json_path, dataset_path, name, tag: Tag, config: DatasetConfig):
    graphI_list: list[Graph24PointI] = []
    for file_name in os.listdir(json_path):
        print(f"Load file: {file_name}")
        file_path = os.path.join(json_path, file_name)
        # Read the file and process the data
        graphI = Graph24PointI(f"{name}I_{len(graphI_list) + 1}", tag)
        graphI.load_from_native_json(file_path)
        graphI.calc_goal().calc_achievements()
        graphI_list.append(graphI)
        
    
    
    print(f"Get {len(graphI_list)} graphs")
    data_list = []
    for i in range(config.total_num):
        graphI = random.choice(graphI_list)
        nx_graph: DiGraph = graphI.convert_to_nx()
        print(f"Sample subgraph {nx_graph}")
        nx_subgraph = sample_graph(nx_graph, config.sampler, config.node_num)
        subgraph = Graph24PointII(f"{name}II_{i + 1}", tag)
        subgraph.load_from_nx(nx_subgraph)
        subgraph.calc_type(graphI)
        data = subgraph.convert_to_pyg()
        print(f"Append subgraph {i + 1}: {data}")
        data_list.append(data)
    
    dataset = SubgraphDataset(data_list)
    dataset.save(dataset_path)
