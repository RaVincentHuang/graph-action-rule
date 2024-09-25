import sys

sys.path.append('/home/vincent/graphrule/src')
from graph.sample import sample_graph
import graph
from graph.store import dataset_build, graph_feature_calc, test_raw
from graph.standard import Graph24PointI
from graph.transform import combine_nodes
from llm.tag import Tag
from utils.config import DatasetConfig
import random
import os
from tqdm import tqdm
import pandas as pd
from networkx import DiGraph
import networkx as nx
import matplotlib.pyplot as plt

path1 = "/home/vincent/graphrule/data/graph/truth_combine_re"
path2 = "/home/vincent/graphrule/data/graph/truth_feature_re"

tag = Tag("calc", "ToT", "24point")
json_path = "/home/vincent/graphrule/data/graph/gpt4o-mini-0.7-p1v5g5_1_re_combine"

for filename in tqdm(os.listdir(json_path), desc='load graphs'):
    file_path = os.path.join(json_path, filename)
    graphI = Graph24PointI.from_json(file_path)
    graphI.calc_goal().calc_achievements()
    nx_graph: DiGraph = graphI.convert_to_nx()
    undir_graph = nx.Graph(nx_graph)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(undir_graph)
    nx.draw(undir_graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=10, font_size=3, font_weight='bold')
    plt.title("Graph Visualization")
    plt.show()
    
    nx_subgraph = sample_graph(nx_graph, "random_walk", 6, lambda x:x)
    break