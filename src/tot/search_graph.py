from math import e
from graph.matching import matching
from graph.model import GraphClassifier, NodeClassifier
from graph.standard import node2node_edge, node_edge2node
from utils.config import SubgraphMatchingConfig
from llm.embedding import get_embedding

from platform import node
import networkx as nx
import torch_geometric.utils

from torch_geometric.data import Data
import torch
from itertools import islice

from tqdm import tqdm
from collections import Counter

def read_lines(file_path: str, start: int, end: int) -> list[str]:
    with open(file_path, 'r') as file:
        return list(islice(file, start, end))


def load_pattern_graph(path) -> list[nx.DiGraph]:
    graphs: list[nx.DiGraph] = []
    # print(f"load {path}")
    graph = nx.DiGraph()
    with open(path, 'r') as file:
        for line in file:
            match line.split():
                case [index, 'p']:
                    index = int(index)
                    if len(graph):
                        print(f"graph {graph.nodes(data=True)}")
                        graphs.append(graph)
                    graph = nx.DiGraph()
                case ['v', node_id, label]:
                    node_id, label = int(node_id), int(label)
                    graph.add_node(node_id, label=label)
                case ['e', u, v, label]:
                    u, v, label = int(u), int(v), int(label)
                    graph.nodes[v]['label'] = node_edge2node(graph.nodes[v]['label'], label)
                    graph.add_edge(u, v)
                case _:
                    continue
    
    print(f"load {len(graphs)} patterns")
    return graphs
            

class SearchGraph:
    def __init__(self, matching_config: SubgraphMatchingConfig) -> None:
        self.graph = nx.DiGraph()
        self.frontier = set()
        self.matching_config = matching_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_model = GraphClassifier(1536 , 128, 2).to(self.device)
        self.class_model.eval()
        self.class_model.load_state_dict(torch.load("/home/vincent/graphrule/model/classifier.pt"))
        self.node_model = NodeClassifier(4, 64, 2, 4).to(self.device)
        self.node_model.eval()
        self.node_model.load_state_dict(torch.load("/home/vincent/graphrule/model/node_classifier.pt"))
        
    def dump_graph(self):
        print("Nodes:")
        for node in self.graph.nodes(data=True):
            print(node)
        print("\nEdges:")
        for edge in self.graph.edges(data=True):
            print(edge)
    
    def _get_graph_class(self, data) -> int:
        return self.class_model(data.to(self.device)).argmax(dim=1).item()
    
    def add_node(self, node_id, label, **kwargs) -> None:
        self.graph.add_node(node_id, label=label, feature=None, **kwargs)
        
    def add_edge(self, u, v, **kwargs) -> None:
        self.graph.add_edge(u, v, **kwargs)
        
    def calc_feature(self, select_feature) -> None:
        for node_id in tqdm(self.graph.nodes, 'calc feature'):
            if not self.graph.nodes(data=True)[node_id]['feature']:
                self.graph.nodes[node_id]['feature'] = get_embedding(select_feature(self.graph.nodes[node_id]))
                
    @staticmethod
    def convert_to_pyg(subgraph) -> Data:
        nx_graph = nx.DiGraph()
        for node_id in subgraph.nodes():
            nx_graph.add_node(node_id, x=subgraph.nodes[node_id]['feature'])
        for u, v in subgraph.edges():
            nx_graph.add_edge(u, v)
        
        return torch_geometric.utils.from_networkx(nx_graph)
        
    def remove_node(self, node_id) -> None:
        self.graph.remove_node(node_id)
        
    def remove_nodes(self, nodes: set[int]) -> None:
        for node_id in nodes:
            self.remove_node(node_id)
    
    def add_frontier(self, index: int):
        self.frontier.add(index)
    
    def fix(self):
        self.frontier = set()
    
    def drop(self):
        for node_id in list(self.frontier):
            self.remove_node(node_id)
    
    def matching_query(self, pattern: nx.DiGraph) -> list[set[int]]:
        mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        reverse_mapping = {i: node for node, i in mapping.items()}
        data_graph: nx.DiGraph = nx.relabel_nodes(self.graph, mapping)
        # print(f"matching query, pattern: {pattern.nodes()}, data: {data_graph.nodes()}")
        result: list[set[int]] = []
        pattern = nx.relabel_nodes(pattern, {node: i for i, node in enumerate(pattern.nodes())})
        embs = matching(pattern, data_graph, self.matching_config)
        if len(embs) > 0:
            print(f"matching query, pattern: {pattern.nodes()}, data: {data_graph.nodes()}, result: {embs}")
        for emb in embs:
            true_emb = set(map(lambda x: reverse_mapping[x], emb))
            if true_emb.intersection(self.frontier):
                result.append(true_emb)
        if len(result):
            print(f"matching query, pattern: {pattern.nodes()}, data: {data_graph.nodes()}, result: {result}")
        return result
    
    def get_subgraph(self, nodes: set[int]) -> nx.DiGraph:
        return self.graph.subgraph(nodes)
    
    def prune_with_classification(self, patterns: list[nx.DiGraph]):
        print(f"start prune, frontier: {self.frontier}")
        self.calc_feature(lambda x: x['last_formula'])
        for i, pattern in tqdm(enumerate(patterns), "prune for each patterns"):
            neg = []
            posi = []
            embs = self.matching_query(pattern)
            for emb in tqdm(embs, f"prune for pattern {i} ", leave=False):
                if not emb.intersection(self.frontier):
                    continue
                subgraph = self.get_subgraph(emb)
                data = SearchGraph.convert_to_pyg(subgraph)
                if self._get_graph_class(data):
                    posi.append(emb.intersection(self.frontier))
                else:
                    neg.append(emb.intersection(self.frontier))
            neg_counter = Counter()
            pos_counter = Counter()
            for emb_set in neg:
                neg_counter.update(emb_set)
            for emb_set in posi:
                pos_counter.update(emb_set)
            need_prune = set()
            for element in neg_counter:
                if neg_counter[element] > pos_counter[element]:
                    need_prune.add(element)
            if need_prune:
                print(f"delete {len(need_prune)} nodes: {need_prune}")
            self.frontier = self.frontier - need_prune
        print(f"end prune, frontier: {self.frontier}")
