
from calendar import c
from enum import IntEnum
import json
import networkx as nx
from networkx import DiGraph
from networkx import is_empty
import torch_geometric.utils

import llm.embedding
from llm.tag import Tag
import sympy
import re

import torch
import torch_geometric.data

class Node:
    def __init__(self, id, value, acc: float, feature=None):
        self.id = id
        self.value = value
        self.feature = feature
        self.acc = acc
    
    def calculate_feature(self, task='24point'):
        # Call OpenAI API to get embedding
        text = ""
        match task:
            case '24point':
                text = self.value[0]
        embedding = llm.embedding.get_embedding(text)
        self.feature = embedding
        
        return self
    
    def __str__(self) -> str:
        return f"Node {self.id}: {self.value}"
    
    def __format__(self, format_spec: str) -> str:
        return f"Node {self.id}: {self.value}"
    
class Edge:
    """Save the index of the source node and the destination node"""
    def __init__(self, src, dst, value=None, feature=None):
        self.src = src
        self.dst = dst
        self.value = value
        self.feature = feature
        
    def __str__(self) -> str:
        return f"Edge {self.src} -> {self.dst}"
    
    def __format__(self, format_spec: str) -> str:
        return f"Edge {self.src} -> {self.dst}"

class BaseGraph:
    def __init__(self, name, tag):
        self.name = name
        self.tag: Tag = tag
        self.nodes: list[Node] = []
        self.edges = []
        
    def load_from_native_json(self, json_path):
        pass
    
    def load_from_json(self, json_path):
        pass
    
    def save_to_json(self, json_path):
        pass
    
    def convert_to_nx(self) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        for node in self.nodes:
            nx_graph.add_node(node.id, value=node.value, acc=node.acc)
        
        for edge in self.edges:
            nx_graph.add_edge(edge.src, edge.dst)
        
        return nx_graph
    
    def load_from_nx(self, nx_graph: nx.DiGraph):
        self.nodes = []
        self.edges = []
        for node_id in nx_graph.nodes:
            node = Node(node_id, nx_graph.nodes[node_id]['value'], nx_graph.nodes[node_id]['acc'])
            self.nodes.append(node)
        
        for edge in nx_graph.edges:
            edge = Edge(edge[0], edge[1])
            self.edges.append(edge)
    
    def __str__(self) -> str:
        return f"Graph {self.name} with {len(self.nodes)} nodes and {len(self.edges)} edges\nnodes: {self.nodes}\nedges: {self.edges}"
    
    def __format__(self, format_spec: str) -> str:
        nodes = "[\n"
        for node in self.nodes:
            nodes += f"\t{node},\n"
        nodes += "]"
        edges = "[\n"
        for edge in self.edges:
            edges += f"\t{edge},\n"
        edges += "]"
        return f"Graph {self.name} with {len(self.nodes)} nodes and {len(self.edges)} edges\nnodes: {nodes}\nedges: {edges}"
    
    # def convert_to_pyg(self):
    #     raise NotImplementedError()
    

class Graph24PointI(BaseGraph):
    def __init__(self, name, tag):
        super().__init__(name, tag)
        self.goal = set()
        self.achievements = set()
        
    def load_from_native_json(self, json_path):
        self.nodes = []
        self.edges = []
        with open(json_path, 'r') as file:
            tot_tree_json = json.load(file)
            node_map = {}
            index = 0
            for node_content, state in tot_tree_json['nodes'].items():
                if state['step'] >= 4:
                    continue
                    
                if state['step'] == 0:
                    node = Node(index, (node_content, node_content, "null"), acc = 0)
                else:
                    node = Node(index, (node_content, state['last_formula'], state['operator']), acc = state['value'])
                self.nodes.append(node)
                node_map[node_content] = index
                index += 1
                assert index == len(self.nodes)
            
            for node_content, state in tot_tree_json['nodes'].items():
                if state['step'] == 0:
                    continue
                if state['parent'] not in node_map or node_content not in node_map:
                    continue
                src, dst = node_map[state['parent']], node_map[node_content]
                edge = Edge(src, dst)
                self.edges.append(edge)
    
    def calc_goal(self):
        for node in self.nodes:
            if node.value[2] == 'null':
                continue
            last_formula = node.value[1]
            first_expr = last_formula.split('=')[0]
            match = re.search(r'\(left:(.*)\)', last_formula)
            if not match:
                continue
            numbers = re.findall(r"(\d+)" , match.group(1))
            # print(first_expr)
            if sympy.simplify(first_expr) == 24 and len(numbers) == 1 and int(numbers[0]) == 24:
                self.goal.add(node.id)
        
        return self
        
    def calc_achievements(self):
        self.achievements = self.goal.copy()
        while True:
            flag = False
            for edge in self.edges:
                if edge.dst in self.achievements and edge.src not in self.achievements:
                    self.achievements.add(edge.src)
                    flag = True
            if not flag:
                break
        
        return self
                
    def save_to_json(self, json_path):
        with open(json_path, 'w') as file:
            tot_tree_json = {
                'name': self.name,
                'tag': str(self.tag),
                'nodes': [],
                'edges': []
            }
            for node in self.nodes:                
                state = {
                    'id' : node.id,
                    'formula': node.value[0],
                    'last_formula': node.value[1],
                    'operator': node.value[2],
                    'acc': node.acc
                }
                tot_tree_json['nodes'].append(state)
            
            for edge in self.edges:
                state = {
                    'src': edge.src,
                    'dst': edge.dst
                }
                tot_tree_json['edges'].append(state)
            
            json.dump(tot_tree_json, file)
    
    def load_from_json(self, json_path):
        self.nodes = []
        self.edges = []
        with open(json_path, 'r') as file:
            tot_tree_json = json.load(file)
            index = 0
            for state in tot_tree_json['nodes'].items():
                node = Node(index, (state['formula'], state['last_formula'], state['operator']), acc = state['value'])
                self.nodes.append(node)
                index += 1
                assert index == len(self.nodes)
            
            for state in tot_tree_json['nodes'].items():
                edge = Edge(state['src'], state['dst'])
                self.edges.append(edge)
                
    

class SubgraphType(IntEnum):
    T0 = 0
    T1 = 1
    T2 = 2
    T3 = 3
    UNKNOWN = 4

class Graph24PointII(BaseGraph):
    def __init__(self, name, tag):
        super().__init__(name, tag)
        self.type: SubgraphType = SubgraphType.UNKNOWN
        
    
    def calc_type(self, graph: Graph24PointI):
        # graph.calc_goal().calc_achievements()
        subgraph = set(map(lambda x: x.id, self.nodes))
        if subgraph.issubset(graph.achievements):
            self.type = SubgraphType.T0
        elif subgraph & graph.achievements:
            self.type = SubgraphType.T1
        elif graph.achievements:
            self.type = SubgraphType.T2
        else:
            self.type = SubgraphType.T3
    
    def convert_to_pyg(self) -> torch_geometric.data.Data:
        
        # x = []
        # edge_index = []
        # for node in self.nodes:
        #     node.calculate_feature()
        #     x.append(node.feature)
        # for edge in self.edges:
        #     edge_index.append([edge.src, edge.dst])
        
        # data = torch_geometric.data.Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index))
        # return data
        
        nx_graph = nx.DiGraph()
        for node in self.nodes:
            if node.feature is None:
                node.calculate_feature()
            nx_graph.add_node(node.id, x=node.feature)
        
        for edge in self.edges:
            nx_graph.add_edge(edge.src, edge.dst)
            
        data = torch_geometric.utils.from_networkx(nx_graph)
        data.y = torch.tensor([self.type], dtype=torch.long)
        
        return data
                
    def __format__(self, format_spec: str) -> str:
        return f"Type: {self.type}\n" + super().__format__(format_spec)
    
    def __str__(self) -> str:
        return f"Type: {self.type}\n" + super().__str__()
