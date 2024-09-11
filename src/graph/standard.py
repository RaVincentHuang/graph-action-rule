
from calendar import c
from enum import IntEnum
import json
from operator import is_
import networkx as nx
from networkx import DiGraph
from networkx import is_empty
import torch_geometric.utils

from task.game24 import ASTNode 

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
        self.edges: list[Edge] = []
        
    def load_from_native_json(self, json_path):
        pass
    
    def load_from_json(self, json_path):
        pass
    
    def save_to_json(self, json_path):
        pass
    
    def convert_to_nx(self) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        for node in self.nodes:
            feature = node.feature if node.feature else None
            nx_graph.add_node(node.id, value=node.value, acc=node.acc, feature=feature)
        
        for edge in self.edges:
            nx_graph.add_edge(edge.src, edge.dst)
        
        return nx_graph
    
    def from_nx(self, nx_graph: nx.DiGraph):
        self.nodes = []
        self.edges = []
        for node_id in nx_graph.nodes:
            node = Node(node_id, nx_graph.nodes[node_id]['value'], nx_graph.nodes[node_id]['acc'], nx_graph.nodes[node_id]['feature'])
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
    def __init__(self, name, index, tag):
        super().__init__(name, tag)
        self.index = index
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
                'index': self.index,
                'nodes': [],
                'edges': [],
                'goal': list(self.goal),
                'achievements': list(self.achievements)
            }
            for node in self.nodes:                
                state = {
                    'id' : node.id,
                    'formula': node.value[0],
                    'last_formula': node.value[1],
                    'operator': node.value[2],
                    'acc': node.acc,
                    'future': node.feature
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
            for state in tot_tree_json['nodes']:
                node = Node(state['id'], (state['formula'], state['last_formula'], state['operator']), acc = state['acc'], feature=state['future'])
                self.nodes.append(node)
            
            for state in tot_tree_json['edges']:
                edge = Edge(state['src'], state['dst'])
                self.edges.append(edge)
                
            self.goal = set(tot_tree_json['goal'])
            self.achievements = set(tot_tree_json['achievements'])
<<<<<<< HEAD:src/graph/standard.py
    
    @staticmethod
    def from_json(json_path) -> 'Graph24PointI':
        with open(json_path, 'r') as file:
            tot_tree_json = json.load(file)
            name = tot_tree_json['name']
            tag = Tag.from_str(tot_tree_json['tag'])
            index = tot_tree_json['index']
            graph = Graph24PointI(name, index, tag)
            graph.load_from_json(json_path)
        return graph
        
            
    def combine(self, graph: 'Graph24PointI') -> 'Graph24PointI':
        res = Graph24PointI(self.name, self.index, self.tag)
        res.nodes = self.nodes.copy()
        res.edges = self.edges.copy()
        
        shift_index = {}
        shift_index[0] = 0
        for node in graph.nodes:
            if node.id != 0:
                node_cpy = Node(node.id + len(self.nodes), node.value, node.acc, node.feature)
                shift_index[node.id] = node.id + len(self.nodes)
                res.nodes.append(node_cpy)
        
        for edge in graph.edges:
            src, dst = shift_index[edge.src], shift_index[edge.dst]
            edge_cpy = Edge(src, dst)
            res.edges.append(edge_cpy)
        
        res.re_index()

        return res
=======
>>>>>>> bcfc9da (add):src/graph/standrad.py
            
    def from_ast(self, roots: list[ASTNode], nums: list[int]):
        self.nodes = []
        self.edges = []
        
        def post_order_traversal(node: ASTNode):
            left, right = None, None
            if node.left:
                left = post_order_traversal(node.left)
            if node.right:
                right = post_order_traversal(node.right)
            if node.leaf:
                return node.value
            stk.append((node.value, left, right))
            return eval(f"{left} {node.value} {right}")
        
        def transform_nums(nums: tuple[int, int, int, int], left, right, res) -> tuple[int, int, int, int]:
            tmp = [x for x in nums if x != -114514]
            # print(tmp)
            # print(left, right, res)
            tmp.remove(left)
            tmp.remove(right)
            tmp.append(res)
            match tmp:
                case [a, b, c, d]:
                    return a, b, c, d
                case [a, b, c]:
                    return a, b, c, -114514
                case [a, b]:
                    return a, b, -114514, -114514
                case [a]:
                    return a, -114514, -114514, -114514
                case _:
                    return -114514, -114514, -114514, -114514
        class HashNode:
            def __init__(self, nums: tuple[int, int, int, int], op=None, left=None, right=None):
                self.nums = nums
                self.op = op
                self.left = left
                self.right = right
                self.value = eval(f"{left} {op} {right}") if op else None
                
            def is_root(self):
                return not self.op
            
            def formula(self):
                nums = [str(x) for x in self.nums if x != -114514]
                left = ", ".join(map(str, nums))
                if self.op is None:
                    return left
                return f"{self.left} {self.op} {self.right} = {self.value} (left: {left}) \n"
        
        node_check: dict[HashNode, int] = {}
        parent: dict[int, int] = {}
        
        nums_tuple: tuple[int, int, int, int] = (nums[0], nums[1], nums[2], nums[3])
        
        node_check[HashNode(nums_tuple)] = 0
        
        for root in roots:
            stk = []
            post_order_traversal(root)
            parent_ptr = 0
            nums_tmp = nums_tuple
            for (op, left, right) in stk:
                res = eval(f"{left} {op} {right}")
                nums_tmp = transform_nums(nums_tmp, left, right, res)
                hash = HashNode(nums_tmp, op, left, right)
                id = node_check[hash] if hash in node_check else len(node_check)
                node_check[HashNode(nums_tmp, op, left, right)] = id
                parent[id] = parent_ptr
                parent_ptr = id
                
        formulas = {}
        for node, id in node_check.items():
            last_formula = node.formula()
            if node.is_root():
                self.nodes.append(Node(id, (last_formula, last_formula, 'null'), acc=0))
                continue
            parent_id = parent[id]
            pre = formulas[parent_id] if parent_id in formulas else ""
            formula = pre + last_formula
            formulas[id] = formula
            self.nodes.append(Node(id, (last_formula, formula, node.op), acc=0))
            self.edges.append(Edge(parent_id, id))
<<<<<<< HEAD:src/graph/standard.py
    
    def re_index(self):
        id_map = {node.id: i for i, node in enumerate(self.nodes)}
        for i, node in enumerate(self.nodes):
            node.id = i
        
        for edge in self.edges:
            edge.src = id_map[edge.src]
            edge.dst = id_map[edge.dst]
=======
>>>>>>> bcfc9da (add):src/graph/standrad.py
            

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
        # print(f"subgraph: {subgraph}, achievements: {graph.achievements}, insert: {subgraph & graph.achievements}")
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

class Graph24PointIII(BaseGraph):
    def __init__(self, name, tag):
        super().__init__(name, tag)
        self.type: SubgraphType = SubgraphType.UNKNOWN
    
    def calc_type(self, graph: Graph24PointI):
        subgraph = set(map(lambda x: x.id, self.nodes))
        if subgraph & graph.achievements:
            self.type = SubgraphType.T1
        else:
            self.type = SubgraphType.T0
    
    def convert_to_pyg(self) -> torch_geometric.data.Data:
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