
from calendar import c
from enum import IntEnum
import json
import networkx as nx
from networkx import DiGraph
from networkx import is_empty
from numpy import mat
import torch_geometric.utils

from task.game24 import ASTNode 

import llm.embedding
from llm.tag import Tag
from task.game24 import calc_exprs_4, calc_exprs_3, calc_exprs_2
import sympy
import re

import torch
import torch_geometric.data
import queue
import math

def node_edge2node(node: int, edge: int) -> int:
    if node == 1:
        return 1
    if node in {2, 3, 4} and edge in {1, 2, 3, 4}:
        return (node - 2) * 4 + edge + 1
    return 14

def node2node_edge(node: int) -> tuple[int, int]:
    if node == 1:
        return 1, 1
    if 2 <= node <= 13:
        node -= 1
        edge = (node - 1) % 4 + 1
        node = (node - 1) // 4 + 2
        return (node, edge)
    return (1, 1)

class Node:
    def __init__(self, id, value, acc: float, feature=None):
        self.id = id
        self.value = value # (formula, last_formula, operator)
        self.feature = feature
        self.acc = acc
    
    def calculate_feature(self, task='24point'):
        # Call OpenAI API to get embedding
        text = ""
        match task:
            case '24point':
                text = self.value[1]
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
    
    def load_from_nx(self, nx_graph: nx.DiGraph):
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
        self.node_label = {}
        
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
    
    def achievements_remove_root(self):
        self.achievements.remove(0)
        return self
    
    def achievements_add_root(self):
        self.achievements.add(0)
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
    
    def calc_node_label(self):
        index_check = {}
        for i, node in enumerate(self.nodes):
            index_check[node.id] = i
        steps = {}
        father = {}
        q = queue.Queue()
        steps[self.nodes[0].id] = 0
        q.put(self.nodes[0].id)
        while not q.empty():
            node_id = q.get()
            node = self.nodes[index_check[node_id]]
            try:
                nums = list(map(lambda x: eval(re.search(r'-?\d+\.?\d*\/?-?\d*\.?\d*', x).group()), node.value[1].split('left: ')[-1].split(')')[0].split()))
                # print(nums)
                match nums:
                    case [num1, num2, num3, num4]:
                        if calc_exprs_4(num1, num2, num3, num4):
                            # self.node_label[node_id] = 3
                            self.node_label[node_id] = 1
                        else:
                            # self.node_label[node_id] = -1
                            self.node_label[node_id] = 0
                    case [num1, num2, num3]:
                        if calc_exprs_3(num1, num2, num3):
                            # self.node_label[node_id] = 2
                            self.node_label[node_id] = 1
                        else:
                            # father_label = self.node_label[father[node_id]]
                            # self.node_label[node_id] = father_label + 1 if father_label != -1 else -1
                            self.node_label[node_id] = 0
                    case [num1, num2]:
                        if calc_exprs_2(num1, num2):
                            self.node_label[node_id] = 1
                        else:
                            # father_label = self.node_label[father[node_id]]
                            # self.node_label[node_id] = father_label + 1 if father_label != -1 else -1
                            self.node_label[node_id] = 0
                            
                    case [num1]:
                        if num1 == 24:
                            # self.node_label[node_id] = 0
                            self.node_label[node_id] = 1
                        else:
                            # father_label = self.node_label[father[node_id]]
                            # self.node_label[node_id] = father_label + 1 if father_label != -1 else -1
                            self.node_label[node_id] = 0

                    case _:
                        # father_label = self.node_label[father[node_id]]
                        # self.node_label[node_id] = father_label + 1 if father_label != -1 else -1
                        self.node_label[node_id] = 0
            except Exception as e:
                print(f"Error in node {node.value[1]} ! {e}")
                # father_label = self.node_label[father[node_id]]
                # self.node_label[node_id] = father_label + 1 if father_label != -1 else -1
                self.node_label[node_id] = 0
            
            for edge in self.edges:
                if edge.src == node_id:
                    father[edge.dst] = node_id
                    steps[edge.dst] = steps[node_id] + 1
                    q.put(edge.dst)

        return self
        
            
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
    
    @staticmethod
    def from_ast(roots: list[ASTNode], nums: list[int], index) -> 'Graph24PointI':
        graph = Graph24PointI("24point", index, Tag("calc", "ToT", "24point"))
        graph.load_from_ast(roots, nums)
        return graph
            
    def load_from_ast(self, roots: list[ASTNode], nums: list[int]):
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
                left = " ".join(map(str, nums))
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
            self.nodes.append(Node(id, (formula, last_formula, node.op), acc=0))
            self.edges.append(Edge(parent_id, id))
            
    def re_index(self):
        id_map = {node.id: i for i, node in enumerate(self.nodes)}
        if len(self.node_label):
            self.node_label = {id_map[k]: v for k, v in self.node_label.items()}
        
        for i, node in enumerate(self.nodes):
            node.id = i
        
        for edge in self.edges:
            edge.src = id_map[edge.src]
            edge.dst = id_map[edge.dst]
            

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
    
class Graph24PointIV(BaseGraph):
    def __init__(self, name, tag):
        super().__init__(name, tag)
        self.label = {}
    
    @staticmethod
    def from_nx(nx_graph: nx.DiGraph) -> 'Graph24PointIV':
        graph = Graph24PointIV("24point", Tag("sample", "ToT", "24point"))
        graph.load_from_nx(nx_graph)
        return graph
    
    def calc_label(self, labels: dict[int, int]):
        self.label = {node.id: labels[node.id] for node in self.nodes}
    
    def convert_to_pyg(self) -> torch_geometric.data.HeteroData:
        data = torch_geometric.data.HeteroData()
        
        id_check = {node.id: i for i, node in enumerate(self.nodes)}
        
        def get_nums(s: str) -> list[float]:
            # print(s, "S")
            nums = s.split('left: ')[-1].split(')')[0].split()
            try:
                nums = list(map(lambda x: float(x), nums))
            except:
                try:
                    nums = list(map(lambda x: eval(re.search(r'-?\d+\.?\d*\/?-?\d*\.?\d*', x).group()), nums))
                except:
                    nums = [0.0, 0.0, 0.0, 0.0]
            # print(nums)
            nums = sorted(nums, reverse=True)
            while len(nums) < 4:
                nums.append(0)
            return nums
        
        data['node'].x = torch.tensor([get_nums(node.value[1]) for node in self.nodes], dtype=torch.float)
        
        op_add_edge_index, op_sub_edge_index, op_mul_edge_index, op_div_edge_index = [], [], [], []
        for edge in self.edges:
            match self.nodes[id_check[edge.dst]].value[2]:
                case '+':
                    op_add_edge_index.append([id_check[edge.src], id_check[edge.dst]])
                case '-':
                    op_sub_edge_index.append([id_check[edge.src], id_check[edge.dst]])
                case '*':
                    op_mul_edge_index.append([id_check[edge.src], id_check[edge.dst]])
                case '/':
                    op_div_edge_index.append([id_check[edge.src], id_check[edge.dst]])
        
        data['node', 'add', 'node'].edge_index = torch.tensor(op_add_edge_index, dtype=torch.long).t().contiguous()
        data['node', 'sub', 'node'].edge_index = torch.tensor(op_sub_edge_index, dtype=torch.long).t().contiguous()
        data['node', 'mul', 'node'].edge_index = torch.tensor(op_mul_edge_index, dtype=torch.long).t().contiguous()
        data['node', 'div', 'node'].edge_index = torch.tensor(op_div_edge_index, dtype=torch.long).t().contiguous()
        data.y = torch.tensor([self.label[node.id] if self.label[node.id] != -1 else 7 for node in sorted(self.nodes, key=lambda n: n.id)], dtype=torch.long)
        return data
    
    def __format__(self, format_spec: str) -> str:
        return f"Type: {self.label}\n" + super().__format__(format_spec)
    
    def __str__(self) -> str:
        return f"Type: {self.label}\n" + super().__str__()