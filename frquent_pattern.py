import sys
sys.path.append('/home/vincent/graphrule/src')

from graph.standard import Edge, Node
from graph.store import build_gspan_data

def node_label(node: Node) -> int:
    return 1

def edge_label(edge: Edge) -> int:
    return 1

graph_path = "/home/vincent/graphrule/data/graph/gpt4o-mini-0.7-p1v5g5_1_re"
target_path = "/home/vincent/graphrule/data/frequent_pattern"

build_gspan_data(graph_path, target_path, node_label, edge_label)
