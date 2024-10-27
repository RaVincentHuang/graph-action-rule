import ctypes
from math import e
import utils
import utils.config
import networkx as nx

libmatching = ctypes.CDLL('build/lib/matching/libmatching.so')

# Define the argument and return types of the matching function
# (char* filter_type, char* order_type, char* engine_type, char* query_graph_file, char* data_graph_file, int order_num, int time_limit) -> int
libmatching.matching.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
libmatching.matching.restype = ctypes.c_int

# TODO add .tmp/

def matching(query: nx.DiGraph, data: nx.DiGraph, config: utils.config.SubgraphMatchingConfig) -> list[set[int]]:
    query_path = ".tmp/query.txt"
    data_path = ".tmp/data.txt"
    emb_path = ".tmp/embs.txt"
    
    # Relabel the nodes of the query and data graphs to have consecutive integers starting from 0
    query = nx.relabel_nodes(query, {node: i for i, node in enumerate(query.nodes())})
    data = nx.relabel_nodes(data, {node: i for i, node in enumerate(data.nodes())})
    
    # Write the query graph to a file
    with open(query_path, 'w') as f:
        f.write(f"t {query.number_of_nodes()} {query.number_of_edges()}\n")
        for node in query.nodes():
            f.write(f"v {node} {query.nodes[node]['label']} {query.degree(node)}\n")
        for u, v in query.edges():
            f.write(f"e {u} {v}\n")

    # Write the data graph to a file
    with open(data_path, 'w') as f:
        f.write(f"t {data.number_of_nodes()} {data.number_of_edges()}\n")
        for node in data.nodes():
            f.write(f"v {node} {data.nodes[node]['label']} {data.degree(node)}\n")
        for u, v in data.edges():
            f.write(f"e {u} {v}\n")
    
    query_path = ctypes.c_char_p(query_path.encode('utf-8'))
    data_path = ctypes.c_char_p(data_path.encode('utf-8'))

    try:
        emb_cnt = libmatching.matching(query_path, data_path, *config.get_ctype())
    except Exception as e:
        print(e, f"Error in {query}, {data}")
        emb_cnt = 0
    
    result = []
    with open(emb_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= emb_cnt:
                break
            nodes = set(map(int, line.strip().split()))
            result.append(nodes)
    # Remove duplicate sets from the result
    result = [set(x) for x in {frozenset(item) for item in result}]
    return result
