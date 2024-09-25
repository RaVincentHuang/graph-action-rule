
from functools import reduce
from .standard import Edge, Graph24PointI

def combine_nodes(graph: Graph24PointI):
    value_check: dict[tuple[str, str, str], int] = {}
    group = {}
    for node in graph.nodes:
        if node.value not in value_check:
            value_check[node.value] = node.id
        group[node.id] = value_check[node.value]
    
    for edge in graph.edges:
        edge.src, edge.dst = group[edge.src], group[edge.dst]

    unique_edges = set()
    for edge in graph.edges:
        if (edge.src, edge.dst) not in unique_edges:
            unique_edges.add((edge.src, edge.dst))
    graph.edges = [Edge(src, dst) for src, dst in unique_edges]
    graph.nodes = [node for node in graph.nodes if node.id == group[node.id]]
    
def combine_graph(graph1: Graph24PointI, graph2: Graph24PointI) -> Graph24PointI:
    return graph1.combine(graph2)

def combine_graphs(graph_list: list[Graph24PointI]):
    return reduce(combine_graph, graph_list)


