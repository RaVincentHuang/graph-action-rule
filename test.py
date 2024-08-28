from re import sub
import sys

from networkx import DiGraph
sys.path.append('/home/vincent/graphrule/src')

from graph.sample import sample_graph
from graph.standrad import Graph24PointI, Graph24PointII
from llm.tag import Tag

from graph.model import GraphClassifier

import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

index = 4
log_jason_path = f"/home/vincent/graphrule/data/ToT/24point/gpt4o-mini-0.7-p1v3g3/game24-{index}_gpt-4o-mini_0.7_propose1_value3_greedy3.json"
graph = Graph24PointI(f"tot_24point_gpt4o_{index}", Tag("gpt-4o-mini", "ToT", "24point"))
graph.load_from_native_json(log_jason_path)
nx_graph: DiGraph = graph.convert_to_nx()
nx_subgraph = sample_graph(nx_graph, "random_walk", 10)
cnt = 0
subgraph = Graph24PointII(f"tot_24point_gpt4o_{index}_{cnt}", Tag("gpt-4o-mini", "ToT", "24point"))
subgraph.load_from_nx(nx_subgraph)
subgraph.calc_type(graph)

data = subgraph.convert_to_pyg()
model = GraphClassifier(data.x.shape[1], 64, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
out = model(data)
loss = F.cross_entropy(out, data.y)
loss.backward()
optimizer.step()
print(data.y)