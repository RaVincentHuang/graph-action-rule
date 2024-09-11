import sys
sys.path.append('/home/vincent/graphrule/src')
from graph.store import dataset_build, graph_future_calc, test_raw
from llm.tag import Tag
from utils.config import DatasetConfig
import random


json_path = "/home/vincent/graphrule/data/raw/ToT/24point/gpt4o-mini-0.7-p1v5g5_1"
graph_path = "/home/vincent/graphrule/data/graph/gpt4o-mini-0.7-p1v5g5_1"
target_path = "/home/vincent/graphrule/data/subgraph"

node_random = lambda x: random.randint(x - 4, x + 4)

# test_raw(json_path, "24point", Tag("gpt-4o-mini", "ToT", "24point"))
# dataset_build(graph_path, target_path, "24point_2", Tag("gpt-4o-mini", "ToT", "24point"), DatasetConfig("random_walk", 6400, 10, node_random))
graph_future_calc(json_path, graph_path, "24point", Tag("gpt-4o-mini", "ToT", "24point"))
