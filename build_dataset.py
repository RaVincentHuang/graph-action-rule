import sys
sys.path.append('/home/vincent/graphrule/src')
from graph.store import dataset_build
from llm.tag import Tag
from utils.config import DatasetConfig


json_path = "/home/vincent/graphrule/data/raw/ToT/24point/gpt4o-mini-0.7-p1v5g5"
target_path = "/home/vincent/graphrule/data/subgraph"

dataset_build(json_path, target_path, "24point", Tag("gpt-4o-mini", "ToT", "24point"), DatasetConfig("random_walk", 6400, 10))

