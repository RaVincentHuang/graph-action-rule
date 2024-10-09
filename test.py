from re import sub
import sys, os
import random

from networkx import DiGraph

sys.path.append(os.path.abspath('src'))
from graph.store import SubgraphDataset
from sklearn.preprocessing import normalize
from graph.sample import sample_graph
from graph.standard import Graph24PointI, Graph24PointII
from llm.tag import Tag
from llm.embedding import get_embedding
from task.game24 import Game24Task
import tot.bfs, tot.bfs_with_rule

from graph.model import GraphClassifier
from graph.matching import matching
from utils.config import LLMConfig, SearchConfig, SubgraphMatchingConfig


import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine

import ctypes
import time

task = Game24Task("/home/vincent/graphrule/data/tasks/24.csv")
search_config = SearchConfig(
    'propose',
    'value',
    'greedy',
    7,
    3,
    1,
    'cot',
    None,
    True,
)

llm_config = LLMConfig(
    "gpt-4o-mini",
    0.7,
    1000,
    1,
    None,
)

cnt_avg, cnt_any = 0, 0
cnt_avg_rule, cnt_any_rule = 0, 0
for i in range(1002, 1007):
    print(f"Task {i}")
    # print(f"std bfs")
    # ys, info = tot.bfs.solve(search_config, llm_config, task, i, False)
    # infos = [task.test_output(i, y) for y in ys]
    # accs = [info['r'] for info in infos]
    # cnt_avg += sum(accs) / len(accs)
    # cnt_any += any(accs)
    # print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
    # print("--------------------------------------------------")
    print(f"rule bfs")
    ys, info = tot.bfs_with_rule.solve(search_config, llm_config, task, i, False)
    infos = [task.test_output(i, y) for y in ys]
    accs = [info['r'] for info in infos]
    cnt_avg_rule += sum(accs) / len(accs)
    cnt_avg_rule += any(accs)
    print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg_rule, 'cnt_any', cnt_avg_rule, '\n')