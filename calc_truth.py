import sys
sys.path.append('/home/vincent/graphrule/src')
from pandas import read_csv

from task.game24 import calc_exprs, ExpressionTreeBuilder, ASTNode
from graph.standard import Graph24PointI
from llm.tag import Tag
from tqdm import tqdm
from sympy import simplify

import pandas as pd

path = "/home/vincent/graphrule/data/tasks/24.csv"
graph_path = "/home/vincent/graphrule/data/graph/truth"
tasks = []

index_check = {}
task_path = "/home/vincent/graphrule/data/tasks/24.csv"
for chunk in pd.read_csv(task_path, usecols=['Rank', 'Puzzles'], chunksize=1):
    for _, row in chunk.iterrows():
        rank, task = row['Rank'], row['Puzzles']
        index_check[task] = rank
        tasks.append(task)

for task in tqdm(tasks):
    index = index_check[task]
    data = list(map(lambda x: int(x), task.split()))
    accs = calc_exprs(*data)
    if not accs:
        continue
    unique_exprs = set()
    for expr in accs:
        unique_exprs.add(expr)
    accs = list(unique_exprs)
    roots = []
    for expr in accs:
        root = ExpressionTreeBuilder().build(expr)
        roots.append(root)
        
    graph = Graph24PointI.from_ast(roots, data, index)
    graph.save_to_json(f"{graph_path}/{graph.name}_{index}.json")
