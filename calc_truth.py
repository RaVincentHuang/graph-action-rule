import sys
sys.path.append('/home/vincent/graphrule/src')
from pandas import read_csv

from task.game24 import calc_exprs, ExpressionTreeBuilder, ASTNode
from graph.standrad import Graph24PointI
from llm.tag import Tag
from tqdm import tqdm

path = "/home/vincent/graphrule/data/tasks/24.csv"
graph_path = "/home/vincent/graphrule/data/graph/truth"
tasks = read_csv(path)['Puzzles']

cnt = 0
for task in tqdm(tasks):
    data = list(map(lambda x: int(x), task.split()))
    accs = calc_exprs(*data)
    if not accs:
        continue
    roots = []
    for expr in accs:
        root = ExpressionTreeBuilder().build(expr)
        roots.append(root)
    graph = Graph24PointI("24point", Tag("calc", "ToT", "24point"))
    graph.from_ast(roots, data)
    graph.save_to_json(f"{graph_path}/{graph.name}_{cnt}.json")
    cnt += 1
