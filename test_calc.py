import sys
sys.path.append('/home/vincent/graphrule/src')
from pandas import read_csv

from task.game24 import calc_exprs, ExpressionTreeBuilder, ASTNode
from graph.standard import Graph24PointI
from llm.tag import Tag
from tqdm import tqdm
from sympy import simplify

import pandas as pd

 
accs = calc_exprs(1, 1, 11, 11)
unique_exprs = set()
for expr in accs:
    unique_exprs.add(expr)
accs = list(unique_exprs)
print(accs)
