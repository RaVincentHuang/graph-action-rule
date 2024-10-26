import itertools
import math

# a op c op c op d
def build_expr_no_bra(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} {num2} {op2} {num3} {op3} {num4}"

# (a op b) op c op d
def build_expr_bra1(num1, num2, num3, num4, op1, op2, op3):
    return f"({num1} {op1} {num2}) {op2} {num3} {op3} {num4}"

# (a op b op c) op d
def build_expr_bra2(num1, num2, num3, num4, op1, op2, op3):
    return f"({num1} {op1} {num2} {op2} {num3}) {op3} {num4}"

# a op (b op c) op d
def build_expr_bra3(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} ({num2} {op2} {num3}) {op3} {num4}"

# a op b op (c op d)
def build_expr_bra4(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} {num2} {op2} ({num3} {op3} {num4})"

# a op (b op c op d)
def build_expr_bra5(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} ({num2} {op2} {num3} {op3} {num4})"

# (a op b) op (c op d)
def build_expr_double_bra1(num1, num2, num3, num4, op1, op2, op3):
    return f"({num1} {op1} {num2}) {op2} ({num3} {op3} {num4})"

# a op (b op (c op d))
def build_expr_double_bra2(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} ({num2} {op2} ({num3} {op3} {num4}))"

# a op ((b op c) op d)
def build_expr_double_bra3(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} (({num2} {op2} {num3}) {op3} {num4})"

# (a op (b op c)) op d
def build_expr_double_bra4(num1, num2, num3, num4, op1, op2, op3):
    return f"({num1} {op1} ({num2} {op2} {num3})) {op3} {num4}"

# ((a op b) op c) op d
def build_expr_double_bra5(num1, num2, num3, num4, op1, op2, op3):
    return f"(({num1} {op1} {num2}) {op2} {num3}) {op3} {num4}"

# a op b op c
def build_expr_3_no_bra(num1, num2, num3, op1, op2):
    return f"{num1} {op1} {num2} {op2} {num3}"

# (a op b) op c
def build_expr_3_bra1(num1, num2, num3, op1, op2):
    return f"({num1} {op1} {num2}) {op2} {num3}"

# a op (b op c)
def build_expr_3_bra2(num1, num2, num3, op1, op2):
    return f"{num1} {op1} ({num2} {op2} {num3})"

def build_expr_2(num1, num2, op):
    return f"{num1} {op} {num2}"

def calc_expr(num1, num2, num3, num4, op1, op2, op3, func):
    try:
        if eval(func(num1, num2, num3, num4, op1, op2, op3)) == 24:
            # print(f"{func(num1, num2, num3, num4, op1, op2, op3)} = 24")
            return func(num1, num2, num3, num4, op1, op2, op3)
        elif abs(eval(func(num1, num2, num3, num4, op1, op2, op3)) - 24) < 1e-3:
            return func(num1, num2, num3, num4, op1, op2, op3)
        return False
    except ZeroDivisionError:
        return False
    
def calc_expr_3(num1, num2, num3, op1, op2, func):
    try:
        if eval(func(num1, num2, num3, op1, op2)) == 24:
            return func(num1, num2, num3, op1, op2)
        elif abs(eval(func(num1, num2, num3, op1, op2)) - 24) < 1e-3:
            return func(num1, num2, num3, op1, op2)
        return False
    except ZeroDivisionError:
        return False
    
def calc_expr_2(num1, num2, op, func):
    try:
        if eval(func(num1, num2, op)) == 24:
            return func(num1, num2, op)
        elif abs(eval(func(num1, num2, op)) - 24) < 1e-3:
            return func(num1, num2, op)
        return False
    except ZeroDivisionError:
        return False
    
def calc_exprs(num1, num2, num3, num4):
    acc = []
    nums = (num1, num2, num3, num4)
    ops = ('+', '-', '*', '/')
    for num_perm in itertools.permutations(nums):
        for op_perm in itertools.product(ops, ops, ops):
            for func in [build_expr_no_bra, build_expr_bra1, build_expr_bra2, build_expr_bra3, build_expr_bra4, build_expr_bra5, build_expr_double_bra1, build_expr_double_bra2, build_expr_double_bra3, build_expr_double_bra4, build_expr_double_bra5]:
                expr = calc_expr(*num_perm, *op_perm, func)
                if expr:
                    acc.append(expr)
    
    return acc

def calc_exprs_4(num1, num2, num3, num4):
    nums = (num1, num2, num3, num4)
    ops = ('+', '-', '*', '/')
    for num_perm in itertools.permutations(nums):
        for op_perm in itertools.permutations(ops, 3):
            for func in [build_expr_no_bra, build_expr_bra1, build_expr_bra2, build_expr_bra3, build_expr_bra4, build_expr_bra5, build_expr_double_bra1, build_expr_double_bra2, build_expr_double_bra3, build_expr_double_bra4, build_expr_double_bra5]:
                if calc_expr(*num_perm, *op_perm, func):
                    return True
    return False

def calc_exprs_3(num1, num2, num3):
    nums = (num1, num2, num3)
    ops = ('+', '-', '*', '/')
    for num_perm in itertools.permutations(nums):
        for op_perm in itertools.permutations(ops, 2):
            for func in [build_expr_3_no_bra, build_expr_3_bra1, build_expr_3_bra2]:
                if calc_expr_3(*num_perm, *op_perm, func):
                    return True
    return False

def calc_exprs_2(num1, num2):
    ops = ('+', '-', '*', '/')
    for op in ops:
        if calc_expr_2(num1, num2, op, build_expr_2):
            return True
    return False


import ast

class ASTNode:
    def __init__(self, value, leaf=False):
        self.value = value
        self.left = None
        self.right = None
        self.leaf = leaf

class ExpressionTreeBuilder(ast.NodeVisitor):
    
    @staticmethod
    def op_str(op):
        if isinstance(op, ast.Add):
            return '+'
        elif isinstance(op, ast.Sub):
            return '-'
        elif isinstance(op, ast.Mult):
            return '*'
        elif isinstance(op, ast.Div):
            return '/'
        else:
            return None
    
    def __init__(self):
        self.root = None

    def visit_BinOp(self, node):
        root = ASTNode(self.op_str(node.op))
        root.left = self.visit(node.left)
        root.right = self.visit(node.right)
        return root

    def visit_Num(self, node):
        return ASTNode(node.n, True)

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        return ASTNode(node.id)

    def build(self, expression) -> ASTNode:
        tree = ast.parse(expression, mode='eval')
        self.root = self.visit(tree.body)
        return self.root
    
from .base import Task, DATA_PATH
from llm.prompt.task24 import standard_prompt, cot_prompt, propose_prompt, value_prompt, value_last_step_prompt

import os
import pandas as pd
import re
import sympy

def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, file)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}
            
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value
    
    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        return ''
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        return []
    