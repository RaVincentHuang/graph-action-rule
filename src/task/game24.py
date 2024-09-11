import itertools

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

def calc_expr(num1, num2, num3, num4, op1, op2, op3, func):
    try:
        if eval(func(num1, num2, num3, num4, op1, op2, op3)) == 24:
            # print(f"{func(num1, num2, num3, num4, op1, op2, op3)} = 24")
            return func(num1, num2, num3, num4, op1, op2, op3)
        return False
    except ZeroDivisionError:
        return False
    
def calc_exprs(num1, num2, num3, num4):
    acc = []
    nums = (num1, num2, num3, num4)
    ops = ('+', '-', '*', '/')
    for num_perm in itertools.permutations(nums):
        for op_perm in itertools.permutations(ops, 3):
            for func in [build_expr_no_bra, build_expr_bra1, build_expr_bra2, build_expr_bra3, build_expr_bra4, build_expr_bra5, build_expr_double_bra1, build_expr_double_bra2, build_expr_double_bra3, build_expr_double_bra4, build_expr_double_bra5]:
                expr = calc_expr(*num_perm, *op_perm, func)
                if expr:
                    acc.append(expr)
    
    return acc

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