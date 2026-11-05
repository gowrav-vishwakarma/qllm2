"""Tools v1 executor (BENCH-TOOL): a calculator run in a bwrap sandbox (no network, read-only /usr, no home, own pid / ipc / uts
namespaces, 2 s timeout). Inside, a whitelist AST evaluator: numbers, + - * / // % **, unary +/-, parentheses; exponents
bounded; nothing else (names, calls, attributes are refused). Returns (ok, value text or error text)."""
from __future__ import annotations

import subprocess

EVAL = r'''
import ast, sys
OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
       ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b, ast.Pow: lambda a, b: a ** b}
def ev(n):
    if isinstance(n, ast.Expression): return ev(n.body)
    if isinstance(n, ast.Constant) and type(n.value) in (int, float): return n.value
    if isinstance(n, ast.UnaryOp) and type(n.op) in (ast.UAdd, ast.USub):
        v = ev(n.operand); return v if isinstance(n.op, ast.UAdd) else -v
    if isinstance(n, ast.BinOp) and type(n.op) in OPS:
        a, b = ev(n.left), ev(n.right)
        if isinstance(n.op, ast.Pow) and (abs(b) > 64 or abs(a) > 1e6): raise ValueError("exponent too large")
        return OPS[type(n.op)](a, b)
    raise ValueError("not arithmetic: " + type(n).__name__)
e = sys.argv[1].replace("^", "**").replace(",", "").replace("$", "").replace("x", "*").replace("×", "*")
v = ev(ast.parse(e, mode="eval"))
v = round(v, 6)
print(int(v) if float(v).is_integer() else v)
'''
BWRAP = ['bwrap', '--ro-bind', '/usr', '/usr', '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
         '--symlink', 'usr/bin', '/bin', '--proc', '/proc', '--dev', '/dev', '--unshare-all', '--die-with-parent', '--new-session',
         '/usr/bin/python3', '-I', '-c', EVAL]


def calculate(expression: str, timeout: float = 2.0):
    expression = str(expression)[:200]
    try:
        r = subprocess.run(BWRAP + [expression], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, 'timeout'
    if r.returncode != 0:
        return False, (r.stderr.strip().splitlines() or ['error'])[-1][:120]
    return True, r.stdout.strip()


if __name__ == '__main__':
    for e in ['48/2', '(5 + 7) * 3', '2**10', '1,200 - 15%4', '__import__("os")', 'open("/etc/passwd")', '9**99999']:
        print(repr(e), calculate(e))
