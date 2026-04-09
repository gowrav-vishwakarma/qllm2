"""AST-based port inference from Python nn.Module code."""
from __future__ import annotations

import ast
from typing import Any, Optional


def _type_annotation_to_str(node: Optional[ast.expr]) -> str:
    if node is None:
        return "Any"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_type_annotation_to_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        base = _type_annotation_to_str(node.value)
        sl = _type_annotation_to_str(node.slice)
        return f"{base}[{sl}]"
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Tuple):
        return ", ".join(_type_annotation_to_str(e) for e in node.elts)
    return "Any"


def _default_to_value(node: ast.expr) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant):
            return -node.operand.value
    if isinstance(node, ast.Name):
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
        return node.id
    return None


def _python_type_to_simple(t: str) -> str:
    t_lower = t.lower().strip()
    if "tensor" in t_lower:
        return "Tensor"
    if t_lower in ("int",):
        return "int"
    if t_lower in ("float",):
        return "float"
    if t_lower in ("bool",):
        return "bool"
    if t_lower in ("str",):
        return "str"
    return t


def infer_ports(code: str) -> dict:
    """Parse module code and extract inputs, outputs, and constructor params."""
    tree = ast.parse(code)

    cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cls = node
            break

    if cls is None:
        raise ValueError("No class definition found in code")

    inputs = []
    outputs = []
    constructor_params = []

    for item in cls.body:
        if isinstance(item, ast.FunctionDef):
            if item.name == "__init__":
                args = item.args
                all_args = args.args[1:]  # skip self
                n_defaults = len(args.defaults)
                n_no_default = len(all_args) - n_defaults

                for i, arg in enumerate(all_args):
                    type_str = _type_annotation_to_str(arg.annotation)
                    simple_type = _python_type_to_simple(type_str)
                    param: dict[str, Any] = {
                        "name": arg.arg,
                        "type": simple_type,
                    }
                    default_idx = i - n_no_default
                    if default_idx >= 0:
                        param["default"] = _default_to_value(args.defaults[default_idx])
                    constructor_params.append(param)

            elif item.name == "forward":
                args = item.args
                all_args = args.args[1:]  # skip self
                n_defaults = len(args.defaults)
                n_no_default = len(all_args) - n_defaults

                for i, arg in enumerate(all_args):
                    type_str = _type_annotation_to_str(arg.annotation)
                    simple_type = _python_type_to_simple(type_str)
                    default_idx = i - n_no_default
                    is_optional = default_idx >= 0
                    inputs.append({
                        "name": arg.arg,
                        "type": simple_type,
                        "optional": is_optional,
                    })

                ret_ann = item.returns
                if ret_ann is not None:
                    ret_type = _type_annotation_to_str(ret_ann)
                    if isinstance(ret_ann, ast.Tuple):
                        for j, elt in enumerate(ret_ann.elts):
                            t = _python_type_to_simple(_type_annotation_to_str(elt))
                            outputs.append({"name": f"out_{j}", "type": t})
                    else:
                        outputs.append({
                            "name": "out",
                            "type": _python_type_to_simple(ret_type),
                        })
                else:
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Return) and stmt.value is not None:
                            if isinstance(stmt.value, ast.Tuple):
                                for j, elt in enumerate(stmt.value.elts):
                                    outputs.append({"name": f"out_{j}", "type": "Tensor"})
                            else:
                                outputs.append({"name": "out", "type": "Tensor"})
                            break

    # Check for PORTS class attribute override
    for item in cls.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "PORTS":
                    try:
                        ports_val = ast.literal_eval(item.value)
                        if isinstance(ports_val, dict):
                            if "inputs" in ports_val:
                                inputs = ports_val["inputs"]
                            if "outputs" in ports_val:
                                outputs = ports_val["outputs"]
                    except (ValueError, TypeError):
                        pass

    if not outputs:
        outputs = [{"name": "out", "type": "Tensor"}]

    return {
        "inputs": inputs,
        "outputs": outputs,
        "constructorParams": constructor_params,
    }
