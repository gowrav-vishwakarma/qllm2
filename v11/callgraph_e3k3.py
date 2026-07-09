"""Static call-graph generator for the production v11_e3_k3_chat path only.

Prints or writes Mermaid flowcharts for train (fused CE + fused E3) and
inference (parallel prompt + recurrent decode). Abandoned branches (E1, E2,
compete, K-loop fallback) are excluded by allowlist.

Usage:
    uv run python -m v11.callgraph_e3k3
    uv run python -m v11.callgraph_e3k3 --write v11/CALLGRAPH_E3K3.md
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

PRESET_NOTE = (
    "Production preset: `v11_e3_k3_chat` (E3 K=3, `gate_content_aware=True`, "
    "`fused_e3=True`, additive write, head decay). Latest release: round-6b-gate."
)


@dataclass(frozen=True)
class Node:
    """Graph node with optional AST location for self-call extraction."""

    gid: str
    label: str
    file: str | None = None
    cls: str | None = None
    func: str | None = None
    module_func: str | None = None


# ── Node registry (E3 K3 path only) ───────────────────────────────────────────

TRAIN_NODES: dict[str, Node] = {
    "train_main": Node("train_main", "v11.train.main", file="v11/train.py", func="main"),
    "trainer_train": Node(
        "trainer_train", "V7Trainer.train", file="v7/train.py", cls="V7Trainer", func="train"
    ),
    "trainer_epoch": Node(
        "trainer_epoch",
        "V7Trainer.train_epoch",
        file="v7/train.py",
        cls="V7Trainer",
        func="train_epoch",
    ),
    "hidden_to_lm": Node(
        "hidden_to_lm",
        "V11LM._hidden_to_lm",
        file="v11/model.py",
        cls="V11LM",
        func="_hidden_to_lm",
    ),
    "ckpt_block": Node(
        "ckpt_block",
        "V11LM._ckpt_block",
        file="v11/model.py",
        cls="V11LM",
        func="_ckpt_block",
    ),
    "collect_route_aux": Node(
        "collect_route_aux",
        "V11LM._collect_route_aux",
        file="v11/model.py",
        cls="V11LM",
        func="_collect_route_aux",
    ),
    "ce_from_lm": Node(
        "ce_from_lm", "V11LM.ce_from_lm", file="v11/model.py", cls="V11LM", func="ce_from_lm"
    ),
    "fused_ce": Node(
        "fused_ce",
        "fused_linear_cross_entropy",
        file="v11/fused_ce.py",
        module_func="fused_linear_cross_entropy",
    ),
    "block_fwd": Node(
        "block_fwd", "V11Block.forward", file="v11/model.py", cls="V11Block", func="forward"
    ),
    "pam_fwd": Node(
        "pam_fwd", "V11PAMLayer.forward", file="v11/model.py", cls="V11PAMLayer", func="forward"
    ),
    "pam_project": Node(
        "pam_project",
        "V11PAMLayer._project",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_project",
    ),
    "pam_fused": Node(
        "pam_fused",
        "V11PAMLayer._forward_multistate_fused",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_forward_multistate_fused",
    ),
    "pam_phase": Node(
        "pam_phase",
        "V11PAMLayer._phase_and_alpha",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_phase_and_alpha",
    ),
    "pam_routing": Node(
        "pam_routing",
        "V11PAMLayer._routing_input",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_routing_input",
    ),
    "pam_gamma_all": Node(
        "pam_gamma_all",
        "V11PAMLayer._gamma_all_and_vprime",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_gamma_all_and_vprime",
    ),
    "pam_chunk": Node(
        "pam_chunk",
        "V11PAMLayer._fused_chunk_step",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_fused_chunk_step",
    ),
    "fused_decay": Node(
        "fused_decay",
        "fused_decay_matrix",
        file="v11/triton_kernels.py",
        module_func="fused_decay_matrix",
    ),
}

INFER_NODES: dict[str, Node] = {
    "generate": Node("generate", "V11LM.generate"),
    "lm_fwd_prompt": Node("lm_fwd_prompt", "V11LM.forward (prompt, parallel)"),
    "lm_fwd_decode": Node("lm_fwd_decode", "V11LM.forward (decode, seq_len=1)"),
    "sample": Node("sample", "sample next token"),
    "block_fwd": Node(
        "block_fwd", "V11Block.forward", file="v11/model.py", cls="V11Block", func="forward"
    ),
    "pam_fwd": Node(
        "pam_fwd", "V11PAMLayer.forward", file="v11/model.py", cls="V11PAMLayer", func="forward"
    ),
    "pam_fwd_par": Node("pam_fwd_par", "V11PAMLayer.forward (parallel prompt)"),
    "pam_fwd_rec": Node("pam_fwd_rec", "V11PAMLayer.forward (recurrent decode)"),
    "pam_project": Node(
        "pam_project",
        "V11PAMLayer._project",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_project",
    ),
    "pam_fused": Node(
        "pam_fused",
        "V11PAMLayer._forward_multistate_fused",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_forward_multistate_fused",
    ),
    "pam_phase": Node(
        "pam_phase",
        "V11PAMLayer._phase_and_alpha",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_phase_and_alpha",
    ),
    "pam_routing": Node(
        "pam_routing",
        "V11PAMLayer._routing_input",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_routing_input",
    ),
    "pam_gamma_all": Node(
        "pam_gamma_all",
        "V11PAMLayer._gamma_all_and_vprime",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_gamma_all_and_vprime",
    ),
    "pam_chunk": Node(
        "pam_chunk",
        "V11PAMLayer._fused_chunk_step",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_fused_chunk_step",
    ),
    "fused_decay": Node(
        "fused_decay",
        "fused_decay_matrix",
        file="v11/triton_kernels.py",
        module_func="fused_decay_matrix",
    ),
    "pam_recurrent": Node(
        "pam_recurrent",
        "V11PAMLayer._recurrent",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_recurrent",
    ),
    "pam_gamma": Node(
        "pam_gamma",
        "V11PAMLayer._gamma_and_vprime",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_gamma_and_vprime",
    ),
    "pam_recur_step": Node(
        "pam_recur_step",
        "V11PAMLayer._recur_step_additive",
        file="v11/model.py",
        cls="V11PAMLayer",
        func="_recur_step_additive",
    ),
    "phase_rotate": Node("phase_rotate", "phase rotate + sum (K=3)"),
}

# Cross-object edges AST cannot resolve.
TRAIN_EXPLICIT: list[tuple[str, str]] = [
    ("train_main", "trainer_train"),
    ("trainer_train", "trainer_epoch"),
    ("trainer_epoch", "hidden_to_lm"),
    ("trainer_epoch", "ce_from_lm"),
    ("hidden_to_lm", "ckpt_block"),
    ("hidden_to_lm", "block_fwd"),
    ("hidden_to_lm", "collect_route_aux"),
    ("ckpt_block", "block_fwd"),
    ("ce_from_lm", "fused_ce"),
    ("block_fwd", "pam_fwd"),
    ("pam_fwd", "pam_project"),
    ("pam_fwd", "pam_fused"),
]

INFER_EXPLICIT: list[tuple[str, str]] = [
    ("generate", "lm_fwd_prompt"),
    ("lm_fwd_prompt", "block_fwd"),
    ("block_fwd", "pam_fwd_par"),
    ("pam_fwd_par", "pam_project"),
    ("pam_fwd_par", "pam_fused"),
    ("generate", "sample"),
    ("sample", "lm_fwd_decode"),
    ("lm_fwd_decode", "block_fwd"),
    ("block_fwd", "pam_fwd_rec"),
    ("pam_fwd_rec", "pam_project"),
    ("pam_fwd_rec", "pam_recurrent"),
    ("pam_recurrent", "pam_phase"),
    ("pam_recurrent", "pam_gamma"),
    ("pam_recurrent", "pam_recur_step"),
    ("pam_recurrent", "phase_rotate"),
    # Prompt path reuses fused E3 internals (same as train PAM).
    ("pam_fused", "pam_phase"),
    ("pam_fused", "pam_gamma_all"),
    ("pam_fused", "pam_chunk"),
    ("pam_chunk", "fused_decay"),
    ("pam_phase", "pam_routing"),
]

# Map class.method -> node gid for AST-derived self-calls.
_TRAIN_METHOD_TO_GID = {
    (n.cls, n.func): gid
    for gid, n in TRAIN_NODES.items()
    if n.cls and n.func
}
_TRAIN_MODULE_TO_GID = {
    n.module_func: gid for gid, n in TRAIN_NODES.items() if n.module_func
}

_INFER_METHOD_TO_GID = {
    (n.cls, n.func): gid
    for gid, n in INFER_NODES.items()
    if n.cls and n.func
}
_INFER_MODULE_TO_GID = {
    n.module_func: gid for gid, n in INFER_NODES.items() if n.module_func
}


class _SelfCallVisitor(ast.NodeVisitor):
    """Collect self.foo() and module-level foo() calls inside one function."""

    def __init__(
        self,
        cls: str | None,
        method_to_gid: dict[tuple[str, str], str],
        module_to_gid: dict[str, str],
    ):
        self.cls = cls
        self.method_to_gid = method_to_gid
        self.module_to_gid = module_to_gid
        self.callees: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and self.cls
            ):
                key = (self.cls, node.func.attr)
                if key in self.method_to_gid:
                    self.callees.add(self.method_to_gid[key])
        elif isinstance(node.func, ast.Name):
            if node.func.id in self.module_to_gid:
                self.callees.add(self.module_to_gid[node.func.id])
        self.generic_visit(node)


def _parse_file(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_class_method(tree: ast.Module, cls_name: str, func_name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == func_name:
                    return item
    return None


def _find_module_function(tree: ast.Module, func_name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    return None


def _walk_function_calls(
    func_node: ast.FunctionDef,
    cls: str | None,
    method_to_gid: dict[tuple[str, str], str],
    module_to_gid: dict[str, str],
) -> set[str]:
    """Visit function body and any nested defs (e.g. _ckpt_block.run)."""
    callees: set[str] = set()

    def visit_node(node: ast.AST) -> None:
        if isinstance(node, ast.FunctionDef):
            visitor = _SelfCallVisitor(cls, method_to_gid, module_to_gid)
            visitor.visit(node)
            callees.update(visitor.callees)
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    visit_node(child)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                visit_node(child)

    visitor = _SelfCallVisitor(cls, method_to_gid, module_to_gid)
    visitor.visit(func_node)
    callees.update(visitor.callees)
    for child in func_node.body:
        if isinstance(child, ast.FunctionDef):
            visit_node(child)
    return callees


def _ast_edges(
    nodes: dict[str, Node],
    method_to_gid: dict[tuple[str, str], str],
    module_to_gid: dict[str, str],
) -> list[tuple[str, str]]:
    """Derive edges from allowlisted methods via AST self/module calls."""
    trees: dict[str, ast.Module] = {}
    edges: list[tuple[str, str]] = []

    for gid, node in nodes.items():
        if not node.file:
            continue
        rel = node.file
        if rel not in trees:
            trees[rel] = _parse_file(REPO_ROOT / rel)

        tree = trees[rel]
        if node.cls and node.func:
            func_node = _find_class_method(tree, node.cls, node.func)
            if func_node is None:
                continue
            callees = _walk_function_calls(
                func_node, node.cls, method_to_gid, module_to_gid
            )
            for callee_gid in callees:
                edges.append((gid, callee_gid))
        elif node.module_func:
            func_node = _find_module_function(tree, node.module_func)
            if func_node is None:
                continue
            callees = _walk_function_calls(
                func_node, None, method_to_gid, module_to_gid
            )
            for callee_gid in callees:
                edges.append((gid, callee_gid))

    return edges


def _merge_edges(*edge_lists: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for edges in edge_lists:
        for edge in edges:
            if edge not in seen:
                seen.add(edge)
                out.append(edge)
    return out


def _reachable(gids: set[str], edges: list[tuple[str, str]]) -> set[str]:
    """Nodes reachable from roots following edges."""
    reachable_set = set(gids)
    changed = True
    while changed:
        changed = False
        for src, dst in edges:
            if src in reachable_set and dst not in reachable_set:
                reachable_set.add(dst)
                changed = True
    return reachable_set


def _to_mermaid(
    title: str,
    nodes: dict[str, Node],
    edges: list[tuple[str, str]],
    roots: list[str],
) -> str:
    reachable_gids = _reachable(set(roots), edges)
    active_edges = [(s, d) for s, d in edges if s in reachable_gids and d in reachable_gids]

    lines = [f"flowchart TD", f"  subgraph {title.replace(' ', '_')} [{title}]"]
    for gid in sorted(reachable_gids, key=lambda g: (roots.index(g) if g in roots else 99, g)):
        label = nodes[gid].label.replace('"', "'")
        lines.append(f'    {gid}["{label}"]')
    for src, dst in active_edges:
        lines.append(f"    {src} --> {dst}")
    lines.append("  end")
    return "\n".join(lines)


def train_graph() -> str:
    ast_edges = _ast_edges(TRAIN_NODES, _TRAIN_METHOD_TO_GID, _TRAIN_MODULE_TO_GID)
    edges = _merge_edges(TRAIN_EXPLICIT, ast_edges)
    return _to_mermaid("Train fused_ce", TRAIN_NODES, edges, roots=["train_main"])


def infer_graph() -> str:
    ast_edges = _ast_edges(INFER_NODES, _INFER_METHOD_TO_GID, _INFER_MODULE_TO_GID)
    edges = _merge_edges(INFER_EXPLICIT, ast_edges)
    return _to_mermaid("Inference generate", INFER_NODES, edges, roots=["generate"])


def markdown_doc() -> str:
    return f"""# E3 K3 Call Graph (production path only)

{PRESET_NOTE}

Regenerate: `uv run python -m v11.callgraph_e3k3 --write v11/CALLGRAPH_E3K3.md`

## Preset locks

| Flag | Value |
|------|-------|
| preset | `v11_e3_k3_chat` |
| `n_states` | 3 (K=3) |
| `gate_content_aware` | True |
| `fused_e3` | True |
| `decay_mode` | `head` |
| `write_mode` | `additive` |
| `routing_content_aware` | False |
| `state_compete` | False |

## Memory state shape (E3 K=3)

| Tensor | Shape | Notes |
|--------|-------|-------|
| PAM state `S` | `[K, B, H, d, d, 2]` | K=3 superposed d×d notebooks per head |
| After full seq (train) | same | returned from fused path |
| Carried in `generate` | same | updated each decode step |

## Train (fused CE + fused E3 parallel)

```mermaid
{train_graph()}
```

**Dispatch** ([`V11PAMLayer.forward`](model.py)): `state is None` and `seq_len > 1` and `n_states > 1` and `fused_e3` → `_forward_multistate_fused`.

## Inference (`V11LM.generate`)

```mermaid
{infer_graph()}
```

**Two phases:**
1. **Prompt** — full sequence, `states=None` → same parallel fused E3 path as training.
2. **Decode** — one token at a time with `states` → `_recurrent` (K-loop over 3 states).

Excluded from both graphs: E1 per-channel, E2 delta, `_forward_multistate` K-loop fallback, competitive routing, flash-PAM, baseline single-state dual-form.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Write markdown (train + infer graphs) to this path",
    )
    args = parser.parse_args()

    if args.write:
        out = args.write if args.write.is_absolute() else REPO_ROOT / args.write
        out.write_text(markdown_doc(), encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print("## Train\n")
        print(train_graph())
        print("\n## Inference\n")
        print(infer_graph())


if __name__ == "__main__":
    main()
