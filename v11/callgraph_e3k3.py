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
class Annot:
    """Human-readable note attached to a call-graph node."""

    phase: str       # e.g. fwd+bwd, infer, setup, custom-bwd
    origin: str      # PyTorch autograd, custom code, Triton kernel, ...
    does: str        # what this call does
    why: str = ""    # design reason (optional)


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

# ── Per-node annotations (phase, origin, what, why) ─────────────────────────

TRAIN_ANNOTATIONS: dict[str, Annot] = {
    "train_main": Annot(
        "setup", "custom script",
        "Parse CLI, build `V11LM`, construct `V7Trainer`, start training loop.",
    ),
    "trainer_train": Annot(
        "setup", "custom (`v7/train.py`)",
        "Print run info; loop epochs calling `train_epoch` + validation checkpoints.",
    ),
    "trainer_epoch": Annot(
        "fwd then bwd", "custom trainer + PyTorch autograd",
        "One training epoch: for each batch run forward (hidden + loss), then "
        "`loss.backward()`, clip grads, `optimizer.step()`.",
        "Orchestrates the two-branch fused-CE path: calls `_hidden_fn` first, "
        "then `ce_from_lm` on the returned hidden — not a single `model.forward()`.",
    ),
    "hidden_to_lm": Annot(
        "fwd+bwd", "custom `V11LM` (PyTorch modules inside)",
        "Forward the **model stack only**: embed → 16× `V11Block` → output norm → "
        "lm_head_proj/norm. Returns pre-logit complex hidden `lm` `[B,T,dim,2]` "
        "(NOT vocab logits).",
        "Split from CE so we never build `[B*T, vocab]` logits (~4 GB). "
        "This half is `torch.compile`'d in production (`--compile --fused_ce`).",
    ),
    "ckpt_block": Annot(
        "fwd+bwd (recompute)", "PyTorch `grad_checkpoint`",
        "Wraps one `V11Block` in activation checkpointing: forward runs normally; "
        "backward **re-runs** the block forward to recompute activations instead of "
        "storing them — trades compute for VRAM.",
        "Optional when `gradient_checkpointing=True` (off in production `--no_grad_ckpt`).",
    ),
    "collect_route_aux": Annot(
        "fwd only (aux scalar)", "custom",
        "Sums optional E3 routing balance aux loss from PAM layers. "
        "With `state_compete=False` (production) this is effectively zero.",
    ),
    "ce_from_lm": Annot(
        "fwd+bwd", "custom `V11LM` method",
        "Takes `lm` hidden from `_hidden_to_lm`, folds tied complex head into one "
        "real matmul `H @ W.T`, calls chunked CE. Returns scalar loss.",
        "Second branch of fused CE: head matmul + softmax are fused/chunked here "
        "instead of inside `forward()`. Keeps logits tensor off GPU.",
    ),
    "fused_ce": Annot(
        "fwd + custom bwd", "custom `torch.autograd.Function` (`v11/fused_ce.py`)",
        "**Forward:** chunk over token rows, compute `logits_chunk = H_chunk @ W.T`, "
        "accumulate `F.cross_entropy` per chunk (uses PyTorch CE on small chunks). "
        "**Backward:** recompute each chunk's logits/softmax, accumulate "
        "`grad_hidden` and `grad_weight` — never stores full `[N,V]` activations.",
        "Not a PyTorch built-in. Custom memory optimization; math is exact vs "
        "materializing full logits + `F.cross_entropy`.",
    ),
    "block_fwd": Annot(
        "fwd+bwd", "PyTorch `nn.Module` (custom `V11Block`)",
        "One transformer-ish block: pre-norm → CGU (channel mixing, no sequence "
        "state) → residual → pre-norm → PAM (sequence memory) → residual.",
    ),
    "pam_fwd": Annot(
        "fwd+bwd", "custom `V11PAMLayer`",
        "PAM entry: `_project` QKV+RoPE, then dispatch. **Train path:** "
        "`state=None`, `seq_len>1` → `_forward_multistate_fused` (parallel chunked).",
    ),
    "pam_project": Annot(
        "fwd+bwd", "PyTorch Linear + custom complex ops",
        "Fused QKV projection, optional RoPE on Q/K, optional QK norm. "
        "Shared by train and infer.",
    ),
    "pam_fused": Annot(
        "fwd+bwd", "custom E3 implementation",
        "Parallel training form for K=3: compute phase routing + all-state decay "
        "once, loop sequence in chunks of 256, call `_fused_chunk_step` per chunk. "
        "O(T) work via chunked matmuls, not O(T²) attention.",
    ),
    "pam_phase": Annot(
        "fwd+bwd", "custom (`phase_proj` Linear)",
        "E3 retrieval: `phase_proj(cabs(x))` → angle φ per (head, state). "
        "At read time each state's output is rotated by `e^{iφ}` before summing.",
    ),
    "pam_routing": Annot(
        "fwd+bwd", "custom helper",
        "Builds router input: `cabs(x)` (magnitude) in production; "
        "`to_real_concat(x)` only if `routing_content_aware=True` (off).",
    ),
    "pam_gamma_all": Annot(
        "fwd+bwd", "custom GSP + decay",
        "All K=3 decay rates + write-protect gate in one pass: `dt_proj` + "
        "`protect_gate` (content-aware GSP). Returns `[K,B,H,T]` decay and "
        "protected values.",
    ),
    "pam_chunk": Annot(
        "fwd+bwd", "custom matmuls",
        "One 256-token chunk: causal dual-form read + write for all K states "
        "collapsed into batched matmuls; updates memory state `[K,B,H,d,d,2]`.",
    ),
    "fused_decay": Annot(
        "fwd+bwd", "Triton kernel (`v11/triton_kernels.py`)",
        "Build causal decay matrix from per-step γ via fused log-cumsum-exp. "
        "Triton `autograd.Function` with PyTorch fallback. Differentiable.",
    ),
}

INFER_ANNOTATIONS: dict[str, Annot] = {
    "generate": Annot(
        "infer only", "custom `V11LM.generate` (`@torch.no_grad`)",
        "Autoregressive loop: process prompt once (parallel), sample tokens, "
        "then decode one token at a time carrying PAM states.",
    ),
    "lm_fwd_prompt": Annot(
        "infer only", "custom `V11LM.forward`",
        "First call: full prompt, `states=None`, `seq_len>1` → parallel fused E3 "
        "(same PAM path as training, but no grad).",
    ),
    "lm_fwd_decode": Annot(
        "infer only", "custom `V11LM.forward`",
        "Per new token: `seq_len=1`, `states` passed → recurrent PAM. "
        "O(1) per token in sequence length.",
    ),
    "sample": Annot(
        "infer only", "PyTorch sampling",
        "Temperature / top-k / top-p / repetition penalty on last logits; "
        "`torch.multinomial` picks next token. No backward.",
    ),
    "block_fwd": Annot(
        "infer only", "PyTorch `nn.Module` eval mode",
        "Same block as train but dropout off, no grad. CGU + PAM per layer.",
    ),
    "pam_fwd_par": Annot(
        "infer only", "custom dispatch (parallel branch)",
        "Prompt pass: `state=None`, `seq_len>1` → `_forward_multistate_fused`.",
    ),
    "pam_fwd_rec": Annot(
        "infer only", "custom dispatch (recurrent branch)",
        "Decode pass: `state` provided or `seq_len==1` → `_recurrent`.",
    ),
    "pam_project": TRAIN_ANNOTATIONS["pam_project"],
    "pam_fused": Annot(
        "infer only", "custom E3 (parallel)",
        "Prompt processing only — identical math to training fused path, no backward.",
    ),
    "pam_phase": TRAIN_ANNOTATIONS["pam_phase"],
    "pam_routing": TRAIN_ANNOTATIONS["pam_routing"],
    "pam_gamma_all": TRAIN_ANNOTATIONS["pam_gamma_all"],
    "pam_chunk": Annot(
        "infer only", "custom matmuls",
        "Chunk step for prompt; updates state carried into decode loop.",
    ),
    "fused_decay": Annot(
        "infer only", "Triton kernel",
        "Same decay-matrix kernel as train; runs under `torch.no_grad()`.",
    ),
    "pam_recurrent": Annot(
        "infer only", "custom E3 recurrent",
        "Token-by-token loop: for each of K=3 states run decay/gate, "
        "`_recur_step_additive`, rotate by phase, sum outputs. "
        "Updates `states` for next step.",
        "Required at decode — parallel form needs full sequence; generation is one token at a time.",
    ),
    "pam_gamma": Annot(
        "infer only", "custom GSP + decay",
        "Per-token, per-state decay + write-protect (one state at a time in K-loop).",
    ),
    "pam_recur_step": Annot(
        "infer only", "custom PAM math",
        "One additive memory step: read `S @ q`, write outer-product `v ⊗ k*` "
        "with decay. O(d²) per head, constant in context length.",
    ),
    "phase_rotate": Annot(
        "infer only", "custom complex ops",
        "Multiply each state's read output by `e^{iφ}` (routing weight × cos/sin) "
        "and sum over K=3 — E3 interference at retrieval.",
    ),
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


def _mermaid_label(label: str, gid: str, annotations: dict[str, Annot]) -> str:
    ann = annotations.get(gid)
    if ann is None:
        return label.replace('"', "'")
    tag = f"{ann.phase} | {ann.origin}"
    return f"{label}<br/>{tag}".replace('"', "'")


def _to_mermaid(
    title: str,
    nodes: dict[str, Node],
    edges: list[tuple[str, str]],
    roots: list[str],
    annotations: dict[str, Annot],
) -> str:
    reachable_gids = _reachable(set(roots), edges)
    active_edges = [(s, d) for s, d in edges if s in reachable_gids and d in reachable_gids]

    lines = [f"flowchart TD", f"  subgraph {title.replace(' ', '_')} [{title}]"]
    for gid in sorted(reachable_gids, key=lambda g: (roots.index(g) if g in roots else 99, g)):
        label = _mermaid_label(nodes[gid].label, gid, annotations)
        lines.append(f'    {gid}["{label}"]')
    for src, dst in active_edges:
        lines.append(f"    {src} --> {dst}")
    lines.append("  end")
    return "\n".join(lines)


def _annotation_table(
    nodes: dict[str, Node],
    edges: list[tuple[str, str]],
    roots: list[str],
    annotations: dict[str, Annot],
) -> str:
    reachable_gids = _reachable(set(roots), edges)
    order = sorted(reachable_gids, key=lambda g: (roots.index(g) if g in roots else 99, g))
    lines = [
        "| Node | Phase | Origin | What it does | Why / notes |",
        "|------|-------|--------|--------------|-------------|",
    ]
    for gid in order:
        node = nodes[gid]
        ann = annotations.get(gid)
        if ann is None:
            lines.append(f"| `{node.label}` | — | — | — | — |")
            continue
        why = ann.why if ann.why else "—"
        lines.append(
            f"| `{node.label}` | {ann.phase} | {ann.origin} | {ann.does} | {why} |"
        )
    return "\n".join(lines)


def train_graph() -> str:
    ast_edges = _ast_edges(TRAIN_NODES, _TRAIN_METHOD_TO_GID, _TRAIN_MODULE_TO_GID)
    edges = _merge_edges(TRAIN_EXPLICIT, ast_edges)
    return _to_mermaid("Train fused_ce", TRAIN_NODES, edges, roots=["train_main"], annotations=TRAIN_ANNOTATIONS)


def infer_graph() -> str:
    ast_edges = _ast_edges(INFER_NODES, _INFER_METHOD_TO_GID, _INFER_MODULE_TO_GID)
    edges = _merge_edges(INFER_EXPLICIT, ast_edges)
    return _to_mermaid("Inference generate", INFER_NODES, edges, roots=["generate"], annotations=INFER_ANNOTATIONS)


def _fused_ce_explainer() -> str:
    return """## Why two branches: `_hidden_to_lm` and `ce_from_lm`?

Production training (`--fused_ce`) does **not** call `V11LM.forward()`. Instead `V7Trainer.train_epoch` runs:

```
loss = ce_from_lm( _hidden_to_lm(input_ids), labels )
loss.backward()
```

| Branch | Returns | Built by | PyTorch? |
|--------|---------|----------|----------|
| `_hidden_to_lm` | Pre-logit hidden `lm` `[B,T,dim,2]` | Custom `V11LM` method | Uses PyTorch modules (`nn.Linear`, etc.) with normal autograd |
| `ce_from_lm` → `fused_linear_cross_entropy` | Scalar CE loss | Custom `v11/fused_ce.py` | **Custom** `autograd.Function`; uses `F.cross_entropy` only on small chunks |

**Why split?** The tied LM head is `logits = lm_r @ E_r.T + lm_i @ E_i.T` → one matmul to `[B*T, vocab]`. With vocab ≈ 50k and B×T ≈ 37k that tensor is ~4 GB, plus softmax and its grad. Materializing it dominates VRAM.

**Forward pass:** `_hidden_to_lm` runs the stack (optionally `torch.compile`'d). `fused_linear_cross_entropy` loops token-rows in chunks (default 4096), computes chunk logits, accumulates CE — peak memory O(chunk×vocab) not O(B×T×vocab).

**Backward pass:** `loss.backward()` first hits `_FusedLinearCE.backward` (custom — recomputes chunk logits/softmax, writes `grad_hidden`, `grad_weight`). Then autograd flows into `_hidden_to_lm` and the whole stack (blocks, PAM, embed). PAM uses PyTorch autograd through matmuls; `fused_decay_matrix` uses a Triton `autograd.Function`.

**Not PyTorch built-ins:** `_hidden_to_lm`, `ce_from_lm`, `fused_linear_cross_entropy`, E3 PAM paths, Triton kernels — all project code. PyTorch provides the autograd engine, `nn.Module`, `F.cross_entropy` (per chunk), `grad_checkpoint`, `torch.compile`, optimizer.

**Alternative (not used in production):** `V11LM.forward()` → full `logits [B,T,V]` → `F.cross_entropy`. Simpler but ~4 GB head memory."""


def _train_step_explainer() -> str:
    return """## One training step (forward → backward)

```mermaid
sequenceDiagram
    participant TE as train_epoch
    participant H as _hidden_to_lm
    participant B as V11Block x16
    participant P as PAM fused E3
    participant CE as fused_linear_cross_entropy
    participant AD as PyTorch autograd

    TE->>H: forward(input_ids)
    H->>B: block.forward (x16)
    B->>P: pam.forward parallel
    P-->>H: lm hidden [B,T,dim,2]
    TE->>CE: ce_from_lm(lm, labels)
    CE-->>TE: scalar loss
    TE->>AD: loss.backward()
    AD->>CE: custom backward (chunked grad to lm + embed)
    AD->>P: standard autograd through matmuls
    AD->>B: standard autograd
    TE->>TE: clip_grad + optimizer.step()
```"""


def markdown_doc() -> str:
    train_edges = _merge_edges(
        TRAIN_EXPLICIT,
        _ast_edges(TRAIN_NODES, _TRAIN_METHOD_TO_GID, _TRAIN_MODULE_TO_GID),
    )
    infer_edges = _merge_edges(
        INFER_EXPLICIT,
        _ast_edges(INFER_NODES, _INFER_METHOD_TO_GID, _INFER_MODULE_TO_GID),
    )
    return f"""# E3 K3 Call Graph (production path only)

{PRESET_NOTE}

Regenerate: `uv run python -m v11.callgraph_e3k3 --write v11/CALLGRAPH_E3K3.md`

{_fused_ce_explainer()}

{_train_step_explainer()}

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

## Train call graph (fused CE + fused E3 parallel)

Node subtitles in the diagram: **phase | origin** (e.g. `fwd+bwd | custom`).

```mermaid
{train_graph()}
```

**PAM dispatch** ([`V11PAMLayer.forward`](model.py)): `state is None` and `seq_len > 1` and `n_states > 1` and `fused_e3` → `_forward_multistate_fused`.

### Train node glossary

{_annotation_table(TRAIN_NODES, train_edges, ["train_main"], TRAIN_ANNOTATIONS)}

## Inference call graph (`V11LM.generate`)

```mermaid
{infer_graph()}
```

**Two phases (both inference-only, no backward):**
1. **Prompt** — full sequence, `states=None` → parallel fused E3 (same PAM math as training).
2. **Decode** — one token at a time with `states` → `_recurrent` (K-loop over 3 states). Parallel form cannot run here because each step only sees one new token.

### Inference node glossary

{_annotation_table(INFER_NODES, infer_edges, ["generate"], INFER_ANNOTATIONS)}

---

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
