"""V12 module registry + dependency resolver.

A trained layer-group is a shippable **module**: its own blocks plus a **card**
declaring identity (module_id@version), provenance, geometry, and an ordered
``requires`` list. Modules stack in dependency order into one inference
checkpoint (see v12/pack.py), which is the foundation for a later marketplace.

Dependency modes:
  - ``prelayer``: the module needs specific frozen substrate module(s) stacked
    beneath it. The resolver includes them and verifies the module's recorded
    ``substrate_hash`` (hash of the concatenated substrate at train time) against
    the resolved stack.
  - ``finetuned``: the module absorbed its base into its own weights, so it is
    standalone (not stacked); the dependency is recorded as lineage only.

Cards are stored BOTH as a sidecar ``<ckpt>.card.json`` next to the checkpoint
AND embedded in the checkpoint ``config`` under ``module_card`` (self-contained,
survives file moves). Reads prefer the sidecar and fall back to the embedded copy.

This is a LOCAL package-manager-style store + resolver (semver-lite version
solving, topological stacking, conflict detection). No network fetch yet.

Note on hashing: V12 registers every buffer with ``persistent=False``, so a
checkpoint's ``state_dict`` contains only parameters. That lets us reproduce
``V12LM._hash_blocks`` from a saved state dict for substrate verification.
"""

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch


# ── Card schema ──────────────────────────────────────────────────────────────

@dataclass
class Requirement:
    """A dependency of a module on another module."""
    module_id: str
    version_spec: str = "*"          # "*", "1.0", ">=1.0", ">=1.0,<2.0"
    mode: str = "prelayer"           # prelayer | finetuned
    # Optional pin: expected self_hash of the resolved dependency module.
    pinned_hash: Optional[str] = None

    @staticmethod
    def from_dict(d: dict) -> "Requirement":
        return Requirement(
            module_id=d["module_id"],
            version_spec=d.get("version_spec", "*"),
            mode=d.get("mode", "prelayer"),
            pinned_hash=d.get("pinned_hash"),
        )


@dataclass
class ModuleCard:
    """Identity + dependency metadata for one module (layer group)."""
    module_id: str
    version: str
    role: str = "group"              # base | group
    skill: Optional[str] = None
    group_id: Optional[str] = None
    attach_mode: str = "sequential"  # sequential | moe
    provenance: str = ""
    description: str = ""
    created: str = ""
    # Geometry (must be consistent across a composed stack).
    dim: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    # This module's OWN blocks.
    n_layers: int = 0
    layer_specs: List[dict] = field(default_factory=list)
    # hash of this module's own blocks (identity) / of the substrate beneath it.
    self_hash: Optional[str] = None
    substrate_hash: Optional[str] = None
    requires: List[Requirement] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["requires"] = [asdict(r) for r in self.requires]
        return d

    @staticmethod
    def from_dict(d: dict) -> "ModuleCard":
        d = dict(d)
        reqs = [Requirement.from_dict(r) for r in d.pop("requires", [])]
        known = {f for f in ModuleCard.__dataclass_fields__ if f != "requires"}
        return ModuleCard(requires=reqs, **{k: v for k, v in d.items() if k in known})


# ── Card I/O (sidecar + embedded) ────────────────────────────────────────────

def sidecar_path(ckpt_path: str) -> str:
    return str(ckpt_path) + ".card.json"


def write_card(ckpt_path: str, card: ModuleCard, *, embed: bool = True):
    """Write the sidecar card and (optionally) embed a copy in the checkpoint."""
    with open(sidecar_path(ckpt_path), "w") as f:
        json.dump(card.to_dict(), f, indent=2)
    if embed:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        cfg["module_card"] = card.to_dict()
        ckpt["config"] = cfg
        tmp = ckpt_path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, ckpt_path)


def read_card(ckpt_path: str) -> Optional[ModuleCard]:
    """Read a card: prefer the sidecar, fall back to the embedded copy."""
    sc = sidecar_path(ckpt_path)
    if os.path.exists(sc):
        with open(sc) as f:
            return ModuleCard.from_dict(json.load(f))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    embedded = (ckpt.get("config") or {}).get("module_card")
    return ModuleCard.from_dict(embedded) if embedded else None


# ── Block hashing (mirrors V12LM._hash_blocks from a saved state dict) ────────

def hash_block_states(sources: List[Tuple[dict, int]]) -> str:
    """SHA-256 over blocks given as (state_dict, block_index) in stack order.

    Reproduces ``V12LM._hash_blocks``: per block, sort params by their
    block-relative name, hash name then raw bytes. state_dict holds only
    parameters (all V12 buffers are non-persistent), so this matches exactly.
    """
    h = hashlib.sha256()
    for state, idx in sources:
        prefix = f"blocks.{idx}."
        rel = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        for name in sorted(rel):
            h.update(name.encode())
            h.update(rel[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def hash_shared_states(state: dict) -> str:
    """SHA-256 over shared-prefix tensors (embed, norms, LM head)."""
    from v12.pack import _SHARED_PREFIXES
    h = hashlib.sha256()
    for key in sorted(state):
        if not any(key.startswith(p) for p in _SHARED_PREFIXES):
            continue
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def hash_substrate_prefix(state: dict, block_indices) -> str:
    """Hash of frozen blocks beneath a group + shared params (full substrate contract)."""
    h = hashlib.sha256()
    idxs = list(block_indices)
    if idxs:
        h.update(hash_block_states([(state, i) for i in idxs]).encode())
    h.update(hash_shared_states(state).encode())
    return h.hexdigest()


# ── semver-lite ───────────────────────────────────────────────────────────────

def version_tuple(v: str) -> Tuple[int, ...]:
    parts = re.split(r"[._]", str(v).strip())
    out = []
    for p in parts:
        m = re.match(r"\d+", p)
        out.append(int(m.group()) if m else 0)
    return tuple(out) or (0,)


def _cmp(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    n = max(len(a), len(b))
    a = a + (0,) * (n - len(a))
    b = b + (0,) * (n - len(b))
    return (a > b) - (a < b)


def satisfies(version: str, spec: str) -> bool:
    """True if ``version`` satisfies a comma-separated constraint spec.

    Supports: "*"/"" (any), bare/"==x" (exact-prefix match), ">=", ">", "<=",
    "<", "!=". Multiple clauses are AND-ed (e.g. ">=1.0,<2.0").
    """
    spec = (spec or "*").strip()
    if spec in ("*", ""):
        return True
    ver = version_tuple(version)
    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        m = re.match(r"^(>=|<=|==|!=|>|<)?\s*(.+)$", clause)
        op, val = m.group(1) or "==", m.group(2).strip()
        c = _cmp(ver, version_tuple(val))
        ok = {
            ">=": c >= 0, "<=": c <= 0, ">": c > 0, "<": c < 0,
            "!=": c != 0, "==": c == 0,
        }[op]
        if not ok:
            return False
    return True


# ── Registry ──────────────────────────────────────────────────────────────────

class Registry:
    """Local module store: root/<module_id>/<version>/model.pt (+ .card.json)."""

    def __init__(self, root: str = "v12_registry"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.index_path = os.path.join(self.root, "index.json")
        self._index = self._load_index()

    def _load_index(self) -> dict:
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self):
        tmp = self.index_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._index, f, indent=2)
        os.replace(tmp, self.index_path)

    def module_dir(self, module_id: str, version: str) -> str:
        return os.path.join(self.root, module_id, str(version))

    def add(self, ckpt: dict, card: ModuleCard, *, overwrite: bool = False) -> str:
        """Store a module checkpoint dict + card; register it. Returns ckpt path."""
        mid, ver = card.module_id, card.version
        entry = self._index.get(mid, {}).get(ver)
        if entry and not overwrite:
            raise ValueError(f"module {mid}@{ver} already registered (use overwrite=True)")
        d = self.module_dir(mid, ver)
        os.makedirs(d, exist_ok=True)
        ckpt_path = os.path.join(d, "model.pt")
        cfg = ckpt.get("config", {})
        cfg["module_card"] = card.to_dict()
        ckpt["config"] = cfg
        torch.save(ckpt, ckpt_path)
        write_card(ckpt_path, card, embed=False)  # ckpt already embeds; just sidecar
        self._index.setdefault(mid, {})[ver] = {
            "ckpt": os.path.relpath(ckpt_path, self.root),
            "card": os.path.relpath(sidecar_path(ckpt_path), self.root),
            "role": card.role,
            "skill": card.skill,
            "created": card.created or datetime.now().isoformat(timespec="seconds"),
        }
        self._save_index()
        return ckpt_path

    def add_from_file(self, src_ckpt_path: str, card: ModuleCard, *, overwrite=False) -> str:
        ckpt = torch.load(src_ckpt_path, map_location="cpu", weights_only=False)
        return self.add(ckpt, card, overwrite=overwrite)

    def versions(self, module_id: str) -> List[str]:
        return sorted(self._index.get(module_id, {}), key=version_tuple)

    def get(self, module_id: str, version: str) -> Tuple[str, ModuleCard]:
        entry = self._index.get(module_id, {}).get(version)
        if not entry:
            raise KeyError(f"module {module_id}@{version} not in registry")
        ckpt_path = os.path.join(self.root, entry["ckpt"])
        return ckpt_path, read_card(ckpt_path)

    def find(self, module_id: str, spec: str = "*") -> Tuple[str, str, ModuleCard]:
        """Return (version, ckpt_path, card) for the highest version satisfying spec."""
        cands = [v for v in self.versions(module_id) if satisfies(v, spec)]
        if not cands:
            have = self.versions(module_id)
            raise KeyError(
                f"no version of '{module_id}' satisfies '{spec}' (have: {have or 'none'})"
            )
        ver = cands[-1]
        ckpt_path, card = self.get(module_id, ver)
        return ver, ckpt_path, card

    def list(self) -> List[Tuple[str, str, dict]]:
        out = []
        for mid, vers in sorted(self._index.items()):
            for ver, entry in sorted(vers.items(), key=lambda kv: version_tuple(kv[0])):
                out.append((mid, ver, entry))
        return out


# ── Dependency resolver ───────────────────────────────────────────────────────

@dataclass
class ModuleRef:
    module_id: str
    version: str
    ckpt_path: str
    card: ModuleCard
    role: str  # base | prelayer

    @property
    def n_layers(self) -> int:
        return self.card.n_layers


@dataclass
class ResolvedPlan:
    stack: List[ModuleRef]          # bottom-to-top (base first)
    lineage: List[dict]             # finetuned deps (recorded, not stacked)
    report: dict                    # verification info + warnings

    def order(self) -> List[str]:
        return [f"{r.module_id}@{r.version}" for r in self.stack]


def _solve(seed: Dict[str, List[Tuple[str, str]]], registry: "Registry",
           max_iter: int = 100):
    """Collect prelayer constraints across the graph and pick a version per module.

    ``seed`` maps module_id -> list of (version_spec, source). Returns
    (chosen: {id: version}, cards: {id: card}, edges: {id: [dep_id...]},
    lineage: [ {module_id, version_spec, from, ...} ]). Raises on conflict.
    """
    constraints: Dict[str, List[Tuple[str, str]]] = {k: list(v) for k, v in seed.items()}
    chosen: Dict[str, str] = {}
    cards: Dict[str, ModuleCard] = {}
    edges: Dict[str, List[str]] = {}
    lineage: List[dict] = []

    for _ in range(max_iter):
        changed = False
        for mid, clauses in list(constraints.items()):
            specs = [c for c, _ in clauses]
            versions = registry.versions(mid)
            cands = [v for v in versions if all(satisfies(v, s) for s in specs)]
            if not cands:
                raise ValueError(
                    f"version conflict for '{mid}': no version satisfies all of "
                    f"{[f'{s} (from {src})' for s, src in clauses]}; have {versions or 'none'}"
                )
            best = cands[-1]
            if chosen.get(mid) == best:
                continue
            chosen[mid] = best
            changed = True
            _, card = registry.get(mid, best)
            cards[mid] = card
            deps = []
            for req in card.requires:
                if req.mode == "finetuned":
                    lineage.append({
                        "from": f"{mid}@{best}", "module_id": req.module_id,
                        "version_spec": req.version_spec, "mode": "finetuned",
                    })
                    continue
                constraints.setdefault(req.module_id, []).append((req.version_spec, mid))
                deps.append(req.module_id)
            edges[mid] = deps
        if not changed:
            break
    else:
        raise ValueError("dependency resolution did not converge (cycle or churn?)")
    return chosen, cards, edges, lineage


def _toposort(target_id: str, edges: Dict[str, List[str]]) -> List[str]:
    """Post-order DFS => dependencies before dependents. Detects cycles."""
    order: List[str] = []
    seen, stack = set(), set()

    def visit(node):
        if node in seen:
            return
        if node in stack:
            raise ValueError(f"dependency cycle detected at '{node}'")
        stack.add(node)
        for dep in edges.get(node, []):
            visit(dep)
        stack.discard(node)
        seen.add(node)
        order.append(node)

    visit(target_id)
    return order


def resolve(target_id: str, constraint: str, registry: "Registry",
            *, force: bool = False) -> ResolvedPlan:
    """Resolve a target module into an ordered, verified module stack.

    Steps: version-solve prelayer constraints -> topological order (deps first) ->
    geometry + single-base checks -> substrate_hash / pinned_hash verification.
    ``force`` downgrades hash/geometry failures to warnings.
    """
    chosen, cards, edges, lineage = _solve({target_id: [(constraint, "<target>")]}, registry)
    ordered_ids = _toposort(target_id, edges)
    return _finalize_plan(ordered_ids, chosen, registry, lineage, force,
                          target_label=f"{target_id}@{chosen[target_id]}")


def resolve_stack(module_specs: List[Tuple[str, str]], registry: "Registry",
                  *, force: bool = False) -> ResolvedPlan:
    """Resolve an ordered LIST of modules (each ``(id, version_spec)``) into a
    single verified stack, including their transitive prelayer dependencies.

    Used to assemble a substrate for training a new group that is not yet a
    registered module (e.g. ``--substrate grammar@1,fact@>=1``).
    """
    seed: Dict[str, List[Tuple[str, str]]] = {}
    for mid, spec in module_specs:
        seed.setdefault(mid, []).append((spec, "<stack>"))
    chosen, cards, edges, lineage = _solve(seed, registry)
    edges = dict(edges)
    edges["<stack>"] = [mid for mid, _ in module_specs]
    ordered_ids = [i for i in _toposort("<stack>", edges) if i != "<stack>"]
    label = ",".join(f"{mid}@{chosen[mid]}" for mid, _ in module_specs)
    return _finalize_plan(ordered_ids, chosen, registry, lineage, force, target_label=label)


def _finalize_plan(ordered_ids, chosen, registry, lineage, force, *, target_label):
    """Build ModuleRefs, run geometry/base/hash verification, return a plan."""
    stack: List[ModuleRef] = []
    for mid in ordered_ids:
        ver = chosen[mid]
        ckpt_path, card = registry.get(mid, ver)
        role = "base" if card.role == "base" else "prelayer"
        stack.append(ModuleRef(mid, ver, ckpt_path, card, role))

    warnings: List[str] = []

    def fail(msg):
        if force:
            warnings.append(msg)
        else:
            raise ValueError(msg)

    # Geometry consistency + exactly one base at the bottom.
    bases = [r for r in stack if r.card.role == "base"]
    if len(bases) != 1:
        fail(f"expected exactly one role=base module, found {len(bases)}: "
             f"{[r.module_id for r in bases]}")
    elif stack[0].card.role != "base":
        fail(f"base module '{bases[0].module_id}' must be at the bottom of the stack, "
             f"got '{stack[0].module_id}' first")
    ref_geom = (stack[0].card.dim, stack[0].card.head_dim, stack[0].card.vocab_size)
    for r in stack[1:]:
        g = (r.card.dim, r.card.head_dim, r.card.vocab_size)
        # vocab may be 0/unset on a pure group module; only compare set values.
        if r.card.dim and (r.card.dim, r.card.head_dim) != ref_geom[:2]:
            fail(f"geometry mismatch: {r.module_id} (dim={r.card.dim},head_dim={r.card.head_dim}) "
                 f"!= base (dim={ref_geom[0]},head_dim={ref_geom[1]})")

    # Substrate-hash verification: for each module, hash everything beneath it.
    state_cache: Dict[str, dict] = {}

    def state_of(ref: ModuleRef) -> dict:
        if ref.module_id not in state_cache:
            ck = torch.load(ref.ckpt_path, map_location="cpu", weights_only=False)
            state_cache[ref.module_id] = ck["model_state_dict"]
        return state_cache[ref.module_id]

    for pos, ref in enumerate(stack):
        # self_hash pin verification against any requirement pinning this module.
        if ref.role == "base":
            continue
        beneath = stack[:pos]
        block_sources = [(state_of(b), j) for b in beneath for j in range(b.n_layers)]
        base_state = state_of(stack[0]) if stack else {}
        if block_sources:
            actual = hashlib.sha256(
                hash_block_states(block_sources).encode()
                + hash_shared_states(base_state).encode()
            ).hexdigest()
        else:
            actual = None
        if ref.card.substrate_hash and actual and ref.card.substrate_hash != actual:
            fail(f"substrate_hash mismatch for {ref.module_id}@{ref.version}: card="
                 f"{ref.card.substrate_hash[:12]} resolved={actual[:12]} "
                 f"(incompatible substrate; use force to override)")

    # pinned_hash: each requirement may pin the exact self_hash of its dep.
    by_id = {r.module_id: r for r in stack}
    for r in stack:
        for req in r.card.requires:
            if req.mode == "prelayer" and req.pinned_hash and req.module_id in by_id:
                dep = by_id[req.module_id]
                if dep.card.self_hash and dep.card.self_hash != req.pinned_hash:
                    fail(f"pinned_hash mismatch: {r.module_id} pins {req.module_id}="
                         f"{req.pinned_hash[:12]} but resolved self_hash="
                         f"{dep.card.self_hash[:12]}")

    report = {
        "target": target_label,
        "stack": [f"{r.module_id}@{r.version} ({r.role}, {r.n_layers}L)" for r in stack],
        "total_layers": sum(r.n_layers for r in stack),
        "lineage": lineage,
        "warnings": warnings,
    }
    return ResolvedPlan(stack=stack, lineage=lineage, report=report)


# ── CLI (inspection helpers used by v12/scripts) ─────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="V12 module registry inspection")
    p.add_argument("--registry", default="v12_registry")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list registered modules")
    rp = sub.add_parser("resolve", help="resolve a target module and print the stack")
    rp.add_argument("target")
    rp.add_argument("--constraint", default="*")
    rp.add_argument("--force", action="store_true")
    sp = sub.add_parser("stack", help="resolve an explicit id@spec[,...] stack")
    sp.add_argument("modules", help='comma list "grammar@1,fact@>=1"')
    sp.add_argument("--force", action="store_true")
    args = p.parse_args()

    reg = Registry(args.registry)
    if args.cmd == "list":
        rows = reg.list()
        if not rows:
            print(f"(registry '{args.registry}' is empty)")
            return
        print(f"{'module_id':<24} {'ver':<8} {'role':<7} {'skill':<14} created")
        print("-" * 70)
        for mid, ver, entry in rows:
            print(f"{mid:<24} {ver:<8} {entry.get('role',''):<7} "
                  f"{str(entry.get('skill','')):<14} {entry.get('created','')}")
    else:
        if args.cmd == "resolve":
            plan = resolve(args.target, args.constraint, reg, force=args.force)
        else:
            specs = [(m.split("@", 1)[0], m.split("@", 1)[1] if "@" in m else "*")
                     for m in args.modules.split(",") if m.strip()]
            plan = resolve_stack(specs, reg, force=args.force)
        print(f"target: {plan.report['target']}")
        print(f"stack ({plan.report['total_layers']} layers, bottom-first):")
        for line in plan.report["stack"]:
            print(f"  - {line}")
        if plan.lineage:
            print("lineage (finetuned, not stacked):")
            for lin in plan.lineage:
                print(f"  - {lin['module_id']} ({lin['version_spec']}) <- {lin['from']}")
        for w in plan.report["warnings"]:
            print(f"[warn] {w}")


if __name__ == "__main__":
    main()
