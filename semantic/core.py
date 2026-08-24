"""semantic.core — named-axis tensor layer over standard PyTorch.

Rank-agnostic: every axis position is a named Dimension. Ops are expressed over
names; the module resolves names -> real permute/reshape/unsqueeze/bmm and runs
on plain torch tensors. Element math dispatches through a pluggable Policy.
"""
import torch
from .policies import Real


class LayoutError(Exception):
    """Shape/layout mismatch. Names both offending dims + expected/actual sizes."""


# --------------------------------------------------------------------------- #
# Dimensions & layouts
# --------------------------------------------------------------------------- #
class Dimension:
    """A named axis. size is symbolic (None) until bound by a real tensor."""

    __slots__ = ("name", "size", "role")

    def __init__(self, name, size=None, role=None):
        self.name = str(name)
        self.size = size
        self.role = role

    def bind(self, size):
        """Bind (or correlate) this dim's size. Raises on disagreement."""
        if self.size is None:
            self.size = int(size)
        elif int(self.size) != int(size):
            raise LayoutError(
                f"dimension '{self.name}' size disagrees: "
                f"expected {self.size}, got {size}"
            )

    def __repr__(self):
        s = self.size if self.size is not None else "?"
        return f"{self.name}({s})"


def _layout_str(layout):
    return "(" + ", ".join(d.name for d in layout) + ")"


class SemanticTensor:
    """Wraps a real torch.Tensor + a named Layout + an element Policy."""

    __slots__ = ("data", "layout", "policy")

    def __init__(self, data, layout, policy=None):
        self.data = data
        self.layout = tuple(layout)
        self.policy = policy if policy is not None else Real()
        self._check_rank()

    def _check_rank(self):
        if self.data.dim() != len(self.layout):
            raise LayoutError(
                f"rank mismatch: tensor has {self.data.dim()} dims, "
                f"layout {_layout_str(self.layout)} has {len(self.layout)}"
            )

    def bind_sizes(self):
        """Bind/correlate every named axis to the real axis size."""
        for dim, size in zip(self.layout, self.data.shape):
            dim.bind(size)

    def __repr__(self):
        return f"Semantic{_layout_str(self.layout)} {tuple(self.data.shape)}"


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def tensor(data, layout, policy=None):
    """Build a SemanticTensor. `data` is a real torch.Tensor (or array)."""
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    t = SemanticTensor(data, layout, policy)
    t.bind_sizes()
    return t


# --------------------------------------------------------------------------- #
# Layout algebra: arrange (permute + split + merge)
# --------------------------------------------------------------------------- #
def arrange(t, new_layout):
    """Re-express t's data with a new named layout.

    Handles permute, split (one dim -> factors) and merge (dims -> one dim).
    Splits require the parent size to be bound and equal to the product of the
    bound factors; unbound factors bind as row-major factors of the parent.
    """
    old = list(t.layout)
    new = list(new_layout)
    # old-name -> list of positions it occupies in `new`
    spans = {}
    for d in old:
        spans[d.name] = [i for i, nd in enumerate(new) if nd.name == d.name]
    # new dims that don't come from any old dim -> must be size 1 or unbound
    for i, nd in enumerate(new):
        if nd.name not in spans:
            raise LayoutError(
                f"arrange: new dim '{nd.name}' has no source in {_layout_str(old)}"
            )
    # Build the target shape and the permutation of the *expanded* old axes.
    # Expand each old dim into its parts (in the order they appear in `new`).
    expanded = []      # (old_dim_index, part_dim)
    shape = []
    for i, nd in enumerate(new):
        old_idx = old.index([d for d in old if d.name == nd.name][0])
        expanded.append((old_idx, nd))
        shape.append(nd.size if nd.size is not None else None)
    # If any part is unbound, factor the old dim size across its parts.
    parts_by_old = {}
    for old_idx, nd in expanded:
        parts_by_old.setdefault(old_idx, []).append(nd)
    for old_idx, parts in parts_by_old.items():
        od = old[old_idx]
        bound = [p for p in parts if p.size is not None]
        unbound = [p for p in parts if p.size is None]
        if od.size is None:
            if len(unbound) > 1:
                raise LayoutError(
                    f"arrange: cannot split unbound dim '{od.name}' into "
                    f"{[p.name for p in unbound]} without a known size"
                )
        else:
            known = 1
            for p in bound:
                if od.size % p.size != 0:
                    raise LayoutError(
                        f"arrange: dim '{od.name}' size {od.size} is not a "
                        f"multiple of factor '{p.name}' size {p.size}"
                    )
                known *= p.size
            if len(unbound) == 1:
                u = unbound[0]
                u.bind(od.size // known)
            elif len(unbound) > 1:
                raise LayoutError(
                    f"arrange: splitting '{od.name}' ({od.size}) into "
                    f"{[p.name for p in unbound]} is under-determined"
                )
        # validate bound factors multiply to od size
        prod = 1
        for p in parts:
            p.bind(p.size)
            prod *= p.size
        if od.size is not None and prod != od.size:
            raise LayoutError(
                f"arrange: factors of '{od.name}' multiply to {prod}, "
                f"but '{od.name}' is {od.size}"
            )
    # Now every new dim has a size; build target shape.
    new_shape = tuple(nd.size for nd in new)
    # Expand old data into per-part shape, then permute to new order.
    old_expanded_shape = []
    for old_idx, od in enumerate(old):
        for p in parts_by_old[old_idx]:
            old_expanded_shape.append(p.size)
    # map old-expanded index -> new index
    new_pos_of_expanded = {}
    for new_idx, (old_idx, nd) in enumerate(expanded):
        key = (old_idx, nd)
        new_pos_of_expanded[key] = new_idx
    # permute: old data reshaped to old_expanded_shape, axes in old-expanded order
    data = t.data.view(old_expanded_shape)
    # build permutation that takes old-expanded order to new order
    perm = []
    for old_idx in range(len(old)):
        for p in parts_by_old[old_idx]:
            perm.append(new_pos_of_expanded[(old_idx, p)])
    data = data.permute(perm).reshape(new_shape)
    return SemanticTensor(data, new, t.policy)


def swap(t, a, b):
    """Permute so that named axes a and b exchange positions."""
    layout = list(t.layout)
    if a.name not in [d.name for d in layout] or b.name not in [d.name for d in layout]:
        raise LayoutError(f"swap: '{a.name}' and '{b.name}' must both be in layout")
    ia, ib = layout.index(a), layout.index(b)
    layout[ia], layout[ib] = layout[ib], layout[ia]
    return arrange(t, layout)


# --------------------------------------------------------------------------- #
# Named broadcast helper
# --------------------------------------------------------------------------- #
def _broadcast_pair(x, y, op):
    """Element op with named broadcast. Shared axes must size-match."""
    if x.policy is not y.policy and not (
        type(x.policy).__name__ == "Real" or type(y.policy).__name__ == "Real"
    ):
        raise LayoutError(f"{op}: mixed policies {type(x.policy).__name__} / "
                          f"{type(y.policy).__name__}")
    names = [d.name for d in x.layout]
    ynames = [d.name for d in y.layout]
    # correlate shared sizes
    for dx in x.layout:
        if dx.name in ynames:
            dy = [d for d in y.layout if d.name == dx.name][0]
            if dx.size is not None and dy.size is not None and dx.size != dy.size:
                raise LayoutError(
                    f"{op}: shared dim '{dx.name}' sizes disagree: "
                    f"{dx.size} vs {dy.size}"
                )
    # canonical layout = x's layout + y-only axes appended
    canon = list(x.layout)
    seen = set(d.name for d in canon)
    for dy in y.layout:
        if dy.name not in seen:
            canon.append(dy)
            seen.add(dy.name)
    xa = arrange(x, canon).data
    ya = arrange(y, canon).data
    return xa, ya, canon


# --------------------------------------------------------------------------- #
# Element ops (policy-dispatched)
# --------------------------------------------------------------------------- #
def add(x, y):
    xa, ya, canon = _broadcast_pair(x, y, "add")
    out = x.policy.add(xa, ya) if type(x.policy).__name__ != "Real" else xa + ya
    # Real policy add == torch add; keep uniform
    if type(x.policy).__name__ == "Real" or type(y.policy).__name__ == "Real":
        # promote: run through the non-Real policy if present
        pol = y.policy if type(x.policy).__name__ == "Real" else x.policy
        out = pol.add(xa, ya) if hasattr(pol, "add") else xa + ya
    return SemanticTensor(out, canon, x.policy)


def scale(x, y):
    """x * y element-wise with named broadcast."""
    xa, ya, canon = _broadcast_pair(x, y, "scale")
    if type(x.policy).__name__ == "Real" and type(y.policy).__name__ == "Real":
        out = xa * ya
    else:
        pol = x.policy if type(y.policy).__name__ == "Real" else y.policy
        out = pol.mul(xa, ya)
    return SemanticTensor(out, canon, x.policy)


def sub(x, y):
    xa, ya, canon = _broadcast_pair(x, y, "sub")
    if type(x.policy).__name__ == "Real" and type(y.policy).__name__ == "Real":
        out = xa - ya
    else:
        pol = x.policy
        out = pol.sub(xa, ya)
    return SemanticTensor(out, canon, x.policy)


def conj(x):
    out = x.policy.conj(x.data)
    return SemanticTensor(out, x.layout, x.policy)


def normalize_vec(x, over):
    """Policy-dispatched per-vector norm over named axis `over`."""
    out = x.policy.norm_vec(x.data, x.layout.index(over))
    return SemanticTensor(out, x.layout, x.policy)


# --------------------------------------------------------------------------- #
# Named matmul / outer / read
# --------------------------------------------------------------------------- #
def contract(x, y, over):
    """Named matmul: sum over shared named axis `over`.

    `over` must appear in both. Other dims are batch axes (unioned by name).
    """
    xn = [d.name for d in x.layout]
    yn = [d.name for d in y.layout]
    if over.name not in xn:
        raise LayoutError(f"contract: dim '{over.name}' missing from x {_layout_str(x.layout)}")
    if over.name not in yn:
        raise LayoutError(f"contract: dim '{over.name}' missing from y {_layout_str(y.layout)}")
    dx = x.layout[xn.index(over.name)]
    dy = y.layout[yn.index(over.name)]
    if dx.size is not None and dy.size is not None and dx.size != dy.size:
        raise LayoutError(
            f"contract: dim '{over.name}' sizes disagree: {dx.size} vs {dy.size}"
        )
    # move `over` to last in each operand
    xa = arrange(x, [d for d in x.layout if d.name != over.name] + [dx])
    ya = arrange(y, [d for d in y.layout if d.name != over.name] + [dy])
    # align batch axes by name: canonical = x batch axes + y-only batch axes
    xb = list(xa.layout)
    yb = list(ya.layout)
    seen = set(d.name for d in xb)
    canon = list(xb)
    for d in yb:
        if d.name not in seen:
            canon.append(d)
            seen.add(d.name)
    xa = arrange(xa, canon + [dx])
    ya = arrange(ya, canon + [dy])
    n = dx.size if dx.size is not None else xa.data.shape[-1]
    # matmul on last dim: x[..., n] * y[..., n, :] -> y last axis is output
    out = torch.matmul(xa.data, ya.data)
    out_layout = canon + [yb[-1]]
    return SemanticTensor(out, out_layout, x.policy)


def outer(x, y, over):
    """Named outer product. over = (A in x, B in y)."""
    A, B = over
    xa = arrange(x, [d for d in x.layout if d.name != A.name] + [A])
    ya = arrange(y, [d for d in y.layout if d.name != B.name] + [B])
    # x: [..., A] -> [..., A, 1]; y: [..., B] -> [..., 1, B]
    # broadcast element-wise through policy mul
    xd = xa.data.unsqueeze(-1)
    yd = ya.data.unsqueeze(-2)
    if type(x.policy).__name__ == "Real" and type(y.policy).__name__ == "Real":
        out = xd * yd
    else:
        out = x.policy.mul(xd, yd)
    out_layout = xa.layout + (B,)
    return SemanticTensor(out, out_layout, x.policy)


def read(S, probe, over, conjugate="probe"):
    """Inner product over `over` (read/collapse). conjugate in
    {"probe","none"} conjugates the probe per policy before the inner product."""
    if conjugate == "probe":
        probe = conj(probe)
    elif conjugate != "none":
        raise LayoutError(f"read: bad conjugate flag {conjugate!r}")
    return contract(S, probe, over)


def write(S, u, k, over, conjugate=True):
    """S + outer(u, conj(k)) over the two named axes `over=(A,B)."""
    A, B = over
    kk = conj(k) if conjugate else k
    delta = outer(u, kk, (A, B))
    return add(S, delta)


# --------------------------------------------------------------------------- #
# Misc named ops
# --------------------------------------------------------------------------- #
def squeeze(t, name):
    layout = [d for d in t.layout if d.name != name]
    if len(layout) == len(t.layout):
        raise LayoutError(f"squeeze: dim '{name}' not in layout")
    data = t.data
    # find axis
    idx = [d.name for d in t.layout].index(name)
    data = data.squeeze(idx)
    return SemanticTensor(data, layout, t.policy)


def softmax(t, over):
    idx = [d.name for d in t.layout].index(over.name)
    data = torch.softmax(t.data, dim=idx)
    return SemanticTensor(data, t.layout, t.policy)
