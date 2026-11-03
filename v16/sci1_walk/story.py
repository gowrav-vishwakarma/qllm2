#!/usr/bin/env python3
"""Word-scale SCI-1 story: 2 heads x 4 axes, the real recurrence.

Reads annotations.json, checks each expect snippet is still on that source
line, runs the toy, and writes trace.json + story.js for the walk page.

The toy is the formulas in RealPAMLayer._stepwise_delta / _rotate_learned /
_delta_gates / _decay, SharedRead, and LM._echo_read. It is not the 385M
checkpoint. Real heads are 8 x 128; the read there divides by sqrt(128).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

AX_H = (
    ["animal", "furry", "sits", "soft"],
    ["meow", "four-legs", "on", "floor"],
)
AX = AX_H[0] + AX_H[1]
TOKENS = ["cat", "sat", "mat", "eos"]
# Next-word targets. EOS has none: the loss stops at the document edge.
TARGETS = ["sat", "mat", "eos", None]
K = 4
D = 8
SQRT_K = math.sqrt(K)
PAM_SCALE = 0.1  # RealBlock.pam_scale init
GATE_BIAS = 1.2785  # silu(1.2785) = 1
BLANK = 0.02

# Rows are cat, sat, mat, eos. EOS is negative so the write gate shuts.
E_RAW = np.array(
    [
        [1.60, 1.40, 0.05, 0.02, 1.50, 1.30, 0.05, 0.02],
        [0.10, 0.08, 1.70, 0.22, 0.05, 0.04, 0.40, 0.05],
        [0.05, 0.08, 0.06, 0.28, 0.04, 0.05, 0.22, 1.70],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    ],
    dtype=np.float64,
)

# Query and key: head 1 reads the first four words, head 2 the last four.
W_QK = np.zeros((2, K, D))
W_QK[0, :, :K] = np.eye(K)
W_QK[1, :, K:] = np.eye(K)

# Value: head 1 files "sits" under an animal/furry token, and "soft" under sits.
# Head 2 files the token's own meow / four-legs / on / floor.
W_V = np.zeros((2, K, D))
W_V[0, 2, 0] = 1.40  # sits from animal
W_V[0, 2, 1] = 1.10  # sits from furry
W_V[0, 3, 2] = 1.20  # soft from sits
W_V[0, 0, 0] = 0.20
W_V[0, 1, 1] = 0.20
W_V[1, :, K:] = np.eye(K)

# Chrono warp. Negative on a head's own words, so a loud token runs a slow clock
# and the animal–furry pair turns a modest slice rather than spinning away.
W_WARP = np.zeros((2, D))
W_WARP[0, :K] = -0.45
W_WARP[1, K:] = -0.35

W_DT = np.full((2, D), 0.02)
DT_BIAS = np.array([-1.15, -1.15])

W_ERASE = np.zeros((2, D))
W_ERASE[0, :K] = 0.70
W_ERASE[1, K:] = 0.70

# Large on every axis: content (positive) writes hard, EOS (all negative) writes ~0.
W_WRITE = np.full((2, D), 1.35)

# Echo projections: identity, so the key is the previous hidden's own words.
W_ECHO_Q = np.eye(D)
W_ECHO_K = np.eye(D)

INV_FREQ = 1.0 / (10000.0 ** (np.arange(K // 2) / (K // 2)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x) + eps)


def rmsnorm_backward(x, dy, eps=1e-6):
    n = x.size
    ms = np.mean(x * x) + eps
    rinv = 1.0 / math.sqrt(ms)
    dot = float(np.dot(dy, x))
    return dy * rinv - x * (dot * rinv ** 3 / n)


def swiglu(x):
    """Diagonal SwiGLU: each word is gated by itself, so heads stay specialised."""
    gate = silu(2.2 * x)
    up = x
    down = 0.15 * gate * up
    return gate, up, gate * up, down


def pack_vec(v):
    return [round(float(x), 5) for x in np.asarray(v).tolist()]


def pack_mat(m):
    return [[round(float(x), 5) for x in row] for row in np.asarray(m).tolist()]


def blank_from(k):
    return [["blank"] * k for _ in range(k)]


def cell_from(token, axes, write, kept, prev_from):
    """Where a notebook cell came from, in words."""
    out = []
    for i, ai in enumerate(axes):
        row = []
        for j, aj in enumerate(axes):
            bits = []
            if abs(write[i, j]) >= BLANK:
                bits.append(f"{token}'s value {ai} × {token}'s key {aj}")
            if abs(kept[i, j]) >= BLANK:
                prev = prev_from[i][j] if prev_from is not None else "blank"
                if prev and prev != "blank":
                    bits.append(f"faded retain of ({prev})")
                else:
                    bits.append("faded retain of the previous page")
            row.append("; ".join(bits) if bits else "blank")
        out.append(row)
    return out


def want_from(axes, dS):
    scale = float(np.max(np.abs(dS))) if dS.size else 0.0
    out = []
    for i, ai in enumerate(axes):
        row = []
        for j, aj in enumerate(axes):
            v = float(dS[i, j])
            # ∂loss/∂cell < 0 means raising the cell lowers the loss.
            if scale < 1e-12 or abs(v) < 0.18 * scale:
                row.append("quiet")
            elif v < 0:
                row.append(f"want more {ai} × {aj}")
            else:
                row.append(f"want less {ai} × {aj}")
        out.append(row)
    return out


def project_heads(u):
    q = np.stack([W_QK[h] @ u for h in (0, 1)])
    k = np.stack([W_QK[h] @ u for h in (0, 1)])
    v = np.stack([W_V[h] @ u for h in (0, 1)])
    return q, k, v


def chrono(u, clock_in):
    """Per-head speed and the exclusive angle. clock_in is shape (2,)."""
    logit = np.clip(W_WARP @ u, -3.0, 3.0)
    g = np.exp(logit)
    tau = clock_in.copy()  # exclusive: the angle applied now
    clock_out = clock_in + g
    return g, tau, clock_out


def rotate_pair(even, odd, phi):
    c, s = math.cos(phi), math.sin(phi)
    return c * even - s * odd, s * even + c * odd, c, s


def rotate_head(vec, tau):
    """Return rotated vec and the per-pair record (before, after, cos, sin, phi)."""
    out = vec.copy()
    pairs = []
    for p in range(K // 2):
        phi = float(tau * INV_FREQ[p])
        e, o = float(vec[2 * p]), float(vec[2 * p + 1])
        er, or_, c, s = rotate_pair(e, o, phi)
        out[2 * p], out[2 * p + 1] = er, or_
        pairs.append(
            {
                "axes": [AX_H[0][2 * p] if False else None, None],  # filled by caller
                "phi": phi,
                "cos": c,
                "sin": s,
                "before": [e, o],
                "after": [er, or_],
            }
        )
    return out, pairs


def gates(u, eos):
    retain = np.exp(-softplus(W_DT @ u + DT_BIAS))
    erase = np.minimum(sigmoid(W_ERASE @ u - 3.0), 0.95)
    write = sigmoid(W_WRITE @ u)
    if eos:
        retain = np.zeros(2)
    return retain, erase, write


def delta_step(S, k_hat, v, q, g, erase, write):
    """One head. S is (K, K) value-rows, key-cols. Returns new S and parts."""
    Sk = S @ k_hat
    erase_term = erase * np.outer(Sk, k_hat)
    erased = S - erase_term
    kept = g * erased
    write_term = write * np.outer(v, k_hat)
    S_new = kept + write_term
    read = S_new @ q
    return S_new, {
        "Sk": Sk,
        "erase_term": erase_term,
        "erased": erased,
        "kept": kept,
        "write_term": write_term,
        "read": read,
    }


def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v), 0.0
    return v / n, n


def ce_grad(logits, target):
    m = float(np.max(logits))
    ex = np.exp(logits - m)
    prob = ex / ex.sum()
    loss = float(-math.log(float(prob[target]) + 1e-12))
    d = prob.copy()
    d[target] -= 1.0
    return loss, d, prob


def prepare_token(index):
    """Embedding through q, k, v, gates. Independent of the notebook."""
    raw = E_RAW[index]
    embed = rmsnorm(raw)
    gate, up, prod, down = swiglu(rmsnorm(embed))
    mixed = embed + down  # cgu_scale 1, diagonal swiglu
    pam_in = rmsnorm(mixed)
    q_raw, k_raw, v = project_heads(pam_in)
    eos = TOKENS[index] == "eos"
    retain, erase, write = gates(pam_in, eos)
    out_gate = np.array([float(silu(GATE_BIAS)), float(silu(GATE_BIAS))])
    return {
        "raw": raw,
        "embed": embed,
        "sw_gate": gate,
        "sw_up": up,
        "sw_prod": prod,
        "sw_down": down,
        "mixed": mixed,
        "pam_in": pam_in,
        "q_raw": q_raw,
        "k_raw": k_raw,
        "v": v,
        "retain": retain,
        "erase": erase,
        "write": write,
        "out_gate": out_gate,
        "eos": eos,
    }


def run(bump=None):
    """Full document. bump = (t, head, i, j, eps) added to S after that write."""
    pre = [prepare_token(t) for t in range(len(TOKENS))]
    S = [np.zeros((K, K)), np.zeros((K, K))]
    clock = np.zeros(2)
    steps = []
    for t, tok in enumerate(pre):
        g, tau, clock_out = chrono(tok["pam_in"], clock)
        heads = []
        for h in (0, 1):
            q_rot, q_pairs = rotate_head(tok["q_raw"][h], tau[h])
            k_rot, k_pairs = rotate_head(tok["k_raw"][h], tau[h])
            # Stamp pair axis names for this head.
            for p, rec in enumerate(q_pairs):
                rec["axes"] = [AX_H[h][2 * p], AX_H[h][2 * p + 1]]
            k_hat, k_norm = unit(k_rot)
            S_prev = S[h].copy()
            S_new, parts = delta_step(
                S_prev, k_hat, tok["v"][h], q_rot,
                float(tok["retain"][h]), float(tok["erase"][h]), float(tok["write"][h]),
            )
            if bump is not None and bump[0] == t and bump[1] == h:
                S_new = S_new.copy()
                S_new[bump[2], bump[3]] += bump[4]
                parts = dict(parts)
                parts["read"] = S_new @ q_rot
            S[h] = S_new
            y = parts["read"] / SQRT_K * float(tok["out_gate"][h])
            heads.append(
                {
                    "q_raw": tok["q_raw"][h],
                    "k_raw": tok["k_raw"][h],
                    "v": tok["v"][h],
                    "q": q_rot,
                    "k": k_rot,
                    "k_hat": k_hat,
                    "k_norm": k_norm,
                    "q_pairs": q_pairs,
                    "k_pairs": k_pairs,
                    "g": float(g[h]),
                    "tau": float(tau[h]),
                    "clock_out": float(clock_out[h]),
                    "retain": float(tok["retain"][h]),
                    "erase": float(tok["erase"][h]),
                    "write": float(tok["write"][h]),
                    "out_gate": float(tok["out_gate"][h]),
                    "S_prev": S_prev,
                    "S": S_new.copy(),
                    "y": y,
                    **parts,
                }
            )
        # Residual add, then the output norm. Echo's out-proj is zero-init, so
        # it does not move the stream. The reader's out-proj is zero-init too.
        y_cat = np.concatenate([heads[0]["y"], heads[1]["y"]])
        stream = tok["mixed"] + PAM_SCALE * y_cat
        hidden = rmsnorm(stream)
        logits = E_RAW @ hidden
        vote = E_RAW @ y_cat  # the notebook's own vote, before the residual
        steps.append(
            {
                "pre": tok,
                "heads": heads,
                "y": y_cat,
                "stream": stream,
                "hidden": hidden,
                "logits": logits,
                "vote": vote,
                "clock_in": clock.copy(),
                "clock_out": clock_out.copy(),
            }
        )
        clock = clock_out
    # Echo, display only (zero-init out-proj, not in the loss).
    echo = echo_pass(steps)
    shared = shared_pass(steps)
    loss = 0.0
    grads_logits = []
    for t, step in enumerate(steps):
        target = TARGETS[t]
        if target is None:
            grads_logits.append(np.zeros(D))
            continue
        li, dlogits, prob = ce_grad(step["logits"], TOKENS.index(target))
        loss += li
        step["prob"] = prob
        grads_logits.append(E_RAW.T @ dlogits)
    return {
        "steps": steps,
        "loss": loss,
        "d_hidden_from_logits": grads_logits,
        "echo": echo,
        "shared": shared,
        "pre": pre,
    }


def echo_pass(steps):
    """k_t = unit(W_k h_{t-1}), v_t = rmsnorm(E[x_t]), same delta rule."""
    S = [np.zeros((K, K)), np.zeros((K, K))]
    out = []
    prev_h = None
    for t, step in enumerate(steps):
        h = step["hidden"]
        if prev_h is None:
            k_full = np.zeros(D)
        else:
            k_full = W_ECHO_K @ prev_h
        if TOKENS[t] == "eos":
            k_full = np.zeros(D)
        v_full = rmsnorm(E_RAW[t])
        q_full = W_ECHO_Q @ h
        # Gates from the hidden, same shapes as the notebook. EOS kills retain.
        retain, erase, write = gates(h, TOKENS[t] == "eos")
        heads = []
        for h_i in (0, 1):
            sl = slice(h_i * K, (h_i + 1) * K)
            k_hat, _ = unit(k_full[sl])
            v = v_full[sl]
            q = q_full[sl]
            S_prev = S[h_i].copy()
            S_new, parts = delta_step(
                S_prev, k_hat, v, q,
                float(retain[h_i]), float(erase[h_i]), float(write[h_i]),
            )
            S[h_i] = S_new
            heads.append(
                {
                    "k_hat": k_hat,
                    "v": v,
                    "q": q,
                    "S_prev": S_prev,
                    "S": S_new.copy(),
                    "retain": float(retain[h_i]),
                    "erase": float(erase[h_i]),
                    "write": float(write[h_i]),
                    **parts,
                }
            )
        out.append({"heads": heads, "prev": None if t == 0 else TOKENS[t - 1]})
        prev_h = h
    return out


def shared_pass(steps):
    """Reader query uses the writer's clock, then dots the writer's S."""
    out = []
    for step in steps:
        heads = []
        for h in (0, 1):
            hh = step["heads"][h]
            # Reader sees the stream after the writer, normalised, split like q.
            u = rmsnorm(step["stream"])
            q_raw = W_QK[h] @ u
            q_rot, pairs = rotate_head(q_raw, hh["tau"])
            for p, rec in enumerate(pairs):
                rec["axes"] = [AX_H[h][2 * p], AX_H[h][2 * p + 1]]
            read = hh["S"] @ q_rot
            y = read / SQRT_K * float(silu(GATE_BIAS))
            heads.append({"q_raw": q_raw, "q": q_rot, "pairs": pairs, "read": read, "y": y, "S": hh["S"]})
        out.append({"heads": heads})
    return out


def backward(result):
    """Local grads of the delta step, including the carry to earlier tokens.

    dS is ∂loss/∂S after the write: the read at this token plus every later read.
    EOS has no target and retention 0, so the push into the previous page is 0.
    """
    steps = result["steps"]
    T = len(steps)
    dS_future = [np.zeros((K, K)), np.zeros((K, K))]
    per_t = [None] * T
    for t in range(T - 1, -1, -1):
        step = steps[t]
        d_hidden = result["d_hidden_from_logits"][t]
        d_stream = rmsnorm_backward(step["stream"], d_hidden)
        # stream = mixed + pam_scale * y, mixed does not depend on S
        d_y = PAM_SCALE * d_stream
        heads_g = []
        new_future = []
        for h in (0, 1):
            hh = step["heads"][h]
            dy = d_y[h * K:(h + 1) * K]
            d_read = dy * hh["out_gate"] / SQRT_K  # ∂L/∂(S q)
            dS = np.outer(d_read, hh["q"]) + dS_future[h]
            g = hh["retain"]
            e = hh["erase"]
            w = hh["write"]
            k = hh["k_hat"]
            v = hh["v"]
            Sprev = hh["S_prev"]
            derased = g * dS
            dg = float(np.sum(dS * hh["kept"]))
            dw = float(np.sum(dS * hh["write_term"]))
            dv = w * (dS @ k)
            dk = w * (dS.T @ v)
            Sk = Sprev @ k
            de = float(-np.sum(derased * np.outer(Sk, k)))
            dSk = -e * (derased @ k)
            dk = dk + (-e) * (derased.T @ Sk) + Sprev.T @ dSk
            dSprev = derased + np.outer(dSk, k)
            dq = hh["S"].T @ d_read
            heads_g.append(
                {
                    "dS": dS,
                    "dS_read": np.outer(d_read, hh["q"]),
                    "dq": dq,
                    "dv": dv,
                    "dk": dk,
                    "d_retain": dg,
                    "d_erase": de,
                    "d_write": dw,
                    "dy": dy,
                }
            )
            new_future.append(dSprev)
        per_t[t] = heads_g
        dS_future = new_future
    return per_t


def top_names(vec, axes, n=2):
    idx = np.argsort(-np.abs(vec))[:n]
    return [axes[i] for i in idx]


def assert_story(result, grads):
    steps = result["steps"]
    cat, sat, mat, eos = steps
    # Specialisation falls out of the split, before the clock turns (cat's tau is 0).
    assert top_names(cat["heads"][0]["k_hat"], AX_H[0]) == ["animal", "furry"], top_names(
        cat["heads"][0]["k_hat"], AX_H[0]
    )
    assert top_names(cat["heads"][1]["k_hat"], AX_H[1]) == ["meow", "four-legs"]
    assert top_names(cat["heads"][0]["v"], AX_H[0])[0] == "sits"
    assert top_names(sat["heads"][0]["k_raw"], AX_H[0])[0] == "sits"
    assert top_names(mat["heads"][1]["k_raw"], AX_H[1])[0] == "floor"
    # The read at cat on head 1 lights sits: the value filed under animal/furry.
    assert int(np.argmax(np.abs(cat["heads"][0]["read"]))) == 2
    # Unit keys, and q/k share the rotation.
    for step in steps:
        for h in (0, 1):
            hh = step["heads"][h]
            assert abs(np.linalg.norm(hh["k_hat"]) - 1.0) < 1e-6
            for qp, kp in zip(hh["q_pairs"], hh["k_pairs"]):
                assert abs(qp["phi"] - kp["phi"]) < 1e-9
                assert abs(qp["cos"] - kp["cos"]) < 1e-9
                er, or_, _, _ = rotate_pair(qp["before"][0], qp["before"][1], qp["phi"])
                assert abs(er - qp["after"][0]) < 1e-9 and abs(or_ - qp["after"][1]) < 1e-9
    # First token has not turned. Later tokens have.
    assert cat["heads"][0]["tau"] == 0.0
    assert sat["heads"][0]["tau"] > 0.2
    # EOS page is blank: retention 0 and the write gate is shut.
    for h in (0, 1):
        assert float(np.max(np.abs(eos["heads"][h]["S"]))) < 1e-4
        assert eos["heads"][h]["retain"] == 0.0
    # Echo at EOS is exactly blank (key zeroed and retention 0).
    for h in (0, 1):
        assert float(np.max(np.abs(result["echo"][-1]["heads"][h]["S"]))) < 1e-8
    # The biggest cell on cat's head-1 page is sits filed under animal or furry,
    # and the parent string names cat and sits.
    S = cat["heads"][0]["S"]
    i, j = np.unravel_index(int(np.argmax(np.abs(S))), S.shape)
    assert AX_H[0][i] == "sits"
    assert AX_H[0][j] in ("animal", "furry")
    # Loss push at cat wants more sits on the animal/furry key.
    dS = grads[0][0]["dS"]
    i, j = np.unravel_index(int(np.argmin(dS)), dS.shape)
    assert AX_H[0][i] == "sits", (AX_H[0][i], dS)
    assert AX_H[0][j] in ("animal", "furry")
    # Gradients stop at EOS: the push from eos into mat's page is ~0,
    # and eos itself has no read-loss.
    assert float(np.max(np.abs(grads[3][0]["dS"]))) < 1e-8
    # Finite difference on cat's sits×animal cell and on mat's page.
    base = result["loss"]
    eps = 1e-5
    for t, h, i, j in ((0, 0, 2, 0), (2, 0, 2, 0), (1, 1, 0, 0)):
        bumped = run((t, h, i, j, eps))["loss"]
        analytic = float(grads[t][h]["dS"][i, j])
        numeric = (bumped - base) / eps
        if abs(analytic) < 1e-8 and abs(numeric) < 1e-6:
            continue
        rel = abs(analytic - numeric) / max(1e-6, abs(numeric))
        assert rel < 1e-3, (t, h, i, j, analytic, numeric, rel)
    # Infer carry matches the document: replaying one token at a time is the same S.
    S_carry = [np.zeros((K, K)), np.zeros((K, K))]
    clock = np.zeros(2)
    for t, step in enumerate(steps):
        assert np.allclose(step["clock_in"], clock)
        for h in (0, 1):
            assert np.allclose(step["heads"][h]["S_prev"], S_carry[h])
            S_carry[h] = step["heads"][h]["S"]
        clock = step["clock_out"]


def build_trace(result, grads):
    steps_j = []
    grad_j = []
    for t, step in enumerate(result["steps"]):
        tok = TOKENS[t]
        pre = step["pre"]
        prev_from = [None, None]
        # Parent strings need the previous step's notebook origins. Recompute
        # while walking, stored on the python step the first time.
        heads_j = {}
        gh = {}
        for h in (0, 1):
            hh = step["heads"][h]
            axes = AX_H[h]
            write = hh["write_term"]
            kept = hh["kept"]
            origin = cell_from(tok, axes, write, kept, hh.get("_prev_from"))
            hh["_origin"] = origin
            pairs = []
            for qp, kp in zip(hh["q_pairs"], hh["k_pairs"]):
                pairs.append(
                    {
                        "axes": qp["axes"],
                        "phi": round(qp["phi"], 5),
                        "cos": round(qp["cos"], 5),
                        "sin": round(qp["sin"], 5),
                        "qBefore": pack_vec(qp["before"]),
                        "qAfter": pack_vec(qp["after"]),
                        "kBefore": pack_vec(kp["before"]),
                        "kAfter": pack_vec(kp["after"]),
                    }
                )
            terms = []
            for i in range(K):
                row = []
                for j in range(K):
                    s = float(hh["S"][i, j])
                    q = float(hh["q"][j])
                    row.append(
                        {
                            "s": round(s, 5),
                            "q": round(q, 5),
                            "prod": round(s * q / SQRT_K, 5),
                            "from": origin[i][j],
                        }
                    )
                terms.append(row)
            heads_j[f"h{h + 1}"] = {
                "qRaw": pack_vec(hh["q_raw"]),
                "kRaw": pack_vec(hh["k_raw"]),
                "v": pack_vec(hh["v"]),
                "q": pack_vec(hh["q"]),
                "k": pack_vec(hh["k"]),
                "kHat": pack_vec(hh["k_hat"]),
                "kNorm": round(hh["k_norm"], 5),
                "g": round(hh["g"], 5),
                "tau": round(hh["tau"], 5),
                "clockOut": round(hh["clock_out"], 5),
                "retain": round(hh["retain"], 5),
                "erase": round(hh["erase"], 5),
                "write": round(hh["write"], 5),
                "outGate": round(hh["out_gate"], 5),
                "pairs": pairs,
                "Sprev": {"v": pack_mat(hh["S_prev"]), "from": hh.get("_prev_from") or blank_from(K)},
                "eraseTerm": {"v": pack_mat(hh["erase_term"]), "from": origin},
                "kept": {"v": pack_mat(kept), "from": origin},
                "writeTerm": {"v": pack_mat(write), "from": cell_from(tok, axes, write, np.zeros_like(write), None)},
                "S": {"v": pack_mat(hh["S"]), "from": origin},
                "read": pack_vec(hh["read"] / SQRT_K),
                "y": pack_vec(hh["y"]),
                "terms": terms,
            }
            g = grads[t][h]
            gh[f"h{h + 1}"] = {
                "dS": {"v": pack_mat(g["dS"]), "from": want_from(axes, g["dS"])},
                "dq": pack_vec(g["dq"]),
                "dv": pack_vec(g["dv"]),
                "dk": pack_vec(g["dk"]),
                "dRetain": round(g["d_retain"], 6),
                "dErase": round(g["d_erase"], 6),
                "dWrite": round(g["d_write"], 6),
                "dy": pack_vec(g["dy"]),
            }
        # Stash origins for the next token's "faded from" phrase.
        if t + 1 < len(result["steps"]):
            for h in (0, 1):
                result["steps"][t + 1]["heads"][h]["_prev_from"] = heads_j[f"h{h + 1}"]["S"]["from"]
        # The stash above is too late: we already built this step's origin
        # without the previous from. Rebuild kept-from using the previous
        # step's stored origin if present. Done in a second pass below.
        echo_h = {}
        for h in (0, 1):
            eh = result["echo"][t]["heads"][h]
            axes = AX_H[h]
            prev = result["echo"][t]["prev"] or "nothing"
            echo_from = []
            for i, ai in enumerate(axes):
                row = []
                for j, aj in enumerate(axes):
                    if abs(eh["write_term"][i, j]) < BLANK and abs(eh["kept"][i, j]) < BLANK:
                        row.append("blank")
                    else:
                        bits = []
                        if abs(eh["write_term"][i, j]) >= BLANK:
                            bits.append(f"{tok}'s embedding {ai} × {prev}'s hidden {aj}")
                        if abs(eh["kept"][i, j]) >= BLANK:
                            bits.append("faded retain of the echo page")
                        row.append("; ".join(bits))
                echo_from.append(row)
            echo_h[f"h{h + 1}"] = {
                "kHat": pack_vec(eh["k_hat"]),
                "v": pack_vec(eh["v"]),
                "q": pack_vec(eh["q"]),
                "retain": round(eh["retain"], 5),
                "erase": round(eh["erase"], 5),
                "write": round(eh["write"], 5),
                "Sprev": {"v": pack_mat(eh["S_prev"]), "from": echo_from},
                "writeTerm": {"v": pack_mat(eh["write_term"]), "from": echo_from},
                "S": {"v": pack_mat(eh["S"]), "from": echo_from},
                "read": pack_vec(eh["read"] / SQRT_K),
            }
        shared_h = {}
        for h in (0, 1):
            sh = result["shared"][t]["heads"][h]
            pairs = []
            for rec in sh["pairs"]:
                pairs.append(
                    {
                        "axes": rec["axes"],
                        "phi": round(rec["phi"], 5),
                        "cos": round(rec["cos"], 5),
                        "sin": round(rec["sin"], 5),
                        "qBefore": pack_vec(rec["before"]),
                        "qAfter": pack_vec(rec["after"]),
                    }
                )
            shared_h[f"h{h + 1}"] = {
                "qRaw": pack_vec(sh["q_raw"]),
                "q": pack_vec(sh["q"]),
                "pairs": pairs,
                "read": pack_vec(sh["read"] / SQRT_K),
                "y": pack_vec(sh["y"]),
            }
        scores = []
        vote = []
        for i, word in enumerate(TOKENS):
            scores.append({"word": word, "score": round(float(step["logits"][i]), 5)})
            vote.append({"word": word, "score": round(float(step["vote"][i]), 5)})
        steps_j.append(
            {
                "token": tok,
                "t": t,
                "target": TARGETS[t],
                "embedRaw": pack_vec(pre["raw"]),
                "embed": pack_vec(pre["embed"]),
                "swiglu": {
                    "gate": pack_vec(pre["sw_gate"]),
                    "up": pack_vec(pre["sw_up"]),
                    "prod": pack_vec(pre["sw_prod"]),
                    "down": pack_vec(pre["sw_down"]),
                },
                "mixed": pack_vec(pre["mixed"]),
                "pamIn": pack_vec(pre["pam_in"]),
                "heads": heads_j,
                "concat": pack_vec(step["y"]),
                "hidden": pack_vec(step["hidden"]),
                "logits": scores,
                "vote": vote,
                "echo": {"prev": result["echo"][t]["prev"], "heads": echo_h},
                "shared": {"heads": shared_h},
            }
        )
        grad_j.append({"heads": gh})
    # Second pass: parent strings now know the previous page.
    for t in range(1, len(steps_j)):
        for key, h in (("h1", 0), ("h2", 1)):
            prev = steps_j[t - 1]["heads"][key]["S"]["from"]
            hh = result["steps"][t]["heads"][h]
            origin = cell_from(TOKENS[t], AX_H[h], hh["write_term"], hh["kept"], prev)
            steps_j[t]["heads"][key]["S"]["from"] = origin
            steps_j[t]["heads"][key]["kept"]["from"] = origin
            for i in range(K):
                for j in range(K):
                    steps_j[t]["heads"][key]["terms"][i][j]["from"] = origin[i][j]
    return {
        "axes": {"h1": AX_H[0], "h2": AX_H[1]},
        "allAxes": AX,
        "tokens": TOKENS,
        "vocab": TOKENS,
        "embedTable": {word: pack_vec(E_RAW[i]) for i, word in enumerate(TOKENS)},
        "sqrtD": round(SQRT_K, 5),
        "pamScale": PAM_SCALE,
        "note": (
            "Scale model of the same math, not the 385M checkpoint. "
            "Real shape is 8 heads × 128. This page uses 2 heads × 4 word-axes. "
            "The real read divides by √128; these cells divide by √4."
        ),
        "steps": steps_j,
        "grads": grad_j,
    }


def load_annotations():
    path = HERE / "annotations.json"
    data = json.loads(path.read_text())
    files = {}
    for sec in data["sections"]:
        files.setdefault(sec["file"], None)
    for line in data["lines"]:
        files.setdefault(line["file"], None)
    cache = {}
    for rel in files:
        text = (ROOT / rel).read_text().splitlines()
        cache[rel] = text
    for line in data["lines"]:
        src = cache[line["file"]]
        n = line["line"]
        if n < 1 or n > len(src):
            raise SystemExit(f"{line['file']}:{n} is past the end of the file")
        got = src[n - 1]
        if line["expect"] not in got:
            raise SystemExit(
                f"{line['file']}:{n} missing {line['expect']!r}\n  got: {got}"
            )
    # Every annotated line must sit inside a declared slice.
    for line in data["lines"]:
        hit = [
            s for s in data["sections"]
            if s["file"] == line["file"] and s["start"] <= line["line"] <= s["end"]
        ]
        if not hit:
            raise SystemExit(f"{line['file']}:{line['line']} is not inside a section")
    slices = []
    by_key = {(a["file"], a["line"]): a for a in data["lines"]}
    for sec in data["sections"]:
        src = cache[sec["file"]]
        rows = []
        for n in range(sec["start"], sec["end"] + 1):
            text = src[n - 1]
            ann = by_key.get((sec["file"], n))
            if ann is None:
                stripped = text.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    sentence = stripped.lstrip("#").strip() or sec["blurb"]
                    ann = {
                        "file": sec["file"],
                        "line": n,
                        "title": "Comment",
                        "viz": "note",
                        "train": sentence,
                        "infer": sentence,
                    }
                else:
                    sentence = sec["blurb"]
                    if sec.get("after") and n >= sec["after"]["line"]:
                        sentence = sec["after"]["text"]
                    ann = {
                        "file": sec["file"],
                        "line": n,
                        "title": sec["title"],
                        "viz": "note",
                        "train": sentence,
                        "infer": sentence,
                    }
            rows.append({"n": n, "text": text, "ann": {k: v for k, v in ann.items() if k != "expect"}})
        slices.append(
            {
                "id": sec["id"],
                "file": sec["file"],
                "title": sec["title"],
                "start": sec["start"],
                "end": sec["end"],
                "lines": rows,
            }
        )
    return slices


def main():
    result = run()
    grads = backward(result)
    assert_story(result, grads)
    trace = build_trace(result, grads)
    slices = load_annotations()
    bundle = {"slices": slices, "trace": trace}
    (HERE / "trace.json").write_text(json.dumps(trace))
    payload = json.dumps(bundle).replace("<", "\\u003c")
    (HERE / "story.js").write_text("window.SCI1 = " + payload + ";\n")
    cat = trace["steps"][0]
    h1 = cat["heads"]["h1"]
    print("asserts ok")
    print("cat h1 key", h1["kHat"])
    print("cat h1 value", h1["v"])
    print("cat h1 S", h1["S"]["v"])
    print("cat h1 read", h1["read"])
    print("cat logits", cat["logits"])
    print("cat vote", cat["vote"])
    print("sat h1 phi", trace["steps"][1]["heads"]["h1"]["pairs"][0]["phi"])
    print("eos |S|", np.max(np.abs(result["steps"][-1]["heads"][0]["S"])))
    print("loss", round(result["loss"], 4))


if __name__ == "__main__":
    main()
