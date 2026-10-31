"""v15 Grove selftest (CPU, ~1 min).

  1. season (parallel) form == token-by-token decode, to round-off, across
     several seasons and with the causal conv on and off;
  2. shares are a simplex (sum to 1) and the state size is O(1);
  3. gradients reach every innate trait (roots, fruits, bias, lam, alpha);
  4. a toy in-context task learns: random key->value pairs per document,
     query a key seen earlier in the SAME document (lookup that must come
     from the organ, since pairs are redrawn per document).
Run: .venv/bin/python -m v15.selftest
"""

from __future__ import annotations

import time

import torch

from v15.grove import GroveConfig, GroveLM, GroveLMConfig, GroveLayer


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        raise SystemExit(f"selftest failed: {name}")


def parity(conv):
    torch.manual_seed(0)
    cfg = GroveConfig(dim=32, groves=2, trees=5, root=8, season=4, conv=conv)
    layer = GroveLayer(cfg).double()
    with torch.no_grad():
        layer.out.weight.normal_()                      # out is zero-init; make it visible
    B, T = 3, 12
    h = torch.randn(B, T, cfg.dim, dtype=torch.double)
    layer._record = []
    o_par, st_par = layer(h)
    n_born = int(sum(int(r[1].sum()) for r in layer._record))
    layer._record = None
    st = None
    outs = []
    for t in range(T):
        o_t, st = layer.step(h[:, t], st)
        outs.append(o_t)
    o_seq = torch.stack(outs, 1)
    err = (o_par - o_seq).abs().max().item()
    check(f"season form == decode (conv={conv})", err < 1e-10, f"max |diff| {err:.2e}")
    check(f"births happened in the parity run (conv={conv})", n_born > 0, f"{n_born} born")
    for name in ('ev', 'roots', 'fruits', 'm_last', 'use'):
        e = (getattr(st_par, name) - getattr(st, name)).abs().max().item()
        check(f"  state '{name}' matches", e < 1e-10, f"{e:.2e}")
    # a second call continues from the state (two more seasons)
    h2 = torch.randn(B, 8, cfg.dim, dtype=torch.double)
    o2, _ = layer(h2, st_par)
    outs = []
    for t in range(8):
        o_t, st = layer.step(h2[:, t], st)
        outs.append(o_t)
    e = (o2 - torch.stack(outs, 1)).abs().max().item()
    check(f"continuation from state (conv={conv})", e < 1e-10, f"{e:.2e}")
    # a length that is not a multiple of the season: outputs equal the decode's
    h3 = torch.randn(B, 10, cfg.dim, dtype=torch.double)
    o3, _ = layer(h3)
    st = None
    outs = []
    for t in range(10):
        o_t, st = layer.step(h3[:, t], st)
        outs.append(o_t)
    e = (o3 - torch.stack(outs, 1)).abs().max().item()
    check(f"partial last season (T=10, C=4) == decode (conv={conv})", e < 1e-10, f"{e:.2e}")


def simplex_and_grads():
    torch.manual_seed(1)
    cfg = GroveConfig(dim=32, groves=2, trees=6, root=8, season=4)
    layer = GroveLayer(cfg)
    with torch.no_grad():
        layer.out.weight.normal_()
    h = torch.randn(2, 16, cfg.dim)
    o, st = layer(h)
    check("shares sum to 1 (softmax over trees)", torch.allclose(st.m_last.sum(-1), torch.ones(2, 2)))
    o.pow(2).mean().backward()
    for name in ('root0', 'fruit0', 'fruit0_born', 'bias', 'bias_born', 'lam_p', 'temp_p', 'alpha_p', 'soil.weight', 'sap.weight'):
        p = dict(layer.named_parameters())[name]
        check(f"gradient reaches {name}", p.grad is not None and p.grad.abs().sum() > 0)
    lm = GroveLM(GroveLMConfig(vocab_size=100, dim=64, n_layers=2, mlp_hidden=128,
                               grove=GroveConfig(groves=2, trees=8, root=16, season=8)))
    check("state is O(1)", lm.state_bytes(1) == 2 * 2 * 8 * (2 * 16 + 3) * 4,
          f"{lm.state_bytes(1)} bytes per sequence for 2 layers")


def toy_lookup(device):
    """Documents: 'k1 v1 k2 v2 ... k8 v8 | q ?' with the 8 pairs redrawn per doc.
    Supervised only on the answer.  Chance = 1/8 (pick any value of the doc)."""
    torch.manual_seed(2)
    V, P, S = 64, 8, 8        # vocab, pairs, season
    sep, q_tok = V - 2, V - 1
    cfg = GroveLMConfig(vocab_size=V, dim=96, n_layers=2, mlp_hidden=192,
                        grove=GroveConfig(groves=4, trees=16, root=24, season=S))
    lm = GroveLM(cfg).to(device)
    opt = torch.optim.AdamW(lm.parameters(), lr=3e-3, weight_decay=0.0)

    def batch(B, g):
        keys = torch.stack([torch.randperm(V // 2 - 1, generator=g)[:P] for _ in range(B)])
        vals = torch.stack([torch.randperm(V // 2 - 1, generator=g)[:P] for _ in range(B)]) + V // 2 - 1
        qi = torch.randint(0, P, (B,), generator=g)
        seq = torch.stack([keys, vals], 2).reshape(B, 2 * P)
        ids = torch.cat([seq, torch.full((B, 1), sep), keys[torch.arange(B), qi][:, None],
                         vals[torch.arange(B), qi][:, None], torch.zeros(B, 32 - 2 * P - 3, dtype=torch.long)], 1)
        labels = torch.full_like(ids, -100)
        labels[:, 2 * P + 2] = vals[torch.arange(B), qi]
        return ids.to(device), labels.to(device)

    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for it in range(400):
        ids, labels = batch(64, g)
        loss, _, _ = lm(ids, labels=labels)
        opt.zero_grad(); loss.backward(); opt.step()
    g = torch.Generator().manual_seed(999)
    ids, labels = batch(256, g)
    with torch.no_grad():
        logits, _ = lm(ids)
        pred = logits[:, 2 * P + 1].argmax(-1)
        acc = (pred == labels[:, 2 * P + 2]).float().mean().item()
    print(f"  toy lookup: final loss {loss.item():.3f}, held-out acc {acc:.3f} "
          f"(chance 1/{P}), {time.time() - t0:.0f}s on {device}")
    check("toy in-context lookup learned (acc >= 0.5)", acc >= 0.5, f"{acc:.3f}")


def main():
    parity(conv=0)
    parity(conv=4)
    simplex_and_grads()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    toy_lookup(device)
    print("v15 selftest PASS")


if __name__ == "__main__":
    main()
