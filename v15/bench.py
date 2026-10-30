"""v15's arm for the v14 micro-bench (external arm, no edits to v14).

    V14_EXT_ARMS="grove=v15.bench:build_grove" ARMS=grove,echo bash v14/run_bench.sh

``build_grove(cfg)`` gets the bench's echo-sized config and returns a GroveLM
whose parameter count matches the echo arm (MLP width solved for parity, as
the bench does for Transformer++), with the bench's interface:
``_hidden_to_lm``, ``ce_from_lm`` (v13's chunked fused CE, same loss
semantics), and the output embedding at ``model.embed.weight``.
"""

from __future__ import annotations

import torch

from v15.grove import GroveConfig, GroveLM, GroveLMConfig

# the grove layout for the micro-bench body (width 384): G*n = width
BENCH_GROVE = dict(groves=6, trees=32, root=64, season=32, conv=4)


class GroveBench(GroveLM):
    def _hidden_to_lm(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h, _ = blk(h)
        return self.norm(h), None

    def ce_from_lm(self, lm, labels, loss_mask=None, ignore_index=-100, chunk=4096,
                   return_nll=False, gemm_dtype=None):
        from v13_sempty.fused_ce import fused_linear_cross_entropy
        B, T, D = lm.shape
        out = fused_linear_cross_entropy(
            lm.reshape(B * T, D), self.embed.weight, labels.reshape(-1),
            mask=(loss_mask.reshape(-1) if loss_mask is not None else None),
            chunk=chunk, ignore_index=ignore_index, return_nll=return_nll,
            gemm_dtype=gemm_dtype)
        if return_nll:
            return out, getattr(out, '_nll', None).reshape(B, T)
        return out


def _grove_params(cfg: GroveLMConfig) -> int:
    with torch.random.fork_rng(devices=[]):
        return sum(p.numel() for p in GroveBench(cfg).parameters())


def build_grove(cfg, grove: dict | None = None):
    """Param-matched GroveBench for the bench's config (a v14 V14Config)."""
    from v14.units import V14LM
    with torch.random.fork_rng(devices=[]):          # sizing must not move the init RNG
        target = sum(p.numel() for p in V14LM(cfg).parameters())
    g = GroveConfig(dim=cfg.dim, **{**BENCH_GROVE, **(grove or {})})
    assert g.groves * g.root == cfg.dim, (g, cfg.dim)
    base = GroveLMConfig(vocab_size=cfg.vocab_size, dim=cfg.dim, n_layers=cfg.n_layers,
                         mlp_hidden=64, grove=g)
    fixed = _grove_params(base) - 3 * cfg.dim * 64 * cfg.n_layers      # everything but the MLPs
    mlp = max(64, round((target - fixed) / (3 * cfg.dim) / cfg.n_layers / 64) * 64)
    model = GroveBench(GroveLMConfig(**{**base.__dict__, 'mlp_hidden': mlp}))
    n = sum(p.numel() for p in model.parameters())
    print(f"  [grove] G={g.groves} K={g.trees} n={g.root} season={g.season} mlp={mlp}: "
          f"{n / 1e6:.2f}M params (echo {target / 1e6:.2f}M), state "
          f"{model.state_bytes(1) / 1024:.0f} KB/seq", flush=True)
    return model
