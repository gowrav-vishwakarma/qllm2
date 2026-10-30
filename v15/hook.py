"""Grove (v15) on the v13_sempty trainer: ``--arch grove --grove_spec ...``.

The trainer keeps the data, loss (chunked fused CE on the tied embedding),
schedule, checkpoints, resume and evals; the only variable is the body.
``GroveTrainLM`` gives GroveLM the interface the trainer and the baseline
wrappers (``baseline_transformer.py``) share: ``config`` (a dataclass, saved
in checkpoints), ``blocks``, ``_hidden_to_lm``, ``ce_from_lm``, ``forward``
-> (logits, None, aux), ``generate`` (through the O(1) decode path, so every
text sample exercises ``step``), ``count_parameters``.

A spec is a comma list ``key=value``: dim, layers, groves, trees, root,
season, conv, mlp (0 = solve for target_m), target_m (total params in
millions the MLP width is solved for; 0 = keep mlp as given).
"""

from __future__ import annotations

from dataclasses import fields
from typing import Dict

import torch
import torch.nn.functional as F

from v15.grove import GroveConfig, GroveLM, GroveLMConfig

# ~102.6M total, the r600 protocol's body size (M1 / Mamba-2 17x768 / T++).
DEFAULT_SPEC = "dim=768,layers=12,groves=12,trees=32,root=64,season=32,conv=4,mlp=0,target_m=102.6"


class GroveTrainLM(GroveLM):
    def __init__(self, cfg: GroveLMConfig):
        super().__init__(cfg)
        self.config = cfg

    def _head_weight(self) -> torch.Tensor:
        return self.embed.weight

    def _hidden_to_lm(self, input_ids, step_offset: int = 0):
        h = self.embed(input_ids)
        for blk in self.blocks:
            h, _ = blk(h)
        return self.norm(h), torch.zeros((), device=input_ids.device)

    def ce_from_lm(self, lm, labels, loss_mask=None, ignore_index=-100,
                   chunk: int = 4096, return_nll: bool = False, gemm_dtype=None):
        from v13_sempty.fused_ce import fused_linear_cross_entropy
        b, t, d = lm.shape
        out = fused_linear_cross_entropy(
            lm.reshape(b * t, d), self._head_weight(), labels.reshape(-1),
            mask=(loss_mask.reshape(-1) if loss_mask is not None else None),
            chunk=chunk, ignore_index=ignore_index, return_nll=return_nll,
            gemm_dtype=gemm_dtype)
        if return_nll:
            return out, getattr(out, '_nll', None).reshape(b, t)
        return out

    def forward(self, input_ids, states=None, step_offset: int = 0, labels=None):
        hidden, aux = self._hidden_to_lm(input_ids)
        return F.linear(hidden, self._head_weight()), None, aux

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0,
                 top_k=50, top_p=0.0, repetition_penalty=1.0, eos_token_id=None):
        """Prompt through the season path once, then one ``step`` per token
        (O(1) state); the two agree to round-off (v15/selftest.py parity)."""
        self.eval()
        generated = input_ids.clone()
        _, states = GroveLM.forward(self, generated)
        nxt = None
        for _ in range(max_new_tokens):
            if nxt is None:
                logits = GroveLM.forward(self, generated)[0][:, -1].float()
            else:
                logits, states = self.step(nxt[:, 0], states)
                logits = logits.float()
            if repetition_penalty != 1.0:
                score = torch.gather(logits, 1, generated)
                score = torch.where(score > 0, score / repetition_penalty,
                                    score * repetition_penalty)
                logits.scatter_(1, generated, score)
            if temperature <= 0.0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = logits.topk(min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = float('-inf')
                if top_p > 0:
                    sl, si = logits.sort(descending=True)
                    p = sl.softmax(dim=-1)
                    sl[p.cumsum(dim=-1) - p >= top_p] = float('-inf')
                    logits = sl.scatter(1, si, sl)
                nxt = torch.multinomial(logits.softmax(dim=-1), 1)
            generated = torch.cat([generated, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
        return generated

    def count_parameters(self) -> Dict[str, int]:
        embed_p = self.embed.weight.numel()
        block_p = sum(p.numel() for b in self.blocks for p in b.parameters())
        grove_p = sum(p.numel() for b in self.blocks for p in b.grove.parameters())
        norm_p = sum(p.numel() for p in self.norm.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {'embedding (tied)': embed_p, 'blocks': block_p, 'of which groves': grove_p,
                'norms': norm_p, 'total': total}


def parse_spec(spec: str) -> dict:
    kv = dict(DEFAULT_SPEC_KV)
    for item in (spec or '').split(','):
        if item.strip():
            k, v = item.split('=')
            kv[k.strip()] = float(v) if k.strip() == 'target_m' else int(v)
    return kv


DEFAULT_SPEC_KV = {k: (float(v) if k == 'target_m' else int(v))
                   for k, v in (i.split('=') for i in DEFAULT_SPEC.split(','))}


def _count(cfg: GroveLMConfig) -> int:
    with torch.random.fork_rng(devices=[]):          # sizing must not move the init RNG
        return sum(p.numel() for p in GroveTrainLM(cfg).parameters())


def build_grove_lm(vocab_size: int, spec: str = DEFAULT_SPEC) -> GroveTrainLM:
    s = parse_spec(spec)
    g = GroveConfig(dim=s['dim'], groves=s['groves'], trees=s['trees'], root=s['root'],
                    season=s['season'], conv=s['conv'])
    mlp = s['mlp']
    if s['target_m'] > 0:
        base = GroveLMConfig(vocab_size=vocab_size, dim=s['dim'], n_layers=s['layers'],
                             mlp_hidden=64, grove=g)
        fixed = _count(base) - 3 * s['dim'] * 64 * s['layers']        # everything but the MLPs
        mlp = max(64, round((s['target_m'] * 1e6 - fixed) / (3 * s['dim']) / s['layers'] / 64) * 64)
    cfg = GroveLMConfig(vocab_size=vocab_size, dim=s['dim'], n_layers=s['layers'],
                        mlp_hidden=mlp, grove=g)
    model = GroveTrainLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"[grove] dim={s['dim']} layers={s['layers']} G={g.groves} K={g.trees} n={g.root} "
          f"season={g.season} conv={g.conv} mlp={mlp}: {n / 1e6:.2f}M params, "
          f"state {model.state_bytes(1) / 1024:.0f} KB/seq", flush=True)
    return model


def config_from_raw(raw: dict) -> GroveLMConfig:
    """A checkpoint's saved config dict -> GroveLMConfig (nested grove dict ok)."""
    names = {f.name for f in fields(GroveLMConfig)}
    kw = {k: v for k, v in raw.items() if k in names}
    if isinstance(kw.get('grove'), dict):
        kw['grove'] = GroveConfig(**kw['grove'])
    return GroveLMConfig(**kw)


def is_grove(raw) -> bool:
    get = raw.get if isinstance(raw, dict) else (lambda k, d=None: getattr(raw, k, d))
    return get('grove') is not None
