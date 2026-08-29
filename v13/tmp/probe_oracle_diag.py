"""Oracle LEAK diagnostic.

The scan showed nright in {0, 128/128, ~10} per seed with the 128/128 case
recovering the value at EVERY position — not the signature of real
addressable memory. This isolates whether the oracle is leaking:

  For ONE example (assoc=8, seed=1002):
   - locate the 8 value-word positions and 8 key-word positions.
   - full scan: which positions' unit keys recover the target value.
   - RANDOM control: replace the final query with a fixed random unit vector
     (not any position's key). If it also recovers -> the query is NOT what
     carries the value -> leak (residual/CGU path or in-chunk contamination).
   - check whether the TARGET value position specifically is in right_at.

Run: .venv/bin/python v13/tmp/probe_oracle_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v13.model import V13LM, V13PAMLayer, get_config      # noqa: E402
from v7.data import get_chat_tokenizer                      # noqa: E402
from memory_probes.behavioral import KEYS, build_example    # noqa: E402


def _encode(tz, text: str) -> list[int]:
    return list(tz.encode(text, add_special_tokens=False))


def positions_of(tz, ex):
    records = ''.join(
        f'Memory record {i+1}: {KEYS[i]} means {ex.candidate_texts[i%8]}.\n'
        for i in range(ex.associations))
    record_ids = _encode(tz, records)
    tk = ex.candidate_token_ids.index(ex.target_token_id)
    target_key = KEYS[tk]
    qids = _encode(tz, f'\nMemory query: {target_key} means')
    filler = ex.context_tokens - len(record_ids) - len(qids)
    before = int(round(filler * ex.target_position))
    # value positions for ALL records + target key position
    vpos = []
    for i in range(ex.associations):
        vpos.append(before + record_ids.index(ex.candidate_token_ids[i % 8]))
    kid = _encode(tz, f' {target_key}')[-1]
    kpos = before + record_ids.index(kid)
    target_value_pos = vpos[tk]
    return vpos, kpos, target_value_pos


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload = torch.load(
        ROOT / 'checkpoints_v13/500m_v13_r1recipe/best_model.pt',
        map_location='cpu', weights_only=False)
    cfg = get_config('v13_e3_k3_selective')
    for k, v in (payload.get('config') or {}).items():
        if hasattr(cfg, k) and k != 'max_seq_len':
            setattr(cfg, k, v)
    cfg.dropout = 0.0
    cfg.gradient_checkpointing = False
    model = V13LM(cfg).to(device).eval()
    model.load_state_dict(payload['model_state_dict'])
    tz = get_chat_tokenizer()

    orig = V13PAMLayer._project
    cap = {}
    swap = set()
    mode = {'s': None, 'rand': None}

    def patch(self, x, step_offset):
        q, k, vv, rk = orig(self, x, step_offset)
        pid = id(self)
        if pid not in cap:
            cap[pid] = k.detach().clone()
        if pid in swap:
            T = q.shape[2]
            q = q.clone()
            if mode['rand'] is not None:
                q[:, :, T - 1] = mode['rand']
            else:
                q[:, :, T - 1] = cap[pid][:, :, mode['s']]
        return q, k, vv, rk

    V13PAMLayer._project = patch

    def run(ex, s=None, rand=None):
        cap.clear(); swap.clear()
        mode['s'] = s; mode['rand'] = rand
        if s is not None or rand is not None:
            swap.update(id(b.pam) for b in model.blocks)
        ids = torch.tensor([ex.prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            lm, _, _ = model._hidden_to_lm(ids)
            lg = (lm[0, -1, :, 0] @ model.embed.embed_real.weight.T
                  + lm[0, -1, :, 1] @ model.embed.embed_imag.weight.T)
        return lg

    ex = build_example(tz, context_tokens=128, target_position=1.0,
                       associations=8, seed=1002, candidate_count=8)
    cand = ex.candidate_token_ids
    tidx = cand.index(ex.target_token_id)
    T = len(ex.prompt_ids)
    vpos, kpos, tvpos = positions_of(tz, ex)
    print(f'target value = {ex.target_text!r} at pos {tvpos}')
    print(f'value positions: {vpos}')
    print(f'target key position: {kpos}')

    base = int(run(ex)[cand].argmax().item()) == tidx
    print(f'\nbase (learned query) correct = {base}')

    scan = [int(run(ex, s=s)[cand].argmax().item()) == tidx for s in range(T)]
    right = [s for s in range(T) if scan[s]]
    print(f'\nscan: nright = {len(right)}/{T}')
    print(f'right_at (full) = {right}')
    print(f'target value pos {tvpos} in right_at: {tvpos in right}')
    print(f'target key pos   {kpos} in right_at: {kpos in right}')
    print(f'value positions in right_at: '
          f'{[p for p in vpos if p in right]}')

    # Random control: one fixed random unit vector (same across all layers).
    rng = torch.Generator().manual_seed(0)
    H, d = model.blocks[0].pam.num_heads, model.blocks[0].pam.head_dim
    randvec = torch.randn(1, H, d, 2, generator=rng)
    randvec = (randvec / randvec.norm(dim=-1, keepdim=True)).to(device)
    sel = run(ex, rand=randvec)[cand]
    r = int(sel.argmax().item()) == tidx
    print(f'\nRANDOM control (fixed unit vector) correct = {r} '
          f'(picks #{int(sel.argmax())}, target #{tidx}, '
          f'margin {sel.max() - sel[1:].max():.4f})')
    z = int(run(ex, rand=torch.zeros(1, H, d, 2).to(device))[cand].argmax().item()) == tidx
    print(f'ZERO control correct = {z}')


if __name__ == '__main__':
    main()
