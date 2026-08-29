"""Oracle readout — the decisive B-vs-C discriminator.

Question: at multi8@ctx128 (recall ~0.133 ≈ chance 0.125), are the stored
VALUES intact-and-addressable (learned query just misaligned -> fix by
retraining with recall data = option B), or DESTROYED by write interference
(-> substrate limit -> fall back to additive = option C)?

Key fact used: the state WRITES are query-independent (the write is built from
the value v and key k, never the query q — only the readout uses q). So
swapping the FINAL-position query does NOT change the state; it only changes
the final readout. For ctx=128 with delta_chunk=128 there is a single chunk
and NO cross-chunk carry, so the final output token depends on the final
query alone. Clean oracle.

Method, per example:
  1. Locate the target value word position vpos and key word position kpos.
  2. Pass 1 (baseline): normal forward -> contrastive accuracy.
  3. Pass 2 (valword oracle): monkeypatch V13PAMLayer._project so the final-position
     query is replaced by the per-layer unit key k^ captured AT vpos. State
     identical to pass 1; only the final readout is "read with the exact
     address the value was written under".
  4. Pass 3 (keyword oracle): same but the oracle key is captured AT kpos.
  Compare accuracies.

Interpretation:
  oracle >> baseline (>0.5)  -> values INTACT & separable; the bottleneck is
     the learned query->address alignment -> RETRAIN with recall data (B).
  oracle ~ chance (~0.125)   -> values DESTROYED by write interference ->
     substrate limit -> additive (C) or memory redesign.

Run: .venv/bin/python v13/tmp/probe_oracle.py [checkpoint.pt]
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
from memory_probes.behavioral import (                      # noqa: E402
    KEYS, build_example,
)


def _encode(tz, text: str) -> list[int]:
    return list(tz.encode(text, add_special_tokens=False))


def find_positions(tz, ex):
    """Return (vpos, kpos) — absolute positions of the target VALUE word and
    KEY word — by reconstructing the record block exactly as build_example."""
    records = []
    for i in range(ex.associations):
        records.append(f'Memory record {i + 1}: {KEYS[i]} means '
                       f'{ex.candidate_texts[i % 8]}.\n')
    record_ids = _encode(tz, ''.join(records))
    target_key = KEYS[ex.candidate_token_ids.index(ex.target_token_id)]
    query_ids = _encode(tz, f'\nMemory query: {target_key} means')
    filler_count = ex.context_tokens - len(record_ids) - len(query_ids)
    before_count = int(round(filler_count * ex.target_position))
    vpos = before_count + record_ids.index(ex.target_token_id)
    kid = _encode(tz, f' {target_key}')[-1]
    kpos = before_count + record_ids.index(kid)
    return vpos, kpos


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / 'checkpoints_v13/500m_v13_r1recipe/best_model.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = get_config('v13_e3_k3_selective')
    for k, v in (payload.get('config') or {}).items():
        if hasattr(cfg, k) and k != 'max_seq_len':
            setattr(cfg, k, v)
    cfg.dropout = 0.0
    cfg.gradient_checkpointing = False
    model = V13LM(cfg).to(device).eval()
    model.load_state_dict(payload['model_state_dict'])
    tz = get_chat_tokenizer()

    orig_project = V13PAMLayer._project
    capture: dict[int, torch.Tensor] = {}
    swap_at: set[int] = set()
    cap_pos: dict[int, int] = {}

    def patched_project(self, x, step_offset):
        queries, keys, values, readout_keys = orig_project(self, x, step_offset)
        pid = id(self)
        if pid in cap_pos:
            capture[pid] = keys[:, :, cap_pos[pid]].detach().clone()
        if pid in swap_at:
            T = queries.shape[2]
            queries = queries.clone()
            queries[:, :, T - 1] = capture[pid]
        return queries, keys, values, readout_keys

    V13PAMLayer._project = patched_project

    def run_pass(ex, use_oracle: bool, mode: str):
        vpos, kpos = find_positions(tz, ex)
        for block in model.blocks:
            cap_pos[id(block.pam)] = vpos if mode == 'valword' else kpos
        swap_at.clear()
        capture.clear()
        ids = torch.tensor([ex.prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            if use_oracle:
                model._hidden_to_lm(ids)                 # capture pass (no swap)
                swap_at.update(id(b.pam) for b in model.blocks)
                lm, _, _ = model._hidden_to_lm(ids)      # swap pass
            else:
                lm, _, _ = model._hidden_to_lm(ids)      # baseline
            lm_last = lm[0, -1]  # [dim,2]
            logits = (lm_last[..., 0] @ model.embed.embed_real.weight.T
                      + lm_last[..., 1] @ model.embed.embed_imag.weight.T)
        return logits

    print(f'ckpt={ckpt.name}')
    print(f'{"assoc":>5} {"ctx":>5} | {"baseline":>9} {"oracle(val)":>12} '
          f'{"oracle(key)":>12} | chance=0.125')
    for associations in (1, 4, 8):
        ctx = 128
        nseeds = 20
        base, val, key = [], [], []
        for seed in range(1000, 1000 + nseeds):
            ex = build_example(tz, context_tokens=ctx, target_position=1.0,
                               associations=associations, seed=seed,
                               candidate_count=8)
            cand = ex.candidate_token_ids
            tidx = cand.index(ex.target_token_id)
            def acc(lg):
                sel = lg[cand].float()
                return int(sel.argmax().item()) == tidx
            base.append(acc(run_pass(ex, False, 'valword')))
            val.append(acc(run_pass(ex, True, 'valword')))
            key.append(acc(run_pass(ex, True, 'keyword')))
        print(f'{associations:>5} {ctx:>5} | {np.mean(base):9.4f} '
              f'{np.mean(val):12.4f} {np.mean(key):12.4f}')
    print()
    print('oracle(val) = final-position query -> the VALUE word\'s own unit key')
    print('  (the exact address the value was written under). >> baseline means')
    print('  the value IS stored & separable; the learned query is just')
    print('  misaligned -> RETRAIN with recall data (option B).')
    print('oracle(key) = query -> the KEY word\'s unit key. High means the value')
    print('  is reachable under the key address (key-conditioned read works).')
    print('oracle ~ chance (~0.125) -> values DESTROYED by write interference')
    print('  -> substrate limit -> additive (option C) or memory redesign.')


if __name__ == '__main__':
    main()
