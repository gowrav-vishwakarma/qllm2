"""Oracle SCAN — the decisive B-vs-C discriminator.

The single-address oracle (value-word key / key-word key) came back WORSE than
baseline, which means the value is NOT stored under those obvious addresses —
the model uses a LEARNED read address. So the right question is:

  Is the value recoverable from the state by reading with SOME address key?

Scan: for each prompt position s, run the full model with the final-position
query replaced by the unit key k^_s (per layer) and score contrastive accuracy.
Take the best. The state writes are query-independent and ctx=128 is a single
delta_chunk (no cross-chunk carry), so each scan forward is a clean "read the
real state with query k^_s" — including the in-chunk dual-form readout term.

  best-address acc >> chance (0.125), nright > 0
      -> at least one fact is INTACT & addressable; the multi-fact failure is
         ROUTING (the learned query can't pick 1 of 8) -> fixable by
         retraining with recall data = option B.
  best-address acc ~ chance, nright == 0
      -> NO single address holds a value; the 8 writes interfere in the shared
         state -> substrate limit = option C (additive) or memory redesign.

Also measures the learned final-query magnitude ‖q_T‖ (qk_norm is OFF, so
learned queries are not unit) so the oracle scale is reported honestly.

Run: .venv/bin/python v13/tmp/probe_oracle_scan.py [checkpoint.pt]
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
from memory_probes.behavioral import build_example          # noqa: E402


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
    capture: dict[int, torch.Tensor] = {}   # id(pam) -> unit keys [H,T,d,2]
    swap_at: set[int] = set()
    scan_s = {'s': None}                    # position to read out with
    qcap: dict[int, torch.Tensor] = {}      # id(pam) -> learned q at final pos
    want_qmag = {'on': False}

    def patched_project(self, x, step_offset):
        queries, keys, values, readout_keys = orig_project(self, x, step_offset)
        pid = id(self)
        if want_qmag['on']:
            qcap[pid] = queries[:, :, -1].detach().clone()
        if pid not in capture:
            capture[pid] = keys.detach().clone()
        if pid in swap_at and scan_s['s'] is not None:
            T = queries.shape[2]
            queries = queries.clone()
            queries[:, :, T - 1] = capture[pid][:, :, scan_s['s']]
        return queries, keys, values, readout_keys

    V13PAMLayer._project = patched_project

    def forward(ex, use_scan: bool, s=None, capture_qmag: bool = False):
        capture.clear()
        swap_at.clear()
        qcap.clear()
        scan_s['s'] = s
        want_qmag['on'] = capture_qmag
        if use_scan:
            swap_at.update(id(b.pam) for b in model.blocks)
        ids = torch.tensor([ex.prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            lm, _, _ = model._hidden_to_lm(ids)
            lm_last = lm[0, -1]
            logits = (lm_last[..., 0] @ model.embed.embed_real.weight.T
                      + lm_last[..., 1] @ model.embed.embed_imag.weight.T)
        want_qmag['on'] = False
        return logits

    def qmag(ex):
        forward(ex, False, capture_qmag=True)
        return float(np.mean([
            torch.sqrt(torch.sum(q * q, dim=-1)).mean().item()
            for q in qcap.values()]))

    print(f'ckpt={ckpt.name}')
    print(f'{"assoc":>5} {"seed":>5} | {"base":>6} {"best":>6} {"@s":>4} '
          f'{"nright":>7} {"qmag":>6}')
    for associations in (4, 8):
        ctx = 128
        for seed in (1000, 1001, 1002):
            ex = build_example(tz, context_tokens=ctx, target_position=1.0,
                               associations=associations, seed=seed,
                               candidate_count=8)
            cand = ex.candidate_token_ids
            tidx = cand.index(ex.target_token_id)
            T = len(ex.prompt_ids)
            qm = qmag(ex)
            base_logits = forward(ex, False)
            base = int(base_logits[cand].argmax().item()) == tidx
            scan_arr = np.zeros(T)
            for s in range(T):
                lg = forward(ex, True, s)
                scan_arr[s] = int(lg[cand].argmax().item()) == tidx
            right = [int(s) for s in range(T) if scan_arr[s] == 1]
            print(f'{associations:>5} {seed:>5} | {int(base):6} '
                  f'{scan_arr.max():6.2f} {int(np.argmax(scan_arr)):4d} '
                  f'{len(right):>4}/{T:<2} {qm:6.3f}   right_at={right[:8]}')
    print()
    print('best = max over prompt positions of contrastive accuracy when the final')
    print('  query is that position\'s unit key; nright = # positions that recover')
    print('  the value. qmag = learned final-query magnitude (unit keys are 1).')
    print('nright > 0 -> a value IS addressable (B: retrain with recall data).')
    print('nright == 0 -> nothing is addressable (C: additive) or redesign.')


if __name__ == '__main__':
    main()
