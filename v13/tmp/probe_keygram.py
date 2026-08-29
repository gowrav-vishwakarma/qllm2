"""Key-gram off-diagonal probe — geometry vs dynamics root-cause check.

Load the 500M v13 ckpt (flag OFF, its trained config) and, on the exact
behavioral multi-fact recall prompts (assoc in {1,4,8}, ctx128, pos1), capture
the PAM unit keys k^ at the fact-key word positions (the WRITE ADDRESSES) and
at the query-key position, per layer. Report the off-diagonal complex inner
products |<k^_i, k^_j>| and compare to a random-unit-vector baseline
(self-calibrating), plus the query/target vs query/other alignment.

Interpretation:
  - fact-key off-diag >> random  -> keys ANGULARLY CLUSTERED (geometry). No
    readout-magnitude fix (two-state) and no retrain will separate them; the
    address space is the bottleneck.
  - fact-key off-diag ~ random   -> keys are ALREADY separable, so the recall
    failure is in the WRITE DYNAMICS (rank-1 interference / erase not firing).
    A retrain with delta actually on (erase_cap 1.0, bias reinit) can fix it.
  - query/target ~ query/other   -> the QUERY side is not aligned to the stored
    address; a retrain must fix the q/k co-geometry, not just the writes.

Run: .venv/bin/python v13/tmp/probe_keygram.py [checkpoint.pt]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v13.model import V13LM, get_config                     # noqa: E402
from v7.data import get_chat_tokenizer                       # noqa: E402
from memory_probes.behavioral import (                       # noqa: E402
    KEYS, FILLER, single_token_values,
)


def _encode(tz, text: str) -> list[int]:
    return list(tz.encode(text, add_special_tokens=False))


def _filler_tokens(tz, count: int) -> list[int]:
    if count <= 0:
        return []
    unit = _encode(tz, FILLER)
    reps = (count + len(unit) - 1) // len(unit)
    return (unit * reps)[:count]


def build_prompt(tz, context_tokens, target_position, associations, seed,
                 candidate_count=8):
    """Mirror memory_probes.behavioral.build_example exactly; also return the
    token position of every fact-key word and the query-key word."""
    candidates = single_token_values(tz)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=candidate_count, replace=False)
    cand_pairs = [candidates[int(i)] for i in chosen]
    target_index = int(rng.integers(min(associations, candidate_count)))
    records = []
    for i in range(associations):
        key = KEYS[i]
        value = cand_pairs[i % candidate_count][0]
        records.append(f'Memory record {i + 1}: {key} means {value}.\n')
    record_ids = _encode(tz, ''.join(records))
    target_key = KEYS[target_index]
    query_ids = _encode(tz, f'\nMemory query: {target_key} means')
    filler_count = context_tokens - len(record_ids) - len(query_ids)
    before = int(round(filler_count * target_position))
    after = filler_count - before
    prompt_ids = (_filler_tokens(tz, before) + record_ids
                  + _filler_tokens(tz, after) + query_ids)
    assert len(prompt_ids) == context_tokens

    # Locate each fact-key word (unique invented word -> exactly one occurrence
    # in its record; the query reuses the target key a second time at the end).
    key_positions = []
    for i in range(associations):
        kid = _encode(tz, f' {KEYS[i]}')[-1]
        key_positions.append(prompt_ids.index(kid))
    qkid = _encode(tz, f' {target_key}')[-1]
    query_pos = len(prompt_ids) - 1 - prompt_ids[::-1].index(qkid)
    return prompt_ids, key_positions, query_pos, target_index


def random_baseline(d: int, trials: int = 200000) -> float:
    """Mean |complex inner product| over random independent unit vectors."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(trials, d)) + 1j * rng.normal(size=(trials, d))
    b = rng.normal(size=(trials, d)) + 1j * rng.normal(size=(trials, d))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return float(np.abs((a * b.conj()).sum(axis=1)).mean())


def cip_mag(a, b):
    """|complex inner product| of two [...,d,2] tensors: a . conj(b)."""
    re = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    im = a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1]
    return (re * re + im * im).sqrt()


@torch.inference_mode()
def capture(model, prompt_ids, device):
    """Walk the model exactly like V13Block.forward (eval) and return, per
    layer, the QUERY projection q [H,T,d,2] and the unit KEY projection k^
    [H,T,d,2] at the PAM input. Retrieval = q . k^ (separate projections)."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    z = model.embed(ids)
    if model.pos_embed is not None:
        z = model.pos_embed(z, step_offset=0)
    z = model.embed_norm(z)
    q_per_layer, k_per_layer = [], []
    for block in model.blocks:
        x = z + block.cgu(block.norm1(z)) * block.cgu_scale
        pam_in = block.norm2(x)
        q, k_unit, _, _ = block.pam._project(pam_in, step_offset=0)
        q_per_layer.append(q[0])    # [H,T,d,2]
        k_per_layer.append(k_unit[0])  # [H,T,d,2]
        x = x + block.pam(pam_in, state=None, step_offset=0)[0] * block.pam_scale
        z = x
    return q_per_layer, k_per_layer

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
    n_layers = cfg.n_layers
    d = cfg.head_dim
    rand = random_baseline(d)

    print(f'ckpt={ckpt.name}  n_layers={n_layers}  head_dim={d}  '
          f'random |<k,k>|={rand:.4f} (1/sqrt(d)={1 / d ** 0.5:.4f})')
    print(f'{"assoc":>5} {"seed":>5} | ' +
          ' | '.join(f'factL{i}' for i in range(n_layers)) +
          ' | q/target  q/other  (readout alignment, query projection vs fact keys)')
    agg = {}
    for associations in (1, 4, 8):
        for seed in (1000, 1001, 1002, 1003):
            prompt_ids, key_pos, query_pos, tidx = build_prompt(
                tz, 128, 1.0, associations, seed)
            ql, kl = capture(model, prompt_ids, device)  # each list of [H,T,d,2]
            # fact-key off-diag mean |cip|, averaged over heads, per layer
            factL = []
            for li in range(n_layers):
                ks = [kl[li][:, p] for p in key_pos]  # [H,d,2] each
                mags = [cip_mag(ks[i], ks[j]).mean().item()
                        for i in range(len(ks))
                        for j in range(i + 1, len(ks))]
                factL.append(float(np.mean(mags)) if mags else float('nan'))
            # query PROJECTION at the query token vs each fact KEY (the real
            # readout score q . k^, averaged over layers then heads).
            q_at = [ql[li][:, query_pos] for li in range(n_layers)]  # [L,H,d,2]
            tgt = [kl[li][:, key_pos[tidx]] for li in range(n_layers)]
            q_tgt = float(cip_mag(torch.stack(q_at), torch.stack(tgt)).mean().item())
            others = [kl[li][:, p] for li in range(n_layers)
                      for p in key_pos if p != key_pos[tidx]]
            if others:
                q_oth = float(np.mean([
                    cip_mag(torch.stack(q_at), torch.stack([o])).mean().item()
                    for o in others]))
            else:
                q_oth = float('nan')
            agg.setdefault(associations, []).append(
                {'seed': seed, 'factL': factL, 'q_tgt': q_tgt, 'q_oth': q_oth})
            print(f'{associations:>5} {seed:>5} | ' +
                  ' | '.join(f'{v:8.4f}' for v in factL) +
                  f' | {q_tgt:8.4f}     {q_oth:8.4f}')
        rows = agg[associations]
        mean_factL = [float(np.mean([r['factL'][li] for r in rows]))
                      for li in range(n_layers)]
        m_qt = float(np.mean([r['q_tgt'] for r in rows]))
        m_qo = float(np.mean([r['q_oth'] for r in rows]))
        ratio = float(np.mean(mean_factL)) / rand
        print(f'  assoc={associations:>2} mean fact off-diag={np.mean(mean_factL):.4f} '
              f'({ratio:.2f}x random)  query/target={m_qt:.4f}  query/other={m_qo:.4f}')
    print()
    print('random baseline (self-calibrating): fact off-diag ~ random means the')
    print('fact keys are ALREADY angularly separable -> write-dynamics problem.')
    print('fact off-diag >> random means the address space is clustered -> geometry')
    print('problem (no readout-magnitude fix or retrain can separate the keys).')


if __name__ == '__main__':
    main()
