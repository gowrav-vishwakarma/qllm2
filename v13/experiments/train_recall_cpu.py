"""CPU training proof: the gated-erase delta (E2b) is a machine-learning fix.

Trains THREE identical small V13 models (same seed, same data, same hyperparams)
differing only in the memory write dynamics:

  additive      S <- g*S + ws*v k*                    (V11 hero dynamics)
  delta_legacy  S <- g*S + b*(ws*v - pred) k*         (current V13 default)
  delta_gated   S <- g*S + (bw*ws*v - be*pred) k*     (E2b: learned erase gate)

ONE-SHOT fact-recall task (no memorization shortcut):
  * 256 subjects x 256 values; every doc draws its facts FRESH (no replacement),
    so a pair stated once is never seen again -> recall MUST come from state.
  * fact:  [THE <S> IS <V> .]
  * query: [WHAT IS THE <S> ? <V>]  -- value predicted from the earlier statement
  * CE is applied ONLY to query-answer value tokens (pure memory signal)

Run:  /home/gowrav/venv-qllm/bin/python -m v13.experiments.train_recall_cpu
      (or --quick for a 100-step sanity pass)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v13.model import V13Config, V13LM

RESULTS_DIR = Path(__file__).parent / 'results'

# ── synthetic vocab layout ──────────────────────────────────────────────────
N_SUBJ = 256
N_VAL = 256
SUBJ_LO, SUBJ_HI = 0, N_SUBJ            # subject tokens: [0, 256)
VAL_LO, VAL_HI = 256, 256 + N_VAL       # value tokens: [256, 512)
THE, IS, WHAT, QMARK, PERIOD = 512, 513, 514, 515, 516
VOCAB = 517


def _ri(rng, n: int) -> int:
    return int(rng.integers(0, n))


def _rand(rng) -> float:
    return float(rng.random())


def make_doc(rng, length: int) -> torch.Tensor:
    """One doc: fresh facts (no replacement), interleaved queries about them."""
    ids: list[int] = []
    stated: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    while len(ids) < length:
        if stated and _rand(rng) < 0.25:
            s, v = stated[int(rng.integers(0, len(stated)))]
            ids += [WHAT, IS, THE, s + SUBJ_LO, QMARK, v + VAL_LO]
        else:
            for _ in range(20):
                s, v = _ri(rng, N_SUBJ), _ri(rng, N_VAL)
                if (s, v) not in used:
                    used.add((s, v))
                    break
            else:
                s, v = _ri(rng, N_SUBJ), _ri(rng, N_VAL)
            ids += [THE, s + SUBJ_LO, IS, v + VAL_LO, PERIOD]
            stated.append((s, v))
    return torch.tensor(ids[:length], dtype=torch.long)


def make_eval_doc(rng, s: int, v: int, gap: int) -> tuple[torch.Tensor, int]:
    """Fresh fact stated first, `gap` filler tokens, then the query.
    Returns (doc, index-of-QMARK): logits at QMARK must score the value."""
    filler: list[int] = []
    while len(filler) < gap:
        s2, v2 = _ri(rng, N_SUBJ), _ri(rng, N_VAL)
        filler += [THE, s2 + SUBJ_LO, IS, v2 + VAL_LO, PERIOD]
    doc = [THE, s + SUBJ_LO, IS, v + VAL_LO, PERIOD] + filler[:gap] + \
          [WHAT, IS, THE, s + SUBJ_LO, QMARK]
    return torch.tensor(doc, dtype=torch.long), len(doc) - 1


def recall_accuracy(model: V13LM, eval_pairs, gaps, seed: int = 7) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    per_gap: dict[int, list[float]] = {g: [] for g in gaps}
    for s, v in eval_pairs:
        for gap in gaps:
            doc, qidx = make_eval_doc(rng, s, v, gap)
            with torch.no_grad():
                logits, _, _ = model(doc.unsqueeze(0))
            logit_q = logits[0, qidx, VAL_LO:VAL_HI]
            per_gap[gap].append(1.0 if int(logit_q.argmax()) == v - VAL_LO else 0.0)
    return {str(g): float(sum(a) / len(a)) for g, a in per_gap.items()}


def erase_gate_stats(model: V13LM, doc: torch.Tensor) -> dict:
    """Mean erase-beta at each layer's real input x, on VALUE vs filler tokens."""
    model.eval()
    with torch.no_grad():
        z = model.embed(doc.unsqueeze(0))
        z = model.embed_norm(z)
        totals = {'value': [], 'filler': []}
        for block in model.blocks:
            pam = block.pam
            if getattr(pam, 'erase_beta_proj', None) is None:
                return {}
            _, eb = pam._gate_betas(z)  # [B,H,T] at this layer's real input
            eb_t = eb.mean(dim=(0, 1)).squeeze(0)  # [T]
            for t in range(doc.shape[0]):
                tok = int(doc[t])
                kind = 'value' if VAL_LO <= tok < VAL_HI else 'filler'
                totals[kind].append(eb_t[t].item())
            z, _ = block(z)
        return {k: float(sum(v) / len(v)) for k, v in totals.items() if v}


def build_model(write_mode: str, erase_gate: bool, seed: int = 0) -> V13LM:
    torch.manual_seed(seed)
    cfg = V13Config(
        vocab_size=VOCAB, dim=96, n_heads=3, head_dim=32, n_layers=3,
        expand=3, dropout=0.0, max_seq_len=256, chunk_size=64,
        gradient_checkpointing=False, use_learned_pos=False, use_rope=True,
        use_gsp=True, fused_qkv=True, qk_norm=False, tie_weights=True,
        activation='swish', decay_mode='head', write_mode=write_mode,
        delta_chunk=32, n_states=3, state_dt_spread=2.0, base_dt_bias=-4.0,
        gate_content_aware=True, protect_gate_bias=-3.0,
        routing_content_aware=False, state_compete=False, phase_init='zero',
        route_balance_lambda=0.0, aux_loss_weight=1.0,
        fused_e3=False, recompute_pam_chunks=False, gamma_floor=0.0,
        gate_surprisal_lambda=0.0,
        vault_state=True, vault_state_idx=0, write_phase_address=True,
        delta_erase_gate=erase_gate,
    )
    return V13LM(cfg)


def answer_mask(ids: torch.Tensor) -> torch.Tensor:
    """1 at target positions t where ids[t] is a value token AND ids[t-1]==QMARK."""
    is_val = (ids >= VAL_LO) & (ids < VAL_HI)
    prev = torch.cat([torch.full((ids.shape[0], 1), -1, dtype=ids.dtype,
                                 device=ids.device), ids[:, :-1]], dim=1)
    return (is_val & (prev == QMARK)).float()


def train_one(name: str, write_mode: str, erase_gate: bool, steps: int,
              seq_len: int, batch: int, lr: float, eval_pairs,
              gaps, log_every: int = 50) -> dict:
    model = build_model(write_mode, erase_gate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(1234)
    model.train()
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        ids = torch.stack([make_doc(rng, seq_len) for _ in range(batch)])
        logits, _, _ = model(ids)
        lm = answer_mask(ids)                       # [B,T]
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                             ids[:, 1:].reshape(-1), reduction='none')
        ce = ce.reshape(batch, seq_len - 1)
        mask = lm[:, 1:].reshape(-1)
        loss = (ce.reshape(-1) * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            acc = recall_accuracy(model, eval_pairs, gaps)
            hist.append({'step': step, 'loss': float(loss), **acc})
            print(f'  [{name}] step {step:4d} ans_loss={loss.item():.3f} ' +
                  ' '.join(f'g{g}={acc[str(g)]:.2f}' for g in gaps), flush=True)
    final_acc = recall_accuracy(model, eval_pairs, gaps)
    gate_stats = erase_gate_stats(model, make_doc(rng, seq_len)) if erase_gate else {}
    return {
        'name': name, 'write_mode': write_mode, 'erase_gate': erase_gate,
        'final_ans_loss': hist[-1]['loss'], 'recall': final_acc,
        'erase_gate_mean': gate_stats or None,
        'wall_s': round(time.time() - t0, 1), 'history': hist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--seq', type=int, default=256)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gaps', type=int, nargs='+', default=[16, 64, 128, 208])
    ap.add_argument('--pairs', type=int, default=16)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.steps, args.batch, args.pairs = 100, 4, 8

    torch.manual_seed(777)
    rng = np.random.default_rng(777)
    eval_pairs = [(int(rng.integers(0, N_SUBJ)), int(rng.integers(0, N_VAL)))
                  for _ in range(args.pairs)]

    print(f'E2b ML proof (one-shot) | steps={args.steps} seq={args.seq} '
          f'batch={args.batch} gaps={args.gaps} chance=1/{N_VAL}={1/N_VAL:.4f}')
    results = []
    for name, wm, eg in [
        ('additive', 'additive', False),
        ('delta_legacy', 'delta', False),
        ('delta_gated(E2b)', 'delta', True),
    ]:
        print(f'\n=== {name} ===', flush=True)
        results.append(train_one(name, wm, eg, args.steps, args.seq, args.batch,
                                 args.lr, eval_pairs, args.gaps))

    print('\n' + '=' * 78)
    print(f"{'model':<18} {'ans_loss':>9} " +
          ' '.join(f'{f"gap{g}":>7}' for g in args.gaps) + '  erase_stats')
    for r in results:
        row = ' '.join(f'{r["recall"][str(g)]:>7.2f}' for g in args.gaps)
        gs = r['erase_gate_mean']
        gst = (f"val={gs['value']:.3f} filler={gs['filler']:.3f}" if gs else '-')
        print(f'{r["name"]:<18} {r["final_ans_loss"]:>9.3f} {row}  {gst}  ({r["wall_s"]}s)')
    print('=' * 78)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f'train_recall_cpu_{int(time.time())}.json'
    out.write_text(json.dumps({'args': vars(args), 'results': results}, indent=2))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
