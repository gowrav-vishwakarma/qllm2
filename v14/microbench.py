r"""B0 micro-bench: where does the memory of today's best body fail?

The v13_sempty recall micro-bench (16 keys, 8 values, a1/a4/a8) is solved
by echo (1.00 everywhere), so it can no longer separate designs. B0 asks
harder questions of the same small model (real arm, 4 layers, hd 64):

  facts   n invented key -> value records (n = 1 .. 64), query some keys.
  bind    e entities, each with 3 attributes (colour / home / owner) in a
          shuffled order; query one attribute of one entity. A query needs
          the entity AND the relation: the case V6 HSB failed on.
  rebind  n keys bound, then n more records rebind random keys; the query
          must return the NEWEST value (old values are the distractors).

Every document holds 4 queries (answers teacher-forced), so 4 supervised
positions per document (V12 finding 18: supervision density). Scoring is
contrastive over 8 candidates drawn from the same document where possible
(hard distractors): chance = 0.125. Words are random single GPT-2 tokens,
redrawn per document, so nothing can be memorised across documents.

Train lengths 512..4096; eval at 1024 / 4096 and 8192 (2x the longest train
length: extrapolation). Every arm trains on byte-identical data.

    .venv/bin/python -m v14.microbench --arms delta,echo --steps 6000 --batch 16

Arms (all real arm, chrono + out_gate + delta):
    delta        the v13 memory without echo
    echo         + E1 echo: the current control body
    n1sum / n1gate / n1phase   echo body with N1 (2 notebooks, sum/gate/phase)
Optional suffixes: @<head_dim> (default 64), :L<n_layers> (default 4).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import re
import subprocess
import time
from dataclasses import dataclass

import torch

from memory_probes.rules import TASKS as RULE_TASKS, build_rule_example
from v13_sempty.config import PAMConfig
from v13_sempty.triton_kernels import set_kernel_enabled
from v14.config import from_pam
from v14.units import V14LM

FILLER = (
    'The archive clerk reviewed an ordinary report and placed it on the shelf. '
    'The weather remained mild while routine work continued through the day. '
    'Several notes described roads, buildings, meals, and uneventful meetings. '
)
RELATIONS = (' colour', ' home', ' owner')
N_QUERIES = 4
N_CANDIDATES = 8
# Small sizes are the easy rung the lookup circuit forms on (the v13
# micro-bench learned a1 first). B0 run 1 (sizes from 8 up, answer-only)
# stayed at chance for every arm: logs/v14_b0_a0f672a_20261029_1232.log.
SIZES = {'facts': (1, 2, 4, 8, 16, 32, 64), 'bind': (1, 2, 4, 8, 16), 'rebind': (2, 4, 8, 16)}
POSITIONS = (0.0, 0.5, 1.0)
# B1: in-context rule tasks from the shared generator (memory_probes/rules.py,
# the v13 session's learning probe). Answers are never in the context.
RULE_SHOTS_TRAIN = (1, 2, 3, 4, 5, 6, 7, 8)
RULE_SHOTS_EVAL = (2, 4, 8)
RULE_SEP = '\n\n'


# ── vocabulary pools ────────────────────────────────────────────────────────

class Pools:
    """Disjoint pools of single-token lowercase words (fixed split)."""

    def __init__(self, tokenizer):
        words = sorted(t for t in tokenizer.get_vocab() if re.fullmatch(r'Ġ[a-z]{4,8}', t))
        random.Random(1234).shuffle(words)
        ids = [tokenizer.convert_tokens_to_ids(w) for w in words]
        self.keys, self.values, self.entities = ids[:600], ids[600:1200], ids[1200:1500]
        enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
        self.relations = [enc(r) for r in RELATIONS]
        assert all(len(r) == 1 for r in self.relations), "relation words must be one token"
        self.relations = [r[0] for r in self.relations]
        self.means = enc(' means')
        self.colon = enc(':')
        self.comma = enc(',')
        self.period_nl = enc('.\n')
        self.query = enc('\nQuery:')
        self.filler_unit = enc(FILLER)

    def filler(self, n: int) -> list[int]:
        if n <= 0:
            return []
        unit = self.filler_unit
        return (unit * (n // len(unit) + 1))[:n]


# ── documents ───────────────────────────────────────────────────────────────

@dataclass
class Doc:
    ids: list[int]                         # length L + 1 (input = ids[:-1])
    answers: list[tuple[int, int, list[int]]]   # (index in ids, target id, 8 candidates)


def _candidates(rng, target, hard, pool):
    """target + up to 7 hard distractors (same doc) + random fill from pool."""
    hard = [h for h in dict.fromkeys(hard) if h != target]
    rng.shuffle(hard)
    cands = [target] + hard[:N_CANDIDATES - 1]
    while len(cands) < N_CANDIDATES:
        w = rng.choice(pool)
        if w not in cands:
            cands.append(w)
    rng.shuffle(cands)
    return cands


def _facts(P, rng, n):
    keys, vals = rng.sample(P.keys, n), rng.sample(P.values, n)
    records = []
    for k, v in zip(keys, vals):
        records += [k] + P.means + [v] + P.period_nl
    asked = rng.sample(range(n), min(N_QUERIES, n))
    queries = [(P.query + [keys[i]] + P.means, vals[i], vals) for i in asked]
    return records, queries


def _bind(P, rng, e):
    ents = rng.sample(P.entities, e)
    vals = rng.sample(P.values, 3 * e)
    table = {(i, r): vals[3 * i + r] for i in range(e) for r in range(3)}
    records = []
    for i, ent in enumerate(ents):
        order = list(range(3))
        rng.shuffle(order)
        records += [ent] + P.colon
        for j, r in enumerate(order):
            records += [P.relations[r], table[(i, r)]] + (P.comma if j < 2 else P.period_nl)
    queries = []
    for _ in range(N_QUERIES):
        i, r = rng.randrange(e), rng.randrange(3)
        hard = [table[(i, rr)] for rr in range(3)] + [table[(ii, r)] for ii in range(e)]
        queries.append((P.query + [ents[i], P.relations[r]], table[(i, r)], hard))
    return records, queries


def _rebind(P, rng, n):
    keys = rng.sample(P.keys, n)
    vals = iter(rng.sample(P.values, 2 * n))
    history = {k: [] for k in keys}
    events = [k for k in keys]
    rng.shuffle(events)
    events += [rng.choice(keys) for _ in range(n)]
    records = []
    for k in events:
        v = next(vals)
        history[k].append(v)
        records += [k] + P.means + [v] + P.period_nl
    rebound = [k for k in keys if len(history[k]) > 1]
    pick = rng.sample(rebound, min(len(rebound), N_QUERIES - N_QUERIES // 4))
    others = [k for k in keys if k not in pick]
    pick += rng.sample(others, min(len(others), N_QUERIES - len(pick)))
    every = [v for hs in history.values() for v in hs]
    queries = [(P.query + [k] + P.means, history[k][-1], history[k][:-1] + every)
               for k in pick]
    return records, queries


FAMILIES = {'facts': _facts, 'bind': _bind, 'rebind': _rebind}


class CachedTokenizer:
    """Memoised ``encode`` for the rule generator, which re-encodes the same
    short words (letters, numbers, labels, candidates) for every example."""

    def __init__(self, tok):
        self.tok, self.cache = tok, {}

    def encode(self, text, add_special_tokens=False):
        key = (text, add_special_tokens)
        if key not in self.cache:
            self.cache[key] = self.tok.encode(text, add_special_tokens=add_special_tokens)
        return self.cache[key]


def build_rules_doc(P, tok, length, seed, tasks=RULE_TASKS, shots=RULE_SHOTS_TRAIN) -> Doc:
    """B1: rule episodes packed into one ``length + 1`` sequence, filler after.

    Each episode is a shared-generator prompt + its answer + a blank line;
    every answer is supervised / scored. Episodes are independent, so the
    memory must also learn to start each episode afresh."""
    rng = random.Random(seed)
    ids, answers = [], []
    sep = tok.encode(RULE_SEP, add_special_tokens=False)
    while True:
        ex = build_rule_example(tok, rng.choice(tasks), rng.choice(shots), rng.randrange(1 << 30))
        episode = ex.prompt_ids + [ex.target_token_id] + sep
        if len(ids) + len(episode) > length + 1:
            break
        answers.append((len(ids) + len(ex.prompt_ids), ex.target_token_id,
                        list(ex.candidate_token_ids)))
        ids += episode
    if not answers:
        raise ValueError(f"no rule episode fits in {length}")
    return Doc(ids + P.filler(length + 1 - len(ids)), answers)


def build_doc(P, family, size, length, position, seed, tok=None) -> Doc:
    """One document of exactly ``length + 1`` tokens."""
    if family == 'rules':
        return build_rules_doc(P, tok, length, seed)
    rng = random.Random(seed)
    records, queries = FAMILIES[family](P, rng, size)
    tail, answers_rel = [], []
    for prompt, target, hard in queries:
        tail += prompt
        answers_rel.append((len(tail), target, _candidates(rng, target, hard, P.values)))
        tail.append(target)
    n_fill = length + 1 - len(records) - len(tail)
    if n_fill < 0:
        raise ValueError(f"{family}{size} does not fit in {length}")
    before = int(round(n_fill * position))
    head = P.filler(before) + records + P.filler(n_fill - before)
    ids = head + tail
    answers = [(len(head) + i, t, c) for i, t, c in answers_rel]
    return Doc(ids, answers)


def batch_tensors(docs, device, aux_weight: float = 0.0):
    """x, y and the loss mask for one equal-length batch.

    Answer positions get weight 1 each. With ``aux_weight = w > 0`` the other
    positions of a document share a total weight ``w * n_answers`` (so the
    loss is answer CE plus w x the mean CE elsewhere, per answer). That LM
    signal is what copy circuits form from: answer-only training left the
    delta arm at ln(600) (it learned only "some value word")."""
    t = torch.tensor([d.ids for d in docs], dtype=torch.long, device=device)
    x, y = t[:, :-1].contiguous(), t[:, 1:].contiguous()
    T = y.shape[1]
    mask = torch.zeros_like(y, dtype=torch.float32)
    for b, d in enumerate(docs):
        n = len(d.answers)
        if aux_weight > 0:
            mask[b] = aux_weight * n / (T - n)
        for idx, target, _ in d.answers:
            assert int(y[b, idx - 1]) == target
            mask[b, idx - 1] = 1.0
    return x, y, mask


# ── model arms ──────────────────────────────────────────────────────────────

ARMS = {
    'delta': dict(echo=False),
    'echo': dict(echo=True),
    'n1sum': dict(echo=True, phase_states=2, phase_read='sum'),
    'n1gate': dict(echo=True, phase_states=2, phase_read='gate'),
    'n1phase': dict(echo=True, phase_states=2, phase_read='phase'),
}


def make_cfg(kind, vocab, head_dim=64, n_layers=4, max_seq_len=8192):
    base = PAMConfig(
        vocab_size=vocab, dim=6 * head_dim, n_heads=6, head_dim=head_dim,
        n_layers=n_layers, expand=3, dropout=0.0, max_seq_len=max_seq_len,
        chunk_size=256, gradient_checkpointing=False, is_complex=False,
        chrono=True, out_gate=True, delta=True)
    return from_pam(base, **ARMS[kind])


def parse_arms(spec):
    out = []
    for tok in spec.split(','):
        tok, hd, nl = tok.strip(), 64, 4
        if ':L' in tok:
            tok, nl = tok.split(':L'); nl = int(nl)
        if '@' in tok:
            tok, hd = tok.split('@'); hd = int(hd)
        if tok not in ARMS:
            raise SystemExit(f"unknown arm '{tok}' (have {list(ARMS)})")
        out.append((f"{tok}@hd{hd}L{nl}", tok, hd, nl))
    return out


# ── train / eval ────────────────────────────────────────────────────────────

def train_plan(steps, batch, seed, families=tuple(SIZES)):
    """Per step: (length, [(family, size, position, doc_seed)])."""
    rng = random.Random(seed)
    if tuple(families) == ('rules',):     # B1: short episodes, no long context needed
        lengths, weights = (512, 1024), (0.5, 0.5)
    else:
        lengths, weights = (512, 1024, 2048, 4096), (0.35, 0.30, 0.20, 0.15)
    plan, doc_seed = [], 10_000_000
    for _ in range(steps):
        L = rng.choices(lengths, weights=weights)[0]
        specs = []
        for _ in range(batch):
            fam = rng.choice(tuple(families))
            size = rng.choice(SIZES[fam]) if fam in SIZES else 0
            specs.append((fam, size, rng.choice(POSITIONS), doc_seed))
            doc_seed += 1
        plan.append((L, specs))
    return plan


@torch.no_grad()
def _score(model, embed, docs, device, eval_batch=8):
    """Hits / total over every answer of ``docs`` (contrastive, 8 candidates)."""
    hit = tot = 0
    for i in range(0, len(docs), eval_batch):
        chunk = docs[i:i + eval_batch]
        x, _, _ = batch_tensors(chunk, device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            lm, _ = model._hidden_to_lm(x)
        h = lm.raw(*lm.layout).float()
        for b, d in enumerate(chunk):
            for idx, target, cands in d.answers:
                scores = h[b, idx - 1] @ embed[cands].float().T
                hit += int(cands[int(scores.argmax())] == target)
                tot += 1
    return hit, tot


def evaluate_rules(model, P, tok, device, trials, length=1024):
    """B1 cells: one (task, shots) per document, held-out seeds (2e9+)."""
    model.eval()
    embed = model.embed.embed.weight
    out = {}
    for task in RULE_TASKS:
        for shots in RULE_SHOTS_EVAL:
            docs = [build_rules_doc(P, tok, length, 2_000_000_000 + 7919 * s,
                                    tasks=(task,), shots=(shots,)) for s in range(trials)]
            hit, tot = _score(model, embed, docs, device)
            out[f"{task}{shots}"] = {'acc': hit / max(tot, 1), 'n': tot}
    model.train()
    return out


def evaluate(model, P, device, ctxs, trials, eval_batch=8, families=tuple(SIZES)):
    """Accuracy per (family, size, ctx), and per position, over 8 candidates."""
    model.eval()
    embed = model.embed.embed.weight
    out = {}
    for fam, sizes in SIZES.items():
        if fam not in families:
            continue
        for size in sizes:
            for ctx in ctxs:
                docs = [(pos, build_doc(P, fam, size, ctx, pos, 2_000_000_000 + 7919 * s + int(pos * 10)))
                        for pos in POSITIONS for s in range(trials)]
                hits = {pos: [0, 0] for pos in POSITIONS}
                for i in range(0, len(docs), eval_batch):
                    chunk = docs[i:i + eval_batch]
                    x, _, _ = batch_tensors([d for _, d in chunk], device)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        lm, _ = model._hidden_to_lm(x)
                    h = lm.raw(*lm.layout).float()                       # [B, T, D]
                    for b, (pos, d) in enumerate(chunk):
                        for idx, target, cands in d.answers:
                            scores = h[b, idx - 1] @ embed[cands].float().T
                            hits[pos][0] += int(cands[int(scores.argmax())] == target)
                            hits[pos][1] += 1
                total = [sum(h[0] for h in hits.values()), sum(h[1] for h in hits.values())]
                out[f"{fam}{size}@{ctx}"] = {
                    'acc': total[0] / total[1],
                    'by_pos': {str(p): h[0] / max(h[1], 1) for p, h in hits.items()},
                }
    model.train()
    return out


def train_arm(label, cfg, P, plan, device, lr, log_every, ctxs, trials, aux_weight=0.0,
              tok=None, families=tuple(SIZES)):
    torch.manual_seed(0)
    model = V14LM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=len(plan), pct_start=0.05,
        anneal_strategy='cos', div_factor=10, final_div_factor=10)
    t0, toks = time.time(), 0

    def make_batch(i):
        L, specs = plan[i]
        return [build_doc(P, *spec[:2], L, *spec[2:], tok=tok) for spec in specs]

    # Build batch i+1 on a CPU thread while the GPU runs batch i.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending = pool.submit(make_batch, 0)
    for step, (L, specs) in enumerate(plan, 1):
        docs = pending.result()
        if step < len(plan):
            pending = pool.submit(make_batch, step)
        x, y, mask = batch_tensors(docs, device, aux_weight)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            lm, _ = model._hidden_to_lm(x)
        loss = model.ce_from_lm(lm, y, loss_mask=mask, chunk=4096, gemm_dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        toks += x.numel()
        if step % log_every == 0 or step == 1:
            print(f"  [{label}] step {step:5d}/{len(plan)} L={L:5d} loss={float(loss):.4f} "
                  f"| {toks / (time.time() - t0):,.0f} tok/s", flush=True)
    pool.shutdown()
    train_s = time.time() - t0
    print(f"  [{label}] trained {n_params / 1e6:.1f}M params in {train_s:.0f}s "
          f"({1000 * train_s / len(plan):.0f} ms/step)", flush=True)
    res = evaluate(model, P, device, ctxs, trials, families=families)
    if 'rules' in families:
        res.update(evaluate_rules(model, P, tok, device, trials))
    return {'params': n_params, 'train_s': train_s, 'ms_per_step': 1000 * train_s / len(plan),
            'results': res}


def table(name, res, ctxs):
    lines = [f"== {name}  (accuracy over 8 candidates; chance 0.125)"]
    if any('@' in k for k in res):
        lines.append("task       " + "".join(f"{c:>8}" for c in ctxs))
        for fam, sizes in SIZES.items():
            for size in sizes:
                if f'{fam}{size}@{ctxs[0]}' not in res:
                    continue
                row = f"{fam}{size:<{10 - len(fam)}}"
                row += "".join(f"{res[f'{fam}{size}@{c}']['acc']:8.2f}" for c in ctxs)
                lines.append(row)
    if any(k.startswith(RULE_TASKS) for k in res):
        lines.append("rule task       " + "".join(f"{f'{n}-shot':>8}" for n in RULE_SHOTS_EVAL))
        for task in RULE_TASKS:
            lines.append(f"{task:<16}" + "".join(
                f"{res[f'{task}{n}']['acc']:8.2f}" for n in RULE_SHOTS_EVAL))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='delta,echo')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--trials', type=int, default=16, help="docs per (task, ctx, position)")
    ap.add_argument('--log_every', type=int, default=250)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--ctxs', default='1024,4096,8192', help="eval context lengths")
    ap.add_argument('--families', default='facts,bind,rebind',
                    help="B0: facts,bind,rebind  |  B1: rules (shared rule generator)")
    ap.add_argument('--aux_weight', type=float, default=0.0,
                    help="LM loss weight on non-answer positions (0 = answer-only)")
    ap.add_argument('--json', default='', help="write all results here")
    args = ap.parse_args()

    set_kernel_enabled(True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    P = Pools(tok)
    tok_rules = CachedTokenizer(tok)
    ctxs = tuple(int(c) for c in args.ctxs.split(','))
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    print(f"B0 micro-bench @ {commit}: {args.steps} steps x B{args.batch}, lr {args.lr}, "
          f"eval ctx {ctxs}, {args.trials} docs x {len(POSITIONS)} positions x "
          f"{N_QUERIES} queries per cell, aux_weight {args.aux_weight}, "
          f"families {args.families}", flush=True)
    families = tuple(f.strip() for f in args.families.split(','))
    unknown = [f for f in families if f not in SIZES and f != 'rules']
    if unknown:
        raise SystemExit(f"unknown families {unknown}")
    plan = train_plan(args.steps, args.batch, args.seed, families)
    all_res = {'commit': commit, 'args': vars(args), 'arms': {}}
    for label, kind, hd, nl in parse_arms(args.arms):
        print(f"\n--- arm {label} ---", flush=True)
        cfg = make_cfg(kind, len(tok), head_dim=hd, n_layers=nl)
        all_res['arms'][label] = train_arm(label, cfg, P, plan, args.device, args.lr,
                                           args.log_every, ctxs, args.trials,
                                           args.aux_weight, tok=tok_rules, families=families)
        print(table(label, all_res['arms'][label]['results'], ctxs), flush=True)
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(all_res, f, indent=1)
    print("\n" + "=" * 64)
    for label, r in all_res['arms'].items():
        print(table(label, r['results'], ctxs) + "\n")


if __name__ == '__main__':
    main()
