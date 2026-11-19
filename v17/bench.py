"""v17 Unified Intelligence vs. Memory Benchmark (IMB-17).

One model per (arm, seed) trained on a single mixed stream (max train len = 1024,
real DCLM-edu prose + reasoning/memory episodes, byte-identical batches across arms)
and evaluated zero-shot on 6 held-out axes (disjoint eval words, held-out rule
parameters, held-out scope directions, and 2x / 4x context-length extrapolation):

  1. prose_lure_new (@512, @2048=2x, @4096=4x):
     Natural prose facts with 2 same-type lures + cleft/reverse wordings
     ("It was V that E chose") on UNSEEN entity words (bag-of-values cap = 0.333).
  2. scope_holdout (@512, @2048=2x):
     4-axis scoped rule tuple (Presence P, Carry C, Held-out Mismatch Scope S on
     domain pairs never paired in training, and Local Precedence Override R).
  3. two_hop_new (@512, @2048=2x):
     4 shuffled 2-hop chains ("A -> B ... B -> C") across filler on UNSEEN words,
     with hop-1 bridges B_j included in the 8 candidates (1-hop lookup scores 0).
  4. unseen_rules (@0, @2048=2x):
     B2 held-out rule parameters never seen in training (shift_wide_new k in {5,9},
     step_wide_new d in {7,13,17}, affine_map_new b in {7,13,17}), reporting both
     8-way accuracy and continuous target logit margin M_rule.
  5. rxk_new (@512, @2048=2x):
     Rule x Fact composition: 4 demos of an UNSEEN letter shift k in {5,9} +
     4 stored code->letter facts + filler + "Apply rule to code K ->".
  6. extrap_recall (@4096 = 4x max train window) + web_nll (65k real DCLM holdout tokens).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memory_probes.spread import split_units  # noqa: E402
from sempyt import NamedTensor  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from v13_sempty.fused_ce import fused_linear_cross_entropy  # noqa: E402
from v14.rules_ood import (  # noqa: E402
    B2_NEW_TASKS,
    B2_TASKS,
    SHIFT_NEW,
    SHIFT_TRAIN,
    build_example as b2_build_example,
)
from v17.model import build_arm  # noqa: E402

HOLDOUT_CACHE = ROOT / ".cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt"


# ── Cached tokenizer & disjoint vocabulary / web-prose pools ─────────────────

class FastTok:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        self.cache: dict[str, list[int]] = {}

    def enc(self, s: str) -> list[int]:
        if s not in self.cache:
            self.cache[s] = self.tok.encode(s, add_special_tokens=False)
        return self.cache[s]

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return self.enc(s)

    def get_vocab(self):
        return self.tok.get_vocab()

    def convert_tokens_to_ids(self, t):
        return self.tok.convert_tokens_to_ids(t)


class IMBPools:
    """Strictly disjoint train vs eval word pools + real DCLM prose split."""

    DOMAINS = (
        ("mass", (" mg", " kg", " oz", " lb", " ton", " gm", " ug", " cwt")),
        ("volume", (" ml", " gal", " pt", " qt", " cup", " tsp", " cc", " bbl")),
        ("length", (" mm", " km", " yd", " ft", " cm", " mi", " nm", " in")),
        ("time", (" ms", " hr", " sec", " min", " day", " wk", " yr", " ns")),
        ("speed", (" mph", " kt", " fps", " rpm", " bps", " mach", " knot", " cpm")),
        ("energy", (" cal", " erg", " btu", " rad", " lux", " amp", " volt", " watt")),
    )
    # Held-out mismatch pairs (query_domain, rule_domain) NEVER shown in training!
    SCOPE_HOLDOUT_PAIRS = {("volume", "mass"), ("mass", "volume"), ("time", "length"), ("length", "time")}

    def __init__(self, ftok: FastTok):
        self.ftok = ftok
        vocab = ftok.get_vocab()
        words = sorted(t for t in vocab if re.fullmatch(r"Ġ[a-z]{4,8}", t))
        rng = random.Random(20261119)
        rng.shuffle(words)
        w_ids = [ftok.convert_tokens_to_ids(w) for w in words]

        # Shared learned word pools (like v14/microbench.py: embeddings are trained, while
        # EVERY document redraws fresh random entity->val / A->B->C / key->val bindings so
        # zero associations can be memorised across documents!).
        self.entities = w_ids[0:240]
        self.hop_words = w_ids[240:640]
        self.keys = w_ids[640:1140]
        self.fact_vals = w_ids[1140:1640]
        fallback_units = w_ids[1640:1900]
        self.train_entities = self.entities
        self.eval_entities = self.entities
        self.train_hop = self.hop_words
        self.eval_hop = self.hop_words
        self.train_keys = self.keys
        self.eval_keys = self.keys

        # 4 prose attribute types x 8 guaranteed single-token values each
        raw_attrs = {
            "city": (" Rome", " Paris", " Tokyo", " London", " Berlin", " Boston", " Madrid", " Austin",
                     " Denver", " Dallas", " Miami", " Oxford", " Dublin", " Moscow"),
            "color": (" red", " blue", " green", " gold", " black", " white", " grey", " pink",
                      " brown", " silver", " purple", " orange"),
            "job": (" chef", " nurse", " pilot", " judge", " clerk", " guard", " baker", " coach",
                    " doctor", " lawyer", " writer", " driver"),
            "pet": (" dog", " cat", " goat", " frog", " duck", " wolf", " bear", " deer",
                    " horse", " sheep", " mouse", " snake"),
        }
        self.attr_types = {
            k: self._pick_single(v, fallback_units, 8) for k, v in raw_attrs.items()
        }

        # Domain units (4 distinct single-token units per domain) + bare period token
        self.dot_id = self._one(".")
        used_u: set[int] = {self.dot_id}
        self.domain_units: dict[str, list[int]] = {}
        for dom, ulist in self.DOMAINS:
            picked = self._pick_single(ulist, fallback_units, 4, exclude=used_u)
            used_u.update(picked)
            self.domain_units[dom] = picked
        self.all_unit_ids = [uid for ulist in self.domain_units.values() for uid in ulist]

        # Single-token lowercase letters a..z (with leading space) for shift / R x K
        self.letters = [chr(ord("a") + i) for i in range(26)]
        self.letter_ids = {ch: self._one(" " + ch) for ch in self.letters}

        # Real DCLM-edu web prose split (strictly disjoint, 20k buffer, <50257)
        if HOLDOUT_CACHE.exists():
            from memory_probes.spread import filler_tokens
            arr = np.asarray(filler_tokens(str(HOLDOUT_CACHE)), dtype=np.int64).reshape(-1)
            arr = arr[arr < 50257]
            self.web_train = arr[:380_000].tolist()
            self.web_eval = arr[400_000:490_000].tolist()
        else:
            unit = ftok.enc(
                " The committee reviewed several scientific reports during the morning session "
                "and recorded clear observations in the archive ledger."
            )
            self.web_train = unit * 4000
            self.web_eval = unit * 1000

    def _one(self, s: str) -> int:
        ids = self.ftok.enc(s)
        assert len(ids) == 1, f"expected single token for {s!r}, got {ids}"
        return ids[0]

    def _pick_single(self, cands: tuple[str, ...], fallback: list[int],
                     n: int, exclude: set[int] | None = None) -> list[int]:
        ex = exclude or set()
        out: list[int] = []
        for s in cands:
            ids = self.ftok.enc(s)
            if len(ids) == 1 and ids[0] not in ex and ids[0] not in out:
                out.append(ids[0])
                if len(out) == n:
                    return out
        for fid in fallback:
            if fid not in ex and fid not in out:
                out.append(fid)
                if len(out) == n:
                    return out
        raise RuntimeError(f"could not pick {n} single tokens from {cands}")

    def prose_slice(self, rng: random.Random, n: int, eval_split: bool = False) -> list[int]:
        if n <= 0:
            return []
        src = self.web_eval if eval_split else self.web_train
        if n >= len(src) - 2:
            out = (src * (n // len(src) + 2))[:n]
            return out
        start = rng.randrange(0, len(src) - n - 1)
        return src[start:start + n]


@dataclass
class EpisodeDoc:
    ids: list[int]                                      # length L + 1
    answers: list[tuple[int, int, list[int]]]           # (target_pos_in_ids, gold_id, 8_cand_ids)
    meta: str = ""


def _pack_to_len(P: IMBPools, rng: random.Random, body: list[int],
                 tail_q_and_ans: list[tuple[list[int], int, list[int]]],
                 L: int, eval_split: bool, pos_frac: float | None = None) -> EpisodeDoc:
    tail_ids: list[int] = []
    rel_ans: list[tuple[int, int, list[int]]] = []
    for q_ids, gold, cands in tail_q_and_ans:
        tail_ids.extend(q_ids)
        rel_ans.append((len(tail_ids), gold, cands))
        tail_ids.append(gold)
    rem = (L + 1) - len(body) - len(tail_ids)
    if rem < 0:
        # Trim front of body if needed (only happens on huge multi-episode packs)
        body = body[-max(0, (L + 1) - len(tail_ids)):]
        rem = (L + 1) - len(body) - len(tail_ids)
    frac = rng.choice((0.05, 0.25, 0.5)) if pos_frac is None else pos_frac
    pre_n = int(round(rem * frac))
    post_n = rem - pre_n
    head = P.prose_slice(rng, pre_n, eval_split) + body + P.prose_slice(rng, post_n, eval_split)
    full_ids = head + tail_ids
    assert len(full_ids) == L + 1
    answers = [(len(head) + idx, g, c) for idx, g, c in rel_ans]
    return EpisodeDoc(full_ids, answers)


def _eight_cands(rng: random.Random, gold: int, hard: list[int], pool: list[int]) -> list[int]:
    out = [gold]
    for h in hard:
        if h not in out and len(out) < 8:
            out.append(h)
    while len(out) < 8:
        w = rng.choice(pool)
        if w not in out:
            out.append(w)
    rng.shuffle(out)
    return out


# ── Generator 1: Prose Entity-Lure + Cleft Binding ───────────────────────────

def gen_prose_lure(P: IMBPools, rng: random.Random, L: int, eval_split: bool) -> EpisodeDoc:
    ent_pool = P.eval_entities if eval_split else P.train_entities
    enc = P.ftok.enc
    atype = rng.choice(tuple(P.attr_types.keys()))
    vals = P.attr_types[atype]
    # Include easy bootstrap rung n_ents=1,2 in training (v14 B0 lesson); eval always tests 3 (2 same-type lures!)
    n_ents = 3 if eval_split else rng.choice((1, 2, 3, 4))
    ents = rng.sample(ent_pool, n_ents)
    picked_vals = rng.sample(vals, n_ents)

    # 3 natural templates per attribute, INCLUDING cleft order (value BEFORE entity!)
    sents = []
    for ent, val in zip(ents, picked_vals):
        style = rng.randrange(3)
        if atype == "city":
            if style == 0:
                s = enc("\nProfile:") + [ent] + enc(" packed up last spring and moved to") + [val] + enc(".")
            elif style == 1:
                s = enc("\nAfter years working abroad,") + [ent] + enc(" finally settled in") + [val] + enc(".")
            else:
                s = enc("\nIt was") + [val] + enc(" that") + [ent] + enc(" chose as a permanent home.")
        elif atype == "color":
            if style == 0:
                s = enc("\nNote:") + [ent] + enc(" drives an old car painted") + [val] + enc(".")
            elif style == 1:
                s = enc("\nDown on the main street,") + [ent] + enc(" bought a") + [val] + enc(" coat.")
            else:
                s = enc("\nIt was a") + [val] + enc(" jacket that") + [ent] + enc(" wore to work.")
        elif atype == "job":
            if style == 0:
                s = enc("\nRecord:") + [ent] + enc(" has worked for ten years as a") + [val] + enc(".")
            elif style == 1:
                s = enc("\nEveryone in town knows that") + [ent] + enc(" serves as a") + [val] + enc(".")
            else:
                s = enc("\nWorking as a") + [val] + enc(" is how") + [ent] + enc(" earns a living.")
        else:
            if style == 0:
                s = enc("\nAt home,") + [ent] + enc(" keeps a tame") + [val] + enc(".")
            elif style == 1:
                s = enc("\nVisitors always notice that") + [ent] + enc(" owns a") + [val] + enc(".")
            else:
                s = enc("\nA quiet") + [val] + enc(" is the pet that") + [ent] + enc(" cares for.")
        sents.append(s)

    rng.shuffle(sents)
    body = [tok_id for s in sents for tok_id in s] + enc("\n\n")

    q_prompts = {
        "city": ("Question: Which city is home to", "?\nAnswer:"),
        "color": ("Question: What color belongs to", "?\nAnswer:"),
        "job": ("Question: What job is held by", "?\nAnswer: a"),
        "pet": ("Question: Which animal lives with", "?\nAnswer: a"),
    }
    p1, p2 = q_prompts[atype]
    n_q = 1 if eval_split else min(2, n_ents)
    q_indices = rng.sample(range(n_ents), n_q)
    tail = []
    for qi in q_indices:
        q_ids = enc("\n" + p1) + [ents[qi]] + enc(p2)
        lures = [picked_vals[j] for j in range(n_ents) if j != qi]
        cands = _eight_cands(rng, picked_vals[qi], lures, vals)
        tail.append((q_ids, picked_vals[qi], cands))
    return _pack_to_len(P, rng, body, tail, L, eval_split, pos_frac=0.1 if eval_split else None)


# ── Generator 2: 4-Axis Scoped Rule (Presence P, Carry C, Scope S, Override R) ─

def gen_scope_doc(P: IMBPools, rng: random.Random, L: int, eval_split: bool,
                  mode: str | None = None) -> EpisodeDoc:
    enc = P.ftok.enc
    doms = [d for d, _ in P.DOMAINS]
    if mode is None:
        mode = rng.choices(("none", "own", "other", "override"), weights=(0.25, 0.35, 0.25, 0.15))[0]

    if eval_split and mode == "other":
        q_dom, r_dom = rng.choice(sorted(P.SCOPE_HOLDOUT_PAIRS))
    elif mode == "other":
        pairs = [(qd, rd) for qd in doms for rd in doms if qd != rd and (qd, rd) not in P.SCOPE_HOLDOUT_PAIRS]
        q_dom, r_dom = rng.choice(pairs)
    else:
        q_dom = rng.choice(doms)
        r_dom = q_dom

    u_rule = rng.choice(P.domain_units[r_dom])
    val_num = f" {rng.randrange(11, 98)}"

    if mode == "none":
        proto = enc("\nSession notes follow below.\n")
        row = enc(f"\nMeasured {q_dom}:{val_num}.\n")
        gold = P.dot_id
        hard = P.domain_units[q_dom]
    elif mode == "own":
        proto = enc(f"\nProtocol: report every {r_dom} reading in") + [u_rule] + enc(".\n")
        row = enc(f"\nMeasured {q_dom}:{val_num}.\n")
        gold = u_rule
        hard = [P.dot_id] + [u for u in P.domain_units[q_dom] if u != u_rule]
    elif mode == "other":
        proto = enc(f"\nProtocol: report every {r_dom} reading in") + [u_rule] + enc(".\n")
        row = enc(f"\nMeasured {q_dom}:{val_num}.\n")
        gold = P.dot_id                                                     # other-domain rule MUST NOT fire!
        hard = [u_rule] + P.domain_units[q_dom]
    else:  # override (precedence R): row states explicit local unit u_local != u_rule
        u_local = rng.choice([u for u in P.domain_units[q_dom] if u != u_rule])
        proto = enc(f"\nProtocol: report every {r_dom} reading in") + [u_rule] + enc(".\n")
        row = enc(f"\nMeasured {q_dom}:{val_num}") + [u_local] + enc(".\n")
        gold = u_local
        hard = [u_rule, P.dot_id] + P.domain_units[q_dom]

    q_ids = row + enc(f"Question: what is the {q_dom}?\nAnswer:{val_num}")
    cands = _eight_cands(rng, gold, hard, P.all_unit_ids)
    doc = _pack_to_len(P, rng, proto, [(q_ids, gold, cands)], L, eval_split, pos_frac=0.05)
    doc.meta = mode
    return doc


# ── Generator 3: Multi-Chain 2-Hop Composition in O(1) State ─────────────────

def gen_two_hop(P: IMBPools, rng: random.Random, L: int, eval_split: bool) -> EpisodeDoc:
    enc = P.ftok.enc
    pool = P.eval_hop if eval_split else P.train_hop
    # Include 1-chain and 2-chain bootstrap rungs in training; eval always tests 4 shuffled chains (8 hops)
    n_chains = 4 if eval_split else rng.choice((1, 2, 3, 4))
    words = rng.sample(pool, 3 * n_chains)
    chains = [(words[3 * i], words[3 * i + 1], words[3 * i + 2]) for i in range(n_chains)]

    hop_sents = []
    for a, b, c in chains:
        v1 = rng.choice((" holds", " keeps", " owns"))
        v2 = rng.choice((" unlocks", " reveals", " opens"))
        hop_sents.append(enc("\nFact:") + [a] + enc(v1) + [b] + enc("."))
        hop_sents.append(enc("\nFact:") + [b] + enc(v2) + [c] + enc("."))
    rng.shuffle(hop_sents)

    # Sprinkle short prose sentences between facts so hops aren't trivially adjacent
    body: list[int] = []
    for hs in hop_sents:
        body.extend(hs)
        body.extend(P.prose_slice(rng, rng.randrange(4, 14), eval_split))

    all_bridges = [b for _, b, _ in chains]
    all_targets = [c for _, _, c in chains]
    n_q = 1 if eval_split else min(2, n_chains)
    asked = rng.sample(range(n_chains), n_q)
    tail = []
    for idx in asked:
        a, b_self, c_gold = chains[idx]
        q_ids = enc("\nQuestion: following both links from") + [a] + enc(", we reach")
        # Include b_self (the 1-hop trap!) + other targets + other bridges in 8 candidates!
        hard = [b_self] + [t for t in all_targets if t != c_gold] + [b for b in all_bridges if b != b_self]
        cands = _eight_cands(rng, c_gold, hard, pool)
        tail.append((q_ids, c_gold, cands))
    return _pack_to_len(P, rng, body, tail, L, eval_split, pos_frac=0.1 if eval_split else None)


# ── Generator 4: B2 Unseen Rule Inference (Adjacent & 2x-Extrap Spread) ──────

def gen_b2_rules(P: IMBPools, rng: random.Random, L: int, eval_split: bool,
                 task: str | None = None, spread: bool | None = None) -> EpisodeDoc:
    enc = P.ftok.enc
    if not eval_split:
        # Train ONLY on B2_TASKS (held-out parameters in B2_NEW_TASKS are NEVER trained!)
        t = rng.choice(B2_TASKS) if task is None else task
        use_spread = (rng.random() < 0.5) if spread is None else spread
        if not use_spread:
            # Pack multiple short rule episodes for high supervision density
            ids: list[int] = []
            ans: list[tuple[int, int, list[int]]] = []
            sep = enc("\n\n")
            while True:
                ex = b2_build_example(P.ftok, t, rng.choice((3, 4, 6, 8)), rng.randrange(1 << 29))
                ep = ex.prompt_ids + [ex.target_token_id] + sep
                if len(ids) + len(ep) > L + 1:
                    break
                ans.append((len(ids) + len(ex.prompt_ids), ex.target_token_id, list(ex.candidate_token_ids)))
                ids.extend(ep)
            if ans:
                ids.extend(P.prose_slice(rng, (L + 1) - len(ids), False))
                return EpisodeDoc(ids, ans, meta=t)

    t = (rng.choice(B2_NEW_TASKS) if task is None else task)
    ex = b2_build_example(P.ftok, t, 8, rng.randrange(1 << 29))
    use_spread = True if spread is True else (False if spread is False else False)
    if not use_spread:
        # Adjacent demos (@0), padded BEFORE the prompt with prose so sequence length is L+1
        q_ids = ex.prompt_ids
        cands = list(ex.candidate_token_ids)
        return _pack_to_len(P, rng, [], [(q_ids, ex.target_token_id, cands)], L, eval_split, pos_frac=1.0)

    # Spread each demonstration unit across real web prose filler via split_units!
    header, units = split_units(ex)
    demos, query_unit = units[:-1], units[-1]
    demo_chunks = [enc("\n" + header + demos[0])] + [enc("\n" + d) for d in demos[1:]]
    q_ids = enc("\n" + query_unit)
    total_demo_toks = sum(len(c) for c in demo_chunks)
    rem = max(0, (L + 1) - total_demo_toks - len(q_ids) - 1)
    gap = rem // max(1, len(demo_chunks))
    body: list[int] = []
    for c in demo_chunks:
        body.extend(c)
        body.extend(P.prose_slice(rng, gap, eval_split))
    return _pack_to_len(P, rng, body, [(q_ids, ex.target_token_id, list(ex.candidate_token_ids))],
                        L, eval_split, pos_frac=0.0)


# ── Generator 5: Rule x Fact Composition (rxk_new on UNSEEN shift {5, 9}) ────

def gen_rxk(P: IMBPools, rng: random.Random, L: int, eval_split: bool) -> EpisodeDoc:
    enc = P.ftok.enc
    k_shift = rng.choice(SHIFT_NEW if eval_split else SHIFT_TRAIN)
    key_pool = P.eval_keys if eval_split else P.train_keys
    usable = [ch for ch in P.letters if ord(ch) - ord("a") + k_shift < 26]

    n_keys = 4 if eval_split else rng.choice((1, 2, 4))
    picks = rng.sample(usable, 4 + n_keys)
    demos, fact_letters = picks[:4], picks[4:4 + n_keys]
    keys = rng.sample(key_pool, n_keys)

    demo_ids = enc("\nTransform rule:\n")
    for ch in demos:
        out_ch = chr(ord(ch) + k_shift)
        demo_ids += [P.letter_ids[ch]] + enc(" ->") + [P.letter_ids[out_ch]] + enc("\n")

    fact_ids = enc("\nStored registers:\n")
    pairs = list(zip(keys, fact_letters))
    rng.shuffle(pairs)
    for k_id, ch in pairs:
        fact_ids += enc("Code") + [k_id] + enc(" stores") + [P.letter_ids[ch]] + enc(".\n")

    # Separate rule demos and stored facts by prose filler!
    mid_gap = max(16, (L // 3))
    body = demo_ids + P.prose_slice(rng, mid_gap, eval_split) + fact_ids

    qi = rng.randrange(n_keys)
    q_key = keys[qi]
    in_ch = fact_letters[qi]
    gold_ch = chr(ord(in_ch) + k_shift)
    gold_id = P.letter_ids[gold_ch]

    # Hard decoys: unshifted copy, trained neighbor shifts k+-1, and shifted OTHER keys!
    hard = [P.letter_ids[in_ch]]
    for dk in (-1, 1, -2, 2):
        ks = k_shift + dk
        if 0 <= ord(in_ch) - ord("a") + ks < 26:
            hard.append(P.letter_ids[chr(ord(in_ch) + ks)])
    for j, oth_ch in enumerate(fact_letters):
        if j != qi:
            hard.append(P.letter_ids[chr(ord(oth_ch) + k_shift)])

    q_ids = enc("\nApply transform rule to code") + [q_key] + enc(" ->")
    cands = _eight_cands(rng, gold_id, hard, list(P.letter_ids.values()))
    return _pack_to_len(P, rng, body, [(q_ids, gold_id, cands)], L, eval_split, pos_frac=0.05)


# ── Generator 6: Episodic Recall (4x Extrap) + Pure Web LM Windows ───────────

def gen_facts_recall(P: IMBPools, rng: random.Random, L: int, eval_split: bool,
                     n_facts: int = 16) -> EpisodeDoc:
    enc = P.ftok.enc
    kpool = P.eval_keys if eval_split else P.train_keys
    n_facts_actual = 16 if eval_split else rng.choice((1, 2, 4, 8, 16))
    keys = rng.sample(kpool, n_facts_actual)
    vals = rng.sample(P.fact_vals, n_facts_actual)
    body: list[int] = []
    for k_id, v_id in zip(keys, vals):
        body += enc("\nDefinition:") + [k_id] + enc(" refers to") + [v_id] + enc(".")
    n_q = 2 if eval_split else min(4, n_facts_actual)
    asked = rng.sample(range(n_facts_actual), min(n_q, n_facts_actual))
    tail = []
    for qi in asked:
        q_ids = enc("\nQuery:") + [keys[qi]] + enc(" refers to")
        cands = _eight_cands(rng, vals[qi], [vals[j] for j in range(n_facts_actual) if j != qi], P.fact_vals)
        tail.append((q_ids, vals[qi], cands))
    return _pack_to_len(P, rng, body, tail, L, eval_split, pos_frac=0.05 if eval_split else None)


def gen_pure_web(P: IMBPools, rng: random.Random, L: int, eval_split: bool) -> EpisodeDoc:
    return EpisodeDoc(P.prose_slice(rng, L + 1, eval_split), [])


# ── Batch construction & Evaluation ──────────────────────────────────────────

def make_train_batch(P: IMBPools, step_seed: int, batch_size: int, L: int) -> list[EpisodeDoc]:
    rng = random.Random(step_seed)
    docs = []
    families = ("prose_lure", "scope", "two_hop", "b2", "rxk", "facts", "web")
    weights = (0.20, 0.15, 0.20, 0.20, 0.13, 0.07, 0.05)
    for b in range(batch_size):
        sub_rng = random.Random(step_seed * 31 + b * 997)
        fam = rng.choices(families, weights=weights)[0]
        if fam == "prose_lure":
            docs.append(gen_prose_lure(P, sub_rng, L, False))
        elif fam == "scope":
            docs.append(gen_scope_doc(P, sub_rng, L, False))
        elif fam == "two_hop":
            docs.append(gen_two_hop(P, sub_rng, L, False))
        elif fam == "b2":
            docs.append(gen_b2_rules(P, sub_rng, L, False))
        elif fam == "rxk":
            docs.append(gen_rxk(P, sub_rng, L, False))
        elif fam == "facts":
            docs.append(gen_facts_recall(P, sub_rng, L, False, n_facts=sub_rng.choice((4, 8, 16))))
        else:
            docs.append(gen_pure_web(P, sub_rng, L, False))
    return docs


def collate_docs(docs: list[EpisodeDoc], device: torch.device, aux_lm_weight: float = 0.6):
    """Equal per-document total answer weight (1.0 per doc, split 1/n_ans across its queries)
    + ``aux_lm_weight`` (0.6) total real-prose LM weight per doc.

    Fixes IMB17-R1's gradient hijack where 25-answer packed B2 docs took 25x the gradient of
    1-answer two_hop / rxk / spread docs and starved token-embedding LM learning.
    """
    arr = torch.tensor([d.ids for d in docs], dtype=torch.long, device=device)
    x, y = arr[:, :-1].contiguous(), arr[:, 1:].contiguous()
    B, T = y.shape
    mask = torch.zeros((B, T), dtype=torch.float32, device=device)
    for b, d in enumerate(docs):
        n_ans = len(d.answers)
        if n_ans == 0:
            mask[b] = 1.0 / T
        else:
            if aux_lm_weight > 0:
                mask[b] = aux_lm_weight / max(1, T - n_ans)
            per_ans = 1.0 / n_ans
            for idx, gold, _ in d.answers:
                assert int(y[b, idx - 1]) == gold
                mask[b, idx - 1] = per_ans
    return x, y, mask


def _extract_hidden_and_embed(model, x: torch.Tensor):
    lm, _ = model._hidden_to_lm(x)
    h = lm.raw(*lm.layout) if isinstance(lm, NamedTensor) else lm
    emb = getattr(model.embed, "embed", model.embed).weight
    return h.float(), emb.float()


@torch.no_grad()
def score_docs(model, docs: list[EpisodeDoc], device: torch.device, eval_bs: int = 8) -> dict:
    model.eval()
    hits, margins = 0, []
    for i in range(0, len(docs), eval_bs):
        chunk = docs[i:i + eval_bs]
        x, _, _ = collate_docs(chunk, device, aux_lm_weight=0.0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            h, emb = _extract_hidden_and_embed(model, x)
        for b, d in enumerate(chunk):
            for idx, gold, cands in d.answers:
                logits_c = h[b, idx - 1] @ emb[cands].T
                gi = cands.index(gold)
                others = torch.cat([logits_c[:gi], logits_c[gi + 1:]])
                m = float((logits_c[gi] - others.max()).item())
                hits += int(m > 0.0)
                margins.append(m)
    n = max(1, len(margins))
    return {"acc": round(hits / n, 4), "margin": round(float(np.mean(margins)), 3), "n": len(margins)}


@torch.no_grad()
def eval_web_nll(model, P: IMBPools, device: torch.device, n_win: int = 48, seq_len: int = 1024) -> float:
    model.eval()
    losses = []
    step_stride = max(1, (len(P.web_eval) - seq_len - 2) // n_win)
    docs = [P.web_eval[i * step_stride:i * step_stride + seq_len + 1] for i in range(n_win)]
    for i in range(0, n_win, 8):
        arr = torch.tensor(docs[i:i + 8], dtype=torch.long, device=device)
        x, y = arr[:, :-1].contiguous(), arr[:, 1:].contiguous()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            lm, _ = model._hidden_to_lm(x)
            h_raw = lm.raw(*lm.layout) if isinstance(lm, NamedTensor) else lm
            emb = getattr(model.embed, "embed", model.embed).weight
            ce = fused_linear_cross_entropy(h_raw.reshape(-1, h_raw.shape[-1]), emb, y.reshape(-1))
        losses.append(float(ce.item()))
    return round(float(np.mean(losses)), 4)


def run_full_eval(model, P: IMBPools, device: torch.device, trials: int = 120) -> dict:
    base_seed = 900_000_000
    res = {}

    # 1. Prose Entity-Lure + Cleft Binding (unseen entities, bag-of-values cap = 0.333)
    for L in (512, 2048, 4096):
        docs = [gen_prose_lure(P, random.Random(base_seed + 10000 + L + i), L, True) for i in range(trials)]
        res[f"prose_lure@{L}"] = score_docs(model, docs, device, eval_bs=4 if L >= 4096 else 8)

    # 2. 4-Axis Scoped Rule Tuple (including strictly held-out mismatch domain pairs!)
    for L in (512, 2048):
        for mode in ("none", "own", "other", "override"):
            docs = [gen_scope_doc(P, random.Random(base_seed + 20000 + L + hash(mode) % 997 + i), L, True, mode=mode)
                    for i in range(trials // 2)]
            res[f"scope_{mode}@{L}"] = score_docs(model, docs, device)
        res[f"scope_4axis_mean@{L}"] = round(float(np.mean([
            res[f"scope_{m}@{L}"]["acc"] for m in ("none", "own", "other", "override")
        ])), 4)

    # 3. 2-Hop Composition in O(1) State (unseen words, hop-1 bridge traps in candidates)
    for L in (512, 2048):
        docs = [gen_two_hop(P, random.Random(base_seed + 30000 + L + i), L, True) for i in range(trials)]
        res[f"two_hop_new@{L}"] = score_docs(model, docs, device)

    # 4. B2 Unseen Rule Parameters (@0 adjacent and @2048 2x-extrap spread across real web prose)
    for sp, tag in ((False, "0"), (True, "2048")):
        L = 512 if not sp else 2048
        docs = []
        for ti, task in enumerate(B2_NEW_TASKS):
            for i in range(trials // 3):
                docs.append(gen_b2_rules(P, random.Random(base_seed + 40000 + int(sp) * 5000 + ti * 500 + i),
                                         L, True, task=task, spread=sp))
        res[f"unseen_rules@{tag}"] = score_docs(model, docs, device)

    # 5. Rule x Fact Composition on UNSEEN letter shifts {5, 9}
    for L in (512, 2048):
        docs = [gen_rxk(P, random.Random(base_seed + 50000 + L + i), L, True) for i in range(trials)]
        res[f"rxk_new@{L}"] = score_docs(model, docs, device)

    # 6. 4x Beyond-Window Episodic Recall (@4096, trained at <=1024) + Real Web Holdout NLL
    docs_rec = [gen_facts_recall(P, random.Random(base_seed + 60000 + i), 4096, True, n_facts=16)
                for i in range(trials)]
    res["extrap_recall@4096"] = score_docs(model, docs_rec, device, eval_bs=4)
    res["web_nll"] = eval_web_nll(model, P, device)

    # Headline Intelligence Index (mean of the 5 non-memorisation intelligence axes at distance)
    res["intelligence_headline"] = round(float(np.mean([
        res["prose_lure@2048"]["acc"],
        res["scope_4axis_mean@2048"],
        res["two_hop_new@2048"]["acc"],
        res["unseen_rules@2048"]["acc"],
        res["rxk_new@2048"]["acc"],
    ])), 4)
    return res


# ── Training loop ────────────────────────────────────────────────────────────

def train_and_eval_arm(arm: str, seed: int, steps: int, batch_size: int, lr: float,
                       P: IMBPools, device: torch.device, trials: int,
                       dim: int, n_layers: int, n_heads: int, head_dim: int, base_mlp: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = build_arm(arm, vocab_size=50257, dim=dim, n_layers=n_layers,
                      n_heads=n_heads, head_dim=head_dim, base_mlp=base_mlp).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_kb = (model.state_bytes(1) / 1024.0) if hasattr(model, "state_bytes") else float("inf")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    warmup = min(200, max(20, steps // 10))
    plan_rng = random.Random(777_000 + seed)
    step_plan = [(plan_rng.choices((512, 1024), weights=(0.45, 0.55))[0], 10_000_000 * seed + s)
                 for s in range(steps)]

    print(f"\n=== ARM {arm} | seed={seed} | params={n_params:,} | O(1) state={state_kb:.1f} KB ===", flush=True)
    model.train()
    t0 = time.perf_counter()
    toks_seen = 0
    running_loss = 0.0

    for step_idx, (L, step_seed) in enumerate(step_plan, 1):
        # Cosine schedule with warmup (min_lr = 0.15 * lr to preserve far rule carry!)
        if step_idx <= warmup:
            cur_lr = lr * (step_idx / warmup)
        else:
            prog = (step_idx - warmup) / max(1, steps - warmup)
            cur_lr = lr * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        docs = make_train_batch(P, step_seed, batch_size, L)
        x, y, mask = collate_docs(docs, device, aux_lm_weight=0.6)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            if hasattr(model, "ce_from_lm"):
                lm, _ = model._hidden_to_lm(x)
                loss = model.ce_from_lm(lm, y, loss_mask=mask)
            else:
                # TransformerLM path
                lm, _ = model._hidden_to_lm(x)
                loss = fused_linear_cross_entropy(
                    lm.reshape(-1, lm.shape[-1]), model.embed.weight,
                    y.reshape(-1), mask=mask.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        toks_seen += batch_size * L
        running_loss += float(loss.item())
        if step_idx % 250 == 0 or step_idx == steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            avg_l = running_loss / (250 if step_idx % 250 == 0 else (step_idx % 250))
            running_loss = 0.0
            print(f"  [{arm} s={seed}] step {step_idx:4d}/{steps} | loss={avg_l:.4f} | "
                  f"lr={cur_lr:.2e} | tok={toks_seen / 1e6:.1f}M | {toks_seen / dt / 1e3:.1f}k tok/s",
                  flush=True)

    train_sec = round(time.perf_counter() - t0, 1)
    eval_res = run_full_eval(model, P, device, trials=trials)
    print(f"  [{arm} s={seed} EVAL] headline={eval_res['intelligence_headline']:.4f} | "
          f"lure@2k={eval_res['prose_lure@2048']['acc']:.3f} | "
          f"scope4@2k={eval_res['scope_4axis_mean@2048']:.3f} (S_hold={eval_res['scope_other@2048']['acc']:.3f}) | "
          f"2hop@2k={eval_res['two_hop_new@2048']['acc']:.3f} | "
          f"rules_new@0/2k={eval_res['unseen_rules@0']['acc']:.3f}/{eval_res['unseen_rules@2048']['acc']:.3f} "
          f"(M={eval_res['unseen_rules@2048']['margin']:+.2f}) | "
          f"rxk_new@2k={eval_res['rxk_new@2048']['acc']:.3f} | "
          f"rec@4k={eval_res['extrap_recall@4096']['acc']:.3f} | "
          f"web_nll={eval_res['web_nll']:.4f}", flush=True)

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "arm": arm,
        "seed": seed,
        "params": n_params,
        "state_kb": None if math.isinf(state_kb) else round(state_kb, 2),
        "tokens_trained": toks_seen,
        "train_sec": train_sec,
        "eval": eval_res,
    }


def main():
    ap = argparse.ArgumentParser(description="v17 IMB-17 Two-Organ Intelligence vs Memory Benchmark")
    ap.add_argument("--arms", default="pow_v17,pam_v2,gdn_like,tpp")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--base_mlp", type=int, default=1024)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT).stdout.strip() or "uncommitted"
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = Path(args.out) if args.out else (ROOT / f"logs/v17_imb17_{git_hash}_{stamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "suite": "v17_IMB17",
        "git_hash": git_hash,
        "timestamp": stamp,
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "args": vars(args),
    }
    print("HEADER " + json.dumps(header), flush=True)

    ftok = FastTok()
    P = IMBPools(ftok)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    runs = []
    # Run paired by seed so seed-by-seed head-to-head is visible immediately!
    for seed in seeds:
        for arm in arms:
            r = train_and_eval_arm(
                arm=arm, seed=seed, steps=args.steps, batch_size=args.batch,
                lr=args.lr, P=P, device=dev, trials=args.trials,
                dim=args.dim, n_layers=args.layers, n_heads=args.heads,
                head_dim=args.head_dim, base_mlp=args.base_mlp,
            )
            runs.append(r)
            # Incremental atomic checkpoint of JSON results after every (arm, seed)
            tmp = out_path.with_suffix(".json.part")
            tmp.write_text(json.dumps({"header": header, "runs": runs}, indent=2) + "\n")
            os.replace(tmp, out_path)

    # Print 3-seed summary table
    print("\n================ 3-SEED SUMMARY (MEAN ± SD) ================", flush=True)
    keys = [
        ("headline", lambda e: e["intelligence_headline"]),
        ("lure@2k", lambda e: e["prose_lure@2048"]["acc"]),
        ("lure@4k", lambda e: e["prose_lure@4096"]["acc"]),
        ("scope4@2k", lambda e: e["scope_4axis_mean@2048"]),
        ("S_hold@2k", lambda e: e["scope_other@2048"]["acc"]),
        ("2hop@2k", lambda e: e["two_hop_new@2048"]["acc"]),
        ("rule_new@2k", lambda e: e["unseen_rules@2048"]["acc"]),
        ("rxk_new@2k", lambda e: e["rxk_new@2048"]["acc"]),
        ("rec@4k", lambda e: e["extrap_recall@4096"]["acc"]),
        ("web_nll", lambda e: e["web_nll"]),
    ]
    summary = {}
    for arm in arms:
        ar = [r for r in runs if r["arm"] == arm]
        summary[arm] = {}
        parts = [f"{arm:14s}"]
        for kname, fn in keys:
            vals = [fn(r["eval"]) for r in ar]
            m, sd = float(np.mean(vals)), float(np.std(vals))
            summary[arm][kname] = {"mean": round(m, 4), "sd": round(sd, 4), "seeds": vals}
            parts.append(f"{kname}={m:.3f}±{sd:.3f}")
        print(" | ".join(parts), flush=True)

    final_payload = {"header": header, "summary": summary, "runs": runs}
    tmp = out_path.with_suffix(".json.part")
    tmp.write_text(json.dumps(final_payload, indent=2) + "\n")
    os.replace(tmp, out_path)
    print(f"\nSaved complete IMB-17 results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
