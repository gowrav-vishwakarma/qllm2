"""sft1 (EXPERIMENTS_SEMPY "sft1", 2026-11-24): chat SFT windows mixed with base-mix replay, for SCI-1's SFT stage.

* SFT source: HuggingFaceTB/smoltalk2 config ``SFT`` (unseen by the base), restricted with v7's policy (blocked multilingual / long /
  tool splits, capped reasoning splits, think traces dropped by default) and v7's ``_should_filter_smoltalk`` ('hard').
* Encoding: ChatML exactly as v7 ``_encode_sft_example`` (system / user / role headers are context; assistant content + its closing
  <|im_end|> are the targets), without its per-conversation padding: conversations are PACKED into ``seq_len + 1`` windows, each one
  followed by <|endoftext|> (not a target) so RESET_AT_EOS clears the O(1) state between conversations. A conversation that does not
  fit in the rest of a window starts the next window (the rest is padded with <|endoftext|>, no target); one longer than a window is
  truncated to it. Labels are the real next tokens; ``loss_mask`` is 1.0 on targets and 0.0 elsewhere (the trainer weights by it).
* Filters (counts ride in the cursor): the run's 13-gram decontam index (DECONTAM_INDEX) and a probe-template 13-gram index built
  from our own probes (protocol docs v1 / v2, rule-probe prompts, needle prompts, the CHECK generation prompts): a conversation with
  any hit is dropped.
* Interleave: conversations are drawn across the allowed splits in proportion to parquet bytes x split cap (think-trace splits get
  weight 0 when think traces are off), by a fixed-seed rng.
* Mix: each yielded window is SFT with probability ``sft_share`` (deterministic rng), else the next base-mix window (replay; a base
  window without ``loss_mask`` gets ones). Cursor: doc_counters['smoltalk2_sft'] = conversations consumed (kept); a resume skips them.
"""
from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

SFT_KEY = 'smoltalk2_sft'


def sft_ids_targets(messages: List[dict], tokenizer, default_system: Optional[str] = None) -> Tuple[List[int], List[int]]:
    """Token ids and 0/1 targets of one conversation, the same layout as v7 ``_encode_sft_example`` (no padding / truncation)."""
    from v7.data import DEFAULT_SYSTEM, IM_END, IM_START
    im_start = tokenizer.convert_tokens_to_ids(IM_START)
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    nl = tokenizer.encode('\n', add_special_tokens=False)

    def enc(text: str) -> List[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    sys_msg = default_system if default_system is not None else DEFAULT_SYSTEM
    for msg in messages:
        if (msg.get('role') or '').strip().lower() == 'system' and (msg.get('content') or '').strip():
            sys_msg = (msg.get('content') or '').strip()
            break
    ids: List[int] = []
    tgt: List[int] = []

    def add(tok_ids: List[int], is_target: bool):
        ids.extend(tok_ids)
        tgt.extend([1 if is_target else 0] * len(tok_ids))

    add([im_start], False); add(enc('system\n' + sys_msg), False); add([im_end], False); add(nl, False)
    has_assistant = False
    for msg in messages:
        role = (msg.get('role') or '').strip().lower()
        content = (msg.get('content') or '').strip()
        if role in ('system', '') or not content:
            continue
        is_assistant = role == 'assistant'
        add([im_start], False)
        add(enc(f"{role}\n"), False)
        add(enc(content), is_assistant)
        add([im_end], is_assistant)
        add(nl, False)
        has_assistant = has_assistant or is_assistant
    if not has_assistant or len(ids) < 8:
        return [], []
    return ids, tgt


def probe_texts(tokenizer, n_docs: int = 3000) -> List[str]:
    """Texts of our probes (their surface forms) for the probe-template filter."""
    out: List[str] = []
    try:
        from v14 import protocol_docs as pd
        for s in range(n_docs):
            out.append(pd.build_doc(random.Random(70_000 + s)).replace(pd.ANSWER_MARK, ''))
            if hasattr(pd, 'build_doc_v2'):
                out.append(pd.build_doc_v2(random.Random(80_000 + s)).replace(pd.ANSWER_MARK, ''))
    except Exception as exc:  # noqa: BLE001
        print(f'[sft_mix] protocol docs unavailable for the probe filter: {exc}', flush=True)
    try:
        from memory_probes.rules import TASKS, build_rule_example
        for t in TASKS:
            for s in range(40):
                ex = build_rule_example(tokenizer, t, 8, 7000 + s)
                out.append(tokenizer.decode(list(ex.prompt_ids)))
    except Exception as exc:  # noqa: BLE001
        print(f'[sft_mix] rule prompts unavailable for the probe filter: {exc}', flush=True)
    try:
        import importlib.util
        import pathlib
        p = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'sci_gen.py'
        spec = importlib.util.spec_from_file_location('_sci_gen_prompts', p)
        src = p.read_text()
        start = src.index('PROMPTS = {')
        ns: Dict = {}
        exec(src[start:src.index('\n}', start) + 2], ns)  # noqa: S102 - our own file, the PROMPTS literal only
        out += [str(v) for v in ns['PROMPTS'].values()]
        del spec
    except Exception as exc:  # noqa: BLE001
        print(f'[sft_mix] CHECK prompts unavailable for the probe filter: {exc}', flush=True)
    return [t for t in out if t]


def build_probe_index(tokenizer, n_docs: int = 3000) -> np.ndarray:
    from v7.decontam import ngram_hashes
    hs = [ngram_hashes(t) for t in probe_texts(tokenizer, n_docs)]
    hs = [h for h in hs if h.size]
    return np.unique(np.concatenate(hs)) if hs else np.zeros(0, dtype=np.uint64)


def _conv_text(messages: List[dict]) -> str:
    return '\n'.join((m.get('content') or '') for m in messages)


def _split_weights(splits: List[str]) -> Dict[str, float]:
    """Interleave weights = parquet bytes per smoltalk2-SFT split (its size on the hub); 1.0 each if the listing fails."""
    try:
        from huggingface_hub import HfApi, HfFileSystem
        from v7.data import SMOLTALK2_REPO
        sha = HfApi().dataset_info(SMOLTALK2_REPO).sha
        fs = HfFileSystem()
        out = {}
        for sp in splits:
            files = fs.glob(f'datasets/{SMOLTALK2_REPO}@{sha}/SFT/{sp}-*.parquet', detail=True)
            out[sp] = float(sum(v.get('size', 0) for v in files.values())) or 1.0
        return out
    except Exception as exc:  # noqa: BLE001
        print(f'[sft_mix] split sizes unavailable ({exc}); equal interleave weights', flush=True)
        return {sp: 1.0 for sp in splits}


def iter_sft_conversations(doc_counters: Dict, skip: int = 0, think_fraction: float = 0.0,
                           decontam=None, probe_index: Optional[np.ndarray] = None, seed: int = 1234) -> Iterator[List[dict]]:
    """Kept smoltalk2-SFT conversations, INTERLEAVED across the allowed splits (each next conversation is drawn from a split chosen
    with probability proportional to its parquet bytes, by a fixed-seed rng; an exhausted split leaves the draw), so a short run
    sees every split in proportion instead of the first splits in sorted order. Deterministic: a resume replays the same order and
    skips the first ``skip`` kept conversations."""
    from v7.data import (_cap_keep, _should_filter_smoltalk, _smoltalk2_row_iter, _smoltalk2_sft_splits,
                         _smoltalk2_split_cap, _think_keep)
    from v7.decontam import ngram_hashes
    allowed = _smoltalk2_sft_splits()
    raw = _split_weights(allowed)
    # expected KEPT volume: bytes x the split cap; think-trace splits weigh 0 when think traces are off (they would all be dropped)
    def _is_think(sp: str) -> bool:
        x = sp.lower()
        return x.endswith('_think') and not x.endswith('no_think')
    w = {sp: raw[sp] * _smoltalk2_split_cap(sp) * (0.0 if (think_fraction <= 0.0 and _is_think(sp)) else 1.0) for sp in allowed}
    allowed = [sp for sp in allowed if w[sp] > 0]
    tot = sum(w[sp] for sp in allowed)
    print('[sft_mix] interleave weights (kept-volume): ' + ', '.join(f'{sp} {w[sp] / tot:.3f}' for sp in allowed), flush=True)
    its = {sp: _smoltalk2_row_iter('SFT', splits=[sp], emit_source=True) for sp in allowed}
    rng = random.Random(seed)
    kept = 0
    while its:
        names = list(its)
        sp = rng.choices(names, weights=[w[n] for n in names])[0]
        try:
            split, msgs = next(its[sp])
        except StopIteration:
            del its[sp]
            continue
        if not _cap_keep(msgs, _smoltalk2_split_cap(split)) or not _think_keep(msgs, think_fraction):
            continue
        if _should_filter_smoltalk(msgs, split, 'hard'):
            continue
        text = _conv_text(msgs)
        if decontam is not None and decontam.hit(text):
            if kept >= skip:
                doc_counters[f'{SFT_KEY}#decon'] = doc_counters.get(f'{SFT_KEY}#decon', 0) + 1
            continue
        if probe_index is not None and probe_index.size:
            h = ngram_hashes(text)
            if h.size and np.isin(h, probe_index).any():
                if kept >= skip:
                    doc_counters[f'{SFT_KEY}#probe'] = doc_counters.get(f'{SFT_KEY}#probe', 0) + 1
                continue
        kept += 1
        if kept <= skip:
            continue
        doc_counters[SFT_KEY] = kept
        yield msgs


def pack_windows(convs: Iterator[List[dict]], tokenizer, seq_len: int,
                 token_counters: Optional[Dict] = None) -> Iterator[Dict[str, torch.Tensor]]:
    """Pack conversations into seq_len + 1 windows (EOS-separated, EOS-padded, targets only on assistant tokens)."""
    eos = tokenizer.eos_token_id
    L = seq_len + 1
    ids: List[int] = []
    tgt: List[int] = []

    def emit():
        nonlocal ids, tgt
        pad = L - len(ids)
        w_ids, w_tgt = ids + [eos] * pad, tgt + [0] * pad
        ids, tgt = [], []
        return {'input_ids': torch.tensor(w_ids[:-1], dtype=torch.long),
                'labels': torch.tensor(w_ids[1:], dtype=torch.long),
                'loss_mask': torch.tensor(w_tgt[1:], dtype=torch.float32)}

    for msgs in convs:
        c_ids, c_tgt = sft_ids_targets(msgs, tokenizer)
        if not c_ids:
            continue
        c_ids, c_tgt = c_ids + [eos], c_tgt + [0]
        if len(c_ids) > L:
            c_ids, c_tgt = c_ids[:L], c_tgt[:L]
        if len(ids) + len(c_ids) > L:
            yield emit()
        ids += c_ids
        tgt += c_tgt
        if token_counters is not None:
            token_counters[SFT_KEY] = token_counters.get(SFT_KEY, 0) + sum(c_tgt)
        if len(ids) == L:
            yield emit()
    if ids:
        yield emit()


class SFTMixDataset(IterableDataset):
    """Windows: SFT with probability ``sft_share`` (deterministic), else the base-mix stream's next window (replay)."""

    def __init__(self, base_ds, tokenizer, seq_len: int, sft_share: float, seed: int,
                 doc_counters: Dict, token_counters: Dict, think_fraction: float = 0.0, decontam_path: str = '',
                 probe_docs: int = 3000):
        super().__init__()
        self.base_ds, self.tok, self.seq_len = base_ds, tokenizer, seq_len
        self.share, self.seed = float(sft_share), int(seed)
        self.doc_counters, self.token_counters = doc_counters, token_counters
        self.think_fraction, self.decontam_path, self.probe_docs = think_fraction, decontam_path, probe_docs
        self.skip = int(doc_counters.get(SFT_KEY, 0))
        # pass-throughs the trainer reads from the pretrain dataset
        for attr in ('pretrain_cached',):
            if hasattr(base_ds, attr):
                setattr(self, attr, getattr(base_ds, attr))

    def __iter__(self):
        decontam = None
        if self.decontam_path:
            from v7.decontam import DecontamFilter
            decontam = DecontamFilter(self.decontam_path)
        probe_index = build_probe_index(self.tok, self.probe_docs)
        print(f'[sft_mix] share {self.share} | skip {self.skip} kept convs | decontam {"on " + decontam.sha[:8] if decontam else "off"}'
              f' | probe 13-grams {probe_index.size:,}', flush=True)
        sft = pack_windows(iter_sft_conversations(self.doc_counters, self.skip, self.think_fraction, decontam, probe_index),
                           self.tok, self.seq_len, self.token_counters)
        base = iter(self.base_ds)
        rng = random.Random(self.seed * 1_000_003 + self.skip)
        while True:
            if rng.random() < self.share:
                try:
                    yield next(sft)
                    continue
                except StopIteration:
                    print('[sft_mix] SFT source exhausted; replay only from here', flush=True)
                    self.share = 0.0
            try:
                b = next(base)
            except StopIteration:
                return
            if 'loss_mask' not in b:
                b = {**b, 'loss_mask': torch.ones_like(b['labels'], dtype=torch.float32)}
            yield b
