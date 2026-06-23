"""Language filler probes — clustered embeddings, not random vectors."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from memory_probes.core import (
    outer_v_kstar,
    pam_step_additive,
    relative_retrieval,
)

ArrayC = np.ndarray

_GPT2_CACHE: Dict[str, Any] = {}


def _load_gpt2():
    if _GPT2_CACHE:
        return _GPT2_CACHE['tokenizer'], _GPT2_CACHE['embed']
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained('gpt2')
    model.eval()
    wte = model.get_input_embeddings().weight.detach().numpy().astype(np.float64)
    _GPT2_CACHE['tokenizer'] = tok
    _GPT2_CACHE['embed'] = wte
    return tok, wte


def make_complex_projector(embed_dim: int, d: int, seed: int = 42) -> Callable[[np.ndarray], ArrayC]:
    rng = np.random.default_rng(seed)
    wr = rng.standard_normal((embed_dim, d))
    wi = rng.standard_normal((embed_dim, d))

    def proj(e: np.ndarray) -> ArrayC:
        z = e @ wr + 1j * (e @ wi)
        n = np.linalg.norm(z)
        return z / (n + 1e-12)

    return proj


def make_k_v_projectors(embed_dim: int, d: int, seed: int = 42) -> Tuple[Callable, Callable]:
    return (
        make_complex_projector(embed_dim, d, seed=seed),
        make_complex_projector(embed_dim, d, seed=seed + 1000),
    )


def token_embed(tokenizer, embed_table: np.ndarray, text: str) -> np.ndarray:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f'empty tokenization for {text!r}')
    return embed_table[ids[0]]


def load_wikitext_token_stream(
    target_tokens: int,
    tokenizer=None,
    cache_tag: str = 'gpt2',
) -> Tuple[List[int], str]:
    cache_path = Path('.cache') / 'v7_tokens' / f'wikitext103_train_stream_{cache_tag}_{target_tokens}.pt'
    if cache_path.exists():
        data = torch.load(cache_path, weights_only=False)
        return data['ids'], 'cache'

    from datasets import load_dataset

    if tokenizer is None:
        tok, _ = _load_gpt2()
    else:
        tok = tokenizer
    print(f'  Loading WikiText-103 train stream ({target_tokens:,} tokens, {cache_tag})...')
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    ids: List[int] = []
    for item in ds:
        line = (item.get('text') or '').strip()
        if not line:
            continue
        ids.extend(tok.encode(line + '\n', add_special_tokens=False))
        if len(ids) >= target_tokens:
            break
    ids = ids[:target_tokens]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'ids': ids}, cache_path)
    return ids, 'wikitext-103-train'


def embed_pair_from_texts(
    tokenizer,
    embed_table: np.ndarray,
    proj_k: Callable[[np.ndarray], ArrayC],
    proj_v: Callable[[np.ndarray], ArrayC],
    key_text: str,
    value_text: str,
) -> Tuple[ArrayC, ArrayC]:
    k = proj_k(token_embed(tokenizer, embed_table, key_text))
    v = proj_v(token_embed(tokenizer, embed_table, value_text))
    return k, v


def language_filler_step(
    S: ArrayC,
    gamma: float,
    token_id: int,
    embed_table: np.ndarray,
    proj_k: Callable[[np.ndarray], ArrayC],
    proj_v: Callable[[np.ndarray], ArrayC],
) -> ArrayC:
    e = embed_table[token_id]
    return pam_step_additive(S, gamma, proj_v(e), proj_k(e))


def mean_key_correlation(keys: ArrayC, sample: int = 500) -> float:
    n = keys.shape[0]
    if n < 2:
        return 0.0
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(min(sample, n * 10)):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        sims.append(abs(np.vdot(keys[i], keys[j])))
    return float(np.mean(sims)) if sims else 0.0


def _run_language_filler_single(
    d: int,
    wiki_ids: List[int],
    embed_table: np.ndarray,
    tok,
    gamma: float,
    seed: int,
    key_text: str,
    value_text: str,
    query_text: str,
    compare_random: bool,
    report_every: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    proj_k, proj_v = make_k_v_projectors(embed_table.shape[1], d, seed=seed)
    k_needle, v_needle = embed_pair_from_texts(
        tok, embed_table, proj_k, proj_v, key_text, value_text)
    k_query = proj_k(token_embed(tok, embed_table, query_text))

    S = outer_v_kstar(v_needle, k_needle)
    baseline = relative_retrieval(S, k_query, v_needle)

    lang_keys = []
    for t, tid in enumerate(wiki_ids):
        if report_every and t > 0 and t % report_every == 0:
            print(f'    ... wiki filler {t:,}/{len(wiki_ids):,}', flush=True)
        lang_keys.append(proj_k(embed_table[tid]))
        S = language_filler_step(S, gamma, tid, embed_table, proj_k, proj_v)

    lang_rel = relative_retrieval(S, k_query, v_needle)
    lang_keys_arr = np.stack(lang_keys[: min(2000, len(lang_keys))])
    lang_cluster = mean_key_correlation(lang_keys_arr)

    out: Dict[str, Any] = {
        'projection_seed': seed,
        'baseline_relative': baseline,
        'language_filler_relative': lang_rel,
        'language_key_mean_corr': lang_cluster,
    }

    if compare_random:
        rng = np.random.default_rng(seed)
        S_rand = outer_v_kstar(v_needle, k_needle)
        for t in range(len(wiki_ids)):
            if report_every and t > 0 and t % report_every == 0:
                print(f'    ... random filler {t:,}/{len(wiki_ids):,}', flush=True)
            k_f = rng.standard_normal(d) + 1j * rng.standard_normal(d)
            k_f = k_f / (np.linalg.norm(k_f) + 1e-12)
            v_f = rng.standard_normal(d) + 1j * rng.standard_normal(d)
            v_f = v_f / (np.linalg.norm(v_f) + 1e-12)
            S_rand = pam_step_additive(S_rand, gamma, v_f, k_f)
        rand_rel = relative_retrieval(S_rand, k_query, v_needle)
        out['random_filler_relative'] = rand_rel
        out['lang_over_random'] = lang_rel / (rand_rel + 1e-12)

    if verbose:
        ratio = out.get('lang_over_random', float('nan'))
        print(f'  seed={seed:3d}  lang={lang_rel:.3f}  rand={out.get("random_filler_relative", 0):.3f}  '
              f'lang/rand={ratio:.2f}x')
    return out


def test_language_filler_seed_sweep(
    d: int,
    filler_tokens: int,
    gamma: float,
    wiki_ids: List[int],
    embed_table: np.ndarray,
    tok,
    source: str,
    n_trials: int,
    seed_start: int,
    key_text: str,
    value_text: str,
    query_text: str,
    compare_random: bool,
    t0: float,
) -> Dict[str, Any]:
    print(f'\n[language] Filler — projection seed sweep ({n_trials} trials)')
    print(f'  needle: {key_text!r} → {value_text!r}  filler: {len(wiki_ids):,} {source} tokens')

    trials = [
        _run_language_filler_single(
            d, wiki_ids, embed_table, tok, gamma, seed_start + i,
            key_text, value_text, query_text, compare_random,
            report_every=0, verbose=True,
        )
        for i in range(n_trials)
    ]

    lang_rels = [t['language_filler_relative'] for t in trials]
    ratios = [t['lang_over_random'] for t in trials if 'lang_over_random' in t]

    def _stats(xs: List[float]) -> Dict[str, float]:
        a = np.array(xs, dtype=np.float64)
        return {
            'mean': float(a.mean()), 'std': float(a.std()),
            'min': float(a.min()), 'max': float(a.max()),
            'median': float(np.median(a)),
        }

    lang_stats = _stats(lang_rels)
    ratio_stats = _stats(ratios) if ratios else {}
    lang_cv = lang_stats['std'] / max(lang_stats['mean'], 1e-6)
    lang_rel_stable = lang_cv < 0.35
    lang_beats_random_all = ratio_stats.get('min', 0) > 1.0 if ratios else False

    print(f'\n  Language rel: mean={lang_stats["mean"]:.2f} std={lang_stats["std"]:.2f} '
          f'[{lang_stats["min"]:.2f}, {lang_stats["max"]:.2f}]')
    if ratios:
        print(f'  Lang/rand:    mean={ratio_stats["mean"]:.2f}x std={ratio_stats["std"]:.2f} '
              f'[{ratio_stats["min"]:.2f}x, {ratio_stats["max"]:.2f}x]')
        print(f'  Language rel stable (CV<35%): {lang_rel_stable}')
        print(f'  Language beats random all seeds: {lang_beats_random_all}')

    return {
        'd': d, 'filler_tokens': filler_tokens, 'gamma': gamma,
        'n_trials': n_trials, 'seed_start': seed_start, 'trials': trials,
        'language_relative_stats': lang_stats,
        'lang_over_random_stats': ratio_stats,
        'language_rel_stable': lang_rel_stable,
        'language_beats_random_all_seeds': lang_beats_random_all,
        'elapsed_s': time.perf_counter() - t0,
    }


def test_language_filler(
    d: int = 64,
    filler_tokens: int = 10000,
    gamma: float | None = None,
    seed: int = 42,
    key_text: str = 'glorp',
    value_text: str = ' banana',
    query_text: str = 'glorp',
    compare_random: bool = True,
    report_every: int = 10000,
    projection_trials: int = 1,
    projection_seed_start: int = 0,
) -> Dict[str, Any]:
    if gamma is None:
        gamma = 0.995
    t0 = time.perf_counter()
    tok, embed_table = _load_gpt2()
    wiki_ids, source = load_wikitext_token_stream(filler_tokens)

    if projection_trials > 1:
        return test_language_filler_seed_sweep(
            d=d, filler_tokens=filler_tokens, gamma=gamma,
            wiki_ids=wiki_ids, embed_table=embed_table, tok=tok, source=source,
            n_trials=projection_trials, seed_start=projection_seed_start,
            key_text=key_text, value_text=value_text, query_text=query_text,
            compare_random=compare_random, t0=t0,
        )

    print(f'\n[language] WikiText needle-in-haystack')
    print(f'  needle: {key_text!r} → {value_text!r}  filler: {len(wiki_ids):,} tokens, seed={seed}')
    out = _run_language_filler_single(
        d, wiki_ids, embed_table, tok, gamma, seed,
        key_text, value_text, query_text, compare_random, report_every, verbose=False,
    )
    out.update({
        'd': d, 'filler_tokens': len(wiki_ids), 'gamma': gamma,
        'key_text': key_text, 'value_text': value_text, 'query_text': query_text,
        'elapsed_s': time.perf_counter() - t0,
    })
    print(f'  wiki rel={out["language_filler_relative"]:.4f}  '
          f'rand={out.get("random_filler_relative", 0):.4f}  '
          f'lang/rand={out.get("lang_over_random", 0):.2f}x')
    return out
