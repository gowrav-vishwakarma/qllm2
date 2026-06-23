"""V11 PAM probes with real language — clustered embeddings, not random vectors.

Two experiments suggested for harder / more realistic interference:

1. **Language filler** — needle association then Wikipedia (or WikiText) tokens as
   filler, not random unit vectors. Language embeddings cluster (E[k_i·k_j] ≠ 0).

2. **Rank on real text** — effective rank of PAM state S_t while streaming real
   token sequences through V11PAMLayer (optional checkpoint).

No trained V11 checkpoint required for the embedding-level language filler test
(GPT-2 pretrained embeddings capture clustering). Full V11 projections optional
via --checkpoint.

Run:
    .venv/bin/python -m v11.pam_math_language --test language-filler --filler-tokens 10000
    .venv/bin/python -m v11.pam_math_language --test language-filler --projection-trials 50
    .venv/bin/python -m v11.pam_math_language --test rank-text --text-tokens 50000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from v11.model import V11Config, V11PAMLayer
from v11.pam_math import (
    effective_rank,
    outer_v_kstar,
    pam_step_additive,
    relative_retrieval,
)

ArrayC = np.ndarray


# ── GPT-2 embeddings + projection to complex PAM keys/values ─────────────────

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
    """Fixed random projection embed_dim → d complex, unit-normalized."""
    rng = np.random.default_rng(seed)
    wr = rng.standard_normal((embed_dim, d))
    wi = rng.standard_normal((embed_dim, d))

    def proj(e: np.ndarray) -> ArrayC:
        z = e @ wr + 1j * (e @ wi)
        n = np.linalg.norm(z)
        return z / (n + 1e-12)

    return proj


def make_k_v_projectors(embed_dim: int, d: int, seed: int = 42) -> Tuple[Callable, Callable]:
    """Independent K and V projections (different seeds)."""
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
    """Load up to target_tokens from WikiText-103 train split."""
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
    """Mean |k_i·k_j| for i≠j (clustering proxy)."""
    n = keys.shape[0]
    if n < 2:
        return 0.0
    idx = np.random.default_rng(0).choice(n, size=min(sample, n * (n - 1) // 2), replace=False)
    # sample random pairs
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(min(sample, n * 10)):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        sims.append(abs(np.vdot(keys[i], keys[j])))
    return float(np.mean(sims)) if sims else 0.0


# ── A10: Language filler (English / Wikipedia) ───────────────────────────────

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
    """One projection-seed trial. wiki_ids preloaded for speed."""
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
    """Needle write from invented/rare tokens, then WikiText filler, then query."""
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

    print(f'\n[A10] Language filler (WikiText needle-in-haystack)')
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
    print(f'\n[A10] Language filler — projection seed sweep ({n_trials} trials)')
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
    # Language absolute score stable if CV < 35%; language always beats random if min ratio > 1
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


# ── A11: Effective rank + spectrum on real text ───────────────────────────────

def _state_singular_values(s_c: ArrayC) -> List[float]:
    """Descending singular values of complex state matrix."""
    return np.linalg.svd(s_c, compute_uv=False).tolist()


def _print_spectrum_snapshot(pos: int, sv: List[float]) -> None:
    if len(sv) < 3:
        print(f'    t={pos:7,} sv={sv}')
        return
    print(f'    t={pos:7,} eff_rank={effective_rank_from_sv(sv):5.1f}  '
          f's1={sv[0]:.3f} s2={sv[1]:.3f} s3={sv[2]:.3f} ... s{len(sv)}={sv[-1]:.4e}')


def effective_rank_from_sv(sv: Sequence[float]) -> float:
    s = np.array(sv, dtype=np.float64)
    s = s / (s.sum() + 1e-12)
    s = s[s > 1e-12]
    return float(np.exp(-np.sum(s * np.log(s)))) if s.size else 0.0

def _gpt2_embed_to_v11_x(token_id: int, embed_table: np.ndarray, proj_768_to_dim: torch.Tensor) -> torch.Tensor:
    """Project GPT-2 token embed to V11 complex input [1,1,dim,2]."""
    e = torch.from_numpy(embed_table[token_id]).double()
    flat = proj_768_to_dim @ e  # [dim*2]
    dim = flat.shape[0] // 2
    return flat.view(1, 1, dim, 2)


def test_rank_real_text(
    text_tokens: int = 50000,
    sample_every: int = 100,
    seed: int = 42,
    checkpoint: Optional[str] = None,
    preset: str = 'v11_e3_k3',
    layer_idx: int = 0,
    head_idx: int = 0,
    compare_random: bool = True,
    report_every: int = 10000,
) -> Dict[str, Any]:
    """Stream real WikiText through V11PAMLayer; log effective rank(S_t) vs position.

    Signature figure: does rank saturate early or keep using degrees of freedom?
    """
    from v11.model import get_config

    t0 = time.perf_counter()
    tok, embed_table = _load_gpt2()

    cfg = get_config(preset)
    cfg.dropout = 0.0
    cfg.gradient_checkpointing = False
    if checkpoint:
        from v7.data import get_chat_tokenizer
        from v11.model import V11LM

        chat_tok = get_chat_tokenizer()
        cfg.vocab_size = len(chat_tok)
        lm = V11LM(cfg)
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        lm.load_state_dict(ckpt['model_state_dict'])
        lm.eval().double()
        block = lm.blocks[layer_idx]
        pam = block.pam
        use_block = True
        wiki_ids, source = load_wikitext_token_stream(
            text_tokens, tokenizer=chat_tok, cache_tag='chat',
        )
        print(f'\n[A11] Rank on real text (V11 block {layer_idx}, checkpoint)')
    else:
        pam = V11PAMLayer(cfg, layer_idx=layer_idx).eval().double()
        block = None
        lm = None
        use_block = False
        wiki_ids, source = load_wikitext_token_stream(text_tokens)
        print(f'\n[A11] Rank on real text (V11PAMLayer only, untrained projections)')

    dim = cfg.dim
    rng = np.random.default_rng(seed)
    proj = torch.from_numpy(
        rng.standard_normal((dim * 2, embed_table.shape[1]))
    ).double() * (1.0 / math.sqrt(embed_table.shape[1]))

    def run_stream(name: str, token_ids: Sequence[int]) -> Dict[str, Any]:
        state = None
        ranks: List[float] = []
        positions: List[int] = []
        singular_values: List[List[float]] = []
        n = len(token_ids)
        for t, tid in enumerate(token_ids):
            if report_every and t > 0 and t % report_every == 0:
                print(f'    ... {name} token {t:,}/{n:,}', flush=True)
            with torch.no_grad():
                if use_block:
                    z = lm.embed(torch.tensor([[tid]])).double()
                    if lm.pos_embed is not None:
                        z = lm.pos_embed(z, step_offset=t)
                    z = lm.embed_norm(z)
                    cgu_out = block.cgu(block.norm1(z))
                    z = z + cgu_out * block.cgu_scale
                    x_in = block.norm2(z)
                    _, state = pam(x_in, state=state, step_offset=t)
                else:
                    x = _gpt2_embed_to_v11_x(tid, embed_table, proj)
                    _, state = pam(x, state=state, step_offset=t)
            if t % sample_every == 0 or t == n - 1:
                if cfg.n_states > 1:
                    s = state[0, 0, head_idx]
                else:
                    s = state[0, head_idx]
                s_c = s[..., 0].numpy() + 1j * s[..., 1].numpy()
                sv = _state_singular_values(s_c)
                ranks.append(effective_rank_from_sv(sv))
                singular_values.append(sv)
                positions.append(t)
        return {
            'name': name,
            'source': source if name == 'wikitext' else 'uniform_random',
            'positions': positions,
            'ranks': ranks,
            'singular_values': singular_values,
            'final_rank': ranks[-1] if ranks else 0.0,
            'max_rank': max(ranks) if ranks else 0.0,
            'head_dim': cfg.head_dim,
        }

    print(f'  tokens={text_tokens:,}, sample_every={sample_every}, head={head_idx}')
    wiki_result = run_stream('wikitext', wiki_ids)

    results: Dict[str, Any] = {
        'text_tokens': text_tokens,
        'sample_every': sample_every,
        'layer_idx': layer_idx,
        'head_idx': head_idx,
        'checkpoint': checkpoint,
        'preset': preset,
        'wikitext': wiki_result,
        'elapsed_s': time.perf_counter() - t0,
    }

    if compare_random:
        rng_t = np.random.default_rng(seed + 1)
        vocab_hi = cfg.vocab_size if checkpoint else embed_table.shape[0]
        rand_ids = rng_t.integers(0, vocab_hi, size=text_tokens).tolist()
        results['random'] = run_stream('random', rand_ids)

    # ASCII mini-plot + spectrum snapshots
    _print_rank_curve(wiki_result['positions'], wiki_result['ranks'], cfg.head_dim)
    print('\n  Singular value snapshots (wikitext):')
    for pos, sv in zip(wiki_result['positions'][:5], wiki_result['singular_values'][:5]):
        _print_spectrum_snapshot(pos, sv)
    if len(wiki_result['positions']) > 10:
        print('    ...')
        for pos, sv in zip(wiki_result['positions'][-3:], wiki_result['singular_values'][-3:]):
            _print_spectrum_snapshot(pos, sv)

    if compare_random:
        print(f'  WikiText  final rank={wiki_result["final_rank"]:.1f} / d={cfg.head_dim}')
        print(f'  Random    final rank={results["random"]["final_rank"]:.1f} / d={cfg.head_dim}')

    return results


def _print_rank_curve(positions: List[int], ranks: List[float], head_dim: int, width: int = 50) -> None:
    if not ranks:
        return
    print('\n  Effective rank vs tokens (wikitext):')
    mx = max(ranks) if ranks else 1.0
    for pos, r in zip(positions[-min(12, len(positions)):], ranks[-min(12, len(ranks)):]):
        bar = int((r / max(mx, 1e-6)) * width)
        print(f'    t={pos:7,} rank={r:5.1f} |{"#" * bar}')
    print(f'    (d={head_dim}, max rank shown={mx:.1f})')


# ── CLI ──────────────────────────────────────────────────────────────────────

TESTS = ('language-filler', 'rank-text', 'both')


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description='V11 PAM language / real-text probes')
    p.add_argument('--test', default='both', choices=TESTS)
    p.add_argument('--output-dir', default='logs/v11/pam_math')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--filler-tokens', type=int, default=10000,
                   help='WikiText tokens after needle (use 100000 for full stress test)')
    p.add_argument('--text-tokens', type=int, default=50000,
                   help='Tokens to stream for rank-text')
    p.add_argument('--sample-every', type=int, default=100)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--checkpoint', type=str, default='',
                   help='Optional V11 checkpoint for rank-text with trained projections')
    p.add_argument('--preset', default='v11_e3_k3')
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--projection-trials', type=int, default=1,
                   help='Sweep projection seeds (e.g. 50) to stress-test language filler')
    p.add_argument('--projection-seed-start', type=int, default=0)
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {'timestamp': datetime.now(timezone.utc).isoformat(), 'seed': args.seed}

    if args.test in ('language-filler', 'both'):
        results['language_filler'] = test_language_filler(
            filler_tokens=args.filler_tokens,
            gamma=args.gamma,
            seed=args.seed,
            projection_trials=args.projection_trials,
            projection_seed_start=args.projection_seed_start,
        )

    if args.test in ('rank-text', 'both'):
        results['rank_text'] = test_rank_real_text(
            text_tokens=args.text_tokens,
            sample_every=args.sample_every,
            seed=args.seed,
            checkpoint=args.checkpoint or None,
            preset=args.preset,
            layer_idx=args.layer,
        )

    tag = args.test.replace('-', '_')
    out_path = out_dir / f'pam_math_{tag}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with out_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
