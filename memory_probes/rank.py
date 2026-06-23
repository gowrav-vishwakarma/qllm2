"""State rank evolution: synthetic writes and real-text streaming."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from memory_probes.core import (
    effective_rank,
    effective_rank_from_sv,
    pam_step_additive,
    rand_unit_complex,
)
from memory_probes.language import load_wikitext_token_stream
from v11.model import V11Config, V11PAMLayer


def test_rank(
    d: int = 64,
    steps: int = 512,
    gamma: float = 0.995,
    mode: str = 'random',
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    S = np.zeros((d, d), dtype=np.complex128)
    ranks: List[float] = []
    print(f'\n[rank] State rank evolution (mode={mode}, gamma={gamma}, steps={steps})')

    if mode == 'random':
        for t in range(steps):
            k = rand_unit_complex(rng, (d,))
            v = rand_unit_complex(rng, (d,))
            S = pam_step_additive(S, gamma, v, k)
            ranks.append(effective_rank(S))
    elif mode == 'overwrite':
        k = rand_unit_complex(rng, (d,))
        for t in range(steps):
            v = rand_unit_complex(rng, (d,))
            S = pam_step_additive(S, gamma, v, k)
            ranks.append(effective_rank(S))
    else:
        raise ValueError(f'unknown mode: {mode}')

    print(f'  rank@1={ranks[0]:.2f}  max={max(ranks):.2f}  final={ranks[-1]:.2f}  (d={d})')
    return {
        'd': d,
        'steps': steps,
        'gamma': gamma,
        'mode': mode,
        'ranks': ranks,
        'final_rank': ranks[-1],
        'max_rank': max(ranks),
    }


def _state_singular_values(s_c: np.ndarray) -> List[float]:
    return np.linalg.svd(s_c, compute_uv=False).tolist()


def _print_spectrum_snapshot(pos: int, sv: List[float]) -> None:
    if len(sv) < 3:
        print(f'    t={pos:7,} sv={sv}')
        return
    print(f'    t={pos:7,} eff_rank={effective_rank_from_sv(sv):5.1f}  '
          f's1={sv[0]:.3f} s2={sv[1]:.3f} s3={sv[2]:.3f} ... s{len(sv)}={sv[-1]:.4e}')


def _gpt2_embed_to_v11_x(token_id: int, embed_table: np.ndarray, proj_768_to_dim: torch.Tensor) -> torch.Tensor:
    e = torch.from_numpy(embed_table[token_id]).double()
    flat = proj_768_to_dim @ e
    dim = flat.shape[0] // 2
    return flat.view(1, 1, dim, 2)


def _print_rank_curve(positions: List[int], ranks: List[float], head_dim: int, width: int = 50) -> None:
    if not ranks:
        return
    print('\n  Effective rank vs tokens (wikitext):')
    mx = max(ranks) if ranks else 1.0
    for pos, r in zip(positions[-min(12, len(positions)):], ranks[-min(12, len(ranks)):]):
        bar = int((r / max(mx, 1e-6)) * width)
        print(f'    t={pos:7,} rank={r:5.1f} |{"#" * bar}')
    print(f'    (d={head_dim}, max rank shown={mx:.1f})')


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
    """Stream real WikiText through V11PAMLayer; log effective rank(S_t) vs position."""
    from memory_probes.language import _load_gpt2
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
        print(f'\n[rank] Rank on real text (V11 block {layer_idx}, checkpoint)')
    else:
        pam = V11PAMLayer(cfg, layer_idx=layer_idx).eval().double()
        block = None
        lm = None
        use_block = False
        wiki_ids, source = load_wikitext_token_stream(text_tokens)
        print('\n[rank] Rank on real text (V11PAMLayer only, untrained projections)')

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
