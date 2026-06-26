"""Cross-architecture comparison on a shared real-text (WikiText) stream.

Streams the same WikiText-103 corpus through three fixed/growing memories and
measures how much of each architecture's state is actually used (effective rank
of the recurrent state), alongside the state's memory cost as a function of
context length t:

  * PAM            -- fixed O(d^2) complex matrix       (rank ceiling = d)
  * Transformer KV -- growing O(t * d) cache            (rank ceiling = min(t, d))
  * Mamba SSM      -- fixed O(d_inner * d_state) state  (rank ceiling = d_state)

The effective-rank ceiling is a *structural* property (state geometry), so this
comparison is meaningful even though PAM/Transformer use untrained projections;
Mamba uses real pretrained weights (state-spaces/mamba-130m-hf by default).

This is a mechanism-level comparison (state utilization + cost), not a behavioral
recall benchmark, which would require trained checkpoints for every architecture.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from memory_probes.adapters import PAMAdapter, TransformerKVAdapter
from memory_probes.core import effective_rank


def _stream_adapter_rank(
    adapter,
    wiki_ids: Sequence[int],
    embed_table: np.ndarray,
    proj_k,
    proj_v,
    gamma: float,
    sample_every: int,
    report_every: int = 0,
    name: str = 'adapter',
) -> Dict[str, Any]:
    """Stream projected GPT-2 embeddings through an associative adapter; track rank."""
    adapter.reset()
    ranks: List[float] = []
    positions: List[int] = []
    n = len(wiki_ids)
    for t, tid in enumerate(wiki_ids):
        if report_every and t > 0 and t % report_every == 0:
            print(f'    ... {name} token {t:,}/{n:,}', flush=True)
        e = embed_table[tid]
        adapter.write(proj_k(e), proj_v(e), gamma)
        if t % sample_every == 0 or t == n - 1:
            ranks.append(effective_rank(adapter.state()))
            positions.append(t)
    return {'positions': positions, 'ranks': ranks,
            'final_rank': ranks[-1], 'max_rank': max(ranks)}


def _stream_mamba_rank(
    model,
    wiki_ids: Sequence[int],
    layer_idx: int,
    sample_every: int,
    device: str,
    report_every: int = 0,
) -> Dict[str, Any]:
    """Step a pretrained Mamba token-by-token; track effective rank of ssm_states[layer]."""
    import torch

    cp = None
    ranks: List[float] = []
    positions: List[int] = []
    n = len(wiki_ids)
    for t in range(n):
        if report_every and t > 0 and t % report_every == 0:
            print(f'    ... mamba token {t:,}/{n:,}', flush=True)
        tok = torch.tensor([[int(wiki_ids[t])]], device=device)
        with torch.no_grad():
            out = model(input_ids=tok, cache_params=cp, use_cache=True,
                        cache_position=torch.tensor([t], device=device))
        cp = out.cache_params
        if t % sample_every == 0 or t == n - 1:
            s = cp.ssm_states[layer_idx][0].float().cpu().numpy()  # [d_inner, d_state]
            ranks.append(effective_rank(s))
            positions.append(t)
    return {'positions': positions, 'ranks': ranks,
            'final_rank': ranks[-1], 'max_rank': max(ranks)}


def test_arch_comparison(
    text_tokens: int = 2000,
    d: int = 64,
    gamma: float = 0.995,
    sample_every: int = 50,
    layer_idx: int = 12,
    mamba_model: str = 'state-spaces/mamba-130m-hf',
    seed: int = 42,
    include_mamba: bool = True,
) -> Dict[str, Any]:
    """Compare PAM vs Transformer-KV vs Mamba state utilization on a shared WikiText stream."""
    from memory_probes.language import _load_gpt2, load_wikitext_token_stream, make_k_v_projectors

    t0 = time.perf_counter()
    tok, embed_table = _load_gpt2()
    proj_k, proj_v = make_k_v_projectors(embed_table.shape[1], d, seed=seed)
    wiki_ids, source = load_wikitext_token_stream(text_tokens)

    print('\n[compare] Cross-architecture state utilization on WikiText')
    print(f'  tokens={text_tokens:,} ({source}), d={d}, gamma={gamma}, sample_every={sample_every}')

    archs: Dict[str, Any] = {}

    pam = _stream_adapter_rank(
        PAMAdapter(d=d), wiki_ids, embed_table, proj_k, proj_v, gamma,
        sample_every, name='pam')
    pam.update({'state_dim': d, 'state_cost': 'fixed O(d^2)',
                'state_numbers': d * d, 'grows_with_t': False})
    archs['pam'] = pam

    tf = _stream_adapter_rank(
        TransformerKVAdapter(d=d), wiki_ids, embed_table, proj_k, proj_v, gamma,
        sample_every, name='transformer')
    tf.update({'state_dim': min(text_tokens, d), 'state_cost': 'grows O(t*d)',
               'state_numbers': text_tokens * d, 'grows_with_t': True})
    archs['transformer'] = tf

    if include_mamba:
        try:
            import torch
            from transformers import AutoTokenizer, MambaForCausalLM

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'  loading {mamba_model} (trained) on {device} ...')
            m_tok = AutoTokenizer.from_pretrained(mamba_model)
            model = MambaForCausalLM.from_pretrained(mamba_model).eval().to(device)
            m_ids, _ = load_wikitext_token_stream(text_tokens, tokenizer=m_tok, cache_tag='mamba')
            n_layers = model.config.num_hidden_layers
            li = min(layer_idx, n_layers - 1)
            mamba = _stream_mamba_rank(model, m_ids, li, sample_every, device,
                                       report_every=max(text_tokens // 5, 1))
            mamba.update({
                'state_dim': model.config.state_size,
                'state_cost': 'fixed O(d_inner*d_state)',
                'state_numbers': model.config.intermediate_size * model.config.state_size,
                'grows_with_t': False, 'model': mamba_model, 'layer_idx': li,
                'hidden_size': model.config.hidden_size, 'state_size': model.config.state_size,
            })
            archs['mamba'] = mamba
        except Exception as e:  # pragma: no cover - network / dependency dependent
            print(f'  SKIP mamba: {type(e).__name__}: {str(e)[:160]}')
            archs['mamba'] = {'skipped': True, 'reason': f'{type(e).__name__}: {str(e)[:160]}'}

    print('\n  Architecture state-utilization comparison:')
    print(f'  {"arch":12s} {"rank_ceiling":>12s} {"eff_rank":>9s} {"utilization":>12s} {"state_cost":>22s}')
    for name, a in archs.items():
        if a.get('skipped'):
            print(f'  {name:12s} {"-":>12s} {"-":>9s} {"-":>12s}   (skipped: {a["reason"][:40]})')
            continue
        ceiling = a['state_dim']
        util = a['final_rank'] / max(ceiling, 1e-9)
        print(f'  {name:12s} {ceiling:>12d} {a["final_rank"]:>9.2f} {util:>11.1%} '
              f'{a["state_cost"]:>22s}')

    return {
        'text_tokens': text_tokens, 'd': d, 'gamma': gamma, 'source': source,
        'layer_idx': layer_idx, 'archs': archs,
        'elapsed_s': time.perf_counter() - t0,
    }
