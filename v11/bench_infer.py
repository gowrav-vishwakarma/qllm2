"""Rough inference-speed benchmark: PAM (V11 E3 K=3) vs a KV-cached transformer.

Motivation
----------
Our training tok/s numbers (bench_step.py) are a *training* metric. The Reddit /
FAQ question is about **inference**. The honest architectural claim is not "PAM
decodes faster than a transformer on a short prompt" — it is:

    PAM decode is O(1)/token with a **fixed-size** recurrent state (no KV cache),
    while a transformer keeps a **KV cache that grows with context length**.

So this script compares, across growing context length L:
  * decode latency per token (ms/token) at steady state
  * the per-sequence **state / KV-cache memory** that must be carried

To be fair to the transformer we give it a real **KV cache** (single-token SDPA
over cached keys/values) — NOT the naive no-cache generate() in
v6/transformer_baseline.py, which recomputes the whole prefix every step.

Both models are ~100M params, matched pipeline. Speed is weight-independent, so
random init is fine (we are timing the forward, not measuring quality).

Run (GPU):
    uv run python -m v11.bench_infer --device cuda --dtype fp16 \
        --context 128,256,512,1024,2048 --decode 32

Run (CPU, smaller):
    uv run python -m v11.bench_infer --device cpu --dtype fp32 \
        --context 128,256,512 --decode 8
"""

import argparse
import contextlib
import json
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from v11.model import V11LM, get_config
from v6.transformer_baseline import TransformerLM, get_transformer_config_100m

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _HAS_SDPA_CTX = True
except ImportError:  # older torch
    _HAS_SDPA_CTX = False


# Global attention-backend policy for the transformer's SDPA calls.
# 'auto'  -> let PyTorch pick (FlashAttention / mem-efficient fused kernel on CUDA)
# 'math'  -> force the pure-PyTorch math path (matmul + softmax + matmul, no fused kernel)
_ATTN_MODE = 'auto'


def _sdpa_ctx():
    if _ATTN_MODE == 'math' and _HAS_SDPA_CTX:
        return sdpa_kernel([SDPBackend.MATH])
    return contextlib.nullcontext()


# ── Transformer KV-cache decode (fair baseline) ──────────────────────────────

class TransformerKVCacheRunner:
    """Wraps a TransformerLM to run prefill + single-token cached decode.

    Cache: per-layer (k, v) of shape [B, H, T, head_dim]; grows by 1 each decode
    step. This is the standard deployment path (what the naive generate() lacks).
    """

    def __init__(self, model: TransformerLM):
        self.m = model
        self.cfg = model.config
        self.H = self.cfg.n_heads
        self.hd = self.cfg.d_model // self.cfg.n_heads

    def _block_step(self, block, x, kcache, vcache):
        # x: [B, 1, C]; kcache/vcache: [B, H, Tpast, hd] or None
        B, T, C = x.shape
        h = block.ln1(x)
        qkv = block.attn.qkv(h)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.H, self.hd).transpose(1, 2)  # [B,H,1,hd]
        k = k.view(B, T, self.H, self.hd).transpose(1, 2)
        v = v.view(B, T, self.H, self.hd).transpose(1, 2)
        if kcache is not None:
            k = torch.cat([kcache, k], dim=2)
            v = torch.cat([vcache, v], dim=2)
        # attend current token to all cached keys (no causal mask needed: 1 query,
        # all keys are past-or-current)
        with _sdpa_ctx():
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + block.attn.out_proj(y)
        x = x + block.ffn(block.ln2(x))
        return x, k, v

    @torch.no_grad()
    def prefill(self, input_ids):
        """Run the full prompt once, return per-layer (k, v) caches + last logits."""
        m = self.m
        B, T = input_ids.shape
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = m.token_embed(input_ids) + m.pos_embed(pos)
        caches = []
        for block in m.blocks:
            C = x.shape[2]
            h = block.ln1(x)
            qkv = block.attn.qkv(h)
            q, k, v = qkv.split(C, dim=2)
            q = q.view(B, T, self.H, self.hd).transpose(1, 2)
            k = k.view(B, T, self.H, self.hd).transpose(1, 2)
            v = v.view(B, T, self.H, self.hd).transpose(1, 2)
            with _sdpa_ctx():
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            x = x + block.attn.out_proj(y)
            x = x + block.ffn(block.ln2(x))
            caches.append((k, v))
        x = m.ln_f(x)
        logits = m.lm_head(x[:, -1:])
        return caches, logits

    @torch.no_grad()
    def decode_step(self, token_id, caches, pos):
        """One cached decode step. token_id: [B,1]. Returns (logits, caches)."""
        m = self.m
        B = token_id.shape[0]
        pos_t = torch.full((1,), pos, dtype=torch.long, device=token_id.device)
        x = m.token_embed(token_id) + m.pos_embed(pos_t)
        new_caches = []
        for block, (kc, vc) in zip(m.blocks, caches):
            x, k, v = self._block_step(block, x, kc, vc)
            new_caches.append((k, v))
        x = m.ln_f(x)
        logits = m.lm_head(x)
        return logits, new_caches


def _cache_bytes(caches):
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for k, v in caches)


def _pam_state_bytes(states):
    total = 0
    for s in states:
        if s is None:
            continue
        if isinstance(s, (list, tuple)):
            for t in s:
                if torch.is_tensor(t):
                    total += t.numel() * t.element_size()
        elif torch.is_tensor(s):
            total += s.numel() * s.element_size()
    return total


def _sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


# ── Benchmarks ───────────────────────────────────────────────────────────────

@torch.no_grad()
def bench_pam(model, L, decode_steps, device, vocab):
    ids = torch.randint(0, vocab, (1, L), device=device)
    # prefill
    _sync(device)
    t0 = time.perf_counter()
    logits, states, _ = model.forward(ids)
    _sync(device)
    prefill_ms = (time.perf_counter() - t0) * 1000

    state_bytes = _pam_state_bytes(states)
    nxt = logits[:, -1:].argmax(-1)
    step = L
    # warmup a couple of decode steps
    for _ in range(2):
        logits, states, _ = model.forward(nxt, states=states, step_offset=step)
        nxt = logits[:, -1:].argmax(-1)
        step += 1
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(decode_steps):
        logits, states, _ = model.forward(nxt, states=states, step_offset=step)
        nxt = logits[:, -1:].argmax(-1)
        step += 1
    _sync(device)
    decode_ms = (time.perf_counter() - t0) * 1000 / decode_steps
    return prefill_ms, decode_ms, state_bytes


@torch.no_grad()
def bench_transformer(runner, L, decode_steps, device, vocab):
    ids = torch.randint(0, vocab, (1, L), device=device)
    _sync(device)
    t0 = time.perf_counter()
    caches, logits = runner.prefill(ids)
    _sync(device)
    prefill_ms = (time.perf_counter() - t0) * 1000

    nxt = logits[:, -1:].argmax(-1)
    pos = L
    for _ in range(2):
        logits, caches = runner.decode_step(nxt, caches, pos)
        nxt = logits[:, -1:].argmax(-1)
        pos += 1
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(decode_steps):
        logits, caches = runner.decode_step(nxt, caches, pos)
        nxt = logits[:, -1:].argmax(-1)
        pos += 1
    _sync(device)
    decode_ms = (time.perf_counter() - t0) * 1000 / decode_steps
    cache_bytes = _cache_bytes(caches)
    return prefill_ms, decode_ms, cache_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--dtype', default='fp16', choices=['fp16', 'bf16', 'fp32'])
    p.add_argument('--preset', default='v11_e3_k3_chat')
    p.add_argument('--context', default='128,256,512,1024,2048',
                   help='comma-separated prompt/context lengths')
    p.add_argument('--decode', type=int, default=32, help='decode steps to time')
    p.add_argument('--max-seq-len', type=int, default=8192,
                   help='positional capacity for both models (random weights; '
                        'we only time forward, so extending beyond 2048 is fair)')
    p.add_argument('--attn', default='auto', choices=['auto', 'math'],
                   help="transformer SDPA backend: 'auto' allows the fused "
                        "FlashAttention kernel; 'math' forces pure-PyTorch "
                        "matmul+softmax+matmul so both models are unfused")
    p.add_argument('--out-json', default='logs/v11/infer_bench.json')
    args = p.parse_args()

    global _ATTN_MODE
    _ATTN_MODE = args.attn
    if args.attn == 'math' and not _HAS_SDPA_CTX:
        print("WARN: this torch lacks torch.nn.attention.sdpa_kernel; "
              "cannot force math backend — results will use the fused kernel.")

    device = args.device
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]
    if device == 'cpu' and dtype is torch.float16:
        dtype = torch.float32  # fp16 matmul is not well supported on CPU

    torch.manual_seed(0)

    # PAM
    pam_cfg = get_config(args.preset)
    pam_cfg.max_seq_len = args.max_seq_len
    pam = V11LM(pam_cfg).to(device=device, dtype=dtype).eval()
    pam_params = pam.count_parameters()['total']

    # Transformer (matched ~100M), vocab aligned to PAM for a fair per-step cost
    tcfg = get_transformer_config_100m()
    tcfg.vocab_size = pam_cfg.vocab_size
    tcfg.max_seq_len = args.max_seq_len
    tf = TransformerLM(tcfg).to(device=device, dtype=dtype).eval()
    tf_params = tf.count_parameters()['total']
    runner = TransformerKVCacheRunner(tf)

    vocab = pam_cfg.vocab_size
    contexts = [int(x) for x in args.context.split(',')]
    max_needed = max(contexts) + args.decode + 4
    if max_needed > args.max_seq_len:
        raise SystemExit(f"max context {max(contexts)} + decode {args.decode} "
                         f"exceeds --max-seq-len {args.max_seq_len}")

    # Warmup (absorb CUDA init / allocator / autotune so first row isn't an outlier)
    bench_pam(pam, 64, 4, device, vocab)
    bench_transformer(runner, 64, 4, device, vocab)

    print(f"device={device} dtype={args.dtype} attn={args.attn}"
          f"{' (pure PyTorch, no fused kernel)' if args.attn == 'math' else ' (fused FlashAttention allowed)'}")
    print(f"PAM {args.preset}: {pam_params/1e6:.1f}M params | "
          f"Transformer: {tf_params/1e6:.1f}M params")
    print()
    header = (f"{'L':>6} | {'PAM pre':>9} {'PAM dec':>9} {'PAM tok/s':>10} {'PAM state':>10} | "
              f"{'TF pre':>9} {'TF dec':>9} {'TF tok/s':>10} {'TF kv$':>10}")
    print(header)
    print('-' * len(header))

    rows = []
    for L in contexts:
        pam_pre, pam_dec, pam_state = bench_pam(pam, L, args.decode, device, vocab)
        tf_pre, tf_dec, tf_kv = bench_transformer(runner, L, args.decode, device, vocab)
        row = {
            'context': L,
            'pam_prefill_ms': pam_pre, 'pam_decode_ms': pam_dec,
            'pam_decode_tok_s': 1000.0 / pam_dec, 'pam_state_bytes': pam_state,
            'tf_prefill_ms': tf_pre, 'tf_decode_ms': tf_dec,
            'tf_decode_tok_s': 1000.0 / tf_dec, 'tf_kv_bytes': tf_kv,
        }
        rows.append(row)
        print(f"{L:>6} | {pam_pre:>8.1f}m {pam_dec:>8.2f}m {1000.0/pam_dec:>10.0f} "
              f"{pam_state/1e6:>9.1f}M | "
              f"{tf_pre:>8.1f}m {tf_dec:>8.2f}m {1000.0/tf_dec:>10.0f} {tf_kv/1e6:>9.1f}M")

    out = {
        'run_at': datetime.now(timezone.utc).isoformat(),
        'device': device, 'dtype': args.dtype, 'attn': args.attn,
        'pam_preset': args.preset, 'pam_params': pam_params, 'tf_params': tf_params,
        'decode_steps': args.decode,
        'note': ('decode_ms = steady-state ms/token; PAM state is fixed-size '
                 '(constant in L), TF kv$ grows linearly with L. Transformer uses '
                 'a real KV cache (fair baseline), not naive recompute.'),
        'rows': rows,
    }
    import os
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == '__main__':
    main()
