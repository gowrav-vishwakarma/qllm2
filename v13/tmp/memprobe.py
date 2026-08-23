#!/usr/bin/env python3
"""Peak-VRAM probe for a full train step at production shape.

Builds the 100M v13_e3_k3_selective model at B18/T2048 and runs ONE real
train step (forward + fused-CE + gate-aux + backward) with gradient
checkpointing ON or OFF, reporting peak allocated/reserved GB. Decides whether
--no_grad_ckpt fits the 24GB 4090 (it is the verified-correct + faster path);
if not, the ckpt path must be fixed to flow gradients (recompute stays for VRAM).

Usage: .venv/bin/python -m v13.tmp.memprobe [--ckpt]
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import get_config, V13LM

B, T = 18, 2048


def main():
    ckpt = '--ckpt' in sys.argv
    cfg = get_config('v13_e3_k3_selective')
    cfg.gradient_checkpointing = ckpt
    print(f"config: grad_ckpt={ckpt} B={B} T={T} dim={cfg.dim} L={cfg.n_layers} "
          f"K={cfg.n_states} fused_e3={cfg.fused_e3}")
    torch.manual_seed(0)
    model = V13LM(cfg).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ids = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')
    lab = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')

    # real training loss path (matches v7/train.py fused_ce + gate aux), under
    # bf16 autocast like the trainer's --amp_dtype auto
    with torch.autocast('cuda', dtype=torch.bfloat16):
        lm, aux, gate_probs = model._hidden_to_lm(ids)
        loss, nll = model.ce_from_lm(lm, lab, chunk=4096, return_nll=True)
        if gate_probs is not None and nll is not None:
            med = nll.flatten().median().detach()
            target = torch.sigmoid(med - nll).detach()
            g = torch.nn.functional.binary_cross_entropy(
                gate_probs.float(), target.expand_as(gate_probs).float(),
                reduction='mean')
            loss = loss + 0.1 * g
    opt.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_rsv = torch.cuda.max_memory_reserved() / 1e9
    print(f"\nRESULT (grad_ckpt={ckpt}):")
    print(f"  loss={float(loss):.4f}")
    print(f"  PEAK alloc (whole step) = {peak_alloc:.2f} GB  (headroom to 23.5GB: {23.5-peak_alloc:+.2f} GB)")
    print(f"  PEAK reserved = {peak_rsv:.2f} GB")
    fit = "FITS" if peak_rsv < 23.3 else "TOO TIGHT / OOM RISK"
    print(f"  VERDICT: {fit}")
    sys.exit(0 if peak_rsv < 23.3 else 1)


if __name__ == '__main__':
    main()
