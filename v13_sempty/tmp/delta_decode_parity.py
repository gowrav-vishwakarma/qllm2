"""Delta decode parity: chunked full-sequence logits == prefill + step-wise logits.

Usage:  PYTHONPATH=. .venv/bin/python v13_sempty/tmp/delta_decode_parity.py
"""
import sys
import torch

sys.path.insert(0, '.')
from v13_sempty.config import get_config   # noqa: E402
from v13_sempty.model import LM            # noqa: E402


def run(dtype, T=200, T0=150, B=2, chrono=True, out_gate=True, seed=0):
    torch.manual_seed(seed)
    cfg = get_config('baseline_real_pm')
    cfg.n_layers, cfg.d_model, cfg.n_heads = 2, 128, 2
    cfg.chrono, cfg.out_gate, cfg.delta = chrono, out_gate, True
    cfg.dropout = 0.0
    cfg.max_seq_len = 512
    model = LM(cfg).cuda().eval()
    # nudge the gates/decay off their init so erase/write/decay are all live
    with torch.no_grad():
        for blk in model.blocks:
            for lin in (blk.pam.erase_proj, blk.pam.write_proj, blk.pam.dt_proj):
                for p in lin.parameters():
                    p.add_(torch.randn_like(p) * 0.5)
    ids = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
        full, _, _ = model(ids)                                   # chunked, whole sequence
        logits, states, _ = model(ids[:, :T0])                    # prefill
        outs = [logits[:, -1]]
        for t in range(T0, T - 1):
            logits, states, _ = model(ids[:, t:t + 1], states=states, step_offset=t)
            outs.append(logits[:, -1])
        step = torch.stack(outs, dim=1)                           # predictions for positions T0-1..T-2
    ref = full[:, T0 - 1:T - 1].float()
    rel = ((ref - step.float()).norm() / ref.norm()).item()
    agree = (ref.argmax(-1) == step.float().argmax(-1)).float().mean().item()
    tol = 1e-4 if dtype == torch.float32 else 3e-2
    ok = rel <= tol
    print(f"{str(dtype).split('.')[-1]:8s} chrono={chrono} out_gate={out_gate} T={T} T0={T0}: "
          f"rel={rel:.2e} argmax_agree={agree:.3f} tol={tol:.0e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    ok = True
    ok &= run(torch.float32)
    ok &= run(torch.float32, chrono=False, out_gate=False)
    ok &= run(torch.float32, T=300, T0=64)         # ragged prefill (64 = one sub-chunk)
    ok &= run(torch.bfloat16)
    print('ALL OK' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)
