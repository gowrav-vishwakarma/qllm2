"""EOS-reset decode parity (Lead 2026-11-24, found by the HF-port parity work): with reset_at_eos, one-token decode across a
<|endoftext|> separator must equal the full-sequence forward (which resets the notebooks AND echo at the separator).
CPU fp32, small M2-style body (delta, chrono, out_gate, echo, share_reads, notebooks at 1 / 3 of 4 blocks).
    python v13_sempty/tmp/eos_decode_parity.py"""
import sys
import torch
sys.path.insert(0, '.')
from v13_sempty.config import get_config   # noqa: E402
from v13_sempty.model import LM            # noqa: E402


def run(T=160, T0=40, B=2, seed=0, eos_at=(70, 71, 120)):
    torch.manual_seed(seed)
    cfg = get_config('baseline_real_pm')
    cfg.n_layers, cfg.dim, cfg.n_heads, cfg.head_dim = 4, 128, 2, 64
    cfg.mlp_hidden = 256
    cfg.chrono = cfg.out_gate = cfg.delta = cfg.echo = cfg.share_reads = True
    cfg.pam_layers = (1, 3)
    cfg.reset_at_eos = True
    cfg.dropout = 0.0
    cfg.max_seq_len = 512
    cfg.gradient_checkpointing = False
    model = LM(cfg).eval()
    with torch.no_grad():
        for blk in model.blocks:
            if blk.pam is None:
                continue
            for lin in (blk.pam.erase_proj, blk.pam.write_proj, blk.pam.dt_proj):
                for p in lin.parameters():
                    p.add_(torch.randn_like(p) * 0.5)
    ids = torch.randint(0, 50000, (B, T))
    for t in eos_at:
        ids[:, t] = cfg.reset_token_id
    with torch.no_grad():
        full, _, _ = model(ids)
        logits, states, _ = model(ids[:, :T0])
        outs = [logits[:, -1]]
        for t in range(T0, T - 1):
            logits, states, _ = model(ids[:, t:t + 1], states=states, step_offset=t)
            outs.append(logits[:, -1])
        step = torch.stack(outs, dim=1)
    ref = full[:, T0 - 1:T - 1]
    d = (ref - step).abs()
    pre = d[:, :eos_at[0] - T0 + 1].max().item()          # before the first separator is fed
    post = d[:, eos_at[0] - T0 + 1:].max().item()         # after it
    ok = pre < 1e-5 and post < max(20 * pre, 1e-5)          # no jump at the separator (bug: post ~300x pre)
    print(f"reset token {cfg.reset_token_id}: max|full - decode| before EOS {pre:.2e}, after EOS {post:.2e} -> {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    ok = run() & run(seed=1, eos_at=(41, 100, 150))
    print('ALL OK' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)
