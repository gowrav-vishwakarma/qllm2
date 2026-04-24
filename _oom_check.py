"""Verify the EffectAlgebraBank index_select fix end-to-end on the medium preset.

Builds the full V8 model with ``e2e_medium_reasoning`` (the exact config
that OOMed in tmux) on a single batch and runs forward+backward+optimizer
step under ``torch.compile(mode='default')``. Reports peak CUDA memory.
Pre-fix this would crash with 38.6 GiB allocation in inductor's compiled
backward; post-fix it should fit comfortably in 24 GiB.
"""
import torch
import v8.train  # installs noise filters + TF32

from v8.config import get_config
from v8.model import V8LM


def main() -> None:
    torch.manual_seed(0)
    cfg = get_config("e2e_medium_reasoning")
    model = V8LM(cfg).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    cmodel = torch.compile(model, mode="default")

    B, T = 3, 2048
    x = torch.randint(0, cfg.backbone.vocab_size, (B, T), device="cuda")
    y = torch.randint(0, cfg.backbone.vocab_size, (B, T), device="cuda")

    torch.cuda.reset_peak_memory_stats()
    logits, _, ponder = cmodel(x, labels=y)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
    ) + ponder
    loss.backward()
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize()

    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(f"PASS: medium preset compile + forward + backward + step OK.")
    print(f"      Peak CUDA memory: {peak_gib:.2f} GiB (24 GiB available)")
    print(f"      Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
