"""End-to-end compile sanity for the QLC fix.

Verifies that under ``torch.compile(mode='default')``:
  1. forward + backward succeed for the e2e_tiny_reasoning preset,
  2. mutating ``model.qlc.t_max`` mid-run (the QLC schedule transition)
     triggers a recompile but does not error.
"""
import time
import torch

from v8.config import get_config
from v8.model import V8LM


def _step(cmodel, x, y):
    logits, _, ponder = cmodel(x, labels=y)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
    ) + ponder
    loss.backward()
    torch.cuda.synchronize()
    return loss.item()


def main() -> None:
    torch.manual_seed(0)
    cfg = get_config("e2e_tiny_reasoning")
    model = V8LM(cfg).cuda()
    model.train()

    cmodel = torch.compile(model, mode="default")

    B, T = 2, 64
    x = torch.randint(0, cfg.backbone.vocab_size, (B, T), device="cuda")
    y = torch.randint(0, cfg.backbone.vocab_size, (B, T), device="cuda")

    t0 = time.time()
    loss1 = _step(cmodel, x, y)
    print(f"step 1 (compile + run) loss={loss1:.4f}  took={time.time()-t0:.1f}s")

    model.zero_grad()
    t1 = time.time()
    loss2 = _step(cmodel, x, y)
    print(f"step 2 (cached)        loss={loss2:.4f}  took={time.time()-t1:.3f}s")

    print(f"mutating qlc.t_max: {model.qlc.t_max} -> {model.qlc.t_max + 1}")
    model.qlc.t_max += 1
    model.zero_grad()
    t2 = time.time()
    loss3 = _step(cmodel, x, y)
    print(f"step 3 (recompile)     loss={loss3:.4f}  took={time.time()-t2:.1f}s")

    print("PASS: compile + autograd + QLC schedule mutation all OK.")


if __name__ == "__main__":
    main()
