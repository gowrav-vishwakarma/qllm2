# v13_sempty

v13 selective PAM language model rewritten on the [sempyt](https://github.com/) named-axis frontend (`/home/gowrav/Development/sempyt/src`).

Same architecture, same parameter names, same training math as `v13`. A v13 `state_dict` loads directly. Forward logits and backward grads match v13 on CPU (see `selftest.py`).

The model is written in **sempyt named-axis style**: `Dim` objects, `.to()` instead of `.view()`, `contract` / `as_complex` / `SplitComplex *` instead of raw `F.linear` and `[..., 0]` bookkeeping. The only `.raw()` exits are things sempyt cannot name — `torch.linalg.solve_triangular` in the delta UT solve, and the chunked fused-CE autograd Function.

## Layout

| File | Role |
|------|------|
| `config.py` | `V13Config`, `PRESETS`, `get_config` (data only) |
| `complex_ops.py` | Split-real complex modules via sempyt `NamedTensor` + `SplitComplex` |
| `model.py` | `V13PAMLayer` / `V13Block` / `V13LM` (fused PAM path is torch at `.raw()`) |
| `fused_ce.py` | Chunked tied-head linear + CE (custom autograd; `grad_weight +=`) |
| `train.py` | Self-contained trainer (no `V7Trainer`) |
| `selftest.py` | Equivalence contract vs `v13` |
| `generate.py` | Prefix completion from a checkpoint |

Triton fused kernels from `v13/triton_kernels.py` are **not** ported. The eager PyTorch / sempyt fallbacks are the reference math (correctness, not peak CUDA throughput).

## sempyt from source

`import sempyt` is resolved via a `.pth` in the qllm2 venv pointing at `/home/gowrav/Development/sempyt/src`. Framework edits there are picked up immediately (no pip install). Later this will be a published package.

## Run (CPU — do not steal a live GPU training job)

```bash
# equivalence vs v13
.venv/bin/python -m v13_sempty.selftest

# synthetic smoke train (CPU)
.venv/bin/python -m v13_sempty.train --preset tiny --dataset synthetic --steps 8 --device cpu

# generate (needs a checkpoint + tokenizer)
.venv/bin/python -m v13_sempty.generate --checkpoint checkpoints_v13_sempty/latest.pt --device cpu
```

Production preset is `v13_e3_k3_selective` (delta fused, E3 K=3, vault, write-phase-address). Use `--device cuda` only when a GPU is free.

## Equivalence bar

Documented in `selftest.py`. Measured on CPU fp32 with shared weights:

- PAM layer + full LM forward/grad: `0.0`
- fused CE vs v13: `0.0`
- parallel-train vs recurrent-infer: `~1e-8` (bar `2e-3`)
