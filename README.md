# QLLM - Phase-First Language Model Research

> **Disclaimer:** Yes, we use AI, Cursor, and every coding agent we can get our hands on to build this. Because... why not? We're building AI with AI.

## Beyond Transformers

This repository explores language models beyond the standard transformer recipe.

- **v6** (current): phase-first, attention-free by default, O(n), memory-augmented backbone focused on autoregressive training, with diffusion scaffolding in place for later work
- **v5** (major breakthrough): mathematically consistent complex-valued LM that fixed V4 and delivered the first strong results
- **v4** (origin of the novelty): complex phase-space tokens, wave interference, O(n) backbone, and the original non-transformer direction
- **v3**: brain-inspired language modeling branch
- **v2**: earlier quantum-inspired language modeling branch

## Project Overview

### **v6 - Phase-First Language Model** (Current)

V6 is the current main line: a **zero-attention language model direction** built around complex-valued tokens, named banks, phase interference, multi-timescale SSMs, and external memory layers. It keeps the phase-native idea from V4, the mathematical cleanup from V5, and pushes toward a longer-context, more modular system.

Core V6 path:

```text
Tokens --> ComplexEmbed
  --> [SemanticBank + ContextBank --> PhaseInterferenceCoupler] x N
  --> MultiTimescaleSSM
  --> WorkingMemory
  --> InternalMemory
  --> [PersistentMemory]
  --> [ExpertMemory]
  --> MemoryFusion
  --> TiedComplexLMHead
```

What defines V6:

- **Zero attention by default**: the main direction is O(n) sequence processing without token-token attention
- **Named banks**: `SemanticBank` and `ContextBank` specialize and interfere through learned phase rotations
- **Multi-timescale SSM**: fast, medium, and slow decay lanes track syntax, paragraph coherence, and durable facts
- **Memory hierarchy**: working, internal, persistent, expert, and optional session memory
- **Transferable memory pipeline**: session memory can compress into persistent memory, persistent memory can become shared expert memory, and `MemoryAdaptation` can distill user-specific memory back toward internal memory
- **Shared backbone scaffold**: the same `PhaseFieldBackbone` is structured so diffusion text and diffusion image can reuse the autoregressive core later, avoiding duplicated implementations

### **v5 - Algebraic Language Model** (Previous Main Version)

V5 was the turning point. It kept the complex-valued hypothesis but fixed the core mathematical inconsistency in V4: V4 created complex states and then destroyed phase information with real-valued nonlinearities. V5 replaced that with **phase-preserving computation end to end**.

- **Key fix**: `modReLU` and `ComplexGatedUnit` replace real-valued GELU/sigmoid paths
- **Key result**: a 28.7M `small-matched` model beat V4's 178M-class results
- **Key lesson**: smaller but mathematically cleaner beat bigger but inconsistent
- **Best documented public result here**: TinyStories val PPL **5.59** at epoch 3 on the full dataset

### **v4 - Quantum Phase-Field LLM** (Novelty Origin)

V4 is where the main novelty first appeared: tokens living in complex phase space, wave-style interference between banks, and an O(n) non-transformer backbone. Even though V5 superseded it mathematically, V4 was the base of the idea and the first version that made this line of work distinct.

### **v2 / v3**

- **v2** explored a quantum-inspired LM framing with superposition, entanglement, and phase coherence themes.
- **v3** explored a brain-inspired LM framing with memory systems, consciousness motifs, and biologically inspired learning ideas.

## V6 Results

### TinyStories

The current V6 line is already strong on TinyStories.

| Run | Model | Dataset | Progress | Best Val PPL | Notes |
| --- | --- | --- | --- | --- | --- |
| `small_matched_full` | `small-matched` | TinyStories full train split | 5/10 epochs | **5.50** | clean no-memory baseline |
| `fulldata_tiny_memory` | `small-matched` | TinyStories full train split | 1/1 epoch | 2.23 | likely overfit / repeating, not the headline result |

Example TinyStories generation:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and make new friends. One day, she went on an adventure in the forest near her house.

### WikiText-103

V6 is also now training on a more realistic corpus, not just TinyStories.

| Epoch | Train PPL | Val PPL |
| --- | --- | --- |
| 1 | 247.45 | 121.94 |
| 2 | 106.70 | 82.86 |
| 3 | 83.45 | 70.64 |
| 4 | 74.32 | **64.91** |

Run details:

- **Model**: `small-matched` (~28.7M)
- **Dataset**: WikiText-103
- **Sequence length**: 512
- **Hardware**: RTX 4090
- **Throughput**: ~46K tok/s
- **Status**: still training, epoch 5 underway

The qualitative signal is also interesting: generations are starting to pick up Wikipedia-style structure, named entities, and section headers.

> The history of the event.
>
> `=== The Summer Olympics ===`
>
> During the Olympics, the men's team competed in the event, where they had won the first round of a silver medal...

## V5 Results

V5 is still an important milestone because it proved the math cleanup mattered.

| Run | Init | Data | Epochs | Best Val PPL | GPU |
| --- | --- | --- | --- | --- | --- |
| `v1-untied` | random | 100k TinyStories | 10 | 21.03 | A6000 |
| `v3-full` | random | 100k TinyStories | 10 | 11.77 | A6000 |
| `v4-ortho` | orthogonal | 100k TinyStories | 10 | 8.00 | RTX 4090 |
| `v5-full-ds` | orthogonal | full TinyStories | 3/10 | **5.59** | RTX 4090 |

The bigger V5 story is not just the number. It is that **preserving phase properly made the architecture genuinely work**.

## Quick Start

### Install

```bash
uv sync
uv sync --extra cuda
```

### Train V6

```bash
# Smoke test (CPU)
python -m v6.train --size tiny --epochs 5 --max_samples 1000 --seq_len 128

# Standard TinyStories training
python -m v6.train --size small-matched --max_samples 9999999 --seq_len 256 \
  --compile --compile_mode reduce-overhead --amp_dtype auto --num_workers 4

# WikiText-103
./scripts/run_v6_wikitext103.sh

# PG-19
./scripts/run_v6_pg19.sh
```

### Generate With V6

```bash
# Autoregressive text
python -m v6.generate --checkpoint checkpoints_v6/best_model.pt \
  --prompt "Once upon a time"

# With persistent memory
python -m v6.generate --checkpoint checkpoints_v6/best_model.pt \
  --persistent_memory user_alice.pt --prompt "Tell me a story"
```

### Diffusion Scaffolding

```bash
# Text diffusion scaffold (not yet a validated training path)
python -m v6.train --mode diffusion_text --size small-matched --epochs 10

# Image diffusion scaffold (planned / experimental)
python -m v6.train --mode diffusion_image --image_encoder patch --image_size 64
```

Those diffusion paths are present to keep the architecture ground-up ready and avoid rebuilding the backbone later. The current validated work is still the autoregressive line; diffusion is planned to be implemented and tested after, or alongside, the current autoregressive push.

### V5 Still Matters

```bash
# V5 smoke test
python -m v5.train --size tiny --epochs 5 --max_samples 1000

# V5 standard training
python -m v5.train --size small-matched --epochs 10 --max_samples 100000 --init_strategy orthogonal
```

Earlier branches are still available in `v4/`, `v3/`, and `v2/`.

## Performance Comparison

The most useful comparison now is across the main non-transformer line itself:

| Feature | v4 | v5 | v6 |
| --- | --- | --- | --- |
| **Core idea** | phase-space tokens + wave interference | mathematically consistent complex LM | phase-first LM with memory hierarchy |
| **Attention** | none in core design | sparse PhaseAttention hybrid | none by default |
| **Sequence complexity** | O(n) | O(n) with parallel scan depth improvements | O(n) |
| **Banks** | dynamic phase banks | algebraic banks | named semantic/context banks |
| **Coupling** | interference-inspired | `AlgebraicFusion` | `PhaseInterferenceCoupler` |
| **Memory** | dual memory ideas | limited / indirect | working + internal + persistent + expert + session |
| **Activation / gating** | real-valued breaks phase | `modReLU` + CGU | inherits V5's phase-preserving core |
| **Modes** | autoregressive | autoregressive | autoregressive now, with diffusion scaffolding in place |
| **Headline result** | novelty proof-of-concept | strong TinyStories breakthrough | current active direction |

`v2` and `v3` remain earlier exploration branches rather than the current main comparison target.

## Key Innovations Across Versions

### V6

- Attention-free by default, with a phase-native backbone aimed at long-context efficiency
- Memory hierarchy that separates per-sequence facts, trained knowledge, user memory, and shared expert memory
- Transferable memory system already implemented in code: session -> persistent -> expert
- Shared backbone designed so future diffusion work can reuse the autoregressive core instead of duplicating the stack
- WikiText-103 training shows the architecture is moving beyond TinyStories-only validation

### V5

- Phase-preserving activations and gating made the complex-valued hypothesis actually coherent
- Orthogonal initialization turned out to matter a lot for stable complex training
- Weight tying with `Re(z * conj(embed))` reduced parameters and stayed algebraically consistent

### V4

- Introduced the novelty: complex phase-space tokens, wave interference framing, O(n) backbone, and philosophy-inspired interpretability metrics

## Model Sizes

### V6 presets

| Size | Complex Dim | Layers | Banks | State Dim | Batch Size |
| --- | --- | --- | --- | --- | --- |
| `tiny` | 64 | 4 | 2 | 128 | 16 |
| `small` | 256 | 8 | 2 | 512 | 8 |
| `small-matched` | 128 | 12 | 2 | 512 | 8 |
| `medium` | 512 | 12 | 2 | 1024 | 4 |
| `large` | 512 | 24 | 2 | 1536 | 2 |
| `xl` | 768 | 32 | 2 | 2048 | 1 |

Memory is off by default in the presets and can be enabled with flags such as `--wm_slots`, `--im_slots`, and persistent/session memory options.

## Project Structure

```text
qllm2/
├── README.md
├── v6/                    # current main line
│   ├── README.md
│   ├── model.py
│   ├── backbone.py
│   ├── diffusion_model.py
│   ├── train.py
│   ├── generate.py
│   └── core/
├── v5/                    # algebraic LM breakthrough
│   ├── README.md
│   ├── EXPERIMENTS.md
│   └── core/
├── v4/                    # original phase-field architecture
├── v3/                    # brain-inspired branch
├── v2/                    # quantum-inspired branch
└── scripts/               # training / eval / benchmark helpers
```

## Development Status

### V6

- Done: named banks, phase interference coupler, multi-timescale SSM, working/internal/persistent/expert memory layers
- Done: autoregressive training and generation
- Done: diffusion code paths scaffolded on the shared backbone to avoid future duplication
- Done: TinyStories and WikiText-103 training support
- In progress: longer runs on WikiText-103 and broader non-TinyStories evaluation
- In progress: better activation of persistent/session/expert memory in practical runs
- In progress: implementing and validating diffusion work after or alongside the current autoregressive effort
- In progress: stronger benchmarking and scale-up experiments

### V5

- Stable reference point for the mathematically consistent complex LM direction
- Important benchmark and ablation base for understanding what V6 inherited

### V4 / V3 / V2

- Historical branches kept for reference, comparison, and idea lineage

## Honest Limitations

This is still research code. We do not want to oversell it.

- No strict apples-to-apples transformer baseline at the same parameter scale and budget yet
- Long-context and downstream evaluations are still incomplete
- V6's transferable memory system exists in code, but the full deployment workflow is not yet the default path
- Diffusion paths exist in the codebase, but they are not yet a tested or benchmarked headline capability
- Pure PyTorch, with obvious room for custom kernels and systems work
- Scaling behavior beyond the currently explored sizes still needs validation

We are not claiming that complex or phase-first models have already beaten transformers broadly. The narrower claim is that this line of work has produced a real, improving non-transformer family with results strong enough to justify serious continued iteration.

## Documentation

- [v6/README.md](v6/README.md): architecture, CLI, memory system, diffusion modes, setup
- [v5/README.md](v5/README.md): algebraic LM details and the V4 -> V5 correction
- [v5/EXPERIMENTS.md](v5/EXPERIMENTS.md): V5 experiment log
- [v4/README.md](v4/README.md): original phase-field architecture
- [v2/README.md](v2/README.md): quantum-inspired branch
- [v3/README.md](v3/README.md): brain-inspired branch

## Research Notes

- `QLLM_CORE_IDEA.pdf`: core idea document
- `v5/paper/`: V5 paper draft
- `QLLM_V2.pdf`: earlier quantum-inspired paper

## Contributing

1. Pick the branch you want to work on: `v6`, `v5`, `v4`, `v3`, or `v2`.
2. Read that version's README first.
3. Run a smoke test before making larger changes.
4. Keep results and claims honest and tied to actual logs.

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0. See [LICENSE](LICENSE) for details.

You may use, study, modify, and share this work for non-commercial purposes. Commercial use is not granted under the default license. If you want to use this work in a commercial setting, contact [gowravvishwakarma@gmail.com](mailto:gowravvishwakarma@gmail.com).

## Acknowledgments

- **v6**: phase-first language modeling, external memory layers, diffusion reuse of the same backbone
- **v5**: complex-valued language modeling, `modReLU`, CGU, Blelloch parallel scan, orthogonal initialization work
- **v4**: phase-field modeling, oscillatory SSM ideas, philosophy-inspired metrics
- **v2**: quantum-inspired modeling direction
- **v3**: neuroscience and brain-inspired modeling direction
- Built on PyTorch and modern deep learning tooling

---

**Current focus**: `v6`
**Previous breakthrough**: `v5`
**Novelty origin**: `v4`
**Last Updated**: 2026-03-10