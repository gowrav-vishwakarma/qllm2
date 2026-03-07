# QLLM - Algebraic & Brain-Inspired Language Models (Quantum Inspired, but V5 is divered intentionally, V7 will be back on Quantum) 

> **Disclaimer:** Yes, we use AI, Cursor, and every coding agent we can get our hands on to build this. Because... why not? We're building AI with AI. It's poetic, it's practical, and honestly, it would be weird if we didn't.

## Beyond Transformers: Novel AI Architecture Research

This repository explores language model architectures beyond traditional transformers. **V5 is the current active version.**

- **v5** (active): Algebraic LM -- complex-valued end-to-end, phase-preserving, SSM + sparse attention hybrid
- **v4** (prior): Quantum Phase-Field LLM -- O(n) backbone, Phase2D math (superseded by v5)
- **v3**: Brain-Inspired Language Model (BLLM)
- **v2**: Quantum-Inspired Language Model (QLLM)

## Project Overview

### **v5 - Algebraic Language Model** (Current)

A ground-up redesign that fixes V4's core mathematical inconsistency: V4 created complex-valued representations then destroyed them with real-valued activations. V5 preserves complex algebraic structure end-to-end.

- **Architecture**: ComplexSSM backbone + multi-bank interference + sparse PhaseAttention
- **Key Fix**: modReLU + ComplexGatedUnit replace real-valued GELU/sigmoid -- phase is preserved, not destroyed
- **Result**: A 28.7M-param V5 model already beats V4's best results at a fraction of the size
- **Training**: Val PPL **5.59** on TinyStories at epoch 3 (full dataset, still improving)
- **Init Research**: Orthogonal initialization is 31% better than random for complex-valued layers
- **Status**: Active Development
- **Parameters**: ~7M (tiny) to ~540M (large)
- **Target Hardware**: Consumer GPUs (RTX 4090) and data-center GPUs (A6000/A100)

### **v4 - Quantum Phase-Field LLM** (Prior)

- **Approach**: Quantum phase-field architecture with Phase2D representation
- **Key Features**: Oscillatory SSM (O(n) linear), Dynamic Phase Bank Interference, Dual Memory System
- **Status**: Superseded by V5 (V4's math was inconsistent -- see V5 README for diagnosis)
- **Parameters**: ~1M (tiny) to ~350M (large)

### **v2 - Quantum-Inspired LLM**

- **Approach**: Quantum mechanics-inspired architecture
- **Key Features**: Quantum superposition, entanglement, phase coherence
- **Status**: Production Ready
- **Parameters**: 5M+ parameters
- **Performance**: 1.43x faster, 4.38x more memory efficient

### **v3 - Brain-Inspired LLM**

- **Approach**: Human brain-inspired architecture
- **Key Features**: Consciousness, memory systems, biologically plausible learning
- **Status**: Production Ready
- **Parameters**: 16.4M parameters
- **Performance**: Human-like learning efficiency, consciousness mechanisms

## Quick Start

### **Prerequisites**

```bash
uv sync
```

### **Choose Your Approach**

#### **Option 1: Algebraic LM (v5) -- Recommended**

```bash
# Smoke test (CPU/MPS -- tiny model, small data)
python -m v5.train --size tiny --epochs 5 --max_samples 1000

# Standard training (needs GPU)
python -m v5.train --size small-matched --epochs 10 --max_samples 100000

# Full dataset training (RTX 4090 / A6000)
python -m v5.train --size small-matched --epochs 10 --init_strategy orthogonal

# Generate text from checkpoint
python -m v5.generate --checkpoint checkpoints_v5/best_model.pt --prompt "Once upon a time"
```

See [v5/README.md](v5/README.md) for full options including init strategies, ablations, A6000 scripts, and config presets.

#### **Option 2: Quantum Phase-Field (v4)**

```bash
cd v4
uv run python train_real.py --dataset tinystories --size medium --epochs 50
```

#### **Option 3: Quantum-Inspired (v2)**

```bash
cd v2
uv run python run_training.py
```

#### **Option 4: Brain-Inspired (v3)**

```bash
cd v3
uv run python train_brain_llm.py
```

## V5 Training Results

### Current best: 28.7M params on TinyStories

V5 has been through several iterations. Each row is the same 28.7M `small-matched` architecture:


| Run        | Init           | Data                           | Epochs  | Best Val PPL | GPU      |
| ---------- | -------------- | ------------------------------ | ------- | ------------ | -------- |
| v1-untied  | random         | 100k samples                   | 10      | 21.03        | A6000    |
| v3-full    | random         | 100k samples                   | 10      | 11.77        | A6000    |
| v4-ortho   | **orthogonal** | 100k samples                   | 10      | 8.00         | RTX 4090 |
| v5-full-ds | **orthogonal** | **full dataset** (474M tokens) | 3 of 10 | **5.59**     | RTX 4090 |


The full-dataset run is still in progress. At epoch 3, val PPL is 5.59 with the curve still dropping and no sign of plateauing. Train/val gap is only ~0.38, so overfitting is not the limiting factor.

**Sample generation** (prompt: `The quick brown`, epoch 3):

> The quick brown dog wanted to go fast. He grabbed the butterfly with his paws and started jogging faster than ever before. He was so so happy that he had done it!

### Why V5 beats V4 at 1/6 the parameters

V4 used complex representations but applied real-valued activations (GELU, sigmoid) that destroyed phase information. V5 fixes this with phase-preserving operations throughout:


| Component  | V4 (broken)                        | V5 (fixed)                                   |
| ---------- | ---------------------------------- | -------------------------------------------- |
| Activation | `GELU(real_part)`                  | `modReLU(magnitude)` -- phase preserved      |
| Gating     | `sigmoid(concat(r,i))`             | ComplexGatedUnit -- gate selects AND rotates |
| Backbone   | Sequential for-loop                | Blelloch parallel scan -- O(log n) depth     |
| LM Head    | Separate projection (12.9M params) | Weight tying: `Re(z * conj(embed))`          |


## Performance Comparison


| Feature                 | v2        | v3        | v4                        | v5                            |
| ----------------------- | --------- | --------- | ------------------------- | ----------------------------- |
| **Representation**      | sin/cos   | N/A       | Phase2D (no trig)         | Complex-valued end-to-end     |
| **Activation**          | Standard  | Standard  | GELU (real)               | modReLU (phase-preserving)    |
| **Sequence complexity** | O(n^2)    | O(n^2)    | O(n) linear               | O(n) linear, O(log n) depth   |
| **Backbone**            | Attention | Attention | Oscillatory SSM           | ComplexSSM + sparse attention |
| **Gating**              | Standard  | Standard  | Real sigmoid              | ComplexGatedUnit              |
| **Memory systems**      | No        | Yes       | Dual (global + episodic)  | SSM state + sparse attention  |
| **FFN layer**           | Yes       | Yes       | Yes                       | None (CGU replaces it)        |
| **Weight tying**        | No        | No        | No                        | Yes (`Re(z * conj(embed))`)   |
| **Init research**       | N/A       | N/A       | N/A                       | 13 strategies benchmarked     |
| **GPU efficiency**      | Medium    | Medium    | High                      | High (parallel scan)          |
| **Interpretability**    | Low       | Medium    | High (philosophy metrics) | Medium (bank specialization)  |


## Key Innovations

### **v5 - Algebraic LM Features**

- **Phase-Preserving Computation**: Every operation maintains complex algebraic structure. modReLU thresholds magnitude while preserving phase direction exactly.
- **ComplexGatedUnit (CGU)**: SwiGLU-like gating in complex space -- the gate simultaneously selects (magnitude) and transforms (phase). One complex gate does the work of two real gates. Replaces the need for a separate FFN layer.
- **ComplexSSM with Parallel Scan**: Selective state space model with complex-valued transitions. Uses Blelloch associative scan for O(log n) parallel depth instead of O(n) sequential. Complex eigenvalues naturally create damped oscillators at different frequencies -- fast oscillators track syntax, slow ones track topics.
- **PhaseAttention**: Sparse attention (every 4th layer) where scores use `Re(q * conj(k))` -- captures both magnitude similarity and phase alignment.
- **Multi-Bank Interference**: Multiple banks process input through different "lenses" with learned routing. AlgebraicFusion combines outputs via phase rotations -- genuine constructive/destructive interference.
- **Weight Tying**: Output logits computed as `Re(z * conj(embed))`, reusing the embedding table. Saves 12.9M params and is algebraically consistent.
- **Orthogonal Initialization**: Benchmarked 21 strategies (kept 13, removed 8 broken ones). Orthogonal is 2x better than random. Norm-preserving isometry prevents vanishing/exploding gradients in complex layers.

### **v4 - Quantum Phase-Field Features**

- Phase2D Math (complex as 2D real vectors, no trig)
- Oscillatory SSM backbone, Dynamic Phase Banks
- Dual Memory System, Byte Patching, Philosophy Metrics

### **v2 - Quantum-Inspired Features**

- Quantum Superposition, Entanglement, Phase Coherence
- Energy-Based Training, Concept Layers

### **v3 - Brain-Inspired Features**

- Consciousness Layer, Memory Systems
- Biologically Plausible Learning, Spiking Neurons

## Project Structure

```
qllm/
├── README.md                    # This file
├── pyproject.toml              # Project configuration
├── QLLM_CORE_IDEA.pdf         # Core idea document
│
├── v5/                         # Algebraic LM (ACTIVE)
│   ├── README.md              # Full v5 documentation
│   ├── EXPERIMENTS.md         # Experiment log and results
│   ├── model.py               # AlgebraicLM -- wires everything
│   ├── train.py               # Training loop (TinyStories, GPT-2 tokenizer)
│   ├── generate.py            # Text generation from checkpoint
│   ├── config.py              # V5Config with presets
│   ├── init.py                # 13 structured init strategies
│   ├── core/
│   │   ├── complex.py         # ComplexLinear, modReLU, ComplexNorm, CGU, ComplexEmbed
│   │   ├── ssm.py             # ComplexSSM with Blelloch parallel scan
│   │   ├── attention.py       # PhaseAttention (sliding window, complex Q/K/V)
│   │   └── bank.py            # AlgebraicBank, ComplexRouter, AlgebraicFusion
│   └── paper/                 # LaTeX preprint
│
├── v4/                         # Quantum Phase-Field LLM (prior version)
│   ├── README.md              # Full v4 documentation
│   ├── model.py               # Main model
│   ├── train_real.py          # Training with real datasets
│   ├── core/                  # Phase2D math, config, registry
│   ├── banks/                 # Phase banks
│   ├── backbone/              # Oscillatory SSM backbone
│   └── ...
│
├── v2/                         # Quantum-Inspired LLM
│   ├── README.md              # v2 documentation
│   ├── quantum_llm_model.py   # Main quantum model
│   └── run_training.py        # Training script
│
├── v3/                         # Brain-Inspired LLM
│   ├── README.md              # v3 documentation
│   ├── brain_inspired_llm.py  # Main brain model
│   └── train_brain_llm.py     # Training script
│
└── scripts/                    # Training, monitoring, benchmarking scripts
```

## Use Cases

### **Choose v5 (Algebraic LM) for:**

- **Phase-preserving language modeling** -- the core research direction
- **Complex-valued architecture research** (modReLU, CGU, complex SSMs)
- **Consumer GPU training** (RTX 4090 / A6000)
- **Initialization strategy research** for complex-valued networks
- **SSM + attention hybrid experiments**

### **Choose v4 (Quantum Phase-Field) for:**

- Reference implementation of Phase2D math
- Understanding what V5 fixed (V4 README documents the original design)

### **Choose v2 (Quantum-Inspired) for:**

- Speed-critical applications
- Quantum computing research
- Energy-efficient processing

### **Choose v3 (Brain-Inspired) for:**

- Consciousness research
- Minimal data scenarios
- Biologically plausible AI

## Scientific Impact

### **v5 - Phase-Preserving Complex-Valued Language Modeling**

- Demonstrates that **mathematical consistency matters more than model size**: a 28.7M model with correct phase-preserving ops outperforms a 178M model with inconsistent ops
- First systematic benchmark of **initialization strategies for complex-valued language models** (13 strategies, 3 seeds each)
- Shows that orthogonal initialization provides a **persistent 31% quality advantage** in complex-valued networks, not just faster early convergence
- ComplexGatedUnit eliminates the need for separate FFN layers in complex-valued architectures
- Blelloch parallel scan enables practical training of complex-valued SSMs

### **v4 - Phase-Field & Linear Complexity Research**

- Phase2D representation, O(n) linear backbone, multi-bank interference
- Philosophy-aligned interpretability metrics

### **v2 - Quantum Computing Research**

- First practical quantum-inspired language model
- Energy-based optimization techniques

### **v3 - Neuroscience Research**

- Consciousness implementation in LLM
- Biologically plausible learning mechanisms

## Development Status

### **v5 - Algebraic LM** (Active)

- ✅ Phase-preserving complex ops (modReLU, CGU, ComplexNorm)
- ✅ ComplexSSM with Blelloch parallel scan
- ✅ Multi-bank interference with learned routing
- ✅ PhaseAttention (sparse, sliding window)
- ✅ Weight tying (`Re(z * conj(embed))`)
- ✅ 13 structured initialization strategies (orthogonal default)
- ✅ Training on TinyStories (full dataset, 474M tokens)
- ✅ Checkpoint save/resume, generation script
- ✅ LaTeX paper draft
- 🔄 Full-dataset training run (epoch 3/10, val PPL 5.59)
- 🔄 Apples-to-apples transformer baseline comparison
- 🔄 Downstream benchmarks
- 🔄 Custom CUDA/Triton kernels for complex ops
- 🔄 Scale-up experiments (medium, large configs)

### **v4 - Quantum Phase-Field LLM** (Superseded)

- ✅ Core architecture complete -- superseded by V5's corrected math

### **v2 - Quantum LLM**

- ✅ Core Architecture, Training, Testing -- Production Ready

### **v3 - Brain-Inspired LLM**

- ✅ Core Architecture, Learning, Training -- Production Ready

## v5 Model Sizes


| Size          | Complex Dim | Layers | Banks | Total Params | Core Params | Use Case                 |
| ------------- | ----------- | ------ | ----- | ------------ | ----------- | ------------------------ |
| tiny          | 64          | 4      | 2     | ~7M          | ~1.5M       | Smoke tests              |
| small-matched | 128         | 12     | 2     | ~28.7M       | ~15.8M      | Fair baseline comparison |
| small         | 256         | 8      | 2     | ~77M         | ~51M        | Standard experiments     |
| medium        | 512         | 12     | 3     | ~260M        | ~210M       | Serious training         |
| large         | 768         | 16     | 3     | ~540M        | ~460M       | Full scale               |


"Complex dim 128" means 128 complex numbers = 256 real values per position. "Core Params" excludes the tied embedding.

## Getting Started

### **1. Clone Repository**

```bash
git clone https://github.com/gowrav-vishwakarma/qllm2.git
cd qllm2
```

### **2. Install Dependencies**

```bash
uv sync                    # CPU / Mac
uv sync --extra cuda       # CUDA machines (installs xformers)
```

### **3. Train v5 (Recommended)**

```bash
# Quick test
python -m v5.train --size tiny --epochs 5 --max_samples 1000

# Real training (GPU)
python -m v5.train --size small-matched --epochs 10 --max_samples 100000 --init_strategy orthogonal
```

### **4. Or try earlier versions**

```bash
cd v4 && uv run python train_real.py --dataset tinystories --size medium
cd v2 && uv run python run_training.py
cd v3 && uv run python train_brain_llm.py
```

## Testing

```bash
# V5 (smoke test)
python -m v5.train --size tiny --epochs 2 --max_samples 500

# V2
cd v2 && uv run python test_quantum_generation.py

# V3
cd v3 && uv run python simple_brain_test.py

# V4
cd v4 && uv run python test_v4.py
```

## Honest Limitations

V5 is early-stage research. We do not want to oversell it.

- No apples-to-apples transformer baseline at the same parameter scale yet
- No long-context or downstream benchmarks yet
- Pure PyTorch, no custom kernels
- Scaling behavior beyond 28.7M is still unknown
- Only tested on TinyStories so far

We are not claiming "complex numbers beat transformers." What we are claiming is narrower: **a mathematically consistent complex-valued LM is substantially better than our earlier inconsistent version, and the training results are strong enough to justify taking the idea seriously.**

## Documentation

- **[v5 README](v5/README.md)**: Algebraic LM -- full architecture, training, init strategies
- **[v5 Experiments](v5/EXPERIMENTS.md)**: Detailed experiment log with all runs and results
- **[v4 README](v4/README.md)**: Quantum Phase-Field LLM documentation
- **[v2 README](v2/README.md)**: Quantum-Inspired LLM documentation
- **[v3 README](v3/README.md)**: Brain-Inspired LLM documentation

## Research Papers

- **QLLM_CORE_IDEA.pdf**: Core idea document
- **v5/paper/**: V5 LaTeX preprint (in progress)
- **QLLM_V2.pdf**: Quantum-Inspired Language Model research paper

## Contributing

1. **Choose your focus**: v5 (active), v4 (prior), v2 (quantum), or v3 (brain-inspired)
2. **Read the respective README**: Understand the architecture
3. **Run tests**: Ensure everything works
4. **Make changes**: Follow the development guidelines
5. **Submit PR**: Include tests and documentation

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0 - see the [LICENSE](LICENSE) file for details.

You may use, study, modify, and share this work for non-commercial purposes, including research, personal projects, and other non-commercial uses allowed by the license.

Commercial use is not granted under the default license. If you want to use this work in a commercial product, service, or business setting, contact [gowravvishwakarma@gmail.com](mailto:gowravvishwakarma@gmail.com) to discuss a separate commercial license.

## Acknowledgments

- **v5**: Complex-valued language modeling, modReLU (Arjovsky et al.), Blelloch parallel scan, orthogonal initialization theory
- **v4**: Phase2D math, oscillatory SSMs, Indian philosophy-inspired metrics
- **v2**: Inspired by quantum computing and quantum mechanics
- **v3**: Inspired by neuroscience and consciousness research
- Built on PyTorch and modern deep learning frameworks

---

**v5 Status**: 🔄 **Active Development** -- Algebraic LM (val PPL 5.59 at epoch 3, training in progress)
**v4 Status**: Superseded by V5
**v2 Status**: ✅ Production Ready -- Quantum-Inspired LLM
**v3 Status**: ✅ Production Ready -- Brain-Inspired LLM
**Last Updated**: 2026-03-06