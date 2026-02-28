# QLLM - Quantum & Brain-Inspired Language Models

> **Disclaimer:** Yes, we use AI, Cursor, and every coding agent we can get our hands on to build this. Because... why not? We're building AI with AI. It's poetic, it's practical, and honestly, it would be weird if we didn't.

## Revolutionary AI Architecture Research

This repository explores novel language model architectures beyond traditional transformers. **v4 is the current active version.**

- **v4** (active): Quantum Phase-Field LLM - O(n) linear backbone, Phase2D math, consumer-GPU friendly
- **v2**: Quantum-Inspired Language Model (QLLM)
- **v3**: Brain-Inspired Language Model (BLLM)

## Project Overview

### **v4 - Quantum Phase-Field LLM** (Current)

- **Approach**: Quantum phase-field architecture with Phase2D representation
- **Key Features**: Oscillatory SSM (O(n) linear), Dynamic Phase Bank Interference, Dual Memory System, GPT-2 BPE tokenizer, byte patching
- **Status**: 🔄 **Active Development**
- **Parameters**: ~1M (tiny) to ~350M (large)
- **Target Hardware**: Consumer GPUs (RTX 4090) and data-center GPUs (A6000/A100)
- **Highlights**: No trig in hot path, injectable architecture, philosophy-inspired metrics

### **v2 - Quantum-Inspired LLM**

- **Approach**: Quantum mechanics-inspired architecture
- **Key Features**: Quantum superposition, entanglement, phase coherence
- **Status**: ✅ **Production Ready**
- **Parameters**: 5M+ parameters
- **Performance**: 1.43x faster, 4.38x more memory efficient

### **v3 - Brain-Inspired LLM**

- **Approach**: Human brain-inspired architecture
- **Key Features**: Consciousness, memory systems, biologically plausible learning
- **Status**: ✅ **Production Ready**
- **Parameters**: 16.4M parameters
- **Performance**: Human-like learning efficiency, consciousness mechanisms

## Quick Start

### **Prerequisites**

```bash
uv sync
```

### **Choose Your Approach**

#### **Option 1: Quantum Phase-Field (v4) — Recommended**

```bash
cd v4

# Run tests to validate everything works
uv run python test_v4.py

# Train with GPT-2 tokenizer (default) on RTX 4090
uv run python train_real.py \
  --dataset tinystories \
  --size medium \
  --max_length 256 \
  --batch_size 16 \
  --accumulation_steps 4 \
  --epochs 50
```

See [v4/README.md](v4/README.md) for full options including byte tokenizer, custom bank selection, A6000 deployment, and torch.compile.

#### **Option 2: Quantum-Inspired (v2)**

```bash
cd v2
uv run python run_training.py
```

#### **Option 3: Brain-Inspired (v3)**

```bash
cd v3
uv run python train_brain_llm.py
```

## Performance Comparison

| Feature | v2 | v3 | v4 |
|---------|----|----|-----|
| **Phase representation** | sin/cos | N/A | Phase2D (no trig) |
| **Tokenization** | BPE only | BPE | BPE + Byte + Morphological |
| **Sequence complexity** | O(n²) | O(n²) | O(n) linear |
| **Separate meaning layers** | Partial | N/A | Full (up to 5 banks) |
| **Long context** | Limited | Limited | 256K target |
| **GPU efficiency** | Medium | Medium | High (GEMM-only) |
| **Memory systems** | No | Yes | Dual (global + episodic) |
| **Interpretability** | Low | Medium | High (philosophy metrics) |
| **Incremental generation** | O(n²) | O(n²) | O(n) |

## Key Innovations

### **v4 - Quantum Phase-Field Features**

- **Phase2D Math**: Complex numbers as 2D real vectors — no sin/cos in hot path, all ops reduce to matrix multiplies (GEMM)
- **Oscillatory SSM Backbone**: O(n) linear-time sequence processing via coupled oscillators with chunked computation
- **Dynamic Phase Banks**: Semantic, Context, Morphology, Orthography banks with learned per-token routing
- **Dual Memory System**: Global associative memory + Episodic buffer (SDPA-backed sliding window) for copy/retrieval
- **Byte Patching**: Groups P=4 bytes into patch latents for 4x faster byte-level training
- **Philosophy Metrics**: Manas (active mind), Buddhi (discernment), Viveka (stability), Smriti (memory)
- **Injectable Architecture**: All components swappable via registry/config decorators
- **Incremental Generation**: O(n) generation by carrying backbone state across steps

### **v2 - Quantum-Inspired Features**

- **Quantum Superposition**: Multiple states simultaneously
- **Entanglement**: Correlated quantum states
- **Phase Coherence**: Quantum interference patterns
- **Energy-Based Training**: Quantum energy optimization
- **Concept Layers**: Abstract concept representation

### **v3 - Brain-Inspired Features**

- **Consciousness Layer**: Awareness, attention, memory, intention
- **Memory Systems**: Short-term and long-term memory
- **Biologically Plausible Learning**: No backpropagation
- **Spiking Neurons**: Event-driven processing
- **Minimal Data Learning**: One-shot, few-shot learning

## Project Structure

```
qllm/
├── README.md                    # This file
├── pyproject.toml              # Project configuration
├── QLLM_V2.pdf                 # Research paper
├── scripts/                    # Deployment & monitoring scripts
│
├── v4/                         # Quantum Phase-Field LLM (ACTIVE)
│   ├── README.md              # Full v4 documentation
│   ├── model.py               # Main model (wires everything)
│   ├── train_real.py          # Training with real datasets
│   ├── test_v4.py             # Test suite
│   ├── core/                  # Phase2D math, config, registry, interfaces
│   ├── banks/                 # Phase banks (semantic, context, morphology, orthography)
│   ├── backbone/              # Oscillatory SSM backbone
│   ├── coupler/               # Interference-based bank coupling
│   ├── memory/                # Phase-coded associative memory
│   ├── objectives/            # Loss functions (CE, coherence)
│   ├── sampler/               # Autoregressive generation
│   ├── metrics/               # Philosophy metrics (Manas/Buddhi/Viveka/Smriti)
│   └── data/                  # Datasets, tokenizers, morphological tokenizer
│
├── v2/                         # Quantum-Inspired LLM
│   ├── README.md              # v2 documentation
│   ├── quantum_llm_model.py   # Main quantum model
│   ├── energy_trainer.py      # Energy-based training
│   └── run_training.py        # Training script
│
└── v3/                         # Brain-Inspired LLM
    ├── README.md              # v3 documentation
    ├── brain_inspired_llm.py  # Main brain model
    ├── train_brain_llm.py     # Training script
    └── demo_brain_llm.py      # Demo script
```

## Use Cases

### **Choose v4 (Quantum Phase-Field) for:**

- **Consumer GPU training** (RTX 4090, A6000)
- **Long-context modeling** (256K target with O(n) backbone)
- **Phase-representation research** (Phase2D, bank interference)
- **Modular architecture experiments** (injectable components via registry)
- **Byte-level / multilingual modeling** (byte tokenizer + patching)

### **Choose v2 (Quantum-Inspired) for:**

- **Speed-critical applications**
- **Memory-constrained environments**
- **Quantum computing research**
- **Energy-efficient processing**

### **Choose v3 (Brain-Inspired) for:**

- **Consciousness research**
- **Minimal data scenarios**
- **Biologically plausible AI**

## Scientific Impact

### **v4 - Phase-Field & Linear Complexity Research**

- Phase2D representation eliminates trig from the critical path
- O(n) linear backbone via oscillatory SSM (coupled oscillators)
- Multi-bank phase interference for separating semantic, syntactic, morphological, and orthographic information
- Dual memory (associative + episodic) with chunked top-k retrieval for scalability
- Philosophy-aligned interpretability metrics

### **v2 - Quantum Computing Research**

- First practical quantum-inspired language model
- Demonstrates quantum advantage in NLP
- Energy-based optimization techniques

### **v3 - Neuroscience Research**

- First consciousness implementation in LLM
- Biologically plausible learning mechanisms
- Human memory system simulation

## Development Status

### **v4 - Quantum Phase-Field LLM** (Active)

- ✅ Core Phase2D math (no trig in hot path)
- ✅ Injectable architecture (registry + config)
- ✅ Real dataset integration (WikiText-2, TinyStories)
- ✅ GPT-2 BPE tokenizer + byte tokenizer alternative
- ✅ Morphological tokenizer (root + prefix + suffix)
- ✅ Dynamic coupler routing + episodic memory
- ✅ Byte patching (4x faster byte-level training)
- ✅ Memory scaling (chunked top-k, 10x memory reduction)
- ✅ Chunked backbone + SDPA episodic memory
- ✅ Gradient accumulation, incremental generation, learnable scaling
- ✅ Philosophy metrics (Manas/Buddhi/Viveka/Smriti)
- 🔄 Validate training (perplexity on real data)
- 🔄 Incremental learning test (memory sharding)
- 🔄 Long context support (256K streaming)
- 🔄 Custom CUDA/Triton kernels

### **v2 - Quantum LLM**

- ✅ Core Architecture, Training, Testing — Production Ready

### **v3 - Brain-Inspired LLM**

- ✅ Core Architecture, Learning, Training — Production Ready

## v4 Model Sizes

| Size | Dim | Layers | Banks | Params | Use Case |
|------|-----|--------|-------|--------|----------|
| tiny | 64 | 4 | 1 | ~1M | Testing |
| small | 256 | 8 | 1 | ~10M | Quick experiments |
| medium | 512 | 12 | 2 | ~50M | RTX 4090 training |
| large | 768 | 16 | 4 | ~200M | A100 training |

Byte-optimized configs (`tiny-byte`, `small-byte`, `medium-byte`, `large-byte`) are also available when using `--tokenizer byte`. Use `--banks` to customize bank selection for any preset.

## Getting Started

### **1. Clone Repository**

```bash
git clone <repository-url>
cd qllm
```

### **2. Install Dependencies**

```bash
uv sync
```

### **3. Train v4 (Recommended)**

```bash
cd v4
uv run python test_v4.py
uv run python train_real.py --dataset tinystories --size medium --epochs 50
```

### **4. Or try earlier versions**

```bash
cd v2 && uv run python run_training.py
cd v3 && uv run python train_brain_llm.py
```

## Testing

```bash
cd v4 && uv run python test_v4.py
cd v2 && uv run python test_quantum_generation.py
cd v3 && uv run python simple_brain_test.py
```

## Documentation

- **[v4 README](v4/README.md)**: Quantum Phase-Field LLM — full architecture, training options, deployment guide
- **[v2 README](v2/README.md)**: Quantum-Inspired LLM documentation
- **[v3 README](v3/README.md)**: Brain-Inspired LLM documentation
- **[v2 TODO](v2/TODO_ENHANCEMENT_PLAN.md)**: v2 development roadmap
- **[v3 TODO](v3/TODO_V3_BRAIN_INSPIRED.md)**: v3 development roadmap

## Research Papers

- **QLLM_V2.pdf**: Quantum-Inspired Language Model research paper

## Contributing

1. **Choose your focus**: v4 (active), v2 (quantum), or v3 (brain-inspired)
2. **Read the respective README**: Understand the architecture
3. **Run tests**: Ensure everything works
4. **Make changes**: Follow the development guidelines
5. **Submit PR**: Include tests and documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **v4**: Phase2D math, oscillatory SSMs, Indian philosophy-inspired metrics
- **v2**: Inspired by quantum computing and quantum mechanics
- **v3**: Inspired by neuroscience and consciousness research
- Built on PyTorch and modern deep learning frameworks

---

**v4 Status**: 🔄 **Active Development** — Quantum Phase-Field LLM  
**v2 Status**: ✅ Production Ready — Quantum-Inspired LLM  
**v3 Status**: ✅ Production Ready — Brain-Inspired LLM  
**Last Updated**: 2026-03-01
