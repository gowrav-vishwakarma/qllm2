# v4 Quantum Phase-Field LLM

A novel language model architecture combining quantum-inspired phase representations with GPU-practical implementations.

## Key Innovation

Unlike traditional transformers or even v2/v3, v4 uses:

- **Phase2D Representation**: Complex numbers as 2D real vectors (no sin/cos in hot path)
- **Morphological Tokenization**: Words split into Root + Affix, where Affix applies a phase rotation (tense/aspect) to the Root (meaning).
- **Multi-Layer Phase Banks**: Separate semantic/context/language/emotion layers that interfere
- **Oscillatory SSM Backbone**: Linear-time sequence processing via coupled oscillators
- **Phase-Coded Memory**: Long-term associative memory with coherence-based retrieval
- **Injectable Architecture**: All components swappable via registry/config

## Quick Start

```bash
cd v4

# Run tests to validate everything works
uv run python test_v4.py

# Train on random data (for testing architecture)
uv run python train.py --size tiny --epochs 2

# Train on REAL data (WikiText-2)
uv run python train_real.py --dataset wikitext2 --size small --epochs 5

# Train on TinyStories (good for small models)
uv run python train_real.py --dataset tinystories --size small --epochs 10 --max_train_samples 5000
```

## Architecture Overview

```
Tokens → Phase2D Embed → Phase Banks → Backbone → Memory → Coupler → LM Head
                           ↓
            [Semantic, Context, Language, Emotion]
                           ↓
                   Interference Coupling
```

### Components

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **Phase2D** | Complex numbers as [real, imag] pairs | `core/phase2d.py` |
| **PhaseBank** | Separate meaning layers | `banks/` |
| **Backbone** | Oscillatory SSM | `backbone/oscillatory_ssm.py` |
| **Coupler** | Interference-based mixing | `coupler/interference.py` |
| **Memory** | Phase-coded associative memory | `memory/phase_associative.py` |
| **Objectives** | CE + coherence + energy losses | `objectives/` |
| **Sampler** | Autoregressive sampling | `sampler/autoregressive.py` |

## Phase2D: The Core Math

Instead of using sin/cos for phase operations (slow on GPU), we represent complex numbers as 2D vectors:

```python
# Complex number z = a + bi represented as
z = torch.tensor([a, b])  # shape: [..., 2]

# Multiplication by i (90° rotation)
i * z = torch.tensor([-b, a])  # Just swap and negate!

# Complex multiplication (a + bi) * (c + di)
result_real = a*c - b*d
result_imag = a*d + b*c
```

All operations reduce to matrix multiplies (GEMM) - perfect for Tensor Cores.

## Injectable Architecture

Every component can be swapped via config:

```python
from v4.core.config import V4Config, BankConfig

config = V4Config(
    dim=256,
    banks={
        'semantic': BankConfig(type='semantic', dim=256),
        'context': BankConfig(type='context', dim=256),
        'my_custom': BankConfig(type='my_custom_bank', dim=256),  # Your bank!
    },
)
```

Register new components with decorators:

```python
from v4.core.registry import register_bank

@register_bank('my_custom_bank', description='My custom phase bank')
class MyCustomBank(PhaseBank):
    ...
```

## Model Sizes

| Size | Dim | Layers | Params | Use Case |
|------|-----|--------|--------|----------|
| tiny | 64 | 4 | ~1M | Testing |
| small | 256 | 8 | ~10M | Quick experiments |
| medium | 512 | 12 | ~50M | Balanced |
| large | 768 | 16 | ~200M | Production |

## Training

### With Real Data (Recommended)

```bash
# WikiText-2 (quick validation)
uv run python train_real.py --dataset wikitext2 --size small --epochs 10

# TinyStories (better for small models)
uv run python train_real.py --dataset tinystories --size small --epochs 20

# Medium model
uv run python train_real.py --dataset tinystories --size medium --epochs 20 --batch_size 4

# Resume training
uv run python train_real.py --dataset tinystories --size small --resume checkpoints_v4_real/best_model.pt
```

### With Random Data (Architecture Testing)

```bash
uv run python train.py --size tiny --epochs 5
```

## File Structure

```
v4/
├── core/                    # Core abstractions
│   ├── phase2d.py          # Phase2D math (the foundation)
│   ├── interfaces.py       # Base classes (PhaseBank, Backbone, etc.)
│   ├── registry.py         # Factory pattern for components
│   └── config.py           # Configuration system
├── banks/                   # Phase bank implementations
│   ├── semantic.py         # Semantic meaning layer
│   ├── context.py          # Context/syntax layer
│   └── language.py         # Language-specific + emotion layers
├── backbone/               # Sequence backbone
│   └── oscillatory_ssm.py  # Oscillatory state-space model
├── coupler/                # Bank coupling
│   └── interference.py     # Interference-based coupling
├── memory/                 # Long-term memory
│   └── phase_associative.py # Phase-coded associative memory
├── objectives/             # Loss functions
│   ├── ce.py              # Cross-entropy
│   └── coherence.py       # Coherence + energy losses
├── sampler/               # Generation strategies
│   └── autoregressive.py  # AR sampling
├── data/                   # Dataset integration
│   ├── datasets.py        # WikiText-2, TinyStories, etc.
│   └── tokenizer.py       # GPT-2 tokenizer wrapper
├── model.py               # Main model (wires everything)
├── train.py               # Training (random data, for testing)
├── train_real.py          # Training with real datasets
└── test_v4.py             # Test suite
```

## Comparison with v2/v3

| Feature | v2 | v3 | v4 |
|---------|----|----|-----|
| Phase representation | sin/cos | N/A | Phase2D (no trig) |
| Separate meaning layers | Partial | N/A | Full (banks) |
| Sequence complexity | O(n²) | O(n²) | O(n) linear |
| Long context | Limited | Limited | 256K target |
| Incremental learning | No | Partial | Full (shards) |
| GPU efficiency | Medium | Medium | High (GEMM-only) |

## Next Steps

1.  **Morphological Tokenizer**: Implement custom tokenizer that splits words into `(root, affix)` pairs.
    *   *Idea*: "walking" → `root="walk"`, `affix="ing"`.
    *   *Mechanism*: Root sets the base phase vector, Affix applies a rotation (IotaBlock) to modify tense/aspect.
2.  **Dataset Integration**: Connect to v3's dataset system
3.  **256K Context**: Implement chunked processing + state management
4.  **Custom Kernels**: Triton kernels for Phase2D ops
5.  **Benchmarking**: Compare with v2/v3 on perplexity/speed

## Status

**v4 is in active development.**

- ✅ Core Phase2D math (no trig in hot path)
- ✅ All interfaces defined (PhaseBank, Coupler, Backbone, Memory, Objectives, Sampler)
- ✅ First implementations of each component
- ✅ Injectable architecture (registry + config)
- ✅ Model wiring
- ✅ Basic training loop
- ✅ Test suite (all tests pass)
- ✅ Real dataset integration (WikiText-2, TinyStories)
- ✅ GPT-2 tokenizer integration
- 🔄 Validate training (run on real data, check perplexity drops)
- 🔄 Incremental learning test (memory sharding)
- 🔄 Long context support (256K streaming)
- 🔄 Custom CUDA/Triton kernels
- 🔄 Landmark measurement module
