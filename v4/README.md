# v4 Quantum Phase-Field LLM

A novel language model architecture combining quantum-inspired phase representations with GPU-practical implementations.

## Key Innovation

Unlike traditional transformers or even v2/v3, v4 uses:

- **Phase2D Representation**: Complex numbers as 2D real vectors (no sin/cos in hot path)
- **Morphological Tokenization**: Words split into Root + Prefix + Suffix, where affixes apply phase rotations (tense/aspect/case) to the Root (meaning)
- **Multi-Layer Phase Banks**: Semantic/Context/Language/Morphology/Orthography layers that interfere
- **Oscillatory SSM Backbone**: Linear-time sequence processing via coupled oscillators (with vectorized scan option)
- **Phase-Coded Memory**: Long-term associative memory with coherence-based retrieval
- **Injectable Architecture**: All components swappable via registry/config
- **Philosophy Metrics**: Manas/Buddhi/Viveka/Smriti inspired metrics for interpretability

## Quick Start

```bash
cd v4

# Run tests to validate everything works
uv run python test_v4.py

# Train on random data (for testing architecture)
uv run python train.py --size tiny --epochs 2

# Train on REAL data (WikiText-2) with speed optimizations
uv run python train_real.py --dataset wikitext2 --size small --epochs 5 --compile

# Train on TinyStories (good for small models)
uv run python train_real.py --dataset tinystories --size small --epochs 10 --max_train_samples 5000
```

## New Features (v4.1)

### 1. Morphological Tokenizer (User Idea)

Data-driven tokenizer that learns root and affix vocabularies from corpus statistics:

```python
from v4.data import MorphologicalTokenizer, get_tokenizer

# Train morphological tokenizer
tokenizer = get_tokenizer(
    tokenizer_type='morphological',
    morph_train_texts=train_texts,
    morph_path='tokenizer_cache/'
)

# Encode returns (root_ids, prefix_ids, suffix_ids)
result = tokenizer.encode("walking quickly")
print(result['root_ids'])    # [walk, quick]
print(result['prefix_ids'])  # [<null>, <null>]
print(result['suffix_ids'])  # [ing, ly]
```

The morphological embedding applies phase rotations:
- **Root**: Base Phase2D vector (core meaning)
- **Prefix**: Pre-rotation operator (semantic modifier)
- **Suffix**: Post-rotation operator (grammatical role)

```python
# Math: z = RotateSuffix(suffix) ⊙ RotatePrefix(prefix) ⊙ EmbedRoot(root)
```

### Full Text Generation (v4.2)

The model now predicts **all three components** (root + prefix + suffix) for complete morphological text generation:

**Architecture:**
```
backbone_out → lm_head → root_logits (16K vocab)
            → prefix_proj → prefix_logits (2K vocab)
            → suffix_proj → suffix_logits (2K vocab)
```

**Training:** The model learns to predict roots (primary) and affixes (secondary with 0.3 weight):
```python
# Automatic in train_real.py with --tokenizer morphological
loss = ce_loss(root_logits, root_targets) + 0.3 * (
    ce_loss(prefix_logits, prefix_targets) +
    ce_loss(suffix_logits, suffix_targets)
)
```

**Generation:** Returns full morphological tuples:
```python
# Generate returns (root_ids, prefix_ids, suffix_ids) for morphological mode
generated = model.generate(
    root_ids=roots, prefix_ids=prefixes, suffix_ids=suffixes,
    max_new_tokens=50
)
gen_roots, gen_prefixes, gen_suffixes = generated

# Decode with full reconstruction
text = tokenizer.decode(
    gen_roots[0], 
    prefix_ids=gen_prefixes[0],
    suffix_ids=gen_suffixes[0]
)
# "The quickly running dog jumped happily"
```

### Morphological Tokenizer Quality & Speed (v4.3)

The morphological tokenizer has been improved for better quality and speed:

**Quality Improvements:**
- Roots are now primarily **full words** (not short n-grams)
- Affixes are selected by **productivity** (how many stems they attach to)
- Punctuation is tokenized separately with proper spacing on decode
- `min_root_len` increased to 3 to avoid single-char roots

**Speed Improvements:**
- **O(L²) parsing** instead of O(|P|·|S|) - bounded affix-length search
- **Word-level parse cache** - repeated words hit cache (100K default size)
- **Pruned n-gram counting** - only top-K words used for n-gram extraction

**Configuration:**
```python
from v4.data.morphological_tokenizer import MorphologicalTokenizerConfig

config = MorphologicalTokenizerConfig(
    root_vocab_size=16000,      # Total root vocabulary
    prefix_vocab_size=512,      # Smaller for quality
    suffix_vocab_size=512,      # Smaller for quality
    min_root_len=3,             # Avoid single-char roots
    max_affix_len=5,            # Max prefix/suffix length
    min_freq=5,                 # Min frequency to include
    parse_cache_size=100000,    # Word parse cache size (0 to disable)
    top_k_words_for_ngrams=10000,  # Top-K words for n-gram extraction
    min_affix_productivity=3,   # Min stems per affix
    word_priority_ratio=0.7,    # Fill 70% of vocab with full words first
)
```

**Cache Reset (when changing tokenizer settings):**
```bash
# Delete tokenizer cache (forces retraining)
rm -rf .cache/morph_tokenizer

# Delete token caches (forces re-tokenization)
rm -rf .cache/tokens/*_morph.pt
```

**Training Tips:**
- Use more training samples for tokenizer training (e.g., `--max_train_samples 50000`)
- The tokenizer trains once and is cached; model training uses the cached tokenizer

### 2. New Phase Banks

**MorphologyPhaseBank**: Focuses on grammatical transformations
```python
config.banks['morphology'] = BankConfig(type='morphology', dim=256)
```

**OrthographyPhaseBank**: Learns script/shape patterns for multilingual support
```python
config.banks['orthography'] = BankConfig(type='orthography', dim=256)
```

### 3. Philosophy Metrics (Option B)

Inspired by Indian philosophical concepts:

| Metric | Concept | Measures |
|--------|---------|----------|
| **Manas** (मनस्) | Active mind | Backbone state magnitude/entropy |
| **Buddhi** (बुद्धि) | Discernment | Logit confidence/margin |
| **Viveka** (विवेक) | Stability | Phase coherence/energy |
| **Smriti** (स्मृति) | Memory | Attention sharpness/hit rate |

Enable during training:
```python
# In forward pass
output = model(input_ids, context={'compute_metrics': True})
print(output.metrics)  # {'manas/magnitude': 0.5, 'buddhi/confidence': 0.8, ...}
```

### 4. Speed & Memory Optimizations

**Memory-efficient attention**: The memory module now uses einsum-based attention computation, reducing memory usage from O(batch × seq × slots × dim) to O(batch × seq × slots).

**Reduced default memory slots**: Small/Medium configs now use fewer memory slots (512/1024) for consumer GPU compatibility.

```bash
# Enable torch.compile
uv run python train_real.py --compile --compile_mode reduce-overhead

# Use more dataloader workers
uv run python train_real.py --num_workers 8

# Enable token caching (default: on)
uv run python train_real.py --cache_dir .cache/tokens

# Vectorized scan in backbone (default: on)
# Enabled automatically; pre-computes projections for faster recurrence
```

## A/B Testing: BPE vs Morphological

Switch between tokenization modes:

```python
from v4 import create_model, V4Config
from v4.core.config import TokenizerConfig

# BPE mode (default)
config = V4Config(
    tokenizer=TokenizerConfig(mode='bpe', bpe_name='gpt2')
)

# Morphological mode
config = V4Config(
    tokenizer=TokenizerConfig(
        mode='morphological',
        root_vocab_size=16000,
        prefix_vocab_size=2000,
        suffix_vocab_size=2000,
    )
)

model = create_model(config=config)

# Switch mode at runtime
model.set_embedding_mode('bpe')  # or 'morphological'
```

### Byte-Level Tokenizer (Multilingual, Tokenizer-Free)

The **byte tokenizer** (`--tokenizer byte`) uses raw UTF-8 bytes as tokens:
- **Vocab size**: 259 (256 bytes + 3 specials: pad/bos/eos)
- **Multilingual**: Works with any language/script without training
- **Tokenizer-free**: No learned segmentation; the model learns structure end-to-end

```bash
# UTF-8 byte-level training (multilingual)
uv run python train_real.py \
  --dataset tinystories \
  --size small \
  --epochs 5 \
  --tokenizer byte \
  --max_length 512 \
  --cache_dir .cache/tokens_byte
```

**Trade-offs:**
- Sequences are ~4x longer than word-level tokenizers (slower per epoch)
- Model learns character/word boundaries implicitly
- Best for validating the core v4 architecture without tokenizer artifacts

**Recommended settings for byte mode:**
- Use smaller `--max_length` (256-512) to fit in memory
- Compare by **steps** or **tokens processed**, not epochs (byte epochs have more batches)
- The model will learn spelling patterns first, then words, then semantics

### Simple char-level tokenizer (ASCII baseline)

For quick architecture tests with ASCII-only text:

```bash
uv run python train_real.py --dataset tinystories --size tiny --epochs 1 --tokenizer simple
```

## Architecture Overview

```mermaid
flowchart LR
  text[Text] --> tokA[Tokenizer_GPT2_BPE]
  text --> tokB[MorphologicalTokenizer]

  tokA --> idsA[input_ids]
  tokB --> roots[root_ids]
  tokB --> prefixes[prefix_ids]
  tokB --> suffixes[suffix_ids]

  idsA --> embedA[Phase2DEmbed]
  roots --> embedB[MorphologyAwareEmbed]
  prefixes --> embedB
  suffixes --> embedB

  embedA --> banks[PhaseBanks]
  embedB --> banks

  banks --> backbone[OscillatorySSM]
  backbone --> mem[PhaseMemory]
  mem --> coupler[InterferenceCoupler]
  coupler --> head[LM_Head]

  head --> rootOut[root_logits]
  head --> prefixOut[prefix_logits]
  head --> suffixOut[suffix_logits]

  rootOut --> metrics[Manas_Buddhi_Viveka_Smriti]
```

### Components

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **Phase2D** | Complex numbers as [real, imag] pairs | `core/phase2d.py` |
| **DualEmbedding** | BPE or Morphological embedding | `core/morphology_embed.py` |
| **PhaseBanks** | Semantic/Context/Language/Morphology/Orthography | `banks/` |
| **Backbone** | Oscillatory SSM (with scan option) | `backbone/oscillatory_ssm.py` |
| **Coupler** | Interference-based mixing | `coupler/interference.py` |
| **Memory** | Phase-coded associative memory | `memory/phase_associative.py` |
| **Metrics** | Philosophy-aligned metrics | `metrics/` |

## Phase2D: The Core Math

Instead of using sin/cos for phase operations (slow on GPU), we represent complex numbers as 2D vectors:

```python
# Complex number z = a + bi represented as
z = torch.tensor([a, b])  # shape: [..., 2]

# Multiplication by i (90° rotation)
i * z = torch.tensor([-b, a])  # Just swap and negate!

# Rotation via Cayley transform (no trig!)
cos_like = (1 - a²) / (1 + a²)
sin_like = (2a) / (1 + a²)
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
        'morphology': BankConfig(type='morphology', dim=256),
        'orthography': BankConfig(type='orthography', dim=256),
    },
)
```

Register new components with decorators:

```python
from v4.core.registry import register_bank

@register_bank('my_custom_bank', description='My custom phase bank')
class MyCustomBank(nn.Module, PhaseBank):
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

### With Speed Optimizations

```bash
# Full speed training
uv run python train_real.py \
    --dataset tinystories \
    --size small \
    --epochs 20 \
    --compile \
    --num_workers 8 \
    --cache_dir .cache/tokens

# Resume training
uv run python train_real.py \
    --dataset tinystories \
    --size small \
    --resume checkpoints_v4_real/best_model.pt
```

### Speed Options

| Option | Default | Description |
|--------|---------|-------------|
| `--compile` | False | Enable torch.compile |
| `--compile_mode` | reduce-overhead | Compile mode |
| `--num_workers` | 4 | DataLoader workers |
| `--no_pin_memory` | False | Disable pinned memory |
| `--no_cache` | False | Disable token caching |
| `--cache_dir` | .cache/v4_tokens | Token cache location |

## File Structure

```
v4/
├── core/                    # Core abstractions
│   ├── phase2d.py          # Phase2D math (the foundation)
│   ├── morphology_embed.py # Morphology-aware embedding
│   ├── interfaces.py       # Base classes (PhaseBank, Backbone, etc.)
│   ├── registry.py         # Factory pattern for components
│   └── config.py           # Configuration system
├── banks/                   # Phase bank implementations
│   ├── semantic.py         # Semantic meaning layer
│   ├── context.py          # Context/syntax layer
│   ├── language.py         # Language-specific layers
│   ├── morphology.py       # Morphology phase bank (NEW)
│   └── orthography.py      # Orthography phase bank (NEW)
├── backbone/               # Sequence backbone
│   └── oscillatory_ssm.py  # Oscillatory SSM (with scan option)
├── coupler/                # Bank coupling
│   └── interference.py     # Interference-based coupling
├── memory/                 # Long-term memory
│   └── phase_associative.py # Phase-coded associative memory
├── objectives/             # Loss functions
│   ├── ce.py              # Cross-entropy
│   └── coherence.py       # Coherence + energy losses
├── sampler/               # Generation strategies
│   └── autoregressive.py  # AR sampling
├── metrics/                # Philosophy metrics (NEW)
│   └── philosophy_metrics.py # Manas/Buddhi/Viveka/Smriti
├── data/                   # Dataset integration
│   ├── datasets.py        # WikiText-2, TinyStories, etc. (with caching)
│   ├── tokenizer.py       # Unified tokenizer interface
│   └── morphological_tokenizer.py # Morphological tokenizer (NEW)
├── model.py               # Main model (wires everything)
├── train.py               # Training (random data, for testing)
├── train_real.py          # Training with real datasets (speed optimized)
└── test_v4.py             # Test suite
```

## Comparison with v2/v3

| Feature | v2 | v3 | v4 |
|---------|----|----|-----|
| Phase representation | sin/cos | N/A | Phase2D (no trig) |
| Tokenization | BPE only | BPE | BPE + Morphological |
| Separate meaning layers | Partial | N/A | Full (5 banks) |
| Sequence complexity | O(n²) | O(n²) | O(n) linear |
| Long context | Limited | Limited | 256K target |
| Incremental learning | No | Partial | Full (shards) |
| GPU efficiency | Medium | Medium | High (GEMM-only) |
| Interpretability | Low | Medium | High (philosophy metrics) |

## Status

**v4 is in active development.**

- ✅ Core Phase2D math (no trig in hot path)
- ✅ All interfaces defined (PhaseBank, Coupler, Backbone, Memory, Objectives, Sampler)
- ✅ Injectable architecture (registry + config)
- ✅ Real dataset integration (WikiText-2, TinyStories)
- ✅ GPT-2 tokenizer integration
- ✅ **Morphological Tokenizer** (root + prefix + suffix)
- ✅ **Morphology-aware embedding** (affix rotations)
- ✅ **Full text generation** (predicts root + prefix + suffix)
- ✅ **MorphologyPhaseBank + OrthographyPhaseBank**
- ✅ **Philosophy metrics** (Manas/Buddhi/Viveka/Smriti)
- ✅ **Speed optimizations** (torch.compile, workers, caching)
- ✅ **Vectorized scan** option for backbone
- 🔄 Validate training (run on real data, check perplexity drops)
- 🔄 Incremental learning test (memory sharding)
- 🔄 Long context support (256K streaming)
- 🔄 Custom CUDA/Triton kernels
