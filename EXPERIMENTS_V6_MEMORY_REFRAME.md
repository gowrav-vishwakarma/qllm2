# V6 Memory Reframe Experiment Log

Structured log of the memory architecture redesign: why the original token-wise memory failed, what we changed, and how the new episodic/objective/bank-role system is designed.

**Date started**: 2026-03-11
**Predecessor**: [EXPERIMENTS_V6.md](EXPERIMENTS_V6.md) (sections 5-13 document the original memory failures)

---

## 1. Why We Did This

### The problem with token-wise memory

V6's original `WorkingMemory` wrote to slots at every token position. Experiments showed this was the architecture's weakest point:

| Evidence | Source | What it showed |
|----------|--------|----------------|
| WM=64/IM=128 memorization | EXPERIMENTS_V6.md §5.3 | Large writable memory collapsed PPL to ~1.2 via memorization |
| Restart fragmentation | EXPERIMENTS_V6.md §5.5 | WM=16/IM=32 produced mid-sentence "Once upon a time" restarts |
| WikiText template locking | EXPERIMENTS_V6.md §5.8 | WM=8 on WikiText-103 caused word loops ("time", "the history of") |
| IM near-zero contribution | EXPERIMENTS_V6.md §5.4 | IM-only was 11.77 vs no-memory 11.83 — essentially no gain |
| No-memory cleaner text | EXPERIMENTS_V6.md §5.6, §5.7 | No-memory produced cleaner, more diverse text despite higher PPL |
| Diversity loss collapse | EXPERIMENTS_V6.md §3 | Bank diversity loss collapsed to zero repeatedly |

### The root cause: wrong supervision

Next-token cross-entropy does not teach the model:
- **What** is worth storing (facts vs function words vs boilerplate)
- **When** a relation is complete enough to commit
- **How** to retrieve usefully without blending irrelevant slots

The model was asked to learn selective memory implicitly from a dense local signal. It responded by memorizing dataset boundary patterns instead.

### The insight: phase as relation signal

V6 already uses phase-coherence similarity for memory retrieval:

```
score = Re(query * conj(key)) / (|query| * |key|)
```

This measures relational alignment in complex space. The SSM's multi-timescale decay lanes act as a natural temporary buffer — slow lanes retain entities for thousands of tokens while fast lanes track local syntax. The architecture has the raw ingredients for relation-sensitive storage; it just needed the right training signal and write policy.

### Inspiration

The Berkeley FlyWire connectome simulation showed that connectivity + circuit dynamics can predict behavior even with simplified neurons. V6's phase core is analogous: the structure (banks + SSM + phase interference) carries information, not just learned weights. The redesign treats V6 less like a transformer alternative and more like a dynamical system that forms stable internal states before committing to memory.

---

## 2. What We Changed

### 2.1 Control Baseline (Phase 1-2)

**Goal**: Establish a clean reference point before extending memory.

| Change | File | What |
|--------|------|------|
| Sparse retrieval everywhere | `v6/core/memory.py` | `PersistentMemoryReader` and `ExpertMemoryReader` now use top-k sparse retrieval (default k=8), matching WM/IM. Previously used dense softmax over all slots, causing the same "blended generic retrieval" diagnosed in Bug 6 |
| Behavioral quality metrics | `v6/train.py` | Added `compute_text_quality()`: measures 3-gram/4-gram repeat rate, restart fragmentation count, unique word ratio. Printed alongside generation samples |
| No-memory as default | `v6/config.py` | `num_wm_slots=0`, `num_im_slots=0` remain the defaults; no-memory is the official control |

### 2.2 Memory-Aligned Training Objectives (Phase 3)

**Goal**: Give the model a supervision signal that actually requires cross-gap reasoning and memory.

**New file**: `v6/objectives.py`

#### Span Corruption (`SpanCorruptionDataset`)

T5-style span masking adapted for V6:
- Corrupts ~15% of tokens in contiguous spans (mean length 3)
- Replaces masked positions with sentinel token IDs
- Returns `loss_mask` (bool): loss computed only on masked positions
- Forces the model to infer missing content from surrounding context
- Directly pressures relational summary formation

```
Original:  [Paris] [is] [the] [capital] [of] [France] [.] [It] [has] ...
Corrupted: [Paris] [is] [SENT] [SENT]   [of] [France] [.] [It] [has] ...
Loss mask: [ 0  ]  [0]  [ 1 ] [  1  ]   [0]  [  0  ]  [0] [0]  [0]  ...
```

#### Delayed Recall (`DelayedRecallDataset`)

Fact-then-cue task:
- Selects a "fact" region early in the sequence (e.g. 8 tokens)
- Places a "recall" region after a configurable gap (default 64 tokens)
- Upweights loss: 3x on recall positions, 2x on fact positions
- Directly trains write and retrieve: the model must store the fact and recover it later

```
Positions:  [0..3] [4..11 FACT] [12..75 gap] [76..83 RECALL] [84..end]
Loss weight: [1x]  [  2x      ] [  1x      ] [    3x        ] [ 1x   ]
```

#### Integration

The trainer now supports `loss_mask` from both datasets. When present, per-token CE is weighted:

```python
per_token_loss = F.cross_entropy(logits, labels, reduction='none')
ce_loss = (per_token_loss * flat_mask).sum() / flat_mask.sum()
```

**CLI**: `--objective span_corruption` or `--objective delayed_recall`

### 2.3 Event-Based Episodic Memory (Phase 4)

**Goal**: Replace eager token-wise writes with selective event-level commits.

**New file**: `v6/core/episodic.py`

#### EventSalienceHead

Scores each position for persistence value:
- **Phase score**: complex projection → magnitude (how "important" the representation is)
- **Novelty**: local contrast against a 5-token moving average (deviations = interesting)
- **Bias**: initialized to -1.0 (start selective, learn to open up)

```python
salience = sigmoid(phase_score + novelty * scale + bias)
```

#### EpisodicMemory

Write flow:
1. **Salience scoring**: EventSalienceHead produces per-position scores
2. **Span detection**: consecutive above-threshold positions grouped into spans
3. **Span pooling**: each span averaged weighted by salience → one event vector
4. **Slot commit**: event key/value projections written to circular buffer slots
5. **No token-wise writes**: only pooled events enter memory

Read flow:
- Phase-coherence retrieval with sparse top-k (same as WM/IM)
- Returns `[B, L, dim, 2]` retrieved representations

**Key difference from old WorkingMemory**:

| | Old WM | Episodic Memory |
|--|--------|----------------|
| Write granularity | Every token | Pooled spans only |
| Write trigger | Gate on every position | Salience threshold + boundary |
| What gets stored | Raw token embedding | Span-averaged event summary |
| Write selectivity | Learned gate (often ~88%) | Phase salience + novelty |
| Slot cycling | Circular per token | Circular per event |

**Config**: `--episodic_slots 16` (0 = disabled). Wired into backbone alongside legacy WM (both can coexist for ablation).

### 2.4 Bank Role Training (Phase 3, point 3)

**Goal**: Push banks toward complementary functional roles, not just "be different."

**Modified file**: `v6/core/bank.py` — added `compute_role_loss()` to `NamedBankPair`

The role loss has two components:

1. **Stability asymmetry**: semantic bank should have lower magnitude variance across the sequence (entity-like = stable), context bank should have higher variance (relation-like = varies by position). Penalizes `relu(sem_var - ctx_var + 0.1)`.

2. **Phase role asymmetry**: semantic bank should have lower phase variance (entities occupy consistent phase regions), context bank should have higher phase variance (relations rotate more). Penalizes `relu(sem_phase_var - ctx_phase_var + 0.05)`.

Both are soft: they don't force symbolic labels, they encourage the banks to develop complementary temporal profiles through training dynamics.

**Config**: `bank_role_weight` (default 0.05), `--bank_role_weight 0.1`. Computed per layer alongside diversity loss.

### 2.5 Two-Pass Model (Phase 5)

**Goal**: Prototype a chunk-level non-causal encoder feeding a causal decoder as a controlled experiment.

**New file**: `v6/two_pass_model.py`

#### ChunkEncoder

- Takes the forward SSM output + bank representations
- Runs a backward SSM layer on the bank output (flipped sequence)
- Merges forward + backward via `ComplexLinear` + `ComplexNorm`
- Pools into a chunk summary weighted by magnitude

#### TwoPassLM

- Pass 1: standard backbone (banks → SSM → optional episodic memory)
- Chunk encoder: produces bidirectional representations + chunk summary
- Pass 2: gated injection of chunk summary into backbone output
- LM head: same tied complex embedding projection

The gate starts near zero (learned `nn.Linear(1,1)` + sigmoid), so the model initially behaves like standard AR and gradually learns to use chunk summaries.

**Config**: `--mode two_pass`. Shares backbone weights (no parameter doubling except the backward SSM and merge projections).

---

## 3. File Changes Summary

| File | Change | Description |
|------|--------|-------------|
| `v6/core/memory.py` | MODIFY | Sparse top-k in PersistentMemoryReader, ExpertMemoryReader |
| `v6/core/episodic.py` | NEW | EventSalienceHead, EpisodicMemory |
| `v6/core/bank.py` | MODIFY | Added `compute_role_loss()` |
| `v6/backbone.py` | MODIFY | Wire episodic memory, bank role loss, expose salience/bank outputs |
| `v6/config.py` | MODIFY | New fields: objective, episodic, bank_role_weight |
| `v6/objectives.py` | NEW | SpanCorruptionDataset, DelayedRecallDataset |
| `v6/model.py` | MODIFY | `create_model()` dispatches `two_pass` mode |
| `v6/two_pass_model.py` | NEW | ChunkEncoder, TwoPassLM |
| `v6/train.py` | MODIFY | Quality metrics, loss_mask support, new CLI flags, dataset wrapping |

**Untouched**: `v6/core/complex.py`, `v6/core/ssm.py`, `v6/core/coupler.py`, `v6/core/diffusion.py`, `v6/core/image_codec.py`, `v6/core/attention.py`, `v6/init.py`, `v6/diffusion_model.py`, `v6/generate.py`

---

## 4. Smoke Test Results (2026-03-11)

All tests run on CPU with `tiny` config (dim=64, 4 layers):

| Test | Result |
|------|--------|
| Autoregressive no-memory forward pass | PASS — logits [2, 32, 50257] |
| Episodic memory (16 slots) forward pass | PASS — logits [2, 32, 50257], 24,770 episodic params |
| Two-pass model forward pass | PASS — logits [2, 32, 50257], chunk summary [2, 1, 64, 2] |
| SpanCorruptionDataset (seq_len=32, rate=0.15) | PASS — 4/32 positions masked |
| DelayedRecallDataset (seq_len=128, gap=16) | PASS — loss weights [1.0, 3.0] |
| Bank role loss computation | PASS — role_loss=0.1469 |
| Quality metrics (repeated text) | PASS — rep3=0.273, uniq=0.615 |
| Backward pass with episodic + bank role (tiny) | PASS — gradients flow through all new modules |

---

## 5. Planned Training Runs

### Run 1: Control baseline (no-memory, next-token, WikiText-103)

```bash
uv run python -m v6.train --size small-matched --dataset wikitext103 \
    --epochs 10 --seq_len 512
```

Purpose: establish PPL + quality baseline on real text without any new features.

### Run 2: Span corruption objective (no-memory)

```bash
uv run python -m v6.train --size small-matched --dataset wikitext103 \
    --objective span_corruption --epochs 10 --seq_len 512
```

Purpose: test whether span corruption improves cross-gap reasoning without adding memory.

### Run 3: Episodic memory + span corruption

```bash
uv run python -m v6.train --size small-matched --dataset wikitext103 \
    --objective span_corruption --episodic_slots 16 --epochs 10 --seq_len 512
```

Purpose: test whether event-based memory outperforms no-memory on span infilling.

### Run 4: Bank role training ablation

```bash
uv run python -m v6.train --size small-matched --dataset wikitext103 \
    --bank_role_weight 0.1 --epochs 10 --seq_len 512
```

Purpose: test whether bank role loss produces meaningful specialization.

### Run 5: Two-pass model

```bash
uv run python -m v6.train --size small-matched --dataset wikitext103 \
    --mode two_pass --epochs 10 --seq_len 512
```

Purpose: test whether bidirectional chunk encoding improves generation quality.

### Success criteria (beyond PPL)

| Metric | How measured | What "better" means |
|--------|-------------|---------------------|
| Val PPL | Standard CE on held-out data | Lower is better, but not sufficient alone |
| repeat_3gram | `compute_text_quality()` | Lower = less repetitive |
| repeat_4gram | `compute_text_quality()` | Lower = less repetitive |
| restart_frag | `compute_text_quality()` | 0 = no mid-sequence restarts |
| unique_word_ratio | `compute_text_quality()` | Higher = more diverse vocabulary |
| Qualitative coherence | Manual inspection of generated text | Topic persistence, entity consistency, narrative arc |

---

## 6. CLI Reference and Experiment Guide

### New flags and their defaults

| Flag | Default | What it controls |
|------|---------|-----------------|
| `--objective` | `next_token` | Training objective: `next_token` (standard AR), `span_corruption` (T5-style infilling), `delayed_recall` (fact-then-cue) |
| `--span_corruption_rate` | `0.15` | Fraction of tokens to mask (only with `--objective span_corruption`) |
| `--span_mean_length` | `3` | Average span length for masking |
| `--delayed_recall_gap` | `64` | Tokens between fact and recall point (only with `--objective delayed_recall`) |
| `--episodic_slots` | not passed (0) | Episodic memory slots; 0 = disabled |
| `--bank_role_weight` | not passed (0.05) | Weight for bank role specialization loss; 0.0 = disabled |
| `--mode two_pass` | `autoregressive` | Enables bidirectional chunk encoder + causal decoder |

### What each option does in plain terms

**`--objective span_corruption`**: Instead of predicting every next token, the model sees a sentence with holes punched in it ("Paris is [MASK] [MASK] of France") and only gets graded on filling those holes. This forces the model to understand the relationship between surrounding words to reconstruct the missing ones. Higher `--span_corruption_rate` = harder task. Longer `--span_mean_length` = bigger gaps to fill.

**`--objective delayed_recall`**: The model reads the full sentence normally, but positions where it should recall something from 64 tokens ago are penalized 3x harder. This directly pressures the memory write/retrieve path: the model needs to store a fact early and recall it later. Larger `--delayed_recall_gap` = must remember across more tokens.

**`--episodic_slots N`**: Adds event-based memory with N slots. Instead of storing every token, a salience head scores positions and only pools high-salience spans into compact event vectors. The model retrieves from these events via phase-coherence. Start with 16; watch for memorization if going above 32.

**`--bank_role_weight W`**: Adds an auxiliary loss encouraging the semantic bank to produce stable outputs (entity-like) and the context bank to produce varying outputs (relation-like). At 0.05 (default) the pressure is gentle. At 0.1 it's moderate. Above 0.2 it may over-constrain.

**`--mode two_pass`**: The model processes each chunk twice -- once forward+backward (to see the whole sentence) and once forward-only (for causal prediction). A gated summary from pass 1 is injected into pass 2. More compute, potentially better coherence.

### Recommended experiment sequence

Run them in this order. Each run isolates one variable against the control.

| Run | Command | Purpose | Compare to |
|-----|---------|---------|-----------|
| 1 (control) | `--dataset wikitext103` | Pure baseline: no memory, next-token, default bank role 0.05 | — |
| 2 | `--dataset wikitext103 --objective span_corruption` | Does span corruption improve quality metrics? | Run 1 |
| 3 | `--dataset wikitext103 --objective span_corruption --episodic_slots 16` | Does episodic memory help span infilling? | Run 2 |
| 4a | `--dataset wikitext103 --bank_role_weight 0.0` | What happens without role pressure? | Run 1 |
| 4b | `--dataset wikitext103 --bank_role_weight 0.1` | What happens with stronger role pressure? | Run 1 |
| 5 | `--dataset wikitext103 --objective span_corruption --episodic_slots 16 --bank_role_weight 0.1` | Full combination | Runs 1-4 |
| 6 | `--dataset wikitext103 --mode two_pass --objective span_corruption` | Does two-pass help? | Run 2 |

### Combinations to avoid

| Combination | Why |
|-------------|-----|
| `--episodic_slots 64+` with `--objective next_token` | Same trap as old WM: lots of writable memory + no memory-aligned supervision = memorization |
| `--mode two_pass` with `--episodic_slots 32+` | Too many new variables at once |
| `--bank_role_weight 0.3+` with `--episodic_slots` | Over-constraining banks while also adding memory |
| `--objective delayed_recall` without any memory | The recall task needs a retrieve mechanism; without memory the model can only rely on SSM slow lanes |

### Tuning guidance

| Parameter | Too low | Sweet spot | Too high |
|-----------|---------|------------|----------|
| `span_corruption_rate` | < 0.05: barely any masking, no pressure | 0.10 - 0.20 | > 0.30: too many holes, model can't learn |
| `span_mean_length` | 1: just single-token masking (not relational) | 2 - 5 | > 8: spans so large the context vanishes |
| `delayed_recall_gap` | < 16: too easy, SSM handles it without memory | 32 - 128 | > 256: may be beyond what the model can learn early |
| `episodic_slots` | 0: no memory (control) | 8 - 16 | > 32: risk memorization |
| `bank_role_weight` | 0.0: diversity-only (collapses easily) | 0.05 - 0.10 | > 0.20: over-constrains bank outputs |

### What "good results" look like

| Signal | What it means |
|--------|---------------|
| PPL drops AND `repeat_3gram` stays low | Model is genuinely learning, not memorizing |
| `restart_frag` stays at 0 | No mid-sentence restarts (the old WM failure mode is gone) |
| `unique_word_ratio` > 0.5 | Diverse vocabulary usage |
| Diversity loss stays > 0.001 through training | Banks are actually specializing, not collapsing |
| Span corruption PPL < next-token PPL on masked positions | The model learns to reason across gaps |
| Episodic run quality > no-memory quality at similar PPL | Event-based memory adds value without memorization |

### What "bad results" look like (and what to do)

| Signal | Diagnosis | Fix |
|--------|-----------|-----|
| PPL drops fast but `repeat_3gram` > 0.3 | Memorization / repetition | Reduce `--episodic_slots`, try `--objective span_corruption` |
| `restart_frag` > 0 | Memory retrieving boundary patterns | Reduce episodic slots or increase salience threshold |
| Diversity loss collapses to 0 quickly | Banks not specializing | Increase `--bank_role_weight` to 0.1-0.15 |
| PPL stays flat with span corruption | Model not learning from masked positions | Reduce `--span_corruption_rate` or `--span_mean_length` |
| Two-pass much worse than single-pass | Backward SSM adding noise | Try longer warmup (`--warmup_steps 500`) or check if gate stays near zero |

---

## 7. Training Run Results

*(To be filled as runs complete)*

---

## 8. Architecture Decision Record

### Decision 1: Fix supervision before scaling memory

**Context**: Memory capacity was the main behavioral control knob, but more memory = worse behavior.

**Decision**: Add memory-aligned objectives (span corruption, delayed recall) before scaling episodic memory.

**Rationale**: The model can't learn what to write if the loss function doesn't reward cross-gap reasoning. Span corruption is the simplest objective that creates this pressure.

### Decision 2: Event-based writes instead of token-wise writes

**Context**: Old WM wrote at every token, cluttering slots with "the", "is", "a".

**Decision**: Pool salient spans into event summaries before writing.

**Rationale**: Relations like "Paris = capital of France" are multi-token constructs. The SSM slow lanes already buffer tokens long enough for the relation to form. The episodic memory waits for salience to cross a threshold, then commits the whole event — not individual words.

### Decision 3: Bank role loss over symbolic labels

**Context**: Banks should specialize, but diversity loss alone collapses too easily.

**Decision**: Push temporal profiles apart (stability vs variability) rather than labeling banks as "entity extractor" vs "relation extractor."

**Rationale**: Symbolic labels are brittle and presuppose linguistic structure. Temporal asymmetry is a softer inductive bias: entities are stable across positions, relations vary. Let the model discover what those roles mean through training.

### Decision 4: Two-pass model as ablation, not default

**Context**: Seeing the whole sentence before memory commit would be ideal.

**Decision**: The two-pass model is a separate `--mode two_pass` ablation, not a replacement.

**Rationale**: The SSM's multi-timescale decay already provides a "see enough of the sentence" buffer. The two-pass model is more powerful but doubles the SSM compute. It should prove its value against the single-pass + episodic baseline before becoming the default.
