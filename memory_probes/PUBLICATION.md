# Memory Probes Paper: Reproducibility and Handoff

This file is the tracked handoff between the Mac development machine and the
RTX 4090 training machine. It is intentionally stored in the repository; the
Cursor plan is not a substitute for this document.

## Paper scope

The primary contribution is an intrinsic benchmark for neural memory
mechanisms. PAM is a worked case study, not the benchmark definition.

The paper separates three levels:

1. **Associative mechanism:** an architecture exposes native `write(key,
   value)` and `read(query)` operations. Binding, persistence, interference,
   and mechanism-level NIAH apply here.
2. **Inspectable state:** an architecture exposes a recurrent state. Effective
   rank, state cost, and utilization apply here.
3. **Trained behavior:** a language model answers passkey, invented-binding,
   multi-needle, and interference prompts. This level tests whether training
   realized the mechanism's capacity.

Unsupported architecture/probe pairs must be reported as unsupported. They
must not be replaced by an artificial associative interface.

## Relationship to Anthropic J-space

Anthropic's July 2026 work, *Verbalizable Representations Form a Global
Workspace in Language Models*, studies a different object. Its averaged
Jacobian lens identifies sparse, token-linked residual-stream directions that
are poised to affect future verbal output. Causal swaps and ablations test
report, modulation, flexible reasoning, selectivity, and broadcast.

This benchmark measures temporal storage properties of explicit memory
mechanisms: capacity, retention, interference, state utilization, cost, and
trained recall. J-space is an interpretability target inside trained models;
Memory Probes is an architecture-evaluation protocol. Limited capacity and
persistence are points of contact, not evidence that PAM state is a J-space.

## Claim policy

- Every manuscript number must trace to a committed result JSON or a clearly
  identified external artifact.
- Single-seed results are diagnostics, not headline evidence.
- Values of `relative_retrieval` may exceed one because the metric is target
  alignment relative to a fresh write; it is not an accuracy or probability.
- Equal-width PAM versus HRR gives PAM more state scalars. Storage-efficiency
  claims must use `binding_matched_bytes`; equal-width results may be shown
  only as a representation-width diagnostic.
- Oracle GSP protection (`p=1`) is a mechanism ceiling, not evidence that a
  trained gate learned to protect useful information.
- Trained and untrained state-utilization numbers must not be placed in the
  same comparison table without an explicit qualification.

## Mac workflow

Run the fast end-to-end validation:

```bash
./scripts/run_memory_probes_publication.sh smoke
```

Run the complete CPU mechanism sweep:

```bash
./scripts/run_memory_probes_publication.sh mac
```

This produces:

- `logs/memory_probes/publication/mac_results.json`
- `memory_probes/paper/figures/capacity.{pdf,png}`
- `memory_probes/paper/figures/persistence.{pdf,png}`
- `memory_probes/paper/figures/rank.{pdf,png}`

The JSON contains raw per-seed records and aggregate mean, standard deviation,
and 95% confidence half-width. Figures are generated only from that JSON.

## RTX 4090 workflow

After committing the Mac-side changes, pull the branch on the GPU machine.
Set the trained checkpoint path and run:

```bash
CHECKPOINT=/path/to/best_model.pt \
TRANSFORMER_CHECKPOINT=/path/to/transformer/best_model.pt \
PRESET=v11_e3_k3_chat \
./scripts/run_memory_probes_publication.sh gpu
```

Optional environment overrides:

```bash
RANK_LAYERS="0 4 8 12 15" \
TEXT_TOKENS=50000 \
GATE_TOKENS=4096 \
PROJECTION_TRIALS=50 \
ARCH_COMPARE_TOKENS=2000 \
CHECKPOINT=/path/to/best_model.pt \
TRANSFORMER_CHECKPOINT=/path/to/transformer/best_model.pt \
./scripts/run_memory_probes_publication.sh gpu
```

The GPU runner persists:

- trained gate and routing diagnostics in `gpu/gates.json`;
- trained rank trajectories for each requested PAM layer;
- the 50-projection language-filler sweep;
- the PAM/KV/Mamba state-utilization diagnostic.
- tokenizer-matched, contrastive next-token behavioral recall for trained V11,
  Transformer (when `TRANSFORMER_CHECKPOINT` is set), and Mamba.

Copy or commit the resulting JSON files before returning to the Mac. Do not
copy model checkpoints into Git.

## Work still required on the RTX 4090

- [ ] Run trained PAM behavioral recall by context length and needle position
      with `scripts/run_memory_behavioral.py`.
- [ ] Run the same contrastive scoring protocol on parameter-comparable trained
      Transformer and Mamba baselines.
- [ ] Repeat behavioral runs across enough seeds/checkpoints to separate
      architecture effects from training variance.
- [ ] Record peak GPU memory, tokens/second, state bytes, parameter count, and
      exact checkpoint identity.
- [ ] Test whether intrinsic capacity/persistence metrics predict behavioral
      recall across models and checkpoints.

The shared runner is implemented, but model training data, parameter count, and
tokenizers still differ unless deliberately matched. Until the GPU artifacts
and those controls exist, the paper must describe Phase B as incomplete and
must not claim behavioral superiority.

## Before an arXiv upload

- [ ] All tests and the Mac sweep pass from a clean checkout using UV.
- [ ] GPU result JSON is present and schema-validated.
- [ ] Every table cell and figure can be regenerated from saved artifacts.
- [ ] The manuscript contains no placeholder results.
- [ ] Citations and author metadata are verified manually.
- [ ] LaTeX builds without shell escape or machine-local paths.
- [ ] The source archive excludes checkpoints, caches, and private data.

## Git handoff

Review changes first. When ready, create a normal commit containing source,
paper files, this handoff, and publication-sized result artifacts. On the GPU
machine:

```bash
git pull
uv sync
./scripts/run_memory_probes_publication.sh smoke
```

Only after the smoke run succeeds should the full `gpu` profile be launched.
