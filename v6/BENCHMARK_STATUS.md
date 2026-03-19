# V6 Benchmark Status and Next-Step Decision Memo

This note is the current benchmark-grounded status of V6 after the first full WikiText-103 run. Its purpose is to answer four questions clearly:

1. Where V6 stands today.
2. Which external comparisons are fair vs unfair.
3. What this run does and does not prove.
4. What benchmark sequence should decide the next experiments.

---

## 1. Current V6 Position

### Headline run

- Model: V6 `small-matched`
- Parameters: `28.7M`
- Dataset: WikiText-103 raw, tokenized with GPT-2 BPE
- Sequence length: `512`
- Batch size: `14`
- Epochs: `20`
- Working memory: `0`
- Internal memory: `0`
- Attention: off
- Hardware: RTX 4090
- Total wall time: `14.27h`

Primary source:
- [WikiText-103 run log](../logs/v6/wikitext103_small_matched_20260310_152631_6ffe838_dirty/v6_autoregressive_small-matched.log)
- [Run metadata](../logs/v6/wikitext103_small_matched_20260310_152631_6ffe838_dirty/RUN_INFO.txt)

Final result:

- Best validation loss: `3.9041`
- Best validation perplexity: `49.61`

This is the first V6 result that clearly shows the architecture training to completion on a real, non-toy encyclopedia corpus.

---

## 2. What The Run Actually Shows

### Positive evidence

- V6 trains stably for 20 full epochs on real long-form text.
- Validation perplexity improves monotonically from `121.94` to `49.61`.
- Throughput is steady at roughly `46k tok/s` after compile warmup.
- Generations clearly shift away from TinyStories-style text and toward Wikipedia/article-style prose.
- The model learns headings, dates, geographic language, historical framing, and list/article cadence.

### Limits of the result

- Final quality is still far from strong published WikiText-103 baselines.
- Generated text remains semantically unstable and factually unreliable.
- The run does not validate the memory system, because `WM=0` and `IM=0`.
- The run does not validate bank specialization strongly.
- The run does not prove long-range factual retention.

### Honest interpretation

This run proves **viability**, not **architectural superiority**.

TinyStories showed V6 can optimize. WikiText-103 shows V6 can survive real text. But this run alone does not show that V6 is a better LM than established alternatives.

---

## 3. Training Curve Summary

Key checkpoints from the run:

| Epoch | Train PPL | Val PPL | Comment |
|---|---:|---:|---|
| 1 | 247.45 | 121.94 | Real learning begins quickly |
| 5 | 69.80 | 61.28 | Clear progress, still weak quality |
| 10 | 60.14 | 53.75 | Gains continue but slow down |
| 15 | 56.26 | 50.59 | Entering shallow plateau |
| 20 | 54.45 | 49.61 | Best checkpoint, small late gains |

Interpretation:

- The first half of training carries most of the useful improvement.
- Epochs `10 -> 20` still help, but only modestly.
- This setup is not diverging, but it is also not on track to close the gap to strong baselines just by adding a few more epochs.

---

## 4. Generation Quality Assessment

### Early training

Early generations are structurally noisy and semantically mixed. They show that the model is absorbing article surface form before content consistency.

### Final training state

By the final epoch, the model has learned a recognizable Wikipedia-like register:

- section headers
- broad historical/article framing
- factual-sounding sentence construction
- date and region language

But it still suffers from:

- invented chronology
- mixed historical entities
- impossible event combinations
- confident but wrong factual composition

### Practical verdict

Generation quality is **better than toy-quality**, but still **below benchmark-quality**.

It can imitate the *form* of encyclopedia text. It cannot yet be trusted on the *content* of encyclopedia text.

---

## 5. Probe Results On The Final WikiText-103 Checkpoint

Internal probe run:
- command: `uv run python scripts/v6_eval_probes.py --checkpoint checkpoints_v6_wikitext103/best_model.pt --verbose`

Observed summary:

| Probe | Result | Takeaway |
|---|---|---|
| Co-reference | nominal `100%` pass | Not robust evidence; some passes are inflated by weak thresholds and tokenization artifacts |
| Fact persistence | `0%` pass | Strong sign that factual carry-over is not working yet |
| Bank specialization | `0.000011` score | Very weak signal; banks are not yet showing meaningful differentiated behavior |
| Working memory utilization | skipped | WM disabled in this run |
| SSM timescale probe | unstable-looking learned decays | Current probe suggests the SSM is not yet behaving like a clean long-memory mechanism |

Interpretation:

- The probe suite does **not** support a strong claim that V6 has solved entity/relation tracking.
- The probe suite does **not** support a strong claim that the named banks are already empirically specialized.
- The strongest negative signal is the fact persistence failure.

---

## 6. Comparison Caveat: Most WikiText-103 Numbers Are Not Directly Comparable

This matters a lot.

Your run is:

- WikiText-103 **raw**
- GPT-2 **BPE** tokenizer
- validation perplexity
- chunked evaluation over fixed-length token blocks

Many published WikiText-103 numbers are:

- **word-level**
- differently preprocessed
- often reported on **test**
- often measured with different context handling

So raw/BPE perplexity should not be naively compared to classic word-level perplexity.

Useful references documenting this mismatch:

- [Hugging Face issue: GPT-2 perplexity number on WikiText-103 doesn't match the paper](https://github.com/huggingface/transformers/issues/483)
- [llm.c PR: GPT-2 evaluation on WikiText-103 and reproduction inconsistencies](https://github.com/karpathy/llm.c/pull/340)

This means every comparison below is tagged as either:

- **orientation only**: useful for scale, but not apples-to-apples
- **closer comparison**: not perfect, but much more relevant

---

## 7. External Baselines With Links

## 7.1 Orientation-only classic WikiText-103 baselines

These are important historically, but they are not directly apples-to-apples against the current V6 run.

| Model | Reported PPL | Notes | Link |
|---|---:|---|---|
| Neural cache model | `40.8` test | Early WikiText-103 benchmark result | [Salesforce WikiText benchmark page](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) |
| AWD-QRNN | `32.0` val / `33.0` test | Strong word-level single-GPU baseline | [Scalable Language Modeling: WikiText-103 on a Single GPU in 12 hours](https://mlsys.org/Conferences/doc/2018/50.pdf) |
| Adaptive input representations | `18.7` | Strong Transformer-era word-level result | [Adaptive Input Representations for Neural Language Modeling](https://arxiv.org/abs/1809.10853) |
| Transformer-XL | `18.3` | Long-context attention baseline | [Transformer-XL](https://arxiv.org/abs/1901.02860) |

Against these, V6 at `49.61` is clearly not competitive.

## 7.2 Closer raw/BPE comparison points

These are still imperfect, but they are much more relevant to the present setup.

From the reproduction work in [llm.c PR #340](https://github.com/karpathy/llm.c/pull/340):

| Model | Reported setting | Validation PPL |
|---|---|---:|
| GPT-2 `124M` | Hugging Face dataset, raw | `30.59` |
| GPT-2 `124M` | Bare minimum preprocessing | `31.04` |
| GPT-2 `355M` | Raw | `22.35` |
| GPT-2 `774M` | Raw | `19.33` |
| GPT-2 `1558M` | Raw | `17.46` |

These numbers come with documented evaluation caveats, but they are still much closer to your pipeline than the classic word-level papers.

### Evaluation Protocol Differences

Perplexity numbers depend heavily on evaluation protocol:

- **Test set with sliding window**: ~14.84 (lower, standard for paper comparisons)
- **Validation set raw/BPE**: ~30.59-31.04 (higher, more conservative)

The ~2x difference comes from:
1. Test vs validation set distribution
2. Sliding-window vs raw evaluation
3. Tokenization and preprocessing differences

We cite the validation PPL (~31) as it uses the same raw/BPE evaluation as our model.

Relative to the closest one:

- V6 `28.7M`: `49.61`
- GPT-2 `124M`: about `30.6`

That is still a large gap, even allowing for parameter-count differences.

---

## 8. Where V6 Stands Right Now

### Strongest justified claim

V6 is now a **real attention-free language-model family**, not just an interesting toy architecture.

### Claims that are justified

- V6 can train stably on both TinyStories and WikiText-103.
- The no-memory path is currently the safest and cleanest V6 configuration.
- V6 learns meaningful surface structure on real encyclopedia data.
- V6 has enough promise to justify further benchmarking.

### Claims that are not justified

- V6 is not yet proven competitive with strong baselines.
- V6 is not yet proven superior to Transformers, QRNNs, or GPT-style LMs.
- V6 is not yet proven to gain from its memory hierarchy on real data.
- V6 is not yet proven to have meaningful bank specialization.
- V6 is not yet proven to solve long-range entity/relation reasoning.

### Best one-line summary

V6 has graduated from **architectural curiosity** to **serious research candidate**, but it has **not** yet graduated to **benchmark-competitive model**.

---

## 9. Benchmarking Policy From This Point On

To avoid misleading ourselves, future V6 comparisons should follow these rules.

### Rule 1: Always report exact evaluation setup

Every result should state:

- tokenizer
- dataset variant (`raw` vs processed)
- split (`val` vs `test`)
- sequence length / context length
- loss normalization details
- batch size
- parameter count
- hardware
- wall-clock training time
- throughput

### Rule 2: Split comparisons into fair buckets

Every benchmark section should have:

- **Bucket A: exact or near-exact pipeline matches**
- **Bucket B: orientation-only historical baselines**

This avoids mixing word-level and BPE perplexities in the same table without warning.

### Rule 3: Keep at least one matched internal baseline

For each important V6 run, also train at least one simple internal baseline with the same tokenizer/data pipeline:

- a same-budget Transformer baseline
- optionally a plain SSM / diagonal recurrent baseline

Without this, external comparisons remain too noisy.

### Rule 4: Measure more than perplexity

Perplexity alone is not enough for V6, especially because its claimed advantages are structural.

Each main run should be scored on:

- validation perplexity
- long-form generation samples
- fact persistence probe
- entity/co-reference probe
- bank specialization probe
- training speed and memory footprint

### Rule 5: Prefer decision experiments over “just train longer”

The next useful experiments are the ones that answer architectural questions, not the ones that merely add compute.

---

## 10. Recommended Next Benchmark Sequence

## Step 1: Same-pipeline Transformer baseline

Train a standard Transformer on:

- WikiText-103 raw
- GPT-2 tokenizer
- seq_len `512`
- similar parameter budget (`~25M–35M`)

Why:

- This is the most important missing comparison.
- It tells us whether V6 is behind because the benchmark is hard, or because the architecture is weak.

Decision rule:

- If same-budget Transformer is much better, V6 is not yet competitive.
- If the gap is modest, V6 becomes much more interesting.

## Step 2: V6 WikiText-103 with small WM only

Run:

- `WM=8`
- `IM=0`

Why:

- This directly tests whether V6’s memory story helps on a richer corpus without the TinyStories memorization failure mode.

Decision rule:

- If `WM=8` improves PPL and improves factual/entity probes, memory remains central.
- If it does not help, keep memory out of the headline architecture for now.

## Step 3: Same-budget V6 medium/no-memory

Scale capacity before scaling complexity:

- larger no-memory V6
- same dataset and tokenizer

Why:

- This tests whether the current weakness is mainly under-capacity rather than wrong design.

## Step 4: PG-19 only after Step 1 or Step 2 gives a win signal

Why:

- PG-19 is ideal for long-range narrative testing.
- But if V6 is not yet competitive on WikiText-103, PG-19 will generate interesting text without clarifying the benchmark question.

---

## 11. Recommended Decision Criteria

Use these thresholds to decide what to do next.

### Continue pushing V6 hard if at least one is true

- Same-budget Transformer gap is small enough to be plausible to close.
- `WM=8` materially improves both PPL and factual/entity probes.
- Medium no-memory V6 narrows the gap sharply.
- PG-19 shows unusually strong long-range narrative behavior relative to the baseline.

### Re-scope V6 if all are true

- Same-budget Transformer wins clearly.
- WM does not help on real text.
- Bank specialization stays weak.
- Fact persistence remains near zero.

In that case, the architecture may still be publishable as an interesting negative/partial result, but not as a new LM winner.

---

## 12. Current Recommendation

Do **not** claim V6 is proven.

Do claim:

- V6 is stable on real text.
- V6 is interesting enough to benchmark seriously.
- The next decision should come from **matched baselines** and **small-memory WikiText-103**, not from more TinyStories work.

My recommended order:

1. same-budget Transformer on the exact WikiText-103 raw/BPE pipeline
2. V6 `WM=8, IM=0` on the same pipeline
3. V6 medium no-memory on the same pipeline
4. only then PG-19 or mixed-corpus scaling
