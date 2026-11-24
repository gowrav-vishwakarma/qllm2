# The package: SCI-1 with a text library

SCI-1 keeps a fixed-size state: 3.67 MB, whatever the length of the text it has read. The model alone ("bare") answers
from that state. The **package** adds a small text library of the same text. At question time the library fetches the
one sentence the question is about, and the model reads that sentence plus the question on its state.

This folder ships a standalone version of the package (`package_lite.py`, `run_package.py`). It needs only
`modeling_pam2.py`, torch and transformers. It is a research tool, not a product: read "Where it works" below before
relying on it.

## Run it

```bash
python run_package.py --demo                          # a 32k-token needle: the model alone vs the package
python run_package.py --demo --demo_tokens 8192 --device cpu
python run_package.py --context notes.txt --question "What colour is the car Helen drives to work?"
python run_package.py --context notes.txt --question "..." --mode package --strict    # abstain when the text organs miss
```

`--mode bare | package | both` (default both). `--form plain` (default) asks `\n\nQuestion: ...\nAnswer:`; `--form raw`
uses your text as is (a cloze such as `"\nThe car Helen drives to work is painted"`); `--form chat` is for SCI-1 Chat.

Output of `python run_package.py --demo` (RTX 4090):

```
demo: a 32768-token haystack, needle at depth 0.41, gold 1679215
read 31,458 tokens in 2.4 s

BARE    (0.1 s): '4.'   right: False
  memory held: model state 3.67 MB (the same at any length)

PACKAGE (0.1 s): '1679215.'   right: True
  fetched by text2: 'One of the special magic numbers for rusty-falcon is: 1679215.'
  memory held: model state 3.67 MB + library text 0.06 MB + word index ~0.30 MB (+ address 32.34 MB int8, used only when the text organs miss)
```

On CPU (16 threads) the 8k demo takes 13 s in total; reading runs at about 1,000 tokens per second.

## What it does

1. **Read** the text once. The model streams it into its state in 2,048-token chunks. The library keeps:
   - the token text (2 bytes per token);
   - a word index (about 9.4 bytes per token, estimated);
   - per token, the model's hidden vector after block 8 (the "address", used only by step F2).
2. **Fetch** one sentence for the question, by text first:
   - **Token organ** (`text0`): the latest exact match of the question's last 4 tokens.
   - **Word organ**: words are lower-cased, "a / an / the" dropped and one plural "s" stripped.
     - `text1`: the longest question suffix of 6 to 3 words.
     - `text2`: any 6- to 3-word run of the question.
     - `text4`: a question word seen 1-2 times in the text AND at most 3 times in the background counts. That is a name or
       a made-up key, never a frame word.
   - **F2** (default mode only): the 2 question tokens the model finds most surprising are matched by cosine against
     every sentence in the address. `--strict` abstains here instead.
3. **Answer:** the sentence, then the question, are streamed on a copy of the state. The model answers greedily and
   stops at a sentence end, a newline or `<|im_end|>`; a "." between digits does not stop it.

**Bare** is the same state plus the question, with the same decoding and no fetch.

## Where it works, and where it does not

- **Works:** the question shares a key or a phrase with the sentence that holds the answer. Examples: a needle, a
  key-value record, a name with a code, a cloze that repeats the sentence's start. At 32k the model alone cannot
  recall such a fact; the package does (measured below).
- **Weak:** natural questions about facts stated in prose. The fetch often finds the right sentence, but this
  386M base model often reads it wrongly. On a 3-sentence note
  ("Marta ... moved to Lisbon ... Daniel keeps a parrot at home ... The car Helen drives to work is painted silver."),
  this checkpoint gives:

  | question | bare | package (fetch path) |
  |---|---|---|
  | Which pet does Daniel have? | `6.5.` | `5.5.` (F2: the wrong sentence) |
  | What does Daniel keep at home? | `The car Helen drives to work is painted silver.` | `The parrot is a symbol of the city of Lisbon.` (F2: the right sentence) |
  | Which city did Marta move to? | `Portugal.` | `Portugal.` (F2: the right sentence) |
  | What colour is the car Helen drives to work? | `blue.` | `blue.` (text2: the right sentence) |
  | cloze `The car Helen drives to work is painted` | `silver.` | `silver.` (text0) |

  SCI-1 Chat answers these questions much better (see its PACKAGE.md). In the research repo's natural-recall probe
  (a fact in prose plus a same-type distractor), this checkpoint picks the right answer among 8 at 0.32 when the
  question comes 64 tokens later. That is no better than picking the distractor (0.335), so it does not bind a fact to
  its owner.
- **Not in this port:** the research repo also applies a rule held in the state to a fetched value (R × K), reads a
  held rule through a "twin" state, and runs BABILong's multi-sentence hops and a text planner chain. Those paths stay
  in the repo (`v16/package.py`, `v14/`). Every measured "package" number below that uses them says so.

## Memory held

The state is 3.67 MB at any length. The library grows with the text: per token, 2 B of text plus about 9.4 B of word
index, plus the F2 address (1,028 B per token at int8). The demo prints all three. The address is the large part; it is
used only when the text organs miss, so with `--strict` it is not needed.

## Parity with the research code

`v14/pkg_port_parity.py` (research repo) runs the port and the repo's `v16/package.py` on the same weights and the same
20 items:
- RULER-style niah single / multikey at 4k and 8k (the BENCH-PKG docs);
- PKG-SUITE recall kv / kvupd at 8k;
- natural-recall prose at 2k (indirect and synonym questions).

Compared per item: the bare answer, the package answer, the fetch path, the fetched row, the F2 row and its two
salient tokens (forced on every item, to check the address), and the state bytes.

**Result on this checkpoint (step 749,500; RTX 4090, fp32; code e337eacc; `logs/v14_pkg_port_parity_rel_e337eacc_20261124_1302.*`):
20 / 20 items equal on every field.** The fetch paths covered `text0`, `text2`, `text4` and `F2`.

## Measured (research repo, same weights)

Research-repo runs on the same weights (base SCI-1, step 749,500, 49.12B tokens; code f1bba05; 2026-11-24). They are descriptive; no other model is
compared here. "Bare" is the model alone on the whole text. The repo's runners are `v14/pkg_suite.py`, `v14/ruler_pkg.py`
and `v16/babilong_eval.py`.

**Package suite** (100 documents per cell; recall 20 per task):
- R × K: a rule shown earlier in the text is applied to a value the library fetches.
- Rewrite: a stated unit protocol is compiled next to the question.
- Told rule held in state: read through the rule twin.
- Recall: needle among 8 / latest of several updates / two-hop at 32k, through the text organs; the two-hop column is a
  TEXT chain of two package answers.

R × K, the rule twin and the two-hop chain use repo paths that are not in this port.

| cell | bare | package |
|---|---|---|
| R × K, 2k / 8k (first token) | 0.00 / 0.00 | 0.97 / 0.78 |
| rewrite: told rule compiled next to the question, 2k / 8k | 0.04 / 0.00 | 0.93 / 0.92 |
| told rule held in state (twin), 2k / 8k | 0.16 / 0.04 | 0.77 / 0.25 |
| recall 32k, free: needle among 8 / latest-wins / two-hop | 0.05 / 0.25 / 0.05 | 1.00 / 1.00 / 0.95 |
| recall 32k, 20-way: kv / kvupd / hop2 | 0.10 / 0.55 / 0.10 | 1.00 / 1.00 / 1.00 |

Memory per 32k session: model state 3.67 MB; library text 0.07 MB + word index ~0.31 MB; F2 address 33.8 MB at int8.

**RULER-style, our own generator** (not the official RULER suite; 50 documents per cell):
- niah_single: one key → value. niah_multikey: the key among 3 look-alike keys. The value is a word, a 7-digit number or
  a uuid.
- VT: chains of 4 variable assignments. The library follows the chain in text (the "planner" column, which is
  library-only); the model then writes the names.
- Library MB includes the F2 address.

| L | task | n | bare | package | planner (VT) | mind MB | library MB |
|---|---|---|---|---|---|---|---|
| 4k | niah_multikey | 50 | 0.04 | 0.46 | - | 3.67 | 4.38 |
| 4k | niah_single | 50 | 0.08 | 0.54 | - | 3.67 | 4.28 |
| 4k | vt | 50 | 0.00 | 0.00 | 1.00 | 3.67 | 0.01 |
| 8k | niah_multikey | 50 | 0.00 | 0.62 | - | 3.67 | 8.62 |
| 8k | niah_single | 50 | 0.00 | 0.44 | - | 3.67 | 8.54 |
| 8k | vt | 50 | 0.00 | 0.00 | 1.00 | 3.67 | 0.02 |
| 16k | niah_multikey | 50 | 0.02 | 0.36 | - | 3.67 | 17.13 |
| 16k | niah_single | 50 | 0.02 | 0.50 | - | 3.67 | 17.06 |
| 16k | vt | 50 | 0.00 | 0.00 | 1.00 | 3.67 | 0.03 |
| 32k | niah_multikey | 50 | 0.00 | 0.40 | - | 3.67 | 34.16 |
| 32k | niah_single | 50 | 0.02 | 0.32 | - | 3.67 | 34.09 |
| 32k | vt | 50 | 0.00 | 0.02 | 1.00 | 3.67 | 0.07 |

The fetch finds the right row on every niah document (path text2). What fails is the read: uuids get garbled and long
numbers cut; a word after "is:" is often lost because the model writes ":" and a newline, which stops the answer.

**BABILong** (qa1-qa5, 100 documents per cell, BABILong's scoring):
- Each cell: bare → package (planner), 100 documents.
- Bare = the model as is on the whole input.
- Package = fetch + read, with the repo's multi-sentence hops (not in this port).
- Planner = a TEXT chain of package answers.

Mind state 3.67 MB at every length; library text 0.06 MB at 32k (no F2 address is used here).

| task | 0k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|
| qa1 | 0.13 → 0.16 (0.16) | 0.07 → 0.38 (0.38) | 0.04 → 0.28 (0.28) | 0.02 → 0.18 (0.18) | 0.02 → 0.12 (0.12) | 0.00 → 0.10 (0.10) |
| qa2 | 0.00 → 0.00 (0.10) | 0.07 → 0.23 (0.20) | 0.06 → 0.23 (0.15) | 0.03 → 0.19 (0.09) | 0.03 → 0.16 (0.09) | 0.03 → 0.13 (0.09) |
| qa3 | 0.02 → 0.01 (0.02) | 0.07 → 0.06 (0.15) | 0.05 → 0.11 (0.13) | 0.05 → 0.08 (0.10) | 0.02 → 0.07 (0.13) | 0.02 → 0.06 (0.05) |
| qa4 | 0.03 → 0.01 (0.15) | 0.01 → 0.16 (0.14) | 0.01 → 0.12 (0.10) | 0.00 → 0.10 (0.10) | 0.01 → 0.07 (0.07) | 0.00 → 0.01 (0.02) |
| qa5 | 0.20 → 0.12 (0.09) | 0.25 → 0.34 (0.34) | 0.13 → 0.26 (0.24) | 0.07 → 0.34 (0.26) | 0.03 → 0.22 (0.17) | 0.00 → 0.18 (0.18) |

- From 2k on, the package beats the model alone in every cell but qa3 at 2k (0.07 → 0.06). qa1 (one supporting fact) drops from 0.38 at 2k to 0.10 at 32k; qa5 holds 0.18-0.34. qa3 (three facts) and qa4 stay at or below 0.16.
- The fetch is not the main loss. On 2k-32k the answer word is in the fetched rows on 0.91 of qa1 docs and 0.79 of qa5 docs, yet the package is right on 0.21 and 0.27. (That is an upper bound for the fetch: the word may sit in the wrong row.) The 386M model often misreads several fetched rows.
- At 0k (the facts only, no haystack) the package does not help: 0.00-0.16 vs bare 0.00-0.20.

Runs: `logs/v14_babilong_pkg_rel_*` in the research repo (code f1bba05).


## The background word counts

`bg_wordcounts_fineweb014.json`: 1.47 MB, 101,252 normalised words with their counts (sha256 eb4f9c16...).
- **Source:** about 6M tokens of one shard of FineWeb (HuggingFaceFW/fineweb, `sample/10BT/014_00000.parquet`), held out
  from the evaluation prose.
- **Built by:** `v14/bg_counts.py` in the research repo.
- **Licence:** FineWeb is released under ODC-By 1.0 (and subject to Common Crawl's terms of use). The file holds derived
  word counts, not text.
- **Used by:** step `text4` only. Point `PACKAGE_BG` at another copy to override the file.
