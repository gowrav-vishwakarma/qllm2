# The package: SCI-1 Chat with a text library

SCI-1 Chat keeps a fixed-size state: 3.67 MB, whatever the length of the text it has read. The model alone ("bare") answers
from that state. The **package** adds a small text library of the same text. At question time the library fetches the
one sentence the question is about, and the model reads that sentence plus the question on its state.

This folder ships a standalone version of the package (`package_lite.py`, `run_package.py`). It needs only
`modeling_pam2.py`, torch and transformers. It is a research tool, not a product: read "Where it works" below before
relying on it.

## Run it

```bash
python run_package.py --demo                          # a 32k-token needle: the model alone vs the package
python run_package.py --demo --demo_tokens 8192 --device cpu
python run_package.py --context notes.txt --question "What colour is the car Helen drives to work?" --form chat
python run_package.py --context notes.txt --question "..." --mode package --strict    # abstain when the text organs miss
```

`--mode bare | package | both` (default both). `--form chat` asks a ChatML user turn (the form this model was tuned on);
`--form plain` (default) asks `\n\nQuestion: ...\nAnswer:`; `--form raw` uses your text as is (a cloze such as
`"\nThe car Helen drives to work is painted"`).

The demo on this checkpoint (RTX 4090), needle at depth 0.41, gold 1679215:
- **8k:** bare `' 1679.'`, package `' 1679215.'`.
- **32k:** bare `':\n'`, package `' 1679215.'`.

Memory held at 32k: model state 3.67 MB, library text 0.06 MB, word index ~0.30 MB, and the address 32.34 MB at int8
(used only when the text organs miss).

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
- **Natural questions:** on a short note this chat model answers them, and the package can correct it. But it adds
  details that are not in the text, so check them. On a 3-sentence note ("Marta ... moved to Lisbon ... Daniel keeps a
  parrot at home ... The car Helen drives to work is painted silver."), this checkpoint gives:

  | question (form) | bare | package (fetch path) |
  |---|---|---|
  | Which pet does Daniel have? (plain) | `The parrot.` | `The parrot.` (F2) |
  | Which pet does Daniel have? (chat) | `Daniel has a parrot named Pica.` | `Daniel has a parrot named Pica.` (F2) |
  | Which city did Marta move to? (plain) | `Lisbon.` | `Lisbon.` (F2) |
  | Which city did Marta move to? (chat) | `Marta moved to Lisbon, Portugal, in the year 2000.` | `Marta moved to Lisbon, Portugal, in the early 1990…` (F2) |
  | What colour is the car Helen drives to work? (plain) | `Blue.` | `Blue` (text2) |
  | What colour is the car Helen drives to work? (chat) | `The car Helen drives to work is a red car.` | `The car Helen drives to work is a silver car.` (text2) |

  The pet's name and the years are invented. On a note this short the bare model sees every sentence anyway; the
  package matters over long texts, where the bare state no longer holds the fact (the demo, and the tables below).
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

**Result on this checkpoint (step 753,315; RTX 4090, fp32; code e337eacc; `logs/v14_pkg_port_parity_sft_e337eacc_20261124_1302.*`):
20 / 20 items equal on every field.** The fetch paths covered `text0`, `text2`, `text4` and `F2`.

## Measured (research repo, same weights)

Research-repo runs on the same weights (SCI-1 Chat, step 753,315, 49.37B tokens; code f1bba05; 2026-11-24). They are descriptive; no other model is
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
| R × K, 2k / 8k (first token) | 0.00 / 0.00 | 0.97 / 0.85 |
| rewrite: told rule compiled next to the question, 2k / 8k | 0.03 / 0.00 | 0.96 / 0.95 |
| told rule held in state (twin), 2k / 8k | 0.15 / 0.01 | 0.80 / 0.22 |
| recall 32k, free: needle among 8 / latest-wins / two-hop | 0.05 / 0.25 / 0.05 | 1.00 / 1.00 / 0.90 |
| recall 32k, 20-way: kv / kvupd / hop2 | 0.10 / 0.60 / 0.05 | 1.00 / 1.00 / 0.95 |

Memory per 32k session: model state 3.67 MB; library text 0.07 MB + word index ~0.31 MB; F2 address 33.8 MB at int8.

**RULER-style, our own generator** (not the official RULER suite; 50 documents per cell):
- niah_single: one key → value. niah_multikey: the key among 3 look-alike keys. The value is a word, a 7-digit number or
  a uuid.
- VT: chains of 4 variable assignments. The library follows the chain in text (the "planner" column, which is
  library-only); the model then writes the names.
- Library MB includes the F2 address.

| L | task | n | bare | package | planner (VT) | mind MB | library MB |
|---|---|---|---|---|---|---|---|
| 4k | niah_multikey | 50 | 0.04 | 0.30 | - | 3.67 | 4.38 |
| 4k | niah_single | 50 | 0.08 | 0.40 | - | 3.67 | 4.28 |
| 4k | vt | 50 | 0.00 | 0.02 | 1.00 | 3.67 | 0.01 |
| 8k | niah_multikey | 50 | 0.00 | 0.40 | - | 3.67 | 8.62 |
| 8k | niah_single | 50 | 0.00 | 0.38 | - | 3.67 | 8.54 |
| 8k | vt | 50 | 0.00 | 0.04 | 1.00 | 3.67 | 0.02 |
| 16k | niah_multikey | 50 | 0.00 | 0.18 | - | 3.67 | 17.13 |
| 16k | niah_single | 50 | 0.00 | 0.34 | - | 3.67 | 17.06 |
| 16k | vt | 50 | 0.00 | 0.00 | 1.00 | 3.67 | 0.03 |
| 32k | niah_multikey | 50 | 0.00 | 0.30 | - | 3.67 | 34.16 |
| 32k | niah_single | 50 | 0.02 | 0.20 | - | 3.67 | 34.09 |
| 32k | vt | 50 | 0.00 | 0.00 | 1.00 | 3.67 | 0.07 |

The fetch finds the right row on every niah document (path text2). What fails is the read: uuids get garbled and long
numbers cut; a word after "is:" is often lost because the model writes ":" and a newline, which stops the answer.

**BABILong** (qa1-qa5, 100 documents per cell, BABILong's scoring):
- Bare = the model as is on the whole input.
- Package = fetch + read, with the repo's multi-sentence hops.
- Planner = a TEXT chain.

Each cell: bare → package (planner), 100 documents. Bare = the model as is on the whole input. Package = fetch + read, with the repo's multi-sentence hops (not in this port). Planner = a TEXT chain of package answers. Mind state 3.67 MB at every length; library text 0.06 MB at 32k (no F2 address is used here).

| task | 0k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|
| qa1 | 0.40 → 0.47 (0.47) | 0.10 → 0.52 (0.52) | 0.04 → 0.40 (0.40) | 0.05 → 0.29 (0.29) | 0.01 → 0.22 (0.22) | 0.00 → 0.12 (0.12) |
| qa2 | 0.33 → 0.34 (0.39) | 0.05 → 0.26 (0.25) | 0.08 → 0.25 (0.25) | 0.04 → 0.18 (0.12) | 0.04 → 0.14 (0.12) | 0.03 → 0.09 (0.08) |
| qa3 | 0.19 → 0.15 (0.16) | 0.06 → 0.10 (0.12) | 0.05 → 0.06 (0.13) | 0.04 → 0.06 (0.12) | 0.04 → 0.05 (0.09) | 0.01 → 0.01 (0.04) |
| qa4 | 0.18 → 0.12 (0.12) | 0.05 → 0.11 (0.09) | 0.01 → 0.09 (0.08) | 0.01 → 0.07 (0.05) | 0.00 → 0.03 (0.03) | 0.00 → 0.01 (0.02) |
| qa5 | 0.62 → 0.63 (0.69) | 0.22 → 0.44 (0.42) | 0.22 → 0.36 (0.33) | 0.15 → 0.44 (0.39) | 0.06 → 0.25 (0.24) | 0.06 → 0.29 (0.26) |

- The Chat model answers the short task far better alone: at 0k, bare 0.18-0.62 vs 0.00-0.20 for the base. With a haystack, bare falls to 0.00-0.06 at 32k.
- From 2k on, the package beats the model alone in every cell but qa3 at 32k (0.01 both). qa1 0.52 at 2k → 0.12 at 32k; qa5 0.44 at 2k and 8k, 0.29 at 32k. Chat is above the base on qa1 and qa5 at most lengths; qa3 / qa4 stay ≤ 0.12.
- As for the base, the answer word is in the fetched rows on 0.91 (qa1) and 0.79 (qa5) of 2k-32k docs; the read of several rows is the loss.

Runs: `logs/v14_babilong_pkg_sft_*` in the research repo (code f1bba05; the Chat 32k cell ran on code e337eacc, same runner).


## The background word counts

`bg_wordcounts_fineweb014.json`: 1.47 MB, 101,252 normalised words with their counts (sha256 eb4f9c16...).
- **Source:** about 6M tokens of one shard of FineWeb (HuggingFaceFW/fineweb, `sample/10BT/014_00000.parquet`), held out
  from the evaluation prose.
- **Built by:** `v14/bg_counts.py` in the research repo.
- **Licence:** FineWeb is released under ODC-By 1.0 (and subject to Common Crawl's terms of use). The file holds derived
  word counts, not text.
- **Used by:** step `text4` only. Point `PACKAGE_BG` at another copy to override the file.
