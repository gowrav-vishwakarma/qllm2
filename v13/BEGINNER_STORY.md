# V13 as a Story — for someone who doesn't know the words yet

Written 2026-08-23. This is the **ground-up** companion to
[`MATH_EXPLAINER.md`](MATH_EXPLAINER.md) (the per-method math reference) and
[`MATRIX_COOKBOOK.md`](MATRIX_COOKBOOK.md) (the op-by-op shape reference).
If you read only one file, read this one first.

We will build everything in order, and we will not use a word before it has
been defined. The story has five parts:

- **Part A** — the math you actually need (numbers, vectors, the two
  products, complex numbers). All with small examples you can do by hand.
- **Part B** — the model's vocabulary, one by one: feature, head, token,
  batch, state, and the memory table.
- **Part C** — the story: a sentence travels through the model. Forward,
  then backward (training), then inference (your prompt, token by token, and
  where your knowledge actually *is*).
- **Part D** — quick answers to the exact questions in the margin.
- **Part E** — the questions you'd still have, answered with the *actual
  code pasted in*: what batch really does, why the memory shape has two
  d's, the exact RoPE formula, the key-normalization NaN, the next-token
  trick, and what x is.

Line numbers refer to `v13/model.py` unless noted.

---

# Part A — the math, from zero

## A1. What is a "feature"?

A feature is **one number in a list of numbers that describes something**.

That's it. Pick a dog photo. A machine could describe it with a list:

```
[brownness, ear_length, has_sitting, is_indoor, ...]   ← e.g. 200 numbers
```

Each number is one *feature*. None of them means "brownness" forever —
they're just 200 dials. The machine learns which dials matter.

A **vector** is just such a list. `v = [0.3, −0.7, 0.1]` is a vector with
3 features (3 "dimensions").

Now the key move the model makes: **it describes a word the same way.**
The word "king" is a list of 384 numbers. The word "queen" is another list
of 384 numbers. "King" and "queen" will have lists that *look similar*
(close in value, similar direction) because both mean "royal person".
"King" and "sock" will have lists that look *different*.

> **Rule to remember: a token is a vector. A vector is a list of numbers.
> Each number is a feature.** Nothing more.

Why 384? A choice. More dials = can describe more different things, but
more numbers to store and learn. 384 is the sweet spot for a ~500M
parameter model.

## A2. One number you'll meet everywhere: the dot product

Given two lists of the same length, the **dot product** is:
multiply matching entries, add everything up.

```
v = [3, 4]
w = [1, 2]
v · w = 3·1 + 4·2 = 11
```

What does 11 *mean*? It measures **how much the two lists "point the same
way"**. Three cases:

- **Pointing the same way** → big positive number.
  `v=[3,4], w=[3,4]`:  `9+16 = 25` (maximum possible for these lengths).
- **Pointing at right angles** → exactly 0.
  `v=[1,0], w=[0,1]`:  `0+0 = 0` (they "agree on nothing").
- **Pointing opposite** → negative.
  `v=[1,0], w=[−1,0]`: `−1`.

The dot product is the **similarity score** of the whole model. "Does this
word look like that word?", "does this query match that stored key?" — the
answer is always a dot product. When we do it to two 384-lists we get one
number. That number is a **score**.

> **Rule: dot product = many numbers in, one score out. It's a *reading*.**

## A3. The other product: the outer product

Now do the *opposite*: take one list `u` (length 2) and one list `v`
(length 2), and make a **table** (2×2) by combining *every* entry of `u`
with *every* entry of `v`:

```
u = [u1, u2]      v = [v1, v2]

u ⊗ v =  | u1·v1   u1·v2 |
         | u2·v1   u2·v2 |
```

That's the **outer product**: two lists in, a whole table out. Each cell
`[i, j]` says: "channel i of `u`, paired with channel j of `v`."

> **Rule: outer product = two lists in, a table out. It's a *writing*.**

### A3.1 Why the table is a memory (the whole trick)

Here's the magic, in three lines. Suppose you want to **remember** the fact
"key k → value v". You *write* it as a table:

```
S = v ⊗ k*          (the * means "flip the sign of the second number
                     of each pair" — complex, A4; for plain numbers, ignore *)
```

Now you *read* with a query q: you multiply the table by q — which is just
**dot products**: the i-th row of the table dotted with q.

Work it out with 2-d numbers, `v=[a,b]`, `k=[c,d]`, `q=[c,d]` (you're
asking with the *same* key you stored):

```
S = | a·c   a·d |
    | b·c   b·d |

read row 1:  a·c·c + a·d·d = a·(c·c + d·d) = a·|k|²
read row 2:  b·c·c + b·d·d = b·(c·c + d·d) = b·|k|²

result = |k|² · [a, b] = |k|² · v
```

**You got back `v`** — the thing you stored — scaled by how well your query
matched the key. Ask with a different key, one at right angles to `k`, and
you get `0`.

So: **a table (matrix) *is* a memory.** Writing = outer product. Reading =
dot products (inner products). That's the entire concept the model is built
on. Everything else (decay, delta, phases) is decoration on these two moves.

One more consequence, important: if you write **two** facts,
`S = v1⊗k1* + v2⊗k2*`, then reading with k1 gives ~v1 and reading with k2
gives ~v2 — the facts **coexist** in the same table, as long as their keys
point different ways. A 64×64 table can hold many facts at once. (When keys
*do* point similar ways they interfere — that's what the "delta rule" of
C4.3 is for.)

## A4. Complex numbers — and what "phase" and "magnitude" actually are

A complex number is **two real numbers glued together**: `z = a + i·b`,
where `i` is just a symbol with the rule `i·i = −1`. Think of `z` as an
**arrow on a piece of paper**:

```
        b (up)
        ^
        |
   a -->+----> (right)
```

The arrow starts at the middle, goes right by `a`, up by `b`. Two facts
about the arrow:

1. **Magnitude** `|z| = √(a² + b²)` — the **length** of the arrow. How far
   it reaches. "How loud."
2. **Phase** `φ = atan2(b, a)` — the **angle** of the arrow, measured from
   the right-pointing axis, in radians (0 to 2π = one full turn). "Which
   way it points."

Same two facts as a clock hand: length of the hand, and where it points.

**The one rule that makes complex numbers useful:**

> **Multiplying two complex numbers multiplies their magnitudes and ADDS
> their phases.**

That's it. That's why the model uses them. (The code stores each complex
number as a pair `[real, imag]` — that's the last `2` you keep seeing — and
does the multiply by hand: `complex_ops.py:29-56`.)

### A4.1 Why "phase alone" matters (with a tiny example)

Make two complex numbers with the **same magnitude (1)** but different
phases:

```
x = 1 + 0i     → magnitude 1, phase 0°   (points right)
y = 0 + 1i     → magnitude 1, phase 90°  (points up)
```

Now multiply them:

```
x · y = i              → magnitude 1, phase 90°
x · x* = 1             → magnitude 1, phase 0°      (x* = conjugate = arrow flipped down)
y · y* = 1             → magnitude 1, phase 0°
```

Read those results:

- `x · x* = 1`, `y · y* = 1`: **a number always "matches" its conjugate
  perfectly.** This is how the memory knows "this is the key I stored."
- `x · y = i`: two numbers pointing at right angles produce a number that
  points at right angles — they **don't cancel, but they don't reinforce
  either.** They're independent "channels."
- Flip one phase by 180° and you *do* get cancellation: `x·(−x*) = −1`.

So **phase is a label**: it decides whether two things reinforce (same
phase), interfere (opposite phase), or ignore each other (orthogonal
phase) — *without changing their loudness (magnitude)*. Two facts stored
with different phase-labels share the same memory table without colliding.

That's the entire reason for complex numbers in this model. Magnitude =
loudness of a signal. Phase = its address/label.

### A4.2 A "rotation" is just a phase change

Multiply by `i` (magnitude 1, phase 90°) and you **rotate** the arrow by
90° without changing its length:

```
[3, 0]·i = [0, 3]     right-arrow of length 3 → up-arrow of length 3
```

You'll see "rotate the output by a learned phase" in the model (C4.4). That
is literally this move, done per channel, by a *learned* angle.

## A5. Matrices are just tables of numbers

You've now met all the ingredients:

- **vector** = a list of numbers (a column).
- **matrix** = a table of numbers (rows × columns). The memory `S` is one.
- **dot product** = vector·vector → one number.
- **outer product** = vector⊗vector → a table.
- **matrix·vector** = every row of the matrix dotted with the vector →
  a vector. (This is what "read the memory with a query" *is*: `S·q`.)
- **matrix·matrix** = every row of the first dotted with every column of
  the second. The training code is mostly this.

Every "mysterious" operation in the model is one of these five. When you see
an operation you don't recognize, ask: *which of the five is it, and which
two lists are being combined?*


---

# Part B — the model's vocabulary, one word at a time

Now that the math words are defined, here are the *model* words. Each one
is a small idea, and each has a concrete size in this model (the production
500M config: `dim=384, H=6, d=64, K=3`).

## B1. Token

The smallest unit the model reads: roughly a word or a piece of one
("king", "##ing", " the"). Text is chopped into a flat list of tokens.
Each token is an integer id (0 … 50260 — the vocabulary has 50,261 of
them). **The model never sees letters; it sees ids.**

## B2. Token vector (the token "becomes" numbers)

An id is a pointer into a lookup table. The table has 50,261 rows, one per
vocabulary word, and each row is a 384-number vector (the embedding).
`embed` (model.py:1558) does the lookup: id 42 → row 42 → a 384-vector.

Why it matters: this is the moment "meaning" enters. Two similar words have
similar rows. The rows are **learned** during training (they start
random and get shaped by the loss). The same table is reused at the very
end to turn vectors back into word-scores (the "tied" head, C5) — one table
for both directions.

## B3. Batch (B)

The GPU learns from *many sentences at once* to use its parallel cores.
`B = 8` means: 8 independent sequences travel through the model together,
side by side, touching nothing of each other. Every shape in the model
starts with `[B, …]` = "8 copies of everything, done independently."

> Think of it as 8 students taking the same exam in a row. Same questions,
> separate answer sheets, graded the same way.

## B4. Sequence length (T)

How many tokens are in the sentence: `T = 2048`. The model processes them
left to right. At each position it has seen only the tokens *before* it
(causality — it must not peek at the answer).

## B5. Head (H) — your guess, refined

You guessed "a head is a dimension slice." **Exactly right, and here's why
it's done.** The 384 features are split into 6 slices of 64
(`H=6, d=64`): the model keeps **6 separate notebooks**, each 64
features wide, each with its own memory table, and the 6 notebooks
cooperate.

Why 6 and not 1? One 384-wide notebook would be a kitchen where every
fact is in one drawer — a fact about "king" and a fact about "sock" would
fight for the same space. Six 64-wide notebooks let the model *specialize*
different drawers (one learns syntax-ish patterns, another content-ish,
another positional) the way separate workspaces do. The code makes the
slice with `view` + `transpose` (model.py:363-371): one big table,
reinterpreted as 6 smaller ones. Nothing is lost — it's the same numbers,
just grouped.

> **A head = one notebook of 64 features. 6 notebooks per layer.**

## B6. State (K) — the model has K notebooks per head

On top of heads, each head keeps `K = 3` memory tables, called **states**.
Think of one head having three notebooks of *different sizes of memory*:

- state 0 = **the vault**: never forgets (its decay is pinned to 1,
  model.py:448-453). Long-term.
- states 1, 2 = **scratchpads**: fade quickly (fast decay). Recent
  context.

So per head there are 3 tables; per layer 6×3 = 18 tables; 16 layers →
288 tables total in the whole model. Each table is 64×64 complex numbers
≈ 8 KB. The entire "memory of the model" while running is a few MB —
**fixed, no matter how long the conversation gets.** That's the point of
the architecture.

> **A state = one notebook. A head = 3 notebooks. A layer = 6 heads.
> A model = 16 layers.**

## B7. The letters, now that you can decode them

| Letter | Word | Meaning | Size |
|--------|------|---------|------|
| `B` | batch | independent sentences at once | 8 |
| `T` | sequence length | tokens in one sentence | 2048 |
| `dim` | width | features per token | 384 |
| `H` | heads | notebooks per layer | 6 |
| `d` | head dim | features per notebook | 64 |
| `K` | states | notebooks per head | 3 |
| `C` | chunk | tokens per parallel work-block (training only) | 128 |
| `V` | vocab | how many words exist | 50261 |
| `2` | real/imag | the complex pair on the end | 2 |

Now decode a real shape, `[K, B, H, d, d, 2]`:
"for each of 3 states, 8 sentences, 6 heads: one 64×64 table of
complex numbers (real, imag)." That's **the memory of one layer**.
## B8. "Feature" vs "channel" vs "dimension" — and dim vs head_dim

The words "feature", "channel", "dimension" all mean the same thing:
"one number in the list." But the *names of two different lengths* are
easy to mix up, so let's be exact:

- `dim = 384` — the length of a **token's** vector (all its features).
- `head_dim = d = 64` — the length of the vector **inside one head**.

They are **not** the same number: 6 heads × 64 features = 384 = dim.
So `dim` is the *whole* (the token), and `head_dim` is the *slice* (one
head's view of it). When the code says `head_dim` it means 64; when it
says `dim` (or `cfg.dim`) it means 384.

## B9. The one shape to hold in your head

Everything in the model is a reshuffling of these two shapes:

```
TOKENS  : [B, T, dim]   "8 sentences × 2048 words × 384 numbers each"
MEMORY  : [K, B, H, d, d]  "3 states × 8 × 6 × 64×64 tables"
```

The forward pass is the story of **how TOKENS get folded into MEMORY
(write), how MEMORY gets unfolded back out (read), and how the result gets
added to the tokens so the next layer can see it.** That's the whole
model in two lines. Part C walks it.

---

# Part C — the story: a sentence travels through the model

We follow one batch of sentences. Every step names the code that does it.
The three big verbs are **write**, **read**, and **add back**.

## C1. The door: tokens → vectors (model.py:1556-1560)

```
ids:  [B, T]            "the sentence as word-ids"
z  = embed(ids)  →  [B, T, dim]    "each word becomes a 384-vector"
```

That's a pure table lookup (B2). The sentence is now a *stack of 2048
384-vectors, 8 high*. No math yet — just fetching rows.

## C2. One layer, step by step (model.py:1457-1484)

A **layer** (block) does two different jobs, one after the other, each
added to the input ("residual" — the input always passes through, so
information can never be destroyed):

```
x ← x + CGU(norm(x))      # JOB 1: rethink the features of EACH word
x ← x + PAM(norm(x))      # JOB 2: let words TALK to each other via memory
```

JOB 1 (CGU, `ComplexGatedUnit`) is a plain "recompute" — a few learned
table-mixes on a single word's 384 numbers. It has no time, no memory;
it just makes each vector a better summary of its word. (The "G" is a
learned on/off switch on part of the output.)

JOB 2 is the memory — the rest of this section.

## C3. PAM: where the token *becomes* query/key/value (model.py:354-410)

The PAM layer takes each word's 384-vector and makes **three new vectors
per word**, with learned tables — like three different "questions" the
word asks:

- **query q** — "what am I *looking for* in the memory right now?"
- **key k** — "what *label* should I be found by?" (the address)
- **value v** — "what is my *content* if I am found?" (the payload)

```
qkv = qkv_proj(x)               # one learned table-mix, 384 → 3×(6×64)
   → queries, keys, values      # [B, H, T, d, 2] each
```

Then two small touches:

- **RoPE** rotates each q and k by an angle that depends on the word's
  *position in the sentence* (model.py:373-384). Because rotation =
  phase (A4.2), after this two words' keys "match" more strongly the
  *closer* they are. That's how the model knows "nearby" without any
  extra machinery.
- **Key normalization** divides each k by its own length so every key has
  length 1 (model.py:399-400). Why: a key that's twice as long would
  write twice as hard and the memory would blow up to NaN. Equal-length
  keys write equally. (This is the 500M-run NaN fix, explainer §6.)

**Semantics of the three words, concretely.** Take the sentence
`The king sat on the throne`. At the word `throne`:

- its **value** v is "a place to sit, associated with royalty";
- its **key** k is the *address* it will be stored under — learned so
  that words like `king`, `royal`, `crown` will produce *similar* keys;
- its **query** q is what `throne` is *asking for* — learned to point at
  the address of "who is this throne for?" so it can *retrieve* the
  earlier `king`.

That's the whole design: **store under a learned address, retrieve with a
learned question.** The addresses and questions are learned from data —
the model discovers that "royal words" should share an address.

## C4. The memory step: WHEN it writes, WHEN it reads

This is the heart. Per head, per state, there's one 64×64 table `S`.
The model walks the sentence left to right. At **each word**, three
things happen, in this order:

### C4.1 WHEN it writes: *after* computing the word's read, *before*
the next word is processed (model.py:1373-1416)

```
1. S ← γ · S                    FORGET
2. old = S · k                  (peek: what's already stored at my address?)
3. u   = βw·v − βe·old          (the delta: my content minus what's there)
4. S ← S + u ⊗ k*               WRITE
```

Line by line, in words:

1. **FORGET**: multiply the whole notebook by `γ` (0…1, learned per word
   and per head). `γ=0.9` = "this notebook fades 10% per word"; `γ=1` =
   "never forget". The vault state has `γ=1` pinned.
2. **PEEK** (an inner product, A2): "what does my notebook currently
   say at *my* address k?" If the notebook already holds something
   close to my value, the peek ≈ v.
3. **DELTA**: the *correction* = what I want to store (v) minus what's
   already there (peek), each scaled by a learned gate (`βw` = how much
   to write, `βe` = how much to trust the peek). If the notebook
   already had it right, u ≈ 0 → **we write nothing.** If it had
   something wrong (a fact that changed: "capital is X" then "capital
   is Y"), u is the fix.
4. **WRITE** (an outer product, A3): add the correction, filed under my
   address k. Only the *direction of k* in the table is touched —
   other addresses are untouched (their keys are orthogonal to mine).

**Why write the *delta* and not the whole v?** Because the notebook
shares space. If you blindly write v every time, every repeat of a fact
adds noise to every other fact. Writing only the *error* means: new
facts land, repeated facts are a no-op, changed facts get corrected.
That one line is what makes long conversations work without the memory
drowning in repetition.

### C4.2 WHEN it reads: *the very next moment*, for the output of this
same word (model.py:1393)

```
5. out = S · q                  READ
```

An inner product: the *updated* notebook (including the write we just
did) dotted with this word's *query*. "Now that I've filed my own
content, what does the notebook say is relevant to *what I'm asking
for*?" The answer `out` is a 64-vector per head — a *mixture of every
fact stored so far, weighted by how well each fact's address matches
my question*.

> **The ordering is the point: read AFTER write, so a word can retrieve
> itself (useful for grammar) but never a future word (causality).**
> And the *next* word, processed after this one, will read the notebook
> that now includes this word. That's how information flows forward
> through the sentence — one write, one read, per word.

### C4.3 A tiny worked example (2 features, 1 state, real numbers)

Notebook starts empty: `S = [[0,0],[0,0]]`.

Word 1: `king`, key `k1=[1,0]`, value `v1=[5,1]`, γ=1, βw=βe=1.

```
peek = S·k1 = [0,0]                 (empty)
u    = v1 − peek = [5,1]
WRITE S += u⊗k1 = [[5,0],[1,0]]    (file [5,1] under address [1,0])
READ  out = S·q1, q1=[1,0] → [5,1]  (it retrieves its own value)
```

Word 2: `sits`, key `k2=[0,1]`, value `v2=[2,7]`, γ=1.

```
peek = S·k2 = [0,0]                 (address [0,1] is empty)
u    = [2,7]
WRITE S += u⊗k2 = [[5,2],[1,7]]    (file [2,7] under address [0,1])
READ  out = S·q2, q2=[0,1] → [2,7]  (it retrieves exactly [2,7])
```

Now the notebook `[[5,2],[1,7]]` holds **both** facts at once. Ask with
`q=[1,0]` (king's address) → `[5,1]`: king's value, *exactly*. Ask with
`q=[0,1]` (sits' address) → `[2,7]`: sits' value, *exactly*. **No leakage**
— because the two keys are *orthogonal* (at right angles: `[1,0]·[0,1] =
0`), each fact lives in its own "lane" of the table.

Now make the keys *similar* — word 3: `royal`, key `k3=[1,1]` (a mix of
king's and sits' directions), value `v3=[9,0]`, γ=1:

```
peek = S·k3 = S·[1,1] = [7,8]            (the notebook already has something here!)
u    = v3 − peek = [9,0] − [7,8] = [2,−8] (only the CORRECTION is written)
WRITE S += u⊗k3 = [[7,4],[−7,−1]]
```

Read back with the same key `q=k3=[1,1]`: `S·k3 = [7+4, −7−1] = [11,−8]` —
not exactly `[9,0]`; the error is `[2,−8]`. That gap is **interference**:
because `royal`'s key overlaps king's and sits' lanes, the other two facts
bleed into its answer. For contrast, a *blind* write (storing the full
value, no delta) of the same word would read back `[25,8]` — an error of
`[16,8]`, about twice as large. Two things in the design keep this survivable:

- **64 dimensions, not 2** (model.py: `head_dim=64`). In 64-D, random keys
  are *nearly* orthogonal, so overlap is small; in 2-D it's huge.
- **The delta rule** (C4.1): only the *error* `u` is written, so a fact
  that's already mostly there doesn't get re-pressed and amplify the
  interference. Blindly writing the full value every time would make the
  leakage grow with every repeated word.

The *mechanism* is exactly what you've now seen by hand: **writes
accumulate in the table; reads dot against it; each fact retrieves by
address match; similar keys share a lane and interfere a little.**

### C4.4 Then the six heads and three states combine (model.py:1106-1116)

Each of the 3 states produced its own `out` (vault, scratchpad 1,
scratchpad 2). The layer rotates each by a *learned angle* (a phase,
A4.2 — so it can boost or cancel a state's contribution) and adds them:

```
out_head = w0·e^{iφ0}·out_vault + w1·e^{iφ1}·out_scratch1 + w2·e^{iφ2}·out_scratch2
```

Then the 6 heads are concatenated back into 384 features (model.py:1230),
pushed through a learned mix (`o_proj`, model.py:1231), and **added to
the input x** (the residual). The layer's output is "the sentence, but
with each word's vector now a better summary of that word *plus* the
relevant facts from earlier words."

## C5. Sixteen layers, then the answer (model.py:1487-1579)

Layers 1…16 do C2–C4. Each layer's memory is *separate* — layer 1 keeps
its 18 tables, layer 2 its 18, etc. Early layers tend to store
local/syntactic facts; deep layers store more abstract ones (the model
learns this division).

After layer 16, each word's 384-vector is turned into **a score for all
50,261 words**: the vector is dotted with every row of the embedding
table (the *tied* head, model.py:1576-1579 — the same table used at the
door, C1). The word with the highest score is the model's guess for
"what comes next." A softmax turns the 50,261 scores into a probability
over the vocabulary.

> **So "knowledge" is not in one place.** It is (a) *baked into the
> weights* — every learned table (embeddings, projections, gates) — and
> (b) *active in the memory tables* S — the running summary of what the
> model has *seen in this conversation*. (a) is the model's book
> learning; (b) is its notebook for this particular discussion.

## C6. Where is YOUR prompt while it runs? (inference)

`generate()` (model.py:1674-1713) has two phases.

**Phase 1 — prefill.** Your whole prompt (say 500 tokens) is fed in at
once, like a training batch with `T=500`. Every layer runs C3–C4 over all
500 tokens (in the fast parallel "chunk" form, C7). At the end, **your
prompt is no longer stored as text — it has been folded into the 288
memory tables** (one set per layer). The text is gone; only its
*compressed meaning* remains, spread across the tables.

**Phase 2 — decode, token by token.** The model guesses word #501. It
feeds *that one word* back through all 16 layers. Each layer's memory
does the *same* read+write as C4 — but now with `T=1`, so each table
just gets one new entry and one new read. **The cost of one new word is
the same whether the conversation is 50 words or 50,000 words**,
because the notebook is a *fixed-size table*, not a growing list. (A
transformer would carry a growing list of past tokens — a KV cache —
that gets longer with every word; this model never does. That's the
reason it exists.)

So to answer "where is the user's discussion while they talk":

> **It's in the memory tables.** The prompt was absorbed into the
> tables during prefill; every new word the user (or the model) adds is
> absorbed the same way — one write, one read, per table. The tables are
> the *only* place the conversation history lives. Clear the tables and
> the model forgets the conversation entirely (it still knows English
> from its weights, C5).

## C7. Why training does the same math differently (the chunk form)

At *inference* C4 is a loop over 2048 words — fine, the GPU is patient,
and the memory is what we want. At *training*, the GPU wants *big
table-multiples*, not loops. So training unrolls the loop over a block of
`C=128` words and does the whole block with a handful of matrix
multiples. The two forms compute **identical numbers** (proven by
`v13/selftest.py` to 1e-7), so the model trained on the fast form runs
correctly on the slow form.

The one genuinely new math in the training form is the **triangular
solve** (model.py:1423-1435, explainer §9.3). It appears *only because of
the delta rule*: when you unroll 128 words at once, word #50's write
depends on word #49's peek, which depends on word #48's write… — a
chain. The chain is a system of 128 linear equations that's *triangular*
(word #n only depends on words < n, by causality), so it has one exact
fast solution. One solve replaces 128 sequential steps. That's the
**UT transform** — the standard trick for parallelizing delta-rule
memories, from the DeltaNet/linear-attention literature.

Everything else in training is just A5's five moves, at big sizes.

## C8. Backward: how the model *learns* (training only)

Training adds one more phase after the forward pass.

1. **Forward** (C1–C5): compute the 50,261 scores for each position.
2. **Loss**: for each position, "how wrong was the guess of the *next*
   word?" — the cross-entropy, computed in chunks so the huge
   `[16384 × 50261]` score table is never built at once (`fused_ce.py`).
   One small number: the average wrongness.
3. **Backward** (`loss.backward()`, v7/train.py:569): PyTorch walks the
   *record* of every operation from step 1 **in reverse** and, at each
   operation, asks: "if I'd nudged *this* table by a hair, how much
   would the loss have changed?" That answer is the **gradient** — a
   same-shape table of nudges. The chain rule multiplies the answers
   together along each path from loss to table.
4. **Step**: nudge every learned table a tiny bit in the direction that
   *lowers* the loss (the optimizer, v7/train.py:573). Repeat a million
   times and the tables become good.

Two things worth knowing:

- **Nothing about backward is special to this model** — it's the same
  chain rule for any PyTorch model. What *is* special is that the
  triangular solve has an *exact* backward (another triangular solve),
  so the delta rule's gradients are exact too, not approximated.
- **The gate-surprisal aux** (v7/train.py:354-431) is a small side loss:
  it tells the *protect gate* "you should have frozen the notebook on
  boring words and written on important ones" — using the per-word
  surprise as a proxy for "important." That's how the gate learns
  selectivity (C4.1's γ and the write-muting).

> **Forward = run the model. Backward = compute the nudges. Step = apply
> the nudges. Inference has no backward — the weights are frozen, only
> the memory tables move.**

---

# Part D — direct answers to your exact questions

## D1. "What is a feature?"

One number in a list of numbers that describes a word (A1). A word =
384 numbers = 384 features. The features don't have fixed human meanings;
the model learns what each one tracks. "Feature", "channel", "dimension"
are the same *kind* of thing (one number), but `dim` (384) and
`head_dim` (64) are two *different lengths* — see B8.

## D2. "An attention head is a dimension slice — is that right?"

Yes — 384 features sliced into 6 pieces of 64 (B5). The refinement: each
slice is not just a view, it's a **separate notebook with its own memory
table**, and the 6 notebooks cooperate. The slicing is done with a `view`
and a `transpose` (model.py:363-371); no numbers are lost.

## D3. "When and where does the memory get written, and read?"

Per layer, per head, per state, **once per word**, in this order
(C4.1, C4.2; code model.py:1373-1416):

```
FORGET   S ← γ·S
PEEK     old = S·k            ← a READ (inner product)
DELTA    u = βw·v − βe·old
WRITE    S ← S + u⊗k*         ← the WRITE (outer product)
READ     out = S·q            ← the READ that becomes the output
```

- **Written** right after the peek, before the next word is processed.
  The write is the *delta* (the correction), not the whole value (C4.1).
- **Read** twice: once as the peek (to compute the delta) and once as the
  output (with the query). The output read is what flows on to the next
  layer.
- **Where**: in the 288 tables `[K,B,H,d,d]`, one per (state, sentence,
  head, layer). The write touches only the slice of the table pointing in
  the key's direction (C4.1 step 4).

## D4. "What about knowledge — where is it?"

Two places (C5):

1. **The weights** (frozen during inference): every learned table —
   embeddings, projections, gates. This is the model's *book learning*:
   how to read English, what "king" means, how grammar works. It is the
   same for every conversation.
2. **The memory tables** (moving during inference): the running summary
   of *this conversation*. This is the model's *notebook*: what the user
   just said, what was established earlier in the discussion.

"Knowledge of the world" = (1). "Knowledge of this conversation" = (2).

## D5. "Where is the user's prompt / discussion while the model
reasons about it?"

In the memory tables (C6). The prompt was absorbed during prefill; every
new token (user's or the model's) is absorbed the same way — one write,
one read, per table. The tables are the *only* place the history lives.
Clear them and the model forgets the conversation (it still knows
English from the weights).

## D6. "What exactly is a phase, and what makes it a phase?"

A complex number is an arrow (A4). Its **magnitude** is the arrow's
*length*; its **phase** is the arrow's *angle*. The phase is a phase
because it is the part of the number that **adds when you multiply**
(`(a+i b)·(c+i d)` has angle = angle1 + angle2), while the magnitude is
the part that *multiplies*. That additivity is the whole job of phase:
it's how signals **line up (reinforce) or cancel** without changing
loudness (A4.1). Concretely in the model:

- a stored fact's address is partly its *phase*; asking with the same
  phase retrieves it, asking with the opposite phase cancels it;
- RoPE (C3) encodes *position* as phase, so "distance between words" is
  a phase difference;
- the K-state combination (C4.4) *rotates* states by learned phases so
  the model can boost one notebook and cancel another.

## D7. "Why does phase alone matter?"

Because it carries *relationships* without carrying *amount*. Two arrows
of equal length but opposite phase sum to zero; same phase sum to twice
the length. So phase is a **sign/alignment label**: it says "this fact
goes with that fact" or "this fact opposes that fact". Magnitude is the
volume; phase is the agreement. A 64-dim complex vector has 64 magnitudes
*and* 64 phases — 128 real numbers — and the phases are where the
model's fine-grained addressing lives (explainer §2, §7).

## D8. "What makes a magnitude a magnitude?"

It's the part of the complex number that **multiplies** (not adds) under
complex multiplication, and equals the Euclidean length `√(a²+b²)` of the
arrow (A4). It's "loudness": how much a signal contributes. In the code
it's `cabs` (`complex_ops.py:53-56`); the protect gate and the betas
operate on magnitudes (`cabs(x)`, model.py:261-274), and the key
normalization makes every key's magnitude exactly 1 (model.py:399-400) so
no key is "louder" than another.

## D9. "B H C S — what are the letters, finally?"

B = batch (8 sentences at once). H = heads (6 notebooks per layer).
C = chunk (128 tokens per parallel block, *training only*). "S" is not a
shape letter — it's the **state** (the memory table itself, `[K,B,H,d,d]`)
or the count of states K=3. Decode any shape by reading it as nested
"for each…" (B7).

## D10. "How does the code flow, in one breath?"

```
FORWARD (training or inference):
  ids → embed → [16 × (CGU then PAM)] → tied head → scores
  PAM = project(q,k,v) → per word: forget, peek, delta-write, read
         → rotate+sum over states → merge heads → add back

TRAINING adds:
  scores vs. next word → loss → backward (chain rule, reverse graph)
  → nudge every learned table → repeat

INFERENCE:
  prefill: run FORWARD once over the whole prompt (memory absorbs it)
  decode:  run FORWARD once per new word (T=1; memory updates in place)
  no backward; weights frozen; only the tables move
```

The code for each line: C1 (embed), C2 (block), C3 (project),
C4 (memory step), C5 (head), C6 (generate), C7 (chunk form), C8
(backward).

---

# Part E — your follow-up questions, with the actual code pasted in

Each answer below pastes the *real* line from the file, names the file and
line, and then explains it in words. No more "see model.py:NNN" — the code
is right here.

## E1. "Batch — do we pick 8 sequences and feed them all at once?"

**Yes, exactly** — but one important correction: they are *stacked side by
side*, not *concatenated into one long sequence*.

The dataloader grabs 8 independent sentences from the dataset. Each
sentence is its own list of 2048 token-ids. They are put into an array with
a NEW first axis:

```
input_ids : [B, T] = [8, 2048]
            row 0 = sentence 0's 2048 ids
            row 1 = sentence 1's 2048 ids
            ...
            row 7 = sentence 7's 2048 ids
```

They are **not** glued into one 16,384-token sequence. Row 0 never talks to
row 1. Every operation the model does is "do this *per row*", so the GPU
works on all 8 rows *simultaneously* (that's the whole point of a batch —
the GPU has thousands of cores and one sentence isn't enough to keep them
busy). All 8 rows use the *same learned tables* (the weights); that's the
only thing they share.

> **Batch = 8 independent sentences, stacked as rows, processed in
> parallel, never mixing with each other.** `B` is just the count of rows.

## E2. "Is T a 1-D array where each token is… what?"

`T` is the *axis that counts tokens*. What sits at each position along that
axis **changes as you go deeper**:

```
stage                    shape            what one (b,t) entry IS
───────────────────────  ───────────────  ───────────────────────────
token ids                [B, T]           ONE integer (the word id)
after embedding          [B, T, dim]      a VECTOR of 384 numbers
complex (what the model uses) [B,T,dim,2] a VECTOR of 384 complex
                                          numbers (384 pairs of real,imag)
```

So: at the *input*, yes, each token position is basically one number (its
id). The *moment* it passes through the embedding table, that one id becomes
a whole 384-number vector. From then on, "the token" means "the 384-vector
at this position", not a single number.

Tiny concrete picture (1 sentence, 4 words, dim=3):

```
ids        [1,4]      = [ [ king,  sits,  on,  the ] ]        (just ids)
embed      [1,4,3]    = [ [0.2,-1.1,0.4],   ← "king" = 3 numbers
                           [0.5, 0.0, 0.9],   ← "sits" = 3 numbers
                           [0.1, 0.7,-0.2],   ← "on"
                           [0.9, 0.3, 0.1] ]  ← "the"
```

The `T` axis is the *rows of that inner list*: 4 positions, each holding a
3-vector. The model's job is to refine those 4 vectors so each one encodes
"this word *given its neighbors*".

## E3. "Why does [K, B, H, d, d, 2] have d TWICE?"

Because the memory is a **table (a matrix)**, not a list. A *list* has one
length; a *table* has two — rows and columns.

- A **list** of token vectors: `[B, T, dim]` — one `dim` (the length of
  each vector). That's your "feature list".
- A **table**: `[d, d]` — `d` rows **and** `d` columns. That's the memory.

The two `d`s play different roles (this goes back to A3, the outer
product):

```
memory S is d×d.  S[i, j]  =  "value-feature i  ↔  key-feature j"
                     ^         ^
                     |         └── which KEY feature (the address channel)
                     └──────────── which VALUE feature (the content channel)
```

When you *write* a fact "key k → value v" you do `S += v ⊗ k*` — the outer
product of a d-vector and a d-vector, which fills **all d×d cells**. When
you *read* with a query q you do `S · q` — every row dotted with q, giving
a d-vector back. So the memory genuinely *is* a d×d table, and that's why
`d` appears twice.

Decode the full shape now: `[K, B, H, d, d, 2]` = "for each of K states, B
sentences, H heads: one **d×d table** of complex numbers (the trailing 2 is
real/imag)." The two `d`s are the table's rows and columns.

## E4. "dim and head_dim — are they the same? (you said yes, I doubt)"

**You're right to doubt — they are NOT the same.** `dim` is the token's
full width; `head_dim` is one head's slice of it. Here's the actual code
that defines both, `v13/model.py:162-166`:

```python
self.num_heads = cfg.n_heads          # 6   ← how many heads
self.head_dim = cfg.head_dim          # 64  ← features INSIDE one head
inner = cfg.n_heads * cfg.head_dim    # 6 × 64 = 384
self.inner_dim = inner
self.dim = cfg.dim                    # 384 ← features in a WHOLE token
```

See it: `inner = 6 × 64 = 384`, and `dim = 384`. So:

```
dim (384)  =  the whole token vector
head_dim (64) = one head's slice
6 heads × 64 head_dim = 384 dim
```

**`dim` is the whole; `head_dim` is the slice.** 6 heads tile the 384-wide
token into 6 slices of 64. (I was sloppy earlier calling them the same —
B8 now states this correctly.) When you read the code, `cfg.dim` / `self.dim`
means 384; `cfg.head_dim` / `self.head_dim` / `d` means 64.

## E5. "`x ← x + CGU(norm(x))` — what IS x, what shape, what info?"

Here's the exact code, `v13/model.py:1457-1484` (the whole block, with the
comments trimmed):

```python
def forward(self, x, pam_state=None, step_offset: int = 0):
    # x arrives as:  [B, T, dim, 2]  = [8, 2048, 384, 2]
    # "for each of 8 sentences, 2048 positions: a 384-complex-number vector"

    cgu_out = self.cgu(self.norm1(x))        # rethink each word alone
    x = x + cgu_out * self.cgu_scale         # ADD it back (residual)

    pam_in = self.norm2(x)
    pam_out, new_state = pam(pam_in, state=pam_state, step_offset=step_offset)
    x = x + pam_out * self.pam_scale         # ADD it back (residual)
    return x, new_state
```

**What `x` is at each moment** (shape is *always* `[B, T, dim, 2]` — the
shape never changes inside a block; only the *information* does):

1. **Entry**: x = "the sentence so far", as 2048 × 384-vectors. After 5
   layers, each word's vector is no longer just "what the word means" — it's
   "what the word means **plus** everything the first 5 layers noticed
   about it and its neighbors". Each layer *adds a refinement*.
2. **`norm1(x)`**: rescales each vector's *loudness* so it's not too big or
   too small (stability), leaving its *direction* untouched. A housekeeping
   step before the computation.
3. **`self.cgu(...)`**: a few learned table-mixes on *one word at a time* —
   no neighbor involved. Output: a proposed refinement of the word's vector.
4. **`x + cgu_out`**: keep the old x, *add* the refinement on top. The
   output is "old understanding + new insight".
5. **PAM** does the same pattern but its refinement *uses the memory*
   (neighbors). `x + pam_out` = "old understanding + what memory says is
   relevant".

> **x is the running summary of the sentence.** Every layer adds a little
> more understanding to it; nothing is ever thrown away, only added.

### E5.1 "Residual — what's the opposite?"

The *opposite* of `x = x + f(x)` is **`x = f(x)`** — replacing x with the
new result and *forgetting the old one*. That's what a plain (non-residual)
network does, and it's a problem:

- **Information loss.** If `f` is a bad guess, you've destroyed the
  original signal and can't get it back. Residual `x + f(x)` guarantees the
  old signal survives no matter what `f` does.
- **The "highway".** When `f(x)` ≈ 0 (the layer has nothing useful to add),
  `x + f(x)` = `x` — the sentence passes through *unchanged*. A stack of 16
  layers can therefore "do nothing" in some layers without any damage. The
  *opposite* (replace) has no such escape hatch: every layer must produce a
  full replacement, and errors compound 16×.
- **Gradients.** Backprop has to carry the loss *back* through 16 layers.
  The `+ x` is a pure pass-through for gradients (its derivative is 1), so
  the signal reaches layer 1 intact. The replace-style `x = f(x)`
  multiplies the gradient by f's derivative 16 times, and it usually
  vanishes to zero. (That's why "residual connections" — He et al., 2015 —
  made deep networks trainable.)

There's also a *scale* on each add (`cgu_scale`, `pam_scale`,
model.py:1448/1455): a single learned number multiplying the refinement.
`pam_scale` starts at **0.1** — "the memory layer's opinion is worth 10%
at first" — so the cheap, reliable CGU path dominates early training and
the memory warms up gradually. A stability trick, not math.

## E6. The qkv line, with the code and the *why* of every piece

The actual code, `v13/model.py:362-367`:

```python
if self.fused_qkv:
    qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, num_heads, head_dim, 2)
    queries = qkv[:, :, 0].transpose(1, 2).contiguous()
    keys    = qkv[:, :, 1].transpose(1, 2).contiguous()
    values  = qkv[:, :, 2].transpose(1, 2).contiguous()
```

**Line 1: `qkv = self.qkv_proj(x)`.** `qkv_proj` is a learned table
(`ComplexLinear(cfg.dim, 3*inner)`, model.py:183) that multiplies each
word's 384-vector by a learned matrix, producing `3 × 6 × 64 = 1152`
complex numbers per word. Shape: `[B, T, 3·H·d·2]` = `[8, 2048, 2304]` —
one flat list of 2304 numbers per word. *Why one table instead of three:*
three separate 384→768 multiplies and one 384→2304 multiply are the same
amount of math; one big multiply is faster on GPU (one kernel launch, better
memory use).

**The `.view(batch_size, seq_len, 3, num_heads, head_dim, 2)`.** `view`
does **no math** — it just *relabels* the flat 2304-number list into a
6-axis shape: "these 2304 numbers are really: 3 groups (q/k/v) × 6 heads ×
64 channels × 2 (real,imag)". The numbers don't move; only the address
labels change. Think of a 2304-slot shelf re-stickered as 3 shelves of 6
bins of 64 pairs.

**`qkv[:, :, 0]`** picks group 0 = the *queries* (group 1 = keys, 2 =
values). Shape: `[B, T, H, d, 2]` = `[8, 2048, 6, 64, 2]`.

**`.transpose(1, 2)`** swaps the *time* and *heads* axes:
`[B, T, H, d, 2] → [B, H, T, d, 2]`. Again no math — just a different
*order of address labels*. **Why bother?** Because the memory step does one
table-multiply **per (sentence, head)** — 8×6 = 48 independent multiplies.
PyTorch's `@` operator treats *all axes before the last two* as "batch
axes" (do this many independent copies). If the shape is `[B, H, T, d, 2]`,
the 48 (B,H) combos are the batch axes and the `(T,d)` × `(d,?)` multiply
happens 48 times in ONE call. If we'd left it `[B, T, H, d, 2]` we'd need
48 separate calls or a slow loop. **transpose moves the "do these
independently" axes to the front.**

**`.contiguous()`** — the transpose is a *view*: the numbers are still in
old memory, just addressed differently (like a photo viewed through a
tilted frame). Some fast kernels need the numbers in *row order*, so
`.contiguous()` makes one physical copy in the new layout. One copy, paid
once, enables all the fast batched multiplies after it.

> **The whole line, in one breath:** multiply each word by one learned
> table to get its (query, key, value); relabel the flat result into
> (3, heads, channels, real/imag); take the q/k/v groups; move heads in
> front of time so the next multiplies batch over (sentence, head); make a
> dense copy. No numbers are invented or lost — only multiplied, relabeled,
> reordered, copied.

## E7. "RoPE: rotate by position — *how*, exactly, formula?"

Two pieces of code. First, the precomputed table,
`v13/complex_ops.py:286-291`:

```python
def build_rope_cache(max_len: int, head_dim: int) -> torch.Tensor:
    """Complex RoPE: e^{i·m·theta_k} for positions m and frequency bands k."""
    inverse_freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * inverse_freqs.unsqueeze(0)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)
```

Then the application, `v13/model.py:382-384`:

```python
rope_positions = self.rope_cache[step_offset:position_end]   # [T, d, 2]
queries = cmul(queries, rope_positions)      # rotate every q by its position
keys    = cmul(keys,    rope_positions)      # rotate every k by its position
```

**The formula, in words.** For a word at position `m` (0, 1, 2, …), for
each channel `j` (0 … 63) of its query/key vector, compute an angle:

```
angle(m, j)  =  m · θ_j        where   θ_j = 1 / 10000^(j/64)
```

and multiply that channel's complex number by `e^{i·angle}` — i.e. **rotate
it by `m·θ_j` radians, leaving its length untouched.** `cos/sin` in the
builder are just the real/imag parts of `e^{i·angle}`:
`e^{i·a} = cos(a) + i·sin(a)`.

**Why this gives "distance".**(Remember A4: multiplying complex numbers
*adds* phases.) The score between word m's query and word s's key is
`q_m · k_s*`. After RoPE, channel j of `q_m` has phase `m·θ_j` and channel
j of `k_s` has phase `s·θ_j`. When we dot them, the `k` gets conjugated
(phase *flipped*), so the combined phase is:

```
m·θ_j  −  s·θ_j  =  (m − s)·θ_j
```

**The absolute positions m and s disappeared; only the difference (m − s)
remains.** So the score depends on *how far apart* the two words are, not
where they sit in the sentence. The word at position 5 and the word at
position 500, *3 words apart*, interact exactly like any other pair 3 words
apart. That's the entire job of RoPE, in one line.

**The `10000^(j/64)` part.** Channel 0 gets `θ = 1` (rotates a full turn
every 2π ≈ 6 positions — a *fast* clock); channel 63 gets
`θ = 1/10000` (a full turn every ~62,800 positions — a *slow* clock). So
the 64 channels are like a **64-digit odometer**: the fast digits change
every few words (fine-grained "is this the next word?"), the slow digits
change every thousands of words (coarse "is this near the start of the
paragraph?"). Different distances light up different patterns of digits.
(10000 and 64 are just chosen constants; the *shape* of the idea — many
clocks at geometrically different speeds — is the essence.)

**Why `step_offset`** (model.py:374, 382): during inference the model feeds
one new word at a time. If the 37th word of the conversation arrives, its
position is 36, so the code slices `rope_cache[36:37]` — the correct angle
for that word. `step_offset` = "how many words came before this chunk".

## E8. "Key normalization — *why* does a long key blow up to NaN? Show the
code."

The code that does the normalization, `v13/complex_ops.py:108-117`:

```python
def cnormalize_vec(x: torch.Tensor) -> torch.Tensor:
    """Per-VECTOR unit norm across the complex head dimension.
    ...
    The delta rule's k-direction eigenvalue is gamma*(1 - beta_e * ||k||^2),
    so unit keys keep it a strict contraction (gamma in [0,1], beta_e in (0,1)).
    """
    mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
    return x / mag.unsqueeze(-1).unsqueeze(-1)
```

It divides each key by its own total length, so every key has length
exactly 1. (`mag` is the key's length: sum the squares of all 64 real+imag
parts, sqrt.)

**Now the NaN, traced through the actual write code.** The memory step,
`v13/model.py:1382` (forget) and `1397-1406` (write):

```python
memory_state = memory_state * decay_factor          # line 1382: S ← γ·S
...
key_conj = torch.stack([key_t[..., 0], -key_t[..., 1]], dim=-1)   # 1397: k*
outer = update ⊗ key_conj                                # 1398-1405
memory_state = memory_state + outer                     # 1406:  S ← S + u⊗k*
```

Focus on the part of the notebook that points *along the key direction* —
call its size `a` (a single number: "how much of S is in the k direction").
One step does two things to `a`:

1. **Forget:** `a ← γ·a` (line 1382 multiplies the whole table by γ).
2. **Write:** the write `u⊗k*` adds, along the k direction, the amount
   `u · (k*·k)`. And `k*·k` is just `||k||²` — the key's length *squared*.
   (Because `k*·k` sums `k[j]·conj(k[j])` over channels, each term
   `|k[j]|²`, total = the squared length.)

So after one step:

```
a  ←  γ·a  +  (stuff) · ||k||²
```

The part of the notebook in the k direction is **replaced by `γ` times
itself plus the new write scaled by `||k||²`**. The *repeated* effect on
the old content is the factor `γ − βe·||k||²` (the erase term inside the
delta, βe, multiplies the peek, which is proportional to `a·||k||²` — the
full derivation is in MATH_EXPLAINER.md §6). **A system of the form
`a ← (γ − βe·||k||²)·a + ...` stays bounded only if
`|γ − βe·||k||²| < 1`** — otherwise `a` multiplies by more than 1 each
step, growing forever: 2×, 4×, 8×, … until it overflows float32 → **NaN**.

Concrete numbers, vault state (γ = 1):

| keys | ‖k‖² | factor `1 − βe·‖k‖²` (βe = 0.05) | per-step effect |
|------|------|-----------------------------------|-----------------|
| unit (normalized) | 1 | 1 − 0.05 = **0.95** | shrinks 5% — stable |
| typical random 64-dim | ≈ 64 | 1 − 3.2 = **−2.2** | flips sign, grows 2.2× — **explodes** |

That's the whole story: a random 64-dim key has `||k||² ≈ 64` (64
channels, each contributing ~1), so the factor is −2.2 and the memory
doubles every step with a sign flip — NaN in a few hundred steps. This is
exactly how the 500M run died at ~57M tokens. Dividing every key by its
own length (`cnormalize_vec`) makes `||k||² = 1`, factor = 1 − βe ∈ (0,1),
a strict shrink every step. **Normalize = force every key to the same
"loudness" so no write can overpower the forget.**

(One subtlety, model.py:103 note: normalizing each *element* separately —
`qk_norm` — is NOT enough; that pushes `||k||² → 64`, not 1. You must
normalize across the whole 64-channel *vector*, which is what
`cnormalize_vec` does: the `.sum(-1)` sums over all 64 channels before the
divide.)

## E9. "So: we write tokens into memory, then predict the next token and
compare with the actual one — where exactly is the *next token* thing?"

You're 90% right, and the missing 10% is *where the comparison happens*.
Here's the exact sequence, with code.

**Step 1 — the input already contains the answers.** The dataloader hands
the model a *whole sentence* as input, and the labels are the *same
sentence shifted by one word* ("next-token labels"). For the sentence
`[The, king, sits, on]`:

```
input_ids : [The,  king,  sits,  on,  ...]
labels    : [ king, sits,  on,  ..., ...]     ← each label = the word at
                                                     the NEXT position
```

So position 0's label is the word at position 1; position 1's label is the
word at position 2; etc. The "next token" is not generated — it's just
"the input, one slot to the right". The shift happens in the data (the
labels array is the ids with the first column dropped and a sentinel
appended).

**Step 2 — the forward pass (C1–C5).** The model processes all T positions
*in parallel* (not one by one — the chunk form, C7). Crucially, because
each position can only read the memory built from positions *before* it
(the causal mask, C7), position 0's output has seen *nothing before it*,
position 1's output has seen only position 0, position 2's output has seen
positions 0–1, … **The memory step's "write after read" ordering (C4) is
what enforces this:** at position t, the read uses the notebook as it was
after writing position t−1 — never position t+1.

So after the forward pass, position t's final 384-vector is a summary of
"the sentence up to and including word t". The head (C5) turns that
summary into **a score for all 50,261 words** — a probability distribution
over "what could come next".

**Step 3 — the comparison (the loss), `v7/train.py:506-514`:**

```python
main_loss = self._raw_model.ce_from_lm(lm, labels, loss_mask=loss_mask, ...)
    # ↑ ce_from_lm: for EVERY position t, take position t's distribution
    #   over 50,261 words and compute how much probability it put on
    #   labels[t] (the actual next word). Average over all positions.
```

Concretely, at position 0 (input was just `The`): the model's
distribution might say `king: 0.4, queen: 0.3, the: 0.1, ...`. The actual
next word was `king`. The loss = "−log(0.4)" — a small number, because the
model was fairly confident about the right answer. At position 1 (input was
`The king`): the distribution says `sits: 0.02` but the actual word was
`sits` → loss = "−log(0.02)" — big, the model was wrong. **The total loss
is the average of these per-position "how surprised was I by the actual
next word" numbers.** (That's what "cross-entropy" / "NLL" means: surprise
measured in bits.)

**Step 4 — backward (C8)** turns that one average number into per-table
nudges, and the optimizer applies them. After millions of such nudges, the
tables assign high probability to the *actual* next words and low
probability to the wrong ones — that's all "learning" is.

> **The "next token" is a property of the *labels*, not of the model.**
> The model just outputs a distribution at every position; training checks
> that distribution against the input's next word and nudges. During
> *inference* there are no labels — the model takes the highest-probability
> word, appends it, and repeats (C6, phase 2).

## E10. The memory step, with the *actual* inner/outer-product code

C4 showed the step in symbols. Here it is in the real code,
`v13/model.py:1373-1416` (`_recur_step_delta`), with the shape of every
line. Shapes shown for **one state** (`[B,H,…]`; the K states are a batch
over the same code).

```python
def _recur_step_delta(self, memory_state, decay_gamma, value_t, key_t, query_t,
                      write_beta_t, erase_beta_t=None):
    # memory_state : [K, B, H, d, d, 2]   the d×d notebook (complex)
    # decay_gamma  : [K, B, H]            one γ per (state, sentence, head)
    # value_t, key_t, query_t : [K, B, H, d, 2]   this word's v, k, q
    # write_beta_t, erase_beta_t : [K, B, H]      the βw, βe gates
```

**Line 1381-1382 — FORGET** (scale the whole table):

```python
decay_factor = decay_gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [K,B,H]→[K,B,H,1,1,1]
memory_state = memory_state * decay_factor                            # [K,B,H,d,d,2]
```

`decay_gamma` is one number per (state, sentence, head), but the notebook
is `d×d×2` per (state, sentence, head). The three `.unsqueeze(-1)` pad the
number to `[K,B,H,1,1,1]` so it *broadcasts* — multiplies every one of the
d×d×2 cells by the same γ. (This is the `unsqueeze` trick from
MATH_EXPLAINER §12: a size-1 axis means "repeat me across these dims".)

**Line 1383-1390 — PEEK: an INNER product** (read the notebook at key k):

```python
predicted_real = (
    memory_state[..., 0] * key_t[..., 0].unsqueeze(-2)
    - memory_state[..., 1] * key_t[..., 1].unsqueeze(-2)
).sum(dim=-1)
predicted_imag = (
    memory_state[..., 0] * key_t[..., 1].unsqueeze(-2)
    + memory_state[..., 1] * key_t[..., 0].unsqueeze(-2)
).sum(dim=-1)
```

This *is* the dot product, written by hand for split-real complex numbers.
`memory_state[..., 0]` is the real part of S (shape `[K,B,H,d,d]`),
`key_t[..., 0].unsqueeze(-2)` is the real part of k as a *row*
`[K,B,H,1,d]`. Multiplying them pairs S's last column with k's entries —
the `d` of the sum — and `.sum(dim=-1)` collapses that `d` to one number
per (row, state, sentence, head). Result `predicted`: **`[K, B, H, d]`** —
one d-vector: "what the notebook currently says at address k". (The
real/imag cross-terms are just the complex rule `(a+ib)(c+id) =
(ac−bd) + i(ad+bc)` — A4 — applied to a table·vector.)

**Line 1392-1396 — DELTA:**

```python
write_expanded = write_beta_t.unsqueeze(-1)      # [K,B,H] → [K,B,H,1]
erase_expanded = erase_beta_t.unsqueeze(-1)
update_real = write_expanded * value_t[..., 0] - erase_expanded * predicted_real
update_imag = write_expanded * value_t[..., 1] - erase_expanded * predicted_imag
update = torch.stack([update_real, update_imag], dim=-1)   # [K,B,H,d,2]
```

Plain elementwise math: `u = βw·v − βe·predicted`, channel by channel.
`predicted` is the notebook's current answer for this key; `v` is what
this word wants to store; `u` is the *correction*.

**Line 1397-1406 — WRITE: an OUTER product** (file the correction under k):

```python
key_conj = torch.stack([key_t[..., 0], -key_t[..., 1]], dim=-1)  # conj(k)
outer_real = (
    update[..., 0].unsqueeze(-1) * key_conj[..., 0].unsqueeze(-2)
    - update[..., 1].unsqueeze(-1) * key_conj[..., 1].unsqueeze(-2)
)
outer_imag = (
    update[..., 0].unsqueeze(-1) * key_conj[..., 1].unsqueeze(-2)
    + update[..., 1].unsqueeze(-1) * key_conj[..., 0].unsqueeze(-2)
)
memory_state = memory_state + torch.stack([outer_real, outer_imag], dim=-1)
```

This *is* `u ⊗ k*`, written by hand. `update[..., 0].unsqueeze(-1)` is the
real part of u as a **column** `[K,B,H,d,1]`; `key_conj[...,
0].unsqueeze(-2)` is the real part of k* as a **row** `[K,B,H,1,d]`.
Multiplying a d-column by a d-row produces a **d×d table** — that's the
outer product (A3), channel i of u paired with channel j of k*, all d×d
cells at once. The result `outer` is `[K,B,H,d,d,2]` — same shape as the
notebook — and gets *added* to it. **This is the write: two d-vectors
(u and k) in, one d×d table added to the memory out.**

**Line 1407-1415 — READ: an INNER product** (the output, at query q):

```python
state_query_real = (
    memory_state[..., 0] * query_t[..., 0].unsqueeze(-2)
    - memory_state[..., 1] * query_t[..., 1].unsqueeze(-2)
)
...
output = torch.stack([state_query_real.sum(dim=-1), state_query_imag.sum(dim=-1)], dim=-1)
return output, memory_state
```

Same shape dance as the peek, but with `query_t` instead of `key_t`, and
the `.sum(dim=-1)` at the end collapses the d into one number per (state,
sentence, head, output-channel): `output` is `[K, B, H, d, 2]` — "the
notebook's answer to this word's question".

> **The whole step, counted:** two inner products (peek at k, read at q),
> one outer product (write u⊗k*), two scalar scales (forget by γ, gate by
> βw/βe), and a handful of shape-paddings so the one-number gates reach the
> d×d table. That's it — nothing else happens in the memory.