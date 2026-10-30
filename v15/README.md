# v15 — Grove: a mind as an economy of living units

v15 is the third lane. v13 matures PAM, v14 adds new mechanisms to it. v15
starts from nothing: no attention, no state-space filter, no phase-associative
matrix memory, no complex numbers. It rethinks the **unit** and lets the
architecture follow from the unit.

**The unit is an organism, not a weight and a bias.** A tree has a *canopy
share* (its part of a conserved budget that the trees of a grove compete for),
a *root* (a prototype it grows toward the inputs it wins) and a *fruit* (a
prototype of what followed the inputs it won). The recurrence of the model is
**selection**: evidence for each tree accumulates, and the grove's shares are
the softmax of the evidence (with full retention this is exactly the Bayesian
posterior over K hypotheses; with less, a forgetful one). Forgetting is
relative: a tree loses share only when another gains. **There is no decay
constant on the state.** Growth happens in **seasons** (every C tokens), which
is what makes training parallel on a GPU and makes decode equal to prefill.

| file | what |
|---|---|
| `EXPERIMENTS_V15.md` | the lab notebook: the meditation it came from, the design and maths, what is new vs known, the pre-registered experiments G0–G3 |
| `EXPERIMENT_INDEX.md` | one row per idea: status, verdict, hash, rerun recipe |
| `SCRATCHPAD.md` | handover: state of play, what runs where, next step |
| `grove.py` | `GroveLayer` (season form + token-by-token `step`), `GroveLM` (RMSNorm → Grove → MLP blocks, tied head) |
| `selftest.py` | parity season == decode, simplex, gradients, O(1) state, toy in-context lookup |

    .venv/bin/python -m v15.selftest        # CPU, ~1 min; prints PASS

Ground rules are v14's (micro-test first, judge on holdout, one variable per
arm, bars written before the run, failed paths removed and indexed, logs named
`logs/v15_<tag>_<hash>_<YYYYMMDD_HHMM>.log`, sync by commit → push origin →
pull). Plain torch for the prototype; the sempyt port comes when the organ
earns an LM run.
