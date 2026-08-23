# V13 — Recall & Reasoning (R&D) analysis

Lab note, 2026-08-23. Goal: verify (against actual code) a set of proposed
levers for better **fact recall / reasoning**, dissect the v11 vs v13
checkpoints/models, and pick what to actually do. The 500M mission run is
**live and untouched** throughout (tmux `v13_500m`, eager B8/C128); decision
point is the 100M Wiki + KV-recall + gate-Δ probe.

Run state at time of writing: step ~2475, loss 4.74 @ 40.5M tok, ~4.8K tok/s,
**below** the v11 round-1 reference at matched tokens. First `latest.pt`
lands at step 5000 (~82M).

## The four proposed levers — code-verified

### Claim 1 — "No recall/reason rows in the mix; add a small slice (90/6/4)."
**VERDICT: TRUE, and it is the biggest missing gradient. But do it in the
NEXT run, not this one.**

- Current launch is `--pretrain_sources dclm,fineweb,smoltalk2_mid
  --pretrain_weights 48,48,4` (see `v13/tmp/launch_v13_500m_r1recipe.sh`).
  100% web/chat. Zero "store now, answer later" loss.
- A synthetic recall curriculum **already exists and is wired**:
  `v7/data.py:1305+` (passkey / single-kv / multi-kv generators, vocab
  deliberately DISJOINT from `memory_probes/behavioral.py` so eval stays
  held-out). The trainer accepts `recall`/`reason` as sources — they are just
  not in this mix.
- Why not this run: 48/48/4 matches r1 **exactly** so this run is a clean
  V13-vs-V11 *quality* A/B. Adding recall now confounds the comparison. The
  old 75/25 synthetic run sat at chance, but that was 11M params on 100%
  synthetic (under-trained) — a small slice on a rich web base is a different
  proposal and the right test of "does the vault learn to store."

### Claim 2 — "Unit-norm keys only in the erase/mass term; raw keys for readout."
**VERDICT: TRUE — this is the most important finding — but the fix is more
involved than "just don't normalize the read key," because the same keys build
the shared state.**

- `v13/model.py:399-400` (`_project`): when `delta_key_norm=True` and
  `write_mode=='delta'`, `keys = cnormalize_vec(keys)` — applied **once** to
  the single shared keys tensor, *after* RoPE + phase addressing.
- That same tensor is used in `_forward_multistate_delta_fused` for **all four**
  of: (a) key-gram mass `key_gram` (lines 985-988), (b) in-chunk query-key
  scores `query_key` (989-992), (c) the erase read `state_key = k@S` (1003-1010),
  (d) **state construction** `S += update ⊗ k` (1096-1103).
- So the readout is `q@S` with `S = Σ update_t k̂_t^H` → retrieval score is
  `q·k̂`, a **pure cosine**. Key magnitude is stripped from retrieval. This is
  exactly the v6 QK-norm / repetition failure (magnitude as an importance
  signal removed), re-introduced as a *side effect of the 500M NaN fix*.
- **The subtlety the proposal missed:** stability and readout share `S`.
  - Stability needs `‖k‖²=1` in the *erase* term (eigenvalue `γ−βe‖k‖²`) and
    in *state construction* (the write is also `k^H`).
  - Readout wants raw `k` so `q·k` keeps magnitude contrast.
  - You cannot just feed raw keys to readout while `S` is built from them —
    `S` is one tensor. The clean split: **unit keys for mass + state
    construction (stability); raw keys for the in-chunk `query_key` score
    (readout contrast).** The carry term `q@S` (1085-1092) still sees
    unit-built `S`, so restoring its magnitude needs either a parallel
    raw-key state or a `‖k‖` rescale of the readout — a real design decision,
    not a one-liner.
- **Do NOT do this preemptively.** The model is currently *beating* r1 (which
  had no key-norm) on CE. Only switch to raw-key readout **if** the 100M probe
  shows CE on-track but recall at chance. Gated on `v13/selftest` +
  ckpt-vs-no-ckpt grad equiv (the non-negotiable).

### Claim 3 — "Supervise the gate with facts, not only surprisal."
**VERDICT: TRUE that it is a coarse proxy, but the gate is supervised in the
right DIRECTION and also gets CE gradient — so this is a lower-priority,
probe-gated lever.**

- `v7/train.py:354-431` (`_gate_surprisal_loss`): target =
  `sigmoid(sign·(median_ce − surprisal)/τ)`, **detached**; BCE vs per-layer
  protect prob; λ=0.1, τ=1.0. It is a filler/content proxy, not a fact signal —
  the gate learns "easy token → freeze," not "name/date → protect." Confirmed.
- **Nuances the proposal understated:**
  1. `sign=+1` (default) is already the **recall-oriented** direction:
     low-surprisal (filler) → high protect (freeze), content → write. So the
     aux pushes the gate the way we want; it is just coarse.
  2. The gate is **not** trained only by the aux. `protected_values =
     values·(1−p)` and `decay_gamma = base·(1−p)+p` (`model.py:454-469`) feed the
     trunk, so the main CE backprops through the protect gate too. The aux is a
     *selectivity prior* on top, not the only signal.
- Action: **only** if the 100M probe shows gate Δ ≈ 0 (no selectivity) do we
  drop λ 0.1→0.05 and/or add a tiny fact-contrastive head. Do not raise λ.

### Claim 4 — "Route facts into the vault; only state 0 has γ≡1, so 2/3 of
writes decay; routing not content-aware, phase init zero."
**VERDICT: PARTIALLY TRUE, and the headline mechanism is MISREAD.**

- Confirmed settings: `vault_state=True, vault_state_idx=0` (state 0 pinned to
  γ=1, `model.py:448-453`); `state_compete=False` → `routing_weights=ones`
  (`model.py:336`); `phase_init='zero'` → phase_proj weights+bias zero
  (`model.py:306-308`). So initially the K=3 states are **undifferentiated**
  (all phase 0, all routing 1) — K=3 degenerates toward K=1 in *specialization*.
- **The misread:** "only state 0 has γ≡1 … 2/3 of writes decay" implies facts
  do **not** reach the vault. They do. `model.py:1020-1023` **broadcasts the
  write to all K states**, and the state update `model.py:1105`
  (`memory_state = memory_state·α + state_chunk`) applies to all K, with
  per-state decay `α`. State 0's `α≡1` → the vault copy **persists**; only the
  copies in states 1/2 decay. So facts DO land in persistent storage.
- **Emergent division of labor (worth noting):** the erase term is per-state
  (`state_key = k@S_k`, 1003-1012), so the net write to a state is
  `βw·v − βe·(k@S_k)`. In the vault (large `S_0`), `k@S_0` is large → the
  vault write is **self-limiting** (writes the residual / error-corrects); in
  the fresh decayed states, `k@S_k` is small → they take the full write. The
  delta rule already routes "residual to vault, full to fresh" — no explicit
  routing needed for the vault to accumulate facts.
- **Valid residual:** with phase_init=zero + no competition, the three states
  are not yet specialized, so the full K=3 *retrieval* capacity isn't exploited.
  `phase_init='spread'` / content-aware routing (`routing_content_aware`,
  `state_compete`) are legitimate **later** levers — but lower priority than
  the claim implies, because the vault is already receiving facts.

## Checkpoint / model dissection

- **v11 round-1 weights are NOT on this disk.** `find` for v11/e3_k3/pretrain
  `.pt` returns only the duplex-TTS v11 saves
  (`checkpoints_v11_duplex_100m_tts_pam_t2s*`) — a different task. So the v11
  dissection is **config-level from code**, not weight-level.
- **v11 `v11_e3_k3_chat`** (`v11/model.py:1345`): `n_states=3,
  state_dt_spread=2.0, vocab=50261, gate_content_aware=True`. **Additive**
  write, **no** delta, **no** vault, **no** write-phase-address, **no**
  delta_key_norm, **no** gate-surprisal. (The `v11_e3_k3_chat_recall` preset at
  1356 adds `gamma_floor=0.98` + `gate_surprisal_lambda=0.1` — the V12 recall
  program — but round-1 used the plain `v11_e3_k3_chat`.)
- **v13 `v13_e3_k3_selective`** = the v11 additive twin **plus** the full
  selective stack: `write_mode=delta`, `vault_state` (state 0),
  `write_phase_address`, `delta_key_norm`, `delta_erase_beta_cap=0.95`,
  `gate_surprisal_lambda=0.1`. This is the "novel" delta.
- **Current v13 weights:** not yet dissectable — ckpt dir is empty until
  step 5000 (`latest.pt`, ~82M). At the 50M/100M probe, measure:
  (1) `protect_gate` bias drift from −3.0 (did selectivity turn on?),
  (2) `phase_proj` weight norm per state (did the K states differentiate from
  the zero init?), (3) per-state effective rank of the PAM state (is one state
  dominating / are 2/3 dead?), (4) KV-recall @2048 vs the 8-way chance 12.5%.

**First dissection — step 5000 / ~82M (2026-08-23 18:11, `latest.pt`):**
- **Wiki PPL 325.76** (247,808 tok, eval_checkpoints B2). Trajectory only —
  the < 25.77 target is at 500M.
- **Protect gate: selectivity has NOT turned on.** Bias −3.0 → −2.69…−2.87
  (all 16 layers), mean protect prob **~0.056** vs 0.047 init. The gate writes
  on ~94% of tokens; gate Δ ≈ 0 so far.
- **Phase: states still undifferentiated.** `phase_proj` wnorm ~0.56–0.73,
  bnorm ~0.003–0.006 every layer — phases ≈ 0, so K=3 reads ≈ 3× the same
  state (redundant capacity, not specialized).
- **Erase (as measured at 82M):** bias +0.013 → βe ≈ 0.50 — see the 164M
  section for the init/correction (the "effectively off, bias ~−3.0" bullet
  originally written here was a misread of this same +0.013 number).

**164M re-dissection (step 10000, 22:54):**
  - **Wiki PPL 211.66** (was 325.76 @82M — 35% drop over 82M, on trajectory).
  - **Erase — CORRECTION, fully settled:** the running preset
    `v13_e3_k3_selective` HAS `delta_erase_gate=True` (`model.py:1835`), so
    `erase_beta_proj` bias **inits to −3.0** (βe ≈ 0.047, "starts
    additive-like", `model.py:223-227`). Both 82M and 164M checkpoints show
    bias +0.01 → **βe ≈ 0.50**: the model **learned erase ON within the first
    82M** (bias −3.0 → +0.01). The original 82M note misread its own output
    (which already printed +0.013) as "still ~−3.0, effectively off". The
    learned erase gate works exactly as designed; cap 0.95 far from active.
  - **Interpretation (corrected):** sub-r1 CE comes from CGU + PAM delta
    writes (βw ≈ 0.5, βe learned to ≈0.5), with the *selective* levers
    (protect/phase) still near-init.
  - **Protect gate: still flat.** Bias −2.69…−2.85, mean protect prob
    0.055–0.064 — essentially unchanged from 82M. The gate-surprisal aux
    (λ0.1) has NOT turned on selectivity in 82M more tokens.
  - **Phase: still undifferentiated, but moving.** `phase_proj` wnorm grew
    ~0.6→0.72–0.95 (slow), bnorm still ~0.003–0.008 → phases still ≈0.
  - **`write_phase_proj` (Stage-6 key-conditioned binding): wnorm ~0.08–0.19,
    bnorm ~0.0001–0.004** — the phase *addressing* of V/Q is still ≈0; the
    "bind V to ψ(k)" mechanism is effectively dormant.
  - **Conclusion:** at 164M the model matches r1 on CE while its novel
    selective machinery (protect, phase, write-phase) is all still near-init.
    It is winning on the *delta write* + CGU, not on selectivity. The recall
    levers (raw-key readout, data slice, gate prior) are the right next-run
    changes; none are indicated by the CE curve, which is on-track.
- **generate() @82M ckpt** (120 tok, T=0.8): coherent English, **no
  repetition loop**, but factually garbled (Cambridge → "FAA / Royal Society
  for Human Services"). Healthy 82M behavior; not a quality verdict.
- **100M verdict (19:10): PASS** — 4.36 @ 100.4M vs r1 4.36@100M (gap 0.0).
  Probe battery above ran on the step-5000 ckpt (82M, the latest saved at
  verdict time); the step-10000 ckpt (~164M) is the next probe opportunity.

## Picks — what to actually do (ordered)

1. **Leave this 500M run alone** (no LR / warmup / batch / data change). It is
   the clean V13-vs-V11 quality A/B and it is winning. Confounding it now
   throws away the comparison.
2. **At 100M, run the full probe**: Wiki PPL **and** KV-recall **and** gate Δ,
   plus the weight dissection above. This is the decision point — CE being
   on-track but recall at chance is the exact signature that points at the
   architecture/data (not LR).
3. **If recall is at chance and CE is good → raw-key readout (split key-norm).**
   The single most likely bounded recall fix: unit keys for mass + state
   (keep the NaN fix), raw keys for the in-chunk query-key score. Gated on
   `selftest` + ckpt-vs-no-ckpt grad equiv. This is a follow-up run, not a
   hot-patch to the live one.
4. **Next run: add a small recall+reason slice (~90/6/4).** The biggest missing
   gradient. Kept small so the web base (and the quality comparison) survives.
5. **Gate: only if Δ≈0 at the probe** → λ 0.1→0.05 and/or a tiny
   fact-contrastive head. Not preemptive.
6. **Phase/routing specialization (`phase_init='spread'`, content-aware
   routing): only if the states measure undifferentiated** (low effective rank
   / one state dominating). The vault already receives facts, so this is a
   capacity/specialization lever, not a "facts aren't stored" fix.

## What NOT to do on this curve
- Raise LR (6e-4) or shorten warmup — CE is already ahead of r1; the gap is
  recall, which is a different gradient.
- Bigger batch "for SNR" — batch is already the *measured* speed optimum and
  larger is slower per token here.
- `gamma_floor=0.98` — the vault already pins state 0 to γ≡1; redundant.
- Turning `delta_key_norm` off without the erase cap — that is the 500M NaN
  path.

## One-line summary
The other model got 3.5/4 right: the data gap (1) and the key-norm readout
loss (2) are real and the top levers; the gate point (3) is a coarse-proxy
nuance, not a missing signal; and the vault point (4) misreads the math —
writes **are** broadcast to the vault, so facts already reach persistent
storage. Ship order: keep this run clean → 100M probe → (if recall flat)
raw-key readout + small recall data slice in the next run.
