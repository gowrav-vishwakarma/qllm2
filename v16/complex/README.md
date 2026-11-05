# v16/complex: the complex PAM arm with the paper body's parts (cx_m2)

User 2026-11-05: keep all complex work here. The v13 code (`v13_sempty/model.py`, `complex_ops.py`) is unchanged; the only shared
touch is the two-line route in `v14/hook.py:build_lm` (a complex config with any ported part builds `CxLM`).

- `model.py`: `CxPAMLayer` (delta via the realified fused kernel + complex decode reference, chrono, out_gate, reset mask, M2
  exposure), `CxSharedRead`, `CxBlock` (sparse notebooks, explicit MLP width), `CxLM`. Not ported: E1 echo.
- `tests/cx_parity.py`: layer parity (realified prefill vs complex stepwise; prefill + decode; chrono / gate identity at init; reset).
- `tests/cx_m2_parity.py`: model parity (CxLM M2 prefill + decode; readers identity at init; build_lm routing).
- Experiment notes: `v16/complex/EXPERIMENTS_CX.md`; lineage rows in `v13_sempty/EXPERIMENT_LINEAGE.tsv` (parent ctl_pam).
