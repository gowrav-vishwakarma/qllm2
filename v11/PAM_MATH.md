# V11 PAM Math Probes — moved

This documentation has moved to **[`memory_probes/README.md`](../memory_probes/README.md)**.

The probe suite is now **`memory_probes/`** — a standalone evaluation framework for recurrent matrix memory, not V11-specific tests.

```bash
./scripts/run_memory_probes.sh
.venv/bin/python -m memory_probes --all
```

Legacy entry points (`v11.pam_math`, `v11.pam_math_language`, `run_v11_pam_math.sh`) still work but are deprecated.
