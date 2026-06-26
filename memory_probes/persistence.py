"""Association persistence vs distance under decay and filler writes."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from memory_probes.core import (
    outer_v_kstar,
    pam_step_additive,
    rand_unit_complex,
    retrieve_score,
)


def test_persistence(
    d: int = 64,
    distances: Sequence[int] = (64, 128, 256, 512, 1024, 2048),
    gammas: Sequence[float] = (0.99, 0.995, 0.999, 1.0),
    filler_writes: int = 1,
    seed: int = 42,
    adapter=None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    k_needle = rand_unit_complex(rng, (d,))
    v_needle = rand_unit_complex(rng, (d,))

    rows: List[Dict[str, Any]] = []
    arch = getattr(adapter, 'name', 'pam') if adapter is not None else 'pam'
    print(f'\n[persistence] Association persistence vs distance ({arch})')
    print(f'  d={d}, filler_writes/token={filler_writes}')
    for gamma in gammas:
        for T in distances:
            if adapter is not None:
                from memory_probes.adapters import adapter_relative_retrieval
                adapter.reset()
                adapter.write(k_needle, v_needle)
                for _ in range(T):
                    for _w in range(filler_writes):
                        k_f = rand_unit_complex(rng, (d,))
                        v_f = rand_unit_complex(rng, (d,))
                        adapter.write(k_f, v_f, gamma)
                rel = adapter_relative_retrieval(adapter, k_needle, v_needle, d)
                score = float(np.abs(np.vdot(adapter.read(k_needle), v_needle)))
                rows.append({'gamma': gamma, 'distance': T, 'score': score, 'relative': rel})
                continue
            S = outer_v_kstar(v_needle, k_needle)
            for _ in range(T):
                for _w in range(filler_writes):
                    k_f = rand_unit_complex(rng, (d,))
                    v_f = rand_unit_complex(rng, (d,))
                    S = pam_step_additive(S, gamma, v_f, k_f)
            score = retrieve_score(S, k_needle, v_needle)
            score0 = retrieve_score(S * 0 + outer_v_kstar(v_needle, k_needle), k_needle, v_needle)
            rel = score / (score0 + 1e-12)
            rows.append({'gamma': gamma, 'distance': T, 'score': score, 'relative': rel})
        sub = [r for r in rows if r['gamma'] == gamma]
        print(f'  gamma={gamma:.4f}: ' +
              ', '.join(f'T={r["distance"]}:{r["relative"]:.3f}' for r in sub[:4]) +
              (' ...' if len(sub) > 4 else ''))

    return {'d': d, 'filler_writes': filler_writes, 'results': rows}
