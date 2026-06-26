"""Binding capacity: matrix PAM vs vector HRR baseline."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np

from memory_probes.core import (
    abs_inner,
    outer_v_kstar,
    rand_unit_complex,
    retrieval_accuracy,
)


def hrr_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution binding in complex domain (FFT)."""
    fa, fb = np.fft.fft(a), np.fft.fft(b)
    return np.fft.ifft(fa * fb)


def hrr_unbind(state: np.ndarray, key: np.ndarray) -> np.ndarray:
    fk = np.fft.fft(key)
    return np.fft.ifft(np.fft.fft(state) * np.conj(fk))


def hrr_binding_capacity(
    rng: np.random.Generator,
    d: int,
    ns: Sequence[int],
    trials: int,
) -> Dict[str, Any]:
    accs: List[float] = []
    for n in ns:
        trial_accs = []
        for _ in range(trials):
            keys = rand_unit_complex(rng, (n, d))
            values = rand_unit_complex(rng, (n, d))
            state = np.zeros(d, dtype=np.complex128)
            for i in range(n):
                state = state + hrr_bind(keys[i], values[i])
            correct = 0
            for i in range(n):
                retrieved = hrr_unbind(state, keys[i])
                sims = np.array([abs_inner(retrieved, values[j]) for j in range(n)])
                if int(np.argmax(sims)) == i:
                    correct += 1
            trial_accs.append(correct / n)
        accs.append(float(np.mean(trial_accs)))
    theory = [1.0 / math.sqrt(max(n, 1)) for n in ns]
    return {
        'ns': list(ns),
        'accuracy': accs,
        'theory_1_sqrt_n': theory,
        'accuracy_at_n1': accs[0] if accs else None,
        'accuracy_at_n_max': accs[-1] if accs else None,
    }


def matrix_binding_capacity(
    rng: np.random.Generator,
    d: int,
    ns: Sequence[int],
    trials: int,
) -> Dict[str, Any]:
    accs: List[float] = []
    for n in ns:
        trial_accs = []
        for _ in range(trials):
            keys = rand_unit_complex(rng, (n, d))
            values = rand_unit_complex(rng, (n, d))
            S = np.zeros((d, d), dtype=np.complex128)
            for i in range(n):
                S += outer_v_kstar(values[i], keys[i])
            trial_accs.append(retrieval_accuracy(keys, values, S))
        accs.append(float(np.mean(trial_accs)))
    return {
        'ns': list(ns),
        'accuracy': accs,
        'accuracy_at_n1': accs[0] if accs else None,
        'accuracy_at_n64': accs[ns.index(64)] if 64 in ns else None,
    }


def adapter_binding_capacity(
    adapter,
    rng: np.random.Generator,
    d: int,
    ns: Sequence[int],
    trials: int,
) -> Dict[str, Any]:
    """Binding capacity using an architecture adapter's associative write/read."""
    from memory_probes.adapters import adapter_retrieval_accuracy

    accs: List[float] = []
    for n in ns:
        trial_accs = []
        for _ in range(trials):
            keys = rand_unit_complex(rng, (n, d))
            values = rand_unit_complex(rng, (n, d))
            adapter.reset()
            for i in range(n):
                adapter.write(keys[i], values[i])
            trial_accs.append(adapter_retrieval_accuracy(adapter, keys, values))
        accs.append(float(np.mean(trial_accs)))
    return {
        'ns': list(ns),
        'accuracy': accs,
        'accuracy_at_n1': accs[0] if accs else None,
        'accuracy_at_n64': accs[ns.index(64)] if 64 in ns else None,
    }


def test_binding(
    d: int = 64,
    max_n: int = 200,
    trials: int = 20,
    seed: int = 42,
    adapter=None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    ns = list(range(1, 65, 4)) + [64] + list(range(65, max_n + 1, 8))
    ns = sorted(set(ns))

    if adapter is not None:
        arch = getattr(adapter, 'name', 'adapter')
        matrix = adapter_binding_capacity(adapter, rng, d, ns, trials)
        theory = [1.0 / math.sqrt(max(n, 1)) for n in ns]
        print(f'\n[capacity] Binding capacity ({arch} adapter)')
        print(f'  d={d}, trials={trials}')
        for i, n in enumerate(ns):
            if n in (1, 16, 64, 128, max_n) or i == len(ns) - 1:
                print(f'  N={n:4d}  {arch}={matrix["accuracy"][i]:.3f}  theory_1/sqrtN={theory[i]:.3f}')
        return {'arch': arch, 'adapter_capacity': matrix,
                'theory_1_sqrt_n': theory, 'd': d, 'trials': trials}

    matrix = matrix_binding_capacity(rng, d, ns, trials)
    vector = hrr_binding_capacity(rng, d, ns, trials)
    print('\n[capacity] Binding capacity (matrix PAM vs vector HRR)')
    print(f'  d={d}, trials={trials}')
    for i, n in enumerate(ns):
        if n in (1, 16, 64, 128, max_n) or i == len(ns) - 1:
            print(f'  N={n:4d}  matrix={matrix["accuracy"][i]:.3f}  '
                  f'vector={vector["accuracy"][i]:.3f}  theory={vector["theory_1_sqrt_n"][i]:.3f}')
    return {'matrix_pam': matrix, 'vector_hrr': vector, 'd': d, 'trials': trials}
