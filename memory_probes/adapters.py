"""Architecture adapters: run the memory probes on PAM, Transformer KV cache, or Mamba SSM.

The probes only need a small associative-memory interface. Each architecture exposes
what it can:

  * Associative tier  -- write(k, v) / read(q)        (PAM matrix, Transformer KV cache)
  * Stateful tier     -- write(...) updates state; state() returns a matrix for rank
                         (all three: PAM `S`, Transformer stacked-K, Mamba `ssm_states`)

Mamba's associative `read` is not native, so it raises `CapabilityError`; probes that
need it (binding, persistence, NIAH) skip Mamba with a clear message. Rank works on all
three because it only needs `write` + `state`.

These adapters are reference shims for *comparison*, not optimized implementations.
Mamba uses the HuggingFace `MambaModel` sequential (slow) path, so `mamba_ssm` /
`causal_conv1d` are NOT required.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from memory_probes.core import (
    mat_vec,
    outer_v_kstar,
    pam_step_additive,
    rand_unit_complex,
)


class CapabilityError(NotImplementedError):
    """Raised when a probe requests an operation the architecture does not support."""


class MemoryAdapter(ABC):
    """Common interface every architecture exposes to the probe suite."""

    #: capability flags -- (associative read/write, stateful state() readout)
    associative: bool = False
    stateful: bool = True
    name: str = 'abstract'

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memory."""

    @abstractmethod
    def write(self, k: np.ndarray, v: np.ndarray, gamma: float = 1.0) -> None:
        """Store an association (key k, value v) with optional decay gamma applied first."""

    def read(self, q: np.ndarray) -> np.ndarray:
        """Retrieve a value estimate for query q. Override if associative."""
        raise CapabilityError(f'{self.name}: associative read() not supported')

    @abstractmethod
    def state(self) -> np.ndarray:
        """Return the current memory as a 2D array (for effective-rank / spectrum)."""


# ── PAM (matrix-memory) reference ────────────────────────────────────────────

class PAMAdapter(MemoryAdapter):
    """Complex matrix memory: S = gamma*S + v k*, y = S q. The reference baseline."""

    associative = True
    stateful = True
    name = 'pam'

    def __init__(self, d: int = 64):
        self.d = d
        self.S = np.zeros((d, d), dtype=np.complex128)

    def reset(self) -> None:
        self.S = np.zeros((self.d, self.d), dtype=np.complex128)

    def write(self, k: np.ndarray, v: np.ndarray, gamma: float = 1.0) -> None:
        self.S = pam_step_additive(self.S, gamma, v, k)

    def read(self, q: np.ndarray) -> np.ndarray:
        return mat_vec(self.S, q)

    def state(self) -> np.ndarray:
        return self.S


# ── Transformer KV cache ─────────────────────────────────────────────────────

class TransformerKVAdapter(MemoryAdapter):
    """Attention KV cache as an associative store.

    write(k, v) appends to the cache (lossless, O(t)). read(q) is scaled dot-product
    attention over the cache: alpha_i = softmax(Re<k_i, q>/sqrt(d)), y = sum_i alpha_i v_i.
    state() returns the stacked key matrix [t, d] for effective-rank.

    KV cache does not decay, so `gamma` is ignored by default (this lossless-but-O(t)
    behavior is exactly the contrast we report against fixed-size memories).
    """

    associative = True
    stateful = True
    name = 'transformer'

    def __init__(self, d: int = 64):
        self.d = d
        self.K: list[np.ndarray] = []
        self.V: list[np.ndarray] = []
        # Temperature for the softmax. The probe uses unit-norm keys/queries, so the
        # matched score is O(1) and mismatches are O(1/sqrt(d)); scaling by sqrt(d)
        # recovers the standard high-fidelity (sharp) attention regime.
        self._scale = math.sqrt(d)

    def reset(self) -> None:
        self.K = []
        self.V = []

    def write(self, k: np.ndarray, v: np.ndarray, gamma: float = 1.0) -> None:
        # KV cache is lossless; gamma intentionally ignored.
        self.K.append(np.asarray(k, dtype=np.complex128))
        self.V.append(np.asarray(v, dtype=np.complex128))

    def read(self, q: np.ndarray) -> np.ndarray:
        if not self.K:
            return np.zeros(self.d, dtype=np.complex128)
        Kmat = np.stack(self.K)                       # [t, d]
        Vmat = np.stack(self.V)                       # [t, d]
        # attention logits use the same conjugate score as PAM: <k_i, q> = conj(k_i).q
        logits = np.real(np.conj(Kmat) @ q) * self._scale  # [t]
        logits -= logits.max()
        alpha = np.exp(logits)
        alpha /= alpha.sum() + 1e-12
        return (alpha[:, None] * Vmat).sum(axis=0)

    def state(self) -> np.ndarray:
        if not self.K:
            return np.zeros((1, self.d), dtype=np.complex128)
        return np.stack(self.K)                        # [t, d]


# ── Mamba SSM (HuggingFace slow path, no mamba_ssm needed) ────────────────────

class MambaAdapter(MemoryAdapter):
    """State-space model memory via HuggingFace `MambaModel` (sequential/slow path).

    Stateful tier only: write(k, ...) ingests Re(k) as an input embedding; state()
    runs a forward over the accumulated stream and returns the layer's SSM state
    `cache_params.ssm_states[layer]` reshaped to a 2D matrix [intermediate, state_size].

    Associative read is not native and raises CapabilityError.
    """

    associative = False
    stateful = True
    name = 'mamba'

    def __init__(self, d: int = 32, n_layers: int = 1, state_size: int = 16,
                 layer_idx: int = 0, seed: int = 0):
        import torch
        from transformers import MambaConfig, MambaModel

        self._torch = torch
        self.d = d
        self.layer_idx = layer_idx
        torch.manual_seed(seed)
        cfg = MambaConfig(
            vocab_size=8,                 # unused; we feed inputs_embeds
            hidden_size=d,
            num_hidden_layers=n_layers,
            state_size=state_size,
            conv_kernel=4,
            expand=2,
            use_cache=True,
        )
        self.model = MambaModel(cfg).eval()
        self._embeds: list[np.ndarray] = []

    def reset(self) -> None:
        self._embeds = []

    def write(self, k: np.ndarray, v: np.ndarray, gamma: float = 1.0) -> None:
        self._embeds.append(np.real(np.asarray(k)).astype(np.float32))

    def state(self) -> np.ndarray:
        torch = self._torch
        if not self._embeds:
            return np.zeros((1, 1), dtype=np.float64)
        x = torch.from_numpy(np.stack(self._embeds)).unsqueeze(0)  # [1, T, d]
        with torch.no_grad():
            out = self.model(inputs_embeds=x, use_cache=True)
        ssm = out.cache_params.ssm_states[self.layer_idx]          # [1, intermediate, state]
        s = ssm[0].to(torch.float64).cpu().numpy()
        return s                                                    # [intermediate, state]


# ── Factory + probe helpers ──────────────────────────────────────────────────

def build_adapter(arch: str, d: int = 64, n_layers: int = 1, seed: int = 0) -> MemoryAdapter:
    arch = arch.lower()
    if arch == 'pam':
        return PAMAdapter(d=d)
    if arch == 'transformer':
        return TransformerKVAdapter(d=d)
    if arch == 'mamba':
        return MambaAdapter(d=d, n_layers=n_layers, seed=seed)
    raise ValueError(f"Unknown arch '{arch}'. Choose pam | transformer | mamba.")


def adapter_retrieval_accuracy(adapter: MemoryAdapter, keys: np.ndarray,
                               values: np.ndarray) -> float:
    """Top-1 retrieval accuracy using the adapter's associative read()."""
    n = keys.shape[0]
    correct = 0
    for i in range(n):
        y = adapter.read(keys[i])
        sims = np.array([abs(np.vdot(y, values[j])) for j in range(n)])
        if int(np.argmax(sims)) == i:
            correct += 1
    return correct / n


def adapter_relative_retrieval(adapter: MemoryAdapter, k: np.ndarray, v: np.ndarray,
                               d: int) -> float:
    """Retrieval score normalized against a fresh single-write baseline of the same type."""
    score = abs(np.vdot(adapter.read(k), v))
    base = type(adapter)(d=d)
    base.write(k, v)
    base_score = abs(np.vdot(base.read(k), v))
    return score / (base_score + 1e-12)
