#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Philosophy Metrics: Manas, Buddhi, Viveka, Smriti

Measurable signals that map to Indian philosophical concepts,
providing interpretable insights into model behavior.

Manas (मनस् - Active Mind):
    The "working memory" or active thought process.
    Measured by: backbone state magnitude, entropy of state activations.
    High Manas = active processing; Low Manas = passive/stable state.

Buddhi (बुद्धि - Discriminative Intelligence):
    The faculty of discernment and decision-making.
    Measured by: logit margin, softmax confidence, prediction entropy.
    High Buddhi = confident decisions; Low Buddhi = uncertainty.

Viveka (विवेक - Discernment/Stability):
    The ability to distinguish true from false, stability of understanding.
    Measured by: phase coherence, energy regularizer values.
    High Viveka = stable, coherent representations; Low = noisy/unstable.

Smriti (स्मृति - Memory/Recall):
    The faculty of memory and retrieval.
    Measured by: memory attention sharpness, top-k hit rate, shard usage.
    High Smriti = effective memory usage; Low = weak memory engagement.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class PhilosophyMetrics:
    """Container for philosophy-aligned metrics"""
    
    # Manas metrics
    manas_magnitude: float = 0.0       # Average state magnitude
    manas_entropy: float = 0.0          # State distribution entropy
    manas_activity: float = 0.0         # Fraction of active dimensions
    
    # Buddhi metrics
    buddhi_confidence: float = 0.0      # Max softmax probability
    buddhi_margin: float = 0.0          # Gap between top-2 logits
    buddhi_entropy: float = 0.0         # Prediction entropy
    
    # Viveka metrics
    viveka_coherence: float = 0.0       # Phase coherence across banks
    viveka_energy: float = 0.0          # Energy regularizer value
    viveka_stability: float = 0.0       # Temporal stability score
    
    # Smriti metrics
    smriti_sharpness: float = 0.0       # Memory attention sharpness
    smriti_hit_rate: float = 0.0        # Top-k memory hit rate
    smriti_coverage: float = 0.0        # Fraction of memory slots used
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {
            'manas/magnitude': self.manas_magnitude,
            'manas/entropy': self.manas_entropy,
            'manas/activity': self.manas_activity,
            'buddhi/confidence': self.buddhi_confidence,
            'buddhi/margin': self.buddhi_margin,
            'buddhi/entropy': self.buddhi_entropy,
            'viveka/coherence': self.viveka_coherence,
            'viveka/energy': self.viveka_energy,
            'viveka/stability': self.viveka_stability,
            'smriti/sharpness': self.smriti_sharpness,
            'smriti/hit_rate': self.smriti_hit_rate,
            'smriti/coverage': self.smriti_coverage,
        }


def compute_manas_metrics(
    backbone_state: torch.Tensor,  # [batch, seq, dim, 2] or [layers, batch, state_dim, 2]
    threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Compute Manas (active mind) metrics from backbone state.
    
    Args:
        backbone_state: Phase2D backbone hidden states
        threshold: Activation threshold for activity detection
    
    Returns:
        Dict with manas metrics
    """
    if backbone_state is None:
        return {'magnitude': 0.0, 'entropy': 0.0, 'activity': 0.0}
    
    # Compute magnitude: sqrt(real^2 + imag^2)
    if backbone_state.dim() == 4 and backbone_state.shape[-1] == 2:
        # [batch, seq, dim, 2] or [layers, batch, dim, 2]
        real = backbone_state[..., 0]
        imag = backbone_state[..., 1]
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
    else:
        magnitude = backbone_state.abs()
    
    # Average magnitude
    avg_magnitude = magnitude.mean().item()
    
    # Entropy of normalized magnitude distribution
    mag_flat = magnitude.flatten()
    mag_norm = F.softmax(mag_flat, dim=-1)
    entropy = -(mag_norm * torch.log(mag_norm + 1e-8)).sum().item()
    
    # Activity: fraction of dimensions above threshold
    activity = (magnitude > threshold).float().mean().item()
    
    return {
        'magnitude': avg_magnitude,
        'entropy': min(entropy / 10.0, 1.0),  # Normalize
        'activity': activity,
    }


def compute_buddhi_metrics(
    logits: torch.Tensor,  # [batch, seq, vocab_size]
) -> Dict[str, float]:
    """
    Compute Buddhi (discriminative intelligence) metrics from logits.
    
    Args:
        logits: Model output logits
    
    Returns:
        Dict with buddhi metrics
    """
    if logits is None:
        return {'confidence': 0.0, 'margin': 0.0, 'entropy': 0.0}
    
    # Flatten batch and sequence dimensions
    logits_flat = logits.view(-1, logits.size(-1))  # [batch*seq, vocab]
    
    # Softmax probabilities
    probs = F.softmax(logits_flat, dim=-1)
    
    # Confidence: max probability
    max_probs = probs.max(dim=-1).values
    confidence = max_probs.mean().item()
    
    # Margin: difference between top-2 logits
    top2_logits = logits_flat.topk(2, dim=-1).values
    margins = top2_logits[:, 0] - top2_logits[:, 1]
    margin = margins.mean().item()
    
    # Entropy of prediction distribution
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
    
    # Normalize entropy by log(vocab_size)
    max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float))
    normalized_entropy = entropy / max_entropy.item()
    
    return {
        'confidence': confidence,
        'margin': margin / 10.0,  # Normalize to ~[0, 1]
        'entropy': normalized_entropy,
    }


def compute_viveka_metrics(
    bank_outputs: Dict[str, torch.Tensor],  # {bank_name: [batch, seq, dim, 2]}
    coupling_loss: Optional[torch.Tensor] = None,
    prev_states: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Compute Viveka (discernment/stability) metrics from bank outputs.
    
    Args:
        bank_outputs: Dict of Phase2D bank outputs
        coupling_loss: Optional coupling loss from coupler
        prev_states: Optional previous states for stability computation
    
    Returns:
        Dict with viveka metrics
    """
    if not bank_outputs:
        return {'coherence': 0.0, 'energy': 0.0, 'stability': 0.0}
    
    # Coherence: average phase agreement across banks
    coherences = []
    bank_list = list(bank_outputs.values())
    
    for i in range(len(bank_list)):
        for j in range(i + 1, len(bank_list)):
            a = bank_list[i]
            b = bank_list[j]
            
            # Compute phase coherence (dot product of normalized phases)
            a_real, a_imag = a[..., 0], a[..., 1]
            b_real, b_imag = b[..., 0], b[..., 1]
            
            # Dot product
            dot_real = (a_real * b_real + a_imag * b_imag).mean()
            
            # Magnitude product
            mag_a = torch.sqrt(a_real ** 2 + a_imag ** 2 + 1e-8).mean()
            mag_b = torch.sqrt(b_real ** 2 + b_imag ** 2 + 1e-8).mean()
            
            coherence = (dot_real / (mag_a * mag_b + 1e-8)).item()
            coherences.append(coherence)
    
    avg_coherence = sum(coherences) / max(len(coherences), 1)
    
    # Energy: average magnitude deviation from unit circle
    energies = []
    for bank_out in bank_outputs.values():
        real, imag = bank_out[..., 0], bank_out[..., 1]
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        energy = ((magnitude - 1.0) ** 2).mean().item()
        energies.append(energy)
    
    avg_energy = sum(energies) / max(len(energies), 1)
    
    # Stability: temporal change (if prev_states provided)
    stability = 1.0
    if prev_states is not None:
        changes = []
        for name, curr in bank_outputs.items():
            if name in prev_states:
                prev = prev_states[name]
                change = ((curr - prev) ** 2).mean().item()
                changes.append(change)
        if changes:
            stability = 1.0 / (1.0 + sum(changes) / len(changes))
    
    return {
        'coherence': (avg_coherence + 1.0) / 2.0,  # Normalize to [0, 1]
        'energy': max(0, 1.0 - avg_energy),  # Lower energy = higher score
        'stability': stability,
    }


def compute_smriti_metrics(
    memory_result: Any,  # MemoryReadResult
    memory_module: Optional[Any] = None,  # Memory module for slot info
) -> Dict[str, float]:
    """
    Compute Smriti (memory) metrics from memory read results.
    
    Args:
        memory_result: MemoryReadResult from memory module
        memory_module: Optional memory module for slot statistics
    
    Returns:
        Dict with smriti metrics
    """
    if memory_result is None:
        return {'sharpness': 0.0, 'hit_rate': 0.0, 'coverage': 0.0}
    
    attention = memory_result.attention  # [batch, seq, num_slots]
    
    # Sharpness: inverse entropy of attention distribution
    attn_flat = attention.view(-1, attention.size(-1))
    entropy = -(attn_flat * torch.log(attn_flat + 1e-8)).sum(dim=-1).mean().item()
    max_entropy = torch.log(torch.tensor(attention.size(-1), dtype=torch.float)).item()
    sharpness = 1.0 - (entropy / max_entropy)
    
    # Hit rate: fraction of positions with significant attention (>threshold)
    threshold = 1.0 / attention.size(-1) * 2  # 2x uniform
    hit_mask = (attention > threshold).any(dim=-1)  # [batch, seq]
    hit_rate = hit_mask.float().mean().item()
    
    # Coverage: fraction of slots receiving any significant attention
    slot_attention = attention.max(dim=1).values.max(dim=0).values  # [num_slots]
    coverage = (slot_attention > threshold).float().mean().item()
    
    return {
        'sharpness': sharpness,
        'hit_rate': hit_rate,
        'coverage': coverage,
    }


class MetricsLogger:
    """
    Logger for philosophy metrics during training/evaluation.
    
    Tracks running statistics and supports periodic logging.
    """
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.step = 0
        
        # Running sums for averaging
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
    
    def update(self, metrics: PhilosophyMetrics) -> None:
        """Update running statistics with new metrics"""
        self.step += 1
        
        for key, value in metrics.to_dict().items():
            if key not in self._sums:
                self._sums[key] = 0.0
                self._counts[key] = 0
            self._sums[key] += value
            self._counts[key] += 1
    
    def get_averages(self) -> Dict[str, float]:
        """Get averaged metrics since last reset"""
        return {
            key: self._sums[key] / max(self._counts[key], 1)
            for key in self._sums
        }
    
    def reset(self) -> None:
        """Reset running statistics"""
        self._sums.clear()
        self._counts.clear()
    
    def should_log(self) -> bool:
        """Check if it's time to log"""
        return self.step % self.log_interval == 0
    
    def format_log(self) -> str:
        """Format metrics for logging"""
        avgs = self.get_averages()
        
        # Group by category
        manas = {k: v for k, v in avgs.items() if k.startswith('manas/')}
        buddhi = {k: v for k, v in avgs.items() if k.startswith('buddhi/')}
        viveka = {k: v for k, v in avgs.items() if k.startswith('viveka/')}
        smriti = {k: v for k, v in avgs.items() if k.startswith('smriti/')}
        
        lines = []
        
        if manas:
            manas_str = ', '.join(f"{k.split('/')[1]}={v:.3f}" for k, v in manas.items())
            lines.append(f"  मनस् (Manas): {manas_str}")
        
        if buddhi:
            buddhi_str = ', '.join(f"{k.split('/')[1]}={v:.3f}" for k, v in buddhi.items())
            lines.append(f"  बुद्धि (Buddhi): {buddhi_str}")
        
        if viveka:
            viveka_str = ', '.join(f"{k.split('/')[1]}={v:.3f}" for k, v in viveka.items())
            lines.append(f"  विवेक (Viveka): {viveka_str}")
        
        if smriti:
            smriti_str = ', '.join(f"{k.split('/')[1]}={v:.3f}" for k, v in smriti.items())
            lines.append(f"  स्मृति (Smriti): {smriti_str}")
        
        return '\n'.join(lines)


def compute_all_metrics(
    logits: torch.Tensor,
    phase_states: torch.Tensor,
    bank_outputs: Dict[str, torch.Tensor],
    memory_result: Any = None,
    backbone_state: Any = None,
    coupling_loss: Optional[torch.Tensor] = None,
    prev_bank_states: Optional[Dict[str, torch.Tensor]] = None,
) -> PhilosophyMetrics:
    """
    Compute all philosophy metrics from model outputs.
    
    Args:
        logits: Model logits [batch, seq, vocab_size]
        phase_states: Phase states from backbone [batch, seq, dim, 2]
        bank_outputs: Dict of bank outputs
        memory_result: Memory read result
        backbone_state: Backbone hidden state
        coupling_loss: Coupling loss from coupler
        prev_bank_states: Previous bank states for stability
    
    Returns:
        PhilosophyMetrics with all computed metrics
    """
    manas = compute_manas_metrics(
        phase_states if backbone_state is None else backbone_state.hidden
    )
    buddhi = compute_buddhi_metrics(logits)
    viveka = compute_viveka_metrics(bank_outputs, coupling_loss, prev_bank_states)
    smriti = compute_smriti_metrics(memory_result)
    
    return PhilosophyMetrics(
        manas_magnitude=manas['magnitude'],
        manas_entropy=manas['entropy'],
        manas_activity=manas['activity'],
        buddhi_confidence=buddhi['confidence'],
        buddhi_margin=buddhi['margin'],
        buddhi_entropy=buddhi['entropy'],
        viveka_coherence=viveka['coherence'],
        viveka_energy=viveka['energy'],
        viveka_stability=viveka['stability'],
        smriti_sharpness=smriti['sharpness'],
        smriti_hit_rate=smriti['hit_rate'],
        smriti_coverage=smriti['coverage'],
    )
