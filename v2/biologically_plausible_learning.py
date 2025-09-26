#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biologically Plausible Learning System
Implements learning mechanisms inspired by biological neural networks
- No backpropagation (biologically implausible)
- Single-pass learning (like human learning)
- Local learning rules (Hebbian learning, spike-timing dependent plasticity)
- Memory consolidation (like human memory formation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import time

class HebbianLearning(nn.Module):
    """
    Hebbian Learning: "Neurons that fire together, wire together"
    Implements local learning rules without backpropagation
    """
    def __init__(self, dim, learning_rate=0.01, decay_rate=0.99):
        super().__init__()
        self.dim = dim
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # Synaptic weights (learned locally)
        self.synaptic_weights = nn.Parameter(torch.randn(dim, dim) * 0.02)
        
        # Activity traces (for correlation learning)
        self.activity_traces = nn.Parameter(torch.zeros(dim))
        self.correlation_matrix = nn.Parameter(torch.zeros(dim, dim))
        
        # Learning history
        self.learning_history = []
        
    def forward(self, x):
        """
        Forward pass with Hebbian learning
        Args:
            x: Input [batch, seq, dim] or [batch, dim]
        """
        # Compute output
        if x.dim() == 3:
            # [batch, seq, dim] -> [batch*seq, dim]
            x_flat = x.view(-1, self.dim)
            output = torch.matmul(x_flat, self.synaptic_weights)
            output = output.view(x.shape)
        else:
            output = torch.matmul(x, self.synaptic_weights)
        
        # Update activity traces (exponential moving average)
        if x.dim() == 3:
            current_activity = torch.mean(x, dim=(0, 1))
        else:
            current_activity = torch.mean(x, dim=0)
        
        self.activity_traces.data = (
            self.decay_rate * self.activity_traces.data + 
            (1 - self.decay_rate) * current_activity
        )
        
        # Hebbian learning rule: Δw = η * x * y
        if self.training:
            self._update_weights(x, output)
        
        return output
    
    def _update_weights(self, x, y):
        """Update synaptic weights using Hebbian rule"""
        if x.dim() == 3:
            # Average over batch and sequence
            x_avg = torch.mean(x, dim=(0, 1))
            y_avg = torch.mean(y, dim=(0, 1))
        else:
            x_avg = torch.mean(x, dim=0)
            y_avg = torch.mean(y, dim=0)
        
        # Hebbian update: Δw = η * x * y^T
        weight_update = torch.outer(x_avg, y_avg) * self.learning_rate
        
        # Apply weight update
        self.synaptic_weights.data += weight_update
        
        # Update correlation matrix
        self.correlation_matrix.data = (
            self.decay_rate * self.correlation_matrix.data + 
            (1 - self.decay_rate) * weight_update
        )
        
        # Store learning history
        self.learning_history.append(weight_update.norm().item())

class SpikeTimingDependentPlasticity(nn.Module):
    """
    Spike-Timing Dependent Plasticity (STDP)
    Learning rule based on timing of pre and post-synaptic spikes
    """
    def __init__(self, dim, tau_plus=20.0, tau_minus=20.0, 
                 a_plus=0.01, a_minus=0.01):
        super().__init__()
        self.dim = dim
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        
        # Synaptic weights
        self.synaptic_weights = nn.Parameter(torch.randn(dim, dim) * 0.02)
        
        # Spike traces
        self.pre_spike_trace = nn.Parameter(torch.zeros(dim))
        self.post_spike_trace = nn.Parameter(torch.zeros(dim))
        
        # Timing windows
        self.timing_window = 50  # ms
        
    def forward(self, x):
        """
        Forward pass with STDP learning
        Args:
            x: Input spikes [batch, seq, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Compute output
        output = torch.matmul(x, self.synaptic_weights)
        
        # Apply STDP learning
        if self.training:
            self._stdp_update(x, output)
        
        return output
    
    def _stdp_update(self, x, y):
        """Apply STDP learning rule"""
        batch_size, seq_len, dim = x.shape
        
        # Detect spikes (threshold crossing)
        x_spikes = (x > 0.5).float()
        y_spikes = (y > 0.5).float()
        
        # Update spike traces
        for t in range(seq_len):
            # Decay traces
            self.pre_spike_trace.data *= math.exp(-1.0 / self.tau_plus)
            self.post_spike_trace.data *= math.exp(-1.0 / self.tau_minus)
            
            # Add new spikes
            current_pre_spikes = x_spikes[:, t, :].mean(dim=0)
            current_post_spikes = y_spikes[:, t, :].mean(dim=0)
            
            self.pre_spike_trace.data += current_pre_spikes
            self.post_spike_trace.data += current_post_spikes
            
            # STDP update
            # LTP: pre before post (strengthening)
            ltp_update = torch.outer(current_pre_spikes, self.post_spike_trace) * self.a_plus
            
            # LTD: post before pre (weakening)
            ltd_update = torch.outer(self.pre_spike_trace, current_post_spikes) * self.a_minus
            
            # Apply updates
            self.synaptic_weights.data += ltp_update - ltd_update

class LocalErrorSignals(nn.Module):
    """
    Local Error Signals for biologically plausible learning
    Uses prediction errors without backpropagation
    """
    def __init__(self, dim, prediction_horizon=3):
        super().__init__()
        self.dim = dim
        self.prediction_horizon = prediction_horizon
        
        # Prediction networks (local)
        self.predictors = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(prediction_horizon)
        ])
        
        # Error signals
        self.error_signals = nn.Parameter(torch.zeros(dim))
        
        # Learning rates for each predictor
        self.learning_rates = nn.Parameter(torch.ones(prediction_horizon) * 0.01)
        
    def forward(self, x):
        """
        Forward pass with local error signals
        Args:
            x: Input [batch, seq, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Make predictions
        predictions = []
        for i, predictor in enumerate(self.predictors):
            pred = predictor(x)
            predictions.append(pred)
        
        # Compute local errors
        if self.training and seq_len > self.prediction_horizon:
            self._compute_local_errors(x, predictions)
        
        # Return weighted combination of predictions
        output = torch.zeros_like(x)
        for i, pred in enumerate(predictions):
            output += pred * (1.0 / len(predictions))
        
        return output
    
    def _compute_local_errors(self, x, predictions):
        """Compute local prediction errors"""
        batch_size, seq_len, dim = x.shape
        
        for i, pred in enumerate(predictions):
            if seq_len > i + 1:
                # Compare prediction with actual future input
                actual_future = x[:, i+1:, :]
                predicted_future = pred[:, :-i-1, :] if i > 0 else pred[:, :-1, :]
                
                # Compute prediction error
                error = torch.mean((actual_future - predicted_future) ** 2, dim=(0, 1))
                
                # Update learning rate based on error
                self.learning_rates.data[i] = torch.clamp(
                    self.learning_rates.data[i] * (1 + error.mean() * 0.1),
                    0.001, 0.1
                )
                
                # Update error signals
                self.error_signals.data = (
                    0.9 * self.error_signals.data + 
                    0.1 * error
                )

class MemoryConsolidation(nn.Module):
    """
    Memory Consolidation inspired by human memory formation
    Transfers important patterns from short-term to long-term memory
    """
    def __init__(self, dim, consolidation_threshold=0.8):
        super().__init__()
        self.dim = dim
        self.consolidation_threshold = consolidation_threshold
        
        # Short-term memory buffer
        self.stm_buffer = nn.Parameter(torch.zeros(100, dim))
        self.stm_importance = nn.Parameter(torch.zeros(100))
        self.stm_usage_count = nn.Parameter(torch.zeros(100))
        
        # Long-term memory
        self.ltm_memory = nn.Parameter(torch.randn(1000, dim) * 0.02)
        self.ltm_strength = nn.Parameter(torch.zeros(1000))
        
        # Consolidation mechanisms
        self.consolidation_gate = nn.Linear(dim, 1)
        self.importance_estimator = nn.Linear(dim, 1)
        
    def forward(self, x, importance_scores=None):
        """
        Process through memory consolidation
        Args:
            x: Input [batch, seq, dim]
            importance_scores: Importance scores for each input
        """
        batch_size, seq_len, dim = x.shape
        
        # Estimate importance if not provided
        if importance_scores is None:
            importance_scores = torch.sigmoid(self.importance_estimator(x))
            importance_scores = importance_scores.squeeze(-1)  # [batch, seq]
        
        # Update STM buffer
        if self.training:
            self._update_stm_buffer(x, importance_scores)
        
        # Retrieve from memory
        memory_retrieved = self._retrieve_from_memory(x)
        
        # Consolidate if threshold reached
        if self.training:
            self._consolidate_memories()
        
        return x + memory_retrieved * 0.3
    
    def _update_stm_buffer(self, x, importance_scores):
        """Update short-term memory buffer"""
        batch_size, seq_len, dim = x.shape
        
        # Find least important STM slot
        min_importance_idx = torch.argmin(self.stm_importance)
        
        # Compute average importance and input
        avg_importance = torch.mean(importance_scores)
        avg_input = torch.mean(x, dim=(0, 1))
        
        # Update STM buffer
        self.stm_buffer.data[min_importance_idx] = avg_input
        self.stm_importance.data[min_importance_idx] = avg_importance
        self.stm_usage_count.data[min_importance_idx] += 1
    
    def _retrieve_from_memory(self, x):
        """Retrieve relevant information from memory"""
        batch_size, seq_len, dim = x.shape
        
        # Compute similarity with STM
        x_avg = torch.mean(x, dim=(0, 1))
        stm_similarities = torch.matmul(self.stm_buffer, x_avg)
        stm_weights = F.softmax(stm_similarities, dim=0)
        
        # Compute similarity with LTM
        ltm_similarities = torch.matmul(self.ltm_memory, x_avg)
        ltm_weights = F.softmax(ltm_similarities, dim=0)
        
        # Retrieve from both memories
        stm_retrieved = torch.matmul(stm_weights, self.stm_buffer)
        ltm_retrieved = torch.matmul(ltm_weights, self.ltm_memory)
        
        return stm_retrieved + ltm_retrieved * 0.5
    
    def _consolidate_memories(self):
        """Consolidate important STM patterns to LTM"""
        # Find important STM patterns
        important_mask = self.stm_importance > self.consolidation_threshold
        important_patterns = self.stm_buffer[important_mask]
        important_strengths = self.stm_importance[important_mask]
        
        if len(important_patterns) > 0:
            # Find weakest LTM slots
            weakest_ltm_indices = torch.argsort(self.ltm_strength)[:len(important_patterns)]
            
            # Consolidate to LTM
            self.ltm_memory.data[weakest_ltm_indices] = important_patterns
            self.ltm_strength.data[weakest_ltm_indices] = important_strengths
            
            # Clear STM buffer
            self.stm_buffer.data[important_mask] = 0
            self.stm_importance.data[important_mask] = 0
            self.stm_usage_count.data[important_mask] = 0

class BiologicallyPlausibleTrainer:
    """
    Trainer that uses biologically plausible learning mechanisms
    No backpropagation, only local learning rules
    """
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        
        # Learning statistics
        self.learning_stats = {
            'hebbian_updates': 0,
            'stdp_updates': 0,
            'consolidation_events': 0,
            'local_errors': []
        }
        
    def train_step(self, x, y, importance_scores=None):
        """
        Single training step using biologically plausible learning
        Args:
            x: Input [batch, seq]
            y: Target [batch, seq]
            importance_scores: Importance scores for memory consolidation
        """
        self.model.train()
        
        # Forward pass
        logits = self.model(x)
        
        # Compute local errors (no backpropagation)
        local_errors = self._compute_local_errors(logits, y)
        
        # Update learning statistics
        self.learning_stats['local_errors'].append(local_errors.mean().item())
        
        # Memory consolidation
        if importance_scores is not None:
            self.model.memory_consolidation(x, importance_scores)
            self.learning_stats['consolidation_events'] += 1
        
        return {
            'logits': logits,
            'local_errors': local_errors,
            'learning_stats': self.learning_stats
        }
    
    def _compute_local_errors(self, logits, targets):
        """Compute local prediction errors"""
        # Simple local error computation
        predictions = torch.argmax(logits, dim=-1)
        errors = (predictions != targets).float()
        return errors
    
    def get_learning_efficiency(self):
        """Get learning efficiency metrics"""
        if len(self.learning_stats['local_errors']) > 0:
            recent_errors = self.learning_stats['local_errors'][-100:]
            efficiency = 1.0 - np.mean(recent_errors)
        else:
            efficiency = 0.0
        
        return {
            'learning_efficiency': efficiency,
            'total_updates': sum(self.learning_stats.values()),
            'consolidation_rate': self.learning_stats['consolidation_events'] / max(1, len(self.learning_stats['local_errors']))
        }

def test_biologically_plausible_learning():
    """Test biologically plausible learning system"""
    print("🧬 Testing Biologically Plausible Learning...")
    
    # Create a simple model with biologically plausible components
    class BiologicallyPlausibleModel(nn.Module):
        def __init__(self, vocab_size=256, dim=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, dim)
            self.hebbian = HebbianLearning(dim)
            self.stdp = SpikeTimingDependentPlasticity(dim)
            self.local_errors = LocalErrorSignals(dim)
            self.memory_consolidation = MemoryConsolidation(dim)
            self.output_proj = nn.Linear(dim, vocab_size)
        
        def forward(self, x):
            embeddings = self.embedding(x)
            hebbian_out = self.hebbian(embeddings)
            stdp_out = self.stdp(hebbian_out)
            error_out = self.local_errors(stdp_out)
            memory_out = self.memory_consolidation(error_out)
            logits = self.output_proj(memory_out)
            return logits
    
    # Create model and trainer
    model = BiologicallyPlausibleModel()
    trainer = BiologicallyPlausibleTrainer(model)
    
    # Test data
    batch_size, seq_len = 4, 16
    x = torch.randint(0, 256, (batch_size, seq_len))
    y = torch.randint(0, 256, (batch_size, seq_len))
    importance_scores = torch.rand(batch_size, seq_len)
    
    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training step
    start_time = time.time()
    result = trainer.train_step(x, y, importance_scores)
    training_time = time.time() - start_time
    
    print(f"⏱️ Training step time: {training_time:.4f}s")
    print(f"📈 Local errors: {result['local_errors'].mean().item():.4f}")
    print(f"🧠 Learning stats: {result['learning_stats']}")
    
    # Learning efficiency
    efficiency = trainer.get_learning_efficiency()
    print(f"📊 Learning efficiency: {efficiency}")
    
    print("✅ Biologically plausible learning test completed!")
    return model, trainer

if __name__ == "__main__":
    test_biologically_plausible_learning()
