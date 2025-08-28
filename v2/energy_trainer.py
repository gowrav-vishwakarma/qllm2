#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy-Based Training for Quantum-Inspired LLM
Implements efficient energy minimization and coherence maximization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnergyBasedTrainer:
    """Trainer that incorporates energy-based training with efficient computations"""
    def __init__(self, model, learning_rate=3e-4, energy_weight=0.1, 
                 coherence_weight=0.05, grad_clip=1.0):
        self.model = model
        self.energy_weight = energy_weight
        self.coherence_weight = coherence_weight
        self.grad_clip = grad_clip
        
        # Optimizer with different learning rates for different components
        self.optimizer = torch.optim.AdamW([
            {'params': model.token_embedding.parameters(), 'lr': learning_rate},
            {'params': model.phase_init, 'lr': learning_rate * 10},
            {'params': model.quantum_layers.parameters(), 'lr': learning_rate},
            {'params': model.output_proj.parameters(), 'lr': learning_rate},
            {'params': model.pos_embedding.parameters(), 'lr': learning_rate},
        ], weight_decay=0.01)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
    def compute_energy(self, phase_repr):
        """
        Compute the energy of the phase representation efficiently.
        Energy is defined as the negative of local phase coherence.
        This avoids the O(n^2) memory issue of pairwise computations.
        """
        phases = torch.angle(phase_repr)
        batch_size, seq_len, dim = phases.shape
        
        # Compute local phase coherence (within a window)
        window_size = min(5, seq_len)  # Small window to avoid O(n^2)
        energy = torch.zeros(batch_size, device=phases.device)
        
        for i in range(seq_len):
            # Define window around current position
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            
            # Get phases in window
            window_phases = phases[:, start:end, :]
            
            # Compute coherence within window
            avg_phase = torch.mean(torch.exp(1j * window_phases), dim=1)
            coherence = torch.abs(avg_phase).mean(dim=1)
            
            # Energy is negative coherence
            energy += -coherence
        
        # Normalize by sequence length
        energy = energy / seq_len
        return energy
    
    def compute_coherence(self, phase_repr):
        """Compute global phase coherence efficiently"""
        phases = torch.angle(phase_repr)
        
        # Compute coherence as magnitude of average phase vector
        avg_phase = torch.mean(torch.exp(1j * phases), dim=1)
        coherence = torch.abs(avg_phase).mean(dim=1)
        return coherence
    
    def training_step(self, inputs, targets):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute standard cross-entropy loss
        shift_logits = outputs[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100
        )
        
        # Extract phase representation
        phase_repr = self.model.get_phase_representation(inputs)
        
        # Compute energy and coherence
        energy = self.compute_energy(phase_repr).mean()
        coherence = self.compute_coherence(phase_repr).mean()
        
        # Combined loss
        total_loss = ce_loss + self.energy_weight * energy - self.coherence_weight * coherence
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'energy': energy.item(),
            'coherence': coherence.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def validate(self, val_loader, device):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        total_ce = 0
        total_energy = 0
        total_coherence = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute losses
                shift_logits = outputs[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100
                )
                
                # Extract phase representation
                phase_repr = self.model.get_phase_representation(inputs)
                
                # Compute metrics
                energy = self.compute_energy(phase_repr).mean()
                coherence = self.compute_coherence(phase_repr).mean()
                
                total_loss += (ce_loss + self.energy_weight * energy - self.coherence_weight * coherence).item()
                total_ce += ce_loss.item()
                total_energy += energy.item()
                total_coherence += coherence.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_ce': total_ce / num_batches,
            'val_energy': total_energy / num_batches,
            'val_coherence': total_coherence / num_batches,
            'val_ppl': math.exp(total_ce / num_batches)
        }