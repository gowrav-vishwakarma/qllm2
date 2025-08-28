#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy-Based Training for Quantum-Inspired LLM
Implements the energy minimization and coherence maximization training approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnergyBasedTrainer:
    """Trainer that incorporates energy-based training"""
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
        """Compute the energy of the phase representation"""
        phases = torch.angle(phase_repr)
        batch_size, seq_len, dim = phases.shape
        phases_flat = phases.view(batch_size, seq_len * dim)
        
        # Compute cosine of phase differences (interference)
        diff = phases_flat.unsqueeze(2) - phases_flat.unsqueeze(1)
        interference = torch.cos(diff)
        
        # Energy is negative interference (lower energy = more coherence)
        energy = -torch.mean(interference, dim=(1, 2))
        return energy
    
    def compute_coherence(self, phase_repr):
        """Compute phase coherence"""
        phases = torch.angle(phase_repr)
        
        # Compute coherence as magnitude of average phase vector
        avg_phase = torch.mean(torch.exp(1j * phases), dim=1)
        coherence = torch.abs(avg_phase).mean(dim=1)
        return coherence
    
    def training_step(self, batch):
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch)
        
        # Compute standard cross-entropy loss
        shift_logits = outputs[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # Extract phase representation
        phase_repr = self.model.get_phase_representation(batch)
        
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
                batch = batch.to(device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute losses
                shift_logits = outputs[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                
                # Extract phase representation
                phase_repr = self.model.get_phase_representation(batch)
                
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