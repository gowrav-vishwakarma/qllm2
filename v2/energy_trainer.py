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

# In quantum_llm_train.py (or in a separate energy_trainer.py file)
class EnergyBasedTrainer:
    def __init__(self, model, learning_rate=1e-4, energy_weight=0.01, coherence_weight=0.005, 
                 grad_clip=1.0, warmup_steps=1000, total_steps=20000):
        self.model = model
        self.energy_weight = energy_weight
        self.coherence_weight = coherence_weight
        self.grad_clip = grad_clip
        
        # Use AdamW optimizer with weight decay
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Learning rate scheduler with cosine decay
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps, eta_min=learning_rate/10
        )
        
        # Warmup scheduler
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lambda step: min(1.0, step / warmup_steps)
        )
        
        self.total_steps = total_steps
        self.current_step = 0
        
    def training_step(self, inputs, targets, scaler=None):
        self.model.train()
        
        # Forward pass with mixed precision if scaler is provided
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = self.model(inputs)
                
                # Calculate cross-entropy loss
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                # Get phase representation for energy calculation
                phase_repr = self.model.get_phase_representation(inputs)
            
            # Calculate energy and coherence losses in full precision to avoid ComplexHalf issues
            with torch.amp.autocast('cuda', enabled=False):
                # Convert phase_repr to full precision to avoid ComplexHalf issues
                phase_repr_fp32 = phase_repr.float()
                
                # Calculate energy loss (negative coherence)
                energy = self._calculate_energy(phase_repr_fp32)
                energy_loss = -energy.mean()
                
                # Calculate coherence loss
                coherence = self._calculate_coherence(phase_repr_fp32)
                coherence_loss = -coherence.mean()
                
                # Combined loss
                loss = ce_loss + self.energy_weight * energy_loss + self.coherence_weight * coherence_loss
        else:
            # Standard precision forward pass
            logits = self.model(inputs)
            
            # Calculate cross-entropy loss
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Get phase representation for energy calculation
            phase_repr = self.model.get_phase_representation(inputs)
            
            # Calculate energy loss (negative coherence)
            energy = self._calculate_energy(phase_repr)
            energy_loss = -energy.mean()
            
            # Calculate coherence loss
            coherence = self._calculate_coherence(phase_repr)
            coherence_loss = -coherence.mean()
            
            # Combined loss
            loss = ce_loss + self.energy_weight * energy_loss + self.coherence_weight * coherence_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping with scaler
            if self.grad_clip > 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Update weights with scaler
            scaler.step(self.optimizer)
            scaler.update()
        else:
            # Standard precision backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Update weights
            self.optimizer.step()
        
        # Update learning rate
        if self.current_step < 1000:  # warmup_steps
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()
        
        self.current_step += 1
        
        return {
            'loss': loss.item(),
            'ce_loss': ce_loss.item(),
            'energy': energy_loss.item(),
            'coherence': coherence_loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
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
                logits = self.model(inputs)
                
                # Calculate cross-entropy loss
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                # Get phase representation for energy calculation
                phase_repr = self.model.get_phase_representation(inputs)
                
                # Calculate energy and coherence in full precision
                phase_repr_fp32 = phase_repr.float()
                energy = self._calculate_energy(phase_repr_fp32)
                energy_loss = -energy.mean()
                
                coherence = self._calculate_coherence(phase_repr_fp32)
                coherence_loss = -coherence.mean()
                
                # Combined loss
                loss = ce_loss + self.energy_weight * energy_loss + self.coherence_weight * coherence_loss
                
                # Accumulate metrics
                total_loss += loss.item()
                total_ce += ce_loss.item()
                total_energy += energy_loss.item()
                total_coherence += coherence_loss.item()
                num_batches += 1
        
        # Calculate perplexity
        val_ppl = math.exp(total_ce / num_batches)
        
        return {
            'val_loss': total_loss / num_batches,
            'val_ce': total_ce / num_batches,
            'val_energy': total_energy / num_batches,
            'val_coherence': total_coherence / num_batches,
            'val_ppl': val_ppl
        }
    
    def _calculate_energy(self, phase_repr):
        # Calculate energy based on interference patterns
        batch_size, seq_len, dim = phase_repr.shape
        
        # Get phases
        phases = torch.angle(phase_repr)
        
        # Calculate pairwise phase differences
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
        
        # Calculate interference patterns
        interference = torch.cos(phase_diff)
        
        # Energy is negative sum of interference (more coherent = lower energy)
        energy = -torch.sum(interference, dim=(1, 2)) / (seq_len * (seq_len - 1))
        
        return energy
    
    def _calculate_coherence(self, phase_repr):
        # Calculate phase coherence
        phases = torch.angle(phase_repr)
        complex_phases = torch.exp(1j * phases)
        
        # Coherence is magnitude of average phase vector
        coherence = torch.abs(torch.mean(complex_phases, dim=1))
        
        return coherence