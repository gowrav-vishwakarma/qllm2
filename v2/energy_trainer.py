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
    """Trainer that incorporates energy-based training with flexible scheduler"""
    def __init__(self, model, learning_rate=3e-4, energy_weight=0.1, 
                 coherence_weight=0.05, grad_clip=1.0, warmup_steps=500, total_steps=10000):
        self.model = model
        self.energy_weight = energy_weight
        self.coherence_weight = coherence_weight
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Optimizer with different learning rates for different components
        self.optimizer = torch.optim.AdamW([
            {'params': model.token_embedding.parameters(), 'lr': learning_rate},
            {'params': model.phase_init, 'lr': learning_rate * 10},
            {'params': model.quantum_layers.parameters(), 'lr': learning_rate},
            {'params': model.output_proj.parameters(), 'lr': learning_rate},
            {'params': model.pos_embedding.parameters(), 'lr': learning_rate},
        ], weight_decay=0.01)
        
        # Use a better scheduler with warmup and cosine decay
        # Make total_steps larger to accommodate multiple epochs
        adjusted_total_steps = max(total_steps, 10000)  # Ensure we have enough steps
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=adjusted_total_steps,
            pct_start=warmup_steps/adjusted_total_steps,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1000,
            cycle_momentum=False
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
        self.current_step += 1
        
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
        
        # Combined loss with reduced energy/coherence impact for stability
        total_loss = ce_loss + self.energy_weight * energy - self.coherence_weight * coherence
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Optimizer step
        self.optimizer.step()
        
        # Step scheduler with safety check
        try:
            self.scheduler.step()
        except ValueError as e:
            # If we've exceeded the total steps, just keep the last learning rate
            if "total steps" in str(e):
                # We've reached the end of the scheduler cycle
                pass
            else:
                raise e
        
        return {
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'energy': energy.item(),
            'coherence': coherence.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def validate(self, val_loader, device):
        """Validation step with progress logging"""
        self.model.eval()
        total_loss = 0
        total_ce = 0
        total_energy = 0
        total_coherence = 0
        num_batches = 0
        
        print("   Running validation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
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
                
                # Print progress every 10 batches
                if batch_idx % 100 == 0:
                    print(f"   Validation batch {batch_idx}/{len(val_loader)}")
        
        print(f"✅ Validation completed: {num_batches} batches processed")
        
        return {
            'val_loss': total_loss / num_batches,
            'val_ce': total_ce / num_batches,
            'val_energy': total_energy / num_batches,
            'val_coherence': total_coherence / num_batches,
            'val_ppl': math.exp(total_ce / num_batches)
        }