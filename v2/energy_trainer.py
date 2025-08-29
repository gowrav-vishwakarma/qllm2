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
    def __init__(self, model, learning_rate=1e-4, energy_weight=0.001, coherence_weight=0.0005, 
                 grad_clip=1.0, warmup_steps=1000, total_steps=20000):
        self.model = model
        self.energy_weight = energy_weight
        self.coherence_weight = coherence_weight
        self.grad_clip = grad_clip
        
        # NaN tracking for fallback mode
        self.nan_count = 0
        self.max_nan_count = 10  # Disable quantum losses after 10 NaN detections
        
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
        
    def training_step(self, inputs, targets, scaler=None, accumulation_steps=1, is_accumulation_step=False):
        self.model.train()
        
        # Zero gradients only on first accumulation step
        if not is_accumulation_step:
            self.optimizer.zero_grad()
        
        # Forward pass with mixed precision if scaler is provided
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = self.model(inputs, training_step=self.current_step)
                
                # Calculate cross-entropy loss
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                # Get phase representation for energy calculation
                phase_repr = self.model.get_phase_representation(inputs)
            
            # Calculate energy and coherence losses in full precision to avoid ComplexHalf issues
            with torch.amp.autocast('cuda', enabled=False):
                # Properly handle complex numbers in full precision
                if phase_repr.is_complex():
                    # Keep complex numbers as complex but in full precision
                    phase_repr_fp32 = phase_repr.to(torch.complex64)
                else:
                    phase_repr_fp32 = phase_repr.float()
                
                # Skip quantum calculations if weights are zero
                if self.energy_weight == 0.0 and self.coherence_weight == 0.0:
                    energy_loss = torch.tensor(0.0, device=phase_repr_fp32.device)
                    coherence_loss = torch.tensor(0.0, device=phase_repr_fp32.device)
                    if self.current_step == 0:  # Only print once at the start
                        print("🔧 Quantum losses disabled - running basic training only")
                else:
                    # Calculate enhanced energy loss (Phase 1.3) with aggressive NaN protection
                    energy = self._calculate_enhanced_energy(phase_repr_fp32)
                    if torch.isnan(energy).any() or torch.isinf(energy).any():
                        print(f"⚠️ NaN/Inf in energy calculation! Using fallback.")
                        energy = torch.zeros_like(energy)
                    energy_loss = -energy.mean()
                    
                    # Calculate enhanced coherence loss (Phase 1.3) with aggressive NaN protection
                    coherence = self._calculate_enhanced_coherence(phase_repr_fp32)
                    if torch.isnan(coherence).any() or torch.isinf(coherence).any():
                        print(f"⚠️ NaN/Inf in coherence calculation! Using fallback.")
                        coherence = torch.ones_like(coherence)
                    coherence_loss = -coherence.mean()
                
                # Calculate semantic loss (encourage meaningful token sequences)
                semantic_loss = self._calculate_semantic_loss(logits, targets)
                
                # Combined loss with quantum focus - semantic loss is secondary
                loss = ce_loss + self.energy_weight * energy_loss + self.coherence_weight * coherence_loss + 0.001 * semantic_loss
                
                # NaN protection for the combined loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.nan_count += 1
                    print(f"⚠️ NaN/Inf detected in loss calculation! (Count: {self.nan_count}/{self.max_nan_count})")
                    print(f"   CE: {ce_loss.item():.4f}, Energy: {energy_loss.item():.4f}, Coherence: {coherence_loss.item():.4f}")
                    
                    if self.nan_count >= self.max_nan_count:
                        print("🚨 Too many NaN detections! Disabling quantum losses for stability.")
                        self.energy_weight = 0.0
                        self.coherence_weight = 0.0
                    
                    # Fallback to cross-entropy only if quantum losses are problematic
                    loss = ce_loss
        else:
            # Standard precision forward pass
            logits = self.model(inputs, training_step=self.current_step)
            
            # Calculate cross-entropy loss
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Get phase representation for energy calculation
            phase_repr = self.model.get_phase_representation(inputs)
            
            # Skip quantum calculations if weights are zero
            if self.energy_weight == 0.0 and self.coherence_weight == 0.0:
                energy_loss = torch.tensor(0.0, device=phase_repr.device)
                coherence_loss = torch.tensor(0.0, device=phase_repr.device)
                if self.current_step == 0:  # Only print once at the start
                    print("🔧 Quantum losses disabled - running basic training only")
            else:
                # Calculate enhanced energy loss (Phase 1.3) with aggressive NaN protection
                energy = self._calculate_enhanced_energy(phase_repr)
                if torch.isnan(energy).any() or torch.isinf(energy).any():
                    print(f"⚠️ NaN/Inf in energy calculation! Using fallback.")
                    energy = torch.zeros_like(energy)
                energy_loss = -energy.mean()
                
                # Calculate enhanced coherence loss (Phase 1.3) with aggressive NaN protection
                coherence = self._calculate_enhanced_coherence(phase_repr)
                if torch.isnan(coherence).any() or torch.isinf(coherence).any():
                    print(f"⚠️ NaN/Inf in coherence calculation! Using fallback.")
                    coherence = torch.ones_like(coherence)
                coherence_loss = -coherence.mean()
            
            # Calculate semantic loss (encourage meaningful token sequences)
            semantic_loss = self._calculate_semantic_loss(logits, targets)
            
            # Combined loss with quantum focus - semantic loss is secondary
            loss = ce_loss + self.energy_weight * energy_loss + self.coherence_weight * coherence_loss + 0.001 * semantic_loss
            
            # NaN protection for the combined loss
            if torch.isnan(loss) or torch.isinf(loss):
                self.nan_count += 1
                print(f"⚠️ NaN/Inf detected in loss calculation! (Count: {self.nan_count}/{self.max_nan_count})")
                print(f"   CE: {ce_loss.item():.4f}, Energy: {energy_loss.item():.4f}, Coherence: {coherence_loss.item():.4f}")
                
                if self.nan_count >= self.max_nan_count:
                    print("🚨 Too many NaN detections! Disabling quantum losses for stability.")
                    self.energy_weight = 0.0
                    self.coherence_weight = 0.0
                
                # Fallback to cross-entropy only if quantum losses are problematic
                loss = ce_loss
        
        # Backward pass with gradient accumulation
        if scaler is not None:
            # Mixed precision backward pass
            scaler.scale(loss / accumulation_steps).backward()
            
            # Only update weights on the last accumulation step
            if not is_accumulation_step:
                # Gradient clipping with scaler
                if self.grad_clip > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Update weights with scaler
                scaler.step(self.optimizer)
                scaler.update()
                
                # Update learning rate (only after optimizer step)
                if self.current_step < 1000:  # warmup_steps
                    self.warmup_scheduler.step()
                else:
                    self.scheduler.step()
                
                self.current_step += 1
        else:
            # Standard precision backward pass
            loss.backward()
            
            # Only update weights on the last accumulation step
            if not is_accumulation_step:
                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Update weights
                self.optimizer.step()
                
                # Update learning rate (only after optimizer step)
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
                
                # Calculate enhanced energy and coherence in full precision
                if phase_repr.is_complex():
                    # Keep complex numbers as complex but in full precision
                    phase_repr_fp32 = phase_repr.to(torch.complex64)
                else:
                    phase_repr_fp32 = phase_repr.float()
                energy = self._calculate_enhanced_energy(phase_repr_fp32)
                energy_loss = -energy.mean()
                
                coherence = self._calculate_enhanced_coherence(phase_repr_fp32)
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
    
    def _calculate_enhanced_energy(self, phase_repr):
        """Enhanced energy calculation with quantum-inspired components (Phase 1.3) - NaN Safe"""
        batch_size, seq_len, dim = phase_repr.shape
        
        # Get phases with NaN protection
        phases = torch.angle(phase_repr)
        phases = torch.nan_to_num(phases, nan=0.0, posinf=math.pi, neginf=-math.pi)
        
        # QUANTUM-INSPIRED ENERGY COMPONENTS with NaN protection
        
        # 1. Local coherence energy (neighboring tokens)
        # This captures the coherence between adjacent tokens
        if seq_len > 1:
            local_phase_diff = phases[:, 1:] - phases[:, :-1]
            local_coherence = torch.cos(local_phase_diff)
            # Average over feature dimension, then sum over sequence
            local_energy = -torch.mean(local_coherence, dim=2).sum(dim=1) / max(seq_len - 1, 1)
        else:
            local_energy = torch.zeros(batch_size, device=phases.device)
        
        # 2. Global interference energy (all token pairs)
        # Optimized computation - O(n) instead of O(n²)
        # Use mean phase as global reference for efficiency
        global_phase_mean = torch.mean(phases, dim=1, keepdim=True)
        global_phase_diff = phases - global_phase_mean
        global_coherence = torch.cos(global_phase_diff)
        # Average over feature dimension, then sum over sequence
        global_energy = -torch.mean(global_coherence, dim=2).sum(dim=1) / max(seq_len, 1)
        
        # 3. Entanglement energy (long-range dependencies)
        # Sample random token pairs for entanglement to avoid O(n²) complexity
        num_pairs = min(20, seq_len // 2)
        if num_pairs > 0:
            # Create random indices for token pairs
            rand_indices = torch.randperm(seq_len, device=phases.device)[:num_pairs]
            if len(rand_indices) >= 2:
                entangled_pairs = phases[:, rand_indices]
                entangled_phase_diff = entangled_pairs[:, 1:] - entangled_pairs[:, :-1]
                # Average over feature dimension, then mean over sequence
                entanglement_energy = -torch.mean(torch.cos(entangled_phase_diff), dim=2).mean(dim=1)
            else:
                entanglement_energy = torch.zeros(batch_size, device=phases.device)
        else:
            entanglement_energy = torch.zeros(batch_size, device=phases.device)
        
        # 4. Phase stability energy with NaN protection
        # Encourage stable phase relationships
        if seq_len > 1:
            phase_variance = torch.var(phases, dim=1, unbiased=False)
            phase_variance = torch.nan_to_num(phase_variance, nan=0.0, posinf=1.0, neginf=0.0)
            stability_energy = -torch.mean(phase_variance, dim=1)
        else:
            stability_energy = torch.zeros(batch_size, device=phases.device)
        
        # Combined energy with learned weights and NaN protection
        energy = (
            local_energy + 
            0.5 * global_energy + 
            0.3 * entanglement_energy +
            0.2 * stability_energy
        )
        
        # Final NaN protection
        energy = torch.nan_to_num(energy, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return energy
    
    def _calculate_enhanced_coherence(self, phase_repr):
        """Enhanced coherence calculation with multi-scale analysis (Phase 1.3) - NaN Safe"""
        phases = torch.angle(phase_repr)
        phases = torch.nan_to_num(phases, nan=0.0, posinf=math.pi, neginf=-math.pi)
        batch_size, seq_len, dim = phases.shape
        
        # Multi-scale coherence calculation with NaN protection
        coherence_scores = []
        
        # 1. Local coherence (neighboring tokens)
        # Calculate coherence between adjacent tokens
        if seq_len > 1:
            local_diff = phases[:, 1:] - phases[:, :-1]
            # Clamp phase differences to prevent overflow in exp(1j * local_diff)
            local_diff = torch.clamp(local_diff, -math.pi, math.pi)
            local_coherence = torch.abs(torch.mean(torch.exp(1j * local_diff), dim=(1, 2)))
        else:
            local_coherence = torch.ones(batch_size, device=phases.device)
        coherence_scores.append(local_coherence)
        
        # 2. Global coherence (all tokens)
        # Calculate overall phase coherence
        global_phases = phases.view(batch_size, -1)
        # Clamp global phases to prevent overflow
        global_phases = torch.clamp(global_phases, -math.pi, math.pi)
        global_coherence = torch.abs(torch.mean(torch.exp(1j * global_phases), dim=1))
        coherence_scores.append(global_coherence)
        
        # 3. Semantic coherence (similar tokens should have similar phases)
        # Sample token pairs for efficiency
        num_pairs = min(50, seq_len // 2)
        if num_pairs > 0:
            rand_indices = torch.randperm(seq_len, device=phases.device)[:num_pairs]
            sampled_phases = phases[:, rand_indices]
            # Clamp sampled phases to prevent overflow
            sampled_phases = torch.clamp(sampled_phases, -math.pi, math.pi)
            semantic_coherence = torch.abs(torch.mean(torch.exp(1j * sampled_phases), dim=(1, 2)))
        else:
            semantic_coherence = torch.ones(batch_size, device=phases.device)
        coherence_scores.append(semantic_coherence)
        
        # 4. Temporal coherence (phase consistency over time)
        # Calculate how phases evolve across the sequence
        if seq_len > 1:
            temporal_diff = phases[:, 1:] - phases[:, :-1]
            # Clamp temporal differences to prevent overflow
            temporal_diff = torch.clamp(temporal_diff, -math.pi, math.pi)
            temporal_coherence = torch.abs(torch.mean(torch.exp(1j * temporal_diff), dim=(1, 2)))
        else:
            temporal_coherence = torch.ones(batch_size, device=phases.device)
        coherence_scores.append(temporal_coherence)
        
        # Combined coherence with weighted average and NaN protection
        combined_coherence = torch.stack(coherence_scores, dim=1)
        weights = torch.tensor([0.4, 0.3, 0.2, 0.1], device=phases.device)  # Local, Global, Semantic, Temporal
        weighted_coherence = torch.sum(combined_coherence * weights.unsqueeze(0), dim=1)
        
        # Final NaN protection
        weighted_coherence = torch.nan_to_num(weighted_coherence, nan=0.0, posinf=1.0, neginf=0.0)
        
        return weighted_coherence
    
    def _calculate_energy(self, phase_repr):
        """Legacy energy calculation - kept for backward compatibility"""
        return self._calculate_enhanced_energy(phase_repr)
    
    def _calculate_coherence(self, phase_repr):
        """Legacy coherence calculation - kept for backward compatibility"""
        return self._calculate_enhanced_coherence(phase_repr)
    
    def _calculate_semantic_loss(self, logits, targets):
        """
        Semantic loss to encourage meaningful token sequences and reduce repetitive patterns
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # NaN protection for logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"⚠️ NaN/Inf detected in logits! Using fallback semantic loss.")
            return torch.tensor(0.0, device=logits.device)
        
        # 1. Repetition penalty loss (simplified and vectorized)
        # Penalize repetitive token sequences
        probs = torch.softmax(logits, dim=-1)
        
        # NaN protection for probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"⚠️ NaN/Inf detected in probabilities! Using fallback semantic loss.")
            return torch.tensor(0.0, device=logits.device)
        
        # Simplified repetition penalty using tensor operations
        repetition_penalty = 0.0
        if seq_len > 1:
            # Use tensor operations instead of Python loops
            recent_tokens = targets[:, -min(10, seq_len-1):]
            # Count unique tokens (simplified approach)
            unique_tokens = torch.unique(recent_tokens)
            repetition_penalty = (len(unique_tokens) / recent_tokens.numel()) * 0.1
        
        # 2. Semantic diversity loss (simplified)
        # Encourage diverse token selection
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=10.0, neginf=-10.0)
        diversity_loss = -torch.mean(entropy)  # Negative because we want to maximize entropy
        
        # 3. Context coherence loss (simplified)
        # Encourage tokens that make sense in context
        context_loss = 0.0
        if seq_len > 2:
            # Simplified context loss using tensor operations
            # Calculate smoothness between adjacent token probabilities
            prob_diff = torch.diff(probs, dim=1)
            prob_diff = torch.nan_to_num(prob_diff, nan=0.0, posinf=1.0, neginf=-1.0)
            context_loss = torch.mean(torch.norm(prob_diff, dim=-1))
            context_loss = torch.nan_to_num(context_loss, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 4. Byte-level coherence loss
        # For byte-level models, encourage valid UTF-8 sequences
        byte_coherence = 0.0
        if vocab_size == 256:  # Byte-level vocabulary
            # Encourage common byte patterns
            common_bytes = [32, 101, 116, 97, 111, 105, 110, 115, 114, 104]  # space, e, t, a, o, i, n, s, r, h
            for byte_val in common_bytes:
                if byte_val < vocab_size:
                    byte_coherence += torch.mean(probs[:, :, byte_val])
        
        # Combined semantic loss with NaN protection
        semantic_loss = (
            0.3 * repetition_penalty +
            0.3 * diversity_loss +
            0.2 * context_loss +
            0.2 * byte_coherence
        )
        
        # Final NaN protection
        semantic_loss = torch.nan_to_num(semantic_loss, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return semantic_loss