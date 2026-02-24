#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Training with Real Data

Usage:
    # Quick test with WikiText-2
    python train_real.py --dataset wikitext2 --size small --epochs 5
    
    # TinyStories (good for small models)
    python train_real.py --dataset tinystories --size small --epochs 10
    
    # Full training
    python train_real.py --dataset tinystories --size medium --epochs 20 --batch_size 4
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from v4.model import create_model, QuantumPhaseFieldLLM
from v4.core.config import V4Config, get_default_config
from v4.core.registry import get_registry
from v4.data import get_wikitext2, get_tinystories, create_dataloaders, get_tokenizer
from v4.metrics import MetricsLogger


class RealDataTrainer:
    """Trainer for real dataset training with speed optimizations"""
    
    def __init__(
        self,
        model: QuantumPhaseFieldLLM,
        config: V4Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        
        # Check if using morphological tokenizer
        self.is_morphological = config.tokenizer.mode == 'morphological'
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # CUDA matmul speedups (safe defaults on RTX/consumer GPUs)
        if self.device.type == 'cuda':
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Apply torch.compile if enabled
        if config.training.compile_model and hasattr(torch, 'compile'):
            print(f"🔧 Compiling model with mode='{config.training.compile_mode}'...")
            print("   Note: first training step may take a long time while graphs/backward compile.")
            try:
                self.model = torch.compile(
                    self.model, 
                    mode=config.training.compile_mode,
                    fullgraph=False,  # Allow graph breaks for flexibility
                )
                print("   ✅ Model compiled successfully")
            except Exception as e:
                print(f"   ⚠️ Compilation failed: {e}")
                print("   Continuing without compilation...")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.training.max_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
        )
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if config.training.mixed_precision and torch.cuda.is_available() else None
        
        # Objectives
        registry = get_registry()
        self.objectives = []
        for obj_cfg in config.objectives:
            obj = registry.create_objective(
                obj_cfg.type,
                weight=obj_cfg.weight,
                **obj_cfg.params
            )
            self.objectives.append(obj)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Philosophy metrics logger
        self.metrics_logger = MetricsLogger(log_interval=50) if config.training.compute_metrics else None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        num_batches = 0
        
        epoch_start = time.time()
        first_step_wall_start: Optional[float] = None
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle both BPE and morphological inputs
            if self.is_morphological:
                root_ids = batch['root_ids'].to(self.device)
                prefix_ids = batch['prefix_ids'].to(self.device)
                suffix_ids = batch['suffix_ids'].to(self.device)
                input_ids = root_ids  # Use root_ids for loss computation
            else:
                input_ids = batch['input_ids'].to(self.device)
                root_ids = prefix_ids = suffix_ids = None

            # Time the very first training step (this is where torch.compile spends time)
            if batch_idx == 0:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                first_step_wall_start = time.time()
            
            # Forward
            with autocast('cuda', enabled=self.scaler is not None):
                # Disable metrics during compiled training (causes graph breaks from .item() calls)
                # Metrics are computed separately at epoch end if needed
                compute_metrics = self.config.training.compute_metrics and not self.config.training.compile_model
                
                if self.is_morphological:
                    output = self.model(
                        root_ids=root_ids,
                        prefix_ids=prefix_ids,
                        suffix_ids=suffix_ids,
                        context={'compute_metrics': compute_metrics}
                    )
                else:
                    output = self.model(
                        input_ids, 
                        context={'compute_metrics': compute_metrics}
                    )
                
                # Compute losses
                total_batch_loss = torch.tensor(0.0, device=self.device)
                
                model_output = {
                    'logits': output.logits,
                    'phase_states': output.phase_states,
                }
                targets = {'token_ids': input_ids}  # Use root_ids for morphological
                context = {'coupling_loss': output.coupling_loss}
                
                batch_ce = 0.0
                for objective in self.objectives:
                    result = objective(model_output, targets, context)
                    total_batch_loss = total_batch_loss + result.loss * objective.weight
                    
                    if objective.name == 'ce':
                        batch_ce = result.loss.item()
                
                # Add prefix/suffix CE loss for morphological mode
                if self.is_morphological and output.prefix_logits is not None:
                    import torch.nn.functional as F
                    # Shift for next-token prediction
                    prefix_shift = output.prefix_logits[:, :-1, :].contiguous()
                    suffix_shift = output.suffix_logits[:, :-1, :].contiguous()
                    prefix_targets = prefix_ids[:, 1:].contiguous()
                    suffix_targets = suffix_ids[:, 1:].contiguous()
                    root_targets = root_ids[:, 1:].contiguous()
                    
                    # Mask: only compute loss where root is NOT padding
                    # (affixes at padded positions shouldn't contribute to loss)
                    pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
                    valid_mask = (root_targets != pad_id).float()  # [batch, seq-1]
                    
                    # Per-token CE loss (reduction='none')
                    prefix_loss_per_token = F.cross_entropy(
                        prefix_shift.view(-1, prefix_shift.size(-1)),
                        prefix_targets.view(-1),
                        reduction='none'
                    ).view_as(root_targets)
                    suffix_loss_per_token = F.cross_entropy(
                        suffix_shift.view(-1, suffix_shift.size(-1)),
                        suffix_targets.view(-1),
                        reduction='none'
                    ).view_as(root_targets)
                    
                    # Apply mask and normalize
                    num_valid = valid_mask.sum().clamp(min=1)
                    prefix_loss = (prefix_loss_per_token * valid_mask).sum() / num_valid
                    suffix_loss = (suffix_loss_per_token * valid_mask).sum() / num_valid
                    
                    # Add with lower weight (affixes are secondary to root)
                    total_batch_loss = total_batch_loss + 0.3 * (prefix_loss + suffix_loss)
            
            # Backward
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                scale_before = self.scaler.get_scale()
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler.get_scale()
                stepped = scale_after >= scale_before  # if scale dropped, step was skipped
            else:
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                self.optimizer.step()
                stepped = True

            # Scheduler should be stepped AFTER a real optimizer step
            if stepped:
                self.scheduler.step()

            # Report first-step compile time (if enabled)
            if batch_idx == 0 and first_step_wall_start is not None:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                first_step_s = time.time() - first_step_wall_start
                if self.config.training.compile_model:
                    print(f"  ⏱️ First step wall time: {first_step_s:.1f}s (includes compile/graph capture)")
                else:
                    print(f"  ⏱️ First step wall time: {first_step_s:.1f}s")
            
            # Track metrics
            total_loss += total_batch_loss.item()
            total_ce_loss += batch_ce
            num_batches += 1
            self.global_step += 1
            
            # Update philosophy metrics
            if self.metrics_logger is not None and output.metrics is not None:
                from v4.metrics import PhilosophyMetrics
                # Convert dict back to PhilosophyMetrics
                m = output.metrics
                philosophy_m = PhilosophyMetrics(
                    manas_magnitude=m.get('manas/magnitude', 0),
                    manas_entropy=m.get('manas/entropy', 0),
                    manas_activity=m.get('manas/activity', 0),
                    buddhi_confidence=m.get('buddhi/confidence', 0),
                    buddhi_margin=m.get('buddhi/margin', 0),
                    buddhi_entropy=m.get('buddhi/entropy', 0),
                    viveka_coherence=m.get('viveka/coherence', 0),
                    viveka_energy=m.get('viveka/energy', 0),
                    viveka_stability=m.get('viveka/stability', 0),
                    smriti_sharpness=m.get('smriti/sharpness', 0),
                    smriti_hit_rate=m.get('smriti/hit_rate', 0),
                    smriti_coverage=m.get('smriti/coverage', 0),
                )
                self.metrics_logger.update(philosophy_m)
            
            # Log progress
            if batch_idx % self.config.training.log_every == 0:
                avg_loss = total_loss / num_batches
                avg_ce = total_ce_loss / num_batches
                ppl = torch.exp(torch.tensor(avg_ce)).item()
                lr = self.scheduler.get_last_lr()[0]
                
                elapsed = time.time() - epoch_start
                samples_per_sec = (batch_idx + 1) * self.config.training.batch_size / elapsed
                tokens_per_sec = samples_per_sec * input_ids.shape[1]
                
                print(f"  [{batch_idx+1:4d}/{len(self.train_loader)}] "
                      f"Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | PPL: {ppl:.2f} | "
                      f"LR: {lr:.2e} | {samples_per_sec:.1f} samples/s | {tokens_per_sec:.0f} tok/s")
        
        avg_loss = total_loss / num_batches
        avg_ce = total_ce_loss / num_batches
        
        # Log philosophy metrics at end of epoch
        if self.metrics_logger is not None:
            print(f"\n🧘 Philosophy Metrics (epoch average):")
            print(self.metrics_logger.format_log())
            self.metrics_logger.reset()
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce,
            'perplexity': torch.exp(torch.tensor(avg_ce)).item(),
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            # Handle both BPE and morphological inputs
            if self.is_morphological:
                root_ids = batch['root_ids'].to(self.device)
                prefix_ids = batch['prefix_ids'].to(self.device)
                suffix_ids = batch['suffix_ids'].to(self.device)
                input_ids = root_ids
                output = self.model(
                    root_ids=root_ids,
                    prefix_ids=prefix_ids,
                    suffix_ids=suffix_ids,
                )
            else:
                input_ids = batch['input_ids'].to(self.device)
                output = self.model(input_ids)
            
            model_output = {
                'logits': output.logits,
                'phase_states': output.phase_states,
            }
            targets = {'token_ids': input_ids}
            context = {}
            
            batch_loss = 0.0
            batch_ce = 0.0
            for objective in self.objectives:
                result = objective(model_output, targets, context)
                batch_loss += result.loss.item() * objective.weight
                
                if objective.name == 'ce':
                    batch_ce = result.loss.item()
            
            total_loss += batch_loss
            total_ce_loss += batch_ce
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_ce = total_ce_loss / num_batches
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce,
            'perplexity': torch.exp(torch.tensor(avg_ce)).item(),
        }
    
    @torch.no_grad()
    def generate_sample(self, prompt: str = "The", max_tokens: int = 50) -> str:
        """Generate a sample from the model"""
        self.model.eval()
        
        # Tokenize prompt - handle both BPE and morphological
        if self.is_morphological:
            # Encode WITHOUT EOS (add_special_tokens=False), then manually add BOS
            # This prevents the model from seeing EOS in the prompt and generating PAD
            root_ids_list, prefix_ids_list, suffix_ids_list = self.tokenizer.encode(
                prompt, add_special_tokens=False
            )
            
            # Prepend BOS token
            bos_id = self.tokenizer.bos_token_id
            null_affix_id = self.tokenizer.null_affix_id
            root_ids_list = [bos_id] + root_ids_list
            prefix_ids_list = [null_affix_id] + prefix_ids_list
            suffix_ids_list = [null_affix_id] + suffix_ids_list
            
            root_ids = torch.tensor([root_ids_list], device=self.device)
            prefix_ids = torch.tensor([prefix_ids_list], device=self.device)
            suffix_ids = torch.tensor([suffix_ids_list], device=self.device)
            
            prompt_len = root_ids.size(1)
            
            # Generate using morphological mode - returns (roots, prefixes, suffixes)
            # Pass bad_token_ids to filter PAD/BOS from sampling
            generated = self.model.generate(
                root_ids=root_ids,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                bad_token_ids=[self.tokenizer.pad_token_id, self.tokenizer.bos_token_id],
            )
            
            # Unpack generated tuple
            gen_roots, gen_prefixes, gen_suffixes = generated
            
            # Decode with all three for full text reconstruction
            text = self.tokenizer.decode(
                gen_roots[0], 
                prefix_ids=gen_prefixes[0] if gen_prefixes is not None else None,
                suffix_ids=gen_suffixes[0] if gen_suffixes is not None else None,
                skip_special_tokens=True
            )
            return prompt, text, prompt_len
        else:
            # BPE tokenizer
            if hasattr(self.tokenizer, 'encode'):
                tokens = self.tokenizer.encode(prompt, return_tensors='pt')
            else:
                tokens = torch.tensor([self.tokenizer(prompt)['input_ids']])
            
            tokens = tokens.to(self.device)
            prompt_len = tokens.size(1)
            
            # Get EOS token id
            eos_id = getattr(self.tokenizer, 'eos_token_id', None)
            
            # Generate
            generated = self.model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                eos_token_id=eos_id,
            )
            
            # Decode
            text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            return prompt, text, prompt_len
    
    def save_checkpoint(self, name: str = 'checkpoint.pt'):
        """Save checkpoint"""
        path = self.checkpoint_dir / name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
        }, path)
        print(f"💾 Saved checkpoint to {path}")
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Training on {self.device}")
        print(f"   Model parameters: {self.model.count_parameters()['total']:,}")
        print(f"   Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"   Val batches: {len(self.val_loader)}")
        
        for epoch in range(self.config.training.max_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.training.max_epochs}")
            print('='*60)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"\n📊 Train | Loss: {train_metrics['loss']:.4f} | "
                  f"CE: {train_metrics['ce_loss']:.4f} | PPL: {train_metrics['perplexity']:.2f}")
            
            # Validate
            if self.val_loader:
                val_metrics = self.validate()
                print(f"📊 Val   | Loss: {val_metrics['loss']:.4f} | "
                      f"CE: {val_metrics['ce_loss']:.4f} | PPL: {val_metrics['perplexity']:.2f}")
                
                # Save best
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
            
            # Generate sample
            try:
                prompt_text, full_text, prompt_len = self.generate_sample("The quick brown", max_tokens=100)
                print(f"\n📝 Prompt: {prompt_text}")
                print(f"   Generated: {full_text}")
            except Exception as e:
                import traceback
                print(f"   (Sample generation failed: {e})")
                traceback.print_exc()
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Final checkpoint
        self.save_checkpoint('final_model.pt')
        print("\n✅ Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train v4 with real data')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'tinystories'],
                        help='Dataset to use')
    parser.add_argument('--size', type=str, default='small',
                        choices=['tiny', 'small', 'medium', 'large',
                                 'tiny-byte', 'small-byte', 'medium-byte', 'large-byte'],
                        help='Model size. Use *-byte variants for byte tokenizer (faster, 2 banks)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--max_train_samples', type=int, default=10000, 
                        help='Max training samples (for quick tests)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (uses default if not set)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v4_real')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    # Speed optimization arguments
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile for speedup')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--no_pin_memory', action='store_true', help='Disable pin_memory')
    parser.add_argument('--no_cache', action='store_true', help='Disable token caching')
    parser.add_argument('--cache_dir', type=str, default='.cache/v4_tokens', help='Token cache directory')
    parser.add_argument('--no_metrics', action='store_true',
                        help='Disable philosophy metrics (faster training)')
    # Tokenizer options
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='bpe',
        choices=['bpe', 'morphological', 'simple', 'byte'],
        help="Tokenizer type: bpe (GPT-2, default), morphological (root+affix), simple (char-level), or byte (UTF-8 multilingual)",
    )
    parser.add_argument('--morph_cache', type=str, default='.cache/morph_tokenizer',
                        help='Path to save/load morphological tokenizer')
    # Byte patching options
    parser.add_argument('--byte_patching', action='store_true', default=True,
                        help='Enable byte patching when using byte tokenizer (default: True)')
    parser.add_argument('--no_byte_patching', action='store_true',
                        help='Disable byte patching')
    parser.add_argument('--byte_patch_size', type=int, default=4,
                        help='Byte patch size (default: 4)')
    parser.add_argument('--byte_decoder_layers', type=int, default=2,
                        help='Number of layers in byte decoder (default: 2)')
    
    args = parser.parse_args()
    
    # Handle byte patching flag
    if args.no_byte_patching:
        args.byte_patching = False
    
    print("="*60)
    print("v4 Quantum Phase-Field LLM - Real Data Training")
    print("="*60)
    
    # Load dataset first (needed for morphological tokenizer training)
    if args.dataset == 'wikitext2':
        train_texts = get_wikitext2('train', max_samples=args.max_train_samples)
        val_texts = get_wikitext2('validation', max_samples=1000)
    elif args.dataset == 'tinystories':
        train_texts = get_tinystories('train', max_samples=args.max_train_samples)
        val_texts = get_tinystories('validation', max_samples=1000)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Load tokenizer
    if args.tokenizer == 'morphological':
        from v4.data import get_morphological_tokenizer
        tokenizer = get_morphological_tokenizer(
            path=args.morph_cache,
            train_texts=train_texts,  # Train on corpus if not cached
        )
        vocab_size = tokenizer.vocab_size
        prefix_vocab_size = tokenizer.prefix_vocab_size
        suffix_vocab_size = tokenizer.suffix_vocab_size
        print(f"🧬 Morphological tokenizer: root={vocab_size}, prefix={prefix_vocab_size}, suffix={suffix_vocab_size}")
        
        # Sanity check: show tokenization examples
        print("\n🔍 Tokenization sanity check:")
        test_phrases = [
            "walking quickly",
            "The quick brown fox.",
            "unhappiness",
            "running, jumping, and playing",
        ]
        for phrase in test_phrases:
            root_ids, prefix_ids, suffix_ids = tokenizer.encode(phrase, add_special_tokens=False)
            reconstructed = tokenizer.decode(root_ids, prefix_ids, suffix_ids, skip_special_tokens=True)
            print(f"   '{phrase}'")
            print(f"      → tokens: {len(root_ids)}")
            # Show first few parses
            words = tokenizer._tokenize_to_words(phrase)
            for j, word in enumerate(words[:4]):
                prefix, root, suffix = tokenizer._parse_word_cached(word)
                print(f"         '{word}' → (prefix='{prefix}', root='{root}', suffix='{suffix}')")
            print(f"      → decoded: '{reconstructed}'")
        print()
    elif args.tokenizer == 'simple':
        tokenizer = get_tokenizer('simple')
        vocab_size = tokenizer.vocab_size
        prefix_vocab_size = 0
        suffix_vocab_size = 0
        print(f"📝 Simple tokenizer: vocab={vocab_size} (char-level)")
    elif args.tokenizer == 'byte':
        tokenizer = get_tokenizer('byte')
        vocab_size = tokenizer.vocab_size
        prefix_vocab_size = 0
        suffix_vocab_size = 0
        print(f"📝 Byte tokenizer: vocab={vocab_size} (UTF-8 multilingual)")
        
        # Sanity check: show byte encoding for a multilingual sample
        print("\n🔍 Byte tokenization sanity check:")
        test_phrases = [
            "Hello world!",
            "café résumé naïve",
            "日本語テスト",
            "🚀 emoji test 🎉",
        ]
        for phrase in test_phrases:
            byte_ids = tokenizer.encode(phrase)
            decoded = tokenizer.decode(byte_ids)
            print(f"   '{phrase}' → {len(byte_ids)} bytes → '{decoded}'")
        print()
    else:
        tokenizer = get_tokenizer('gpt2')
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
        prefix_vocab_size = 0
        suffix_vocab_size = 0
    
    # Create dataloaders with speed optimizations
    # Map tokenizer arg to tokenizer_type for caching
    # Include patch size in cache key to avoid collisions between different patch sizes
    if args.tokenizer == 'morphological':
        tok_type = 'morphological'
    elif args.tokenizer == 'byte':
        if args.byte_patching:
            tok_type = f'byte_p{args.byte_patch_size}'
        else:
            tok_type = 'byte'
    else:
        tok_type = 'bpe'
    
    train_loader, val_loader = create_dataloaders(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        tokenizer_type=tok_type,
    )
    
    # Create config
    config = get_default_config(args.size)
    config.vocab_size = vocab_size
    config.training.max_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.checkpoint_dir = args.checkpoint_dir
    
    # Tokenizer mode configuration
    if args.tokenizer == 'morphological':
        config.tokenizer.mode = 'morphological'
        config.tokenizer.root_vocab_size = vocab_size
        config.tokenizer.prefix_vocab_size = prefix_vocab_size
        config.tokenizer.suffix_vocab_size = suffix_vocab_size
    elif args.tokenizer == 'byte':
        config.tokenizer.mode = 'byte'
        # Configure byte patching
        config.tokenizer.byte_patching.enabled = args.byte_patching
        config.tokenizer.byte_patching.patch_size = args.byte_patch_size
        config.tokenizer.byte_patching.decoder_layers = args.byte_decoder_layers
    else:
        config.tokenizer.mode = 'bpe'
    
    # Speed options
    config.training.compile_model = args.compile
    config.training.compile_mode = args.compile_mode
    config.training.num_workers = args.num_workers
    config.training.pin_memory = not args.no_pin_memory
    config.training.use_token_cache = not args.no_cache
    config.training.compute_metrics = not args.no_metrics
    
    if args.lr:
        config.training.learning_rate = args.lr
    
    # CRITICAL: Set CE objective to ignore padding tokens
    # Without this, model learns to predict PAD everywhere
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    for obj_cfg in config.objectives:
        if obj_cfg.type == 'ce':
            obj_cfg.params['ignore_index'] = pad_token_id
            print(f"   CE ignore_index set to {pad_token_id} (pad_token_id)")
    
    print(f"\n📋 Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Size: {args.size}")
    print(f"   Tokenizer: {args.tokenizer}")
    print(f"   Dim: {config.dim}")
    print(f"   Backbone layers: {config.backbone.num_layers}")
    print(f"   Banks: {list(config.banks.keys())}")
    print(f"   Vocab size: {vocab_size}")
    if args.tokenizer == 'morphological':
        print(f"   Prefix vocab: {prefix_vocab_size}")
        print(f"   Suffix vocab: {suffix_vocab_size}")
    print(f"   Max length: {args.max_length}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   LR: {config.training.learning_rate}")
    print(f"   Metrics: {'off' if args.no_metrics else 'on'}")
    
    # Create model
    model = create_model(config=config)
    
    # Create trainer
    trainer = RealDataTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
    )
    
    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint['global_step']
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"📥 Resumed from {args.resume}")
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
