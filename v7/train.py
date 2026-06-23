"""
V7 training script.

Usage:
    uv run python -m v7.train --preset tiny --epochs 2 --max_samples 100    # smoke
    uv run python -m v7.train --preset medium --epochs 10                   # WikiText-103
    uv run python -m v7.train --preset medium --dataset tinystories         # TinyStories
    uv run python -m v7.train --resume checkpoints_v7/best_model.pt        # resume
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import time
import urllib.request
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

warnings.filterwarnings(
    'ignore',
    message=r'.*Online softmax is disabled.*Inductor.*split the reduction.*',
    category=UserWarning,
    module=r'torch\._inductor\.lowering',
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from v7.model import V7LM, V7Config, get_config, PRESETS
from v7.model import ComplexLinear as V7ComplexLinear
from v7.simple_model import LeanPAMLM, LeanPAMConfig, get_lean_config, LEAN_PRESETS
from v7.simple_model import ComplexLinear as LeanComplexLinear
from v7.data import (
    load_wikitext103,
    load_tinystories,
    TeeLogger,
    compute_text_quality,
    resolve_amp_dtype,
    build_lr_scheduler,
    build_param_groups,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Discord Notifications ─────────────────────────────────────────────────────

def _notify_discord(content: str) -> None:
    hook = os.environ.get("DISCORD_HOOK", "").strip()
    if not hook:
        return
    if len(content) > 2000:
        content = content[:1997] + "..."
    try:
        payload = json.dumps({"content": content}).encode("utf-8")
        req = urllib.request.Request(
            hook, data=payload, method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "qllm2-v7-notify/1.0",
            },
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[Discord] Webhook send failed: {e}", file=sys.stderr)


def _notify_discord_long(text: str, *, limit: int = 1900) -> None:
    hook = os.environ.get("DISCORD_HOOK", "").strip()
    if not hook:
        return
    lines = text.splitlines(keepends=True)
    chunk: List[str] = []
    chunk_len = 0
    for line in lines:
        if chunk and chunk_len + len(line) > limit:
            _notify_discord("".join(chunk))
            chunk, chunk_len = [], 0
        chunk.append(line)
        chunk_len += len(line)
    if chunk:
        _notify_discord("".join(chunk))


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg


def _notify_training_failure(status: str, exc: Optional[BaseException] = None) -> None:
    parts = [
        f"**V7 Training {status}**",
        f"Host: {os.uname().nodename}",
        f"Command: {' '.join(sys.argv)}",
    ]
    if exc is not None:
        parts.append(f"Error: {type(exc).__name__}: {exc}")
    _notify_discord("\n".join(parts))


# ── Multi-Scale Loss ──────────────────────────────────────────────────────────

def compute_multi_scale_loss(
    aux_outputs: Dict,
    labels: torch.Tensor,
    n_layers: int,
    aux_weight: float = 0.1,
) -> torch.Tensor:
    """Compute weighted sum of per-layer auxiliary CE losses with shifted labels.

    Each aux head predicts tokens at a temporal offset matching its layer's
    memory span.  Global layers predict far ahead (harder, lower weight);
    step layers predict next-token (easier, higher weight).
    """
    if not aux_outputs:
        return torch.tensor(0.0, device=labels.device)

    total = torch.tensor(0.0, device=labels.device)
    for layer_idx, (logits, offset) in aux_outputs.items():
        if offset >= labels.shape[1]:
            continue
        shifted = labels[:, offset:]
        trimmed = logits[:, :shifted.shape[1]]
        if trimmed.numel() == 0:
            continue
        layer_loss = F.cross_entropy(
            trimmed.reshape(-1, trimmed.size(-1)),
            shifted.reshape(-1),
        )
        frac = layer_idx / max(n_layers - 1, 1)
        w = math.exp(-2.0 * (1.0 - frac))
        total = total + w * layer_loss

    return aux_weight * total


# ── Trainer ───────────────────────────────────────────────────────────────────

class V7Trainer:
    def __init__(
        self,
        model: V7LM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        tokenizer,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        gradient_clip: float = 1.0,
        max_epochs: int = 10,
        checkpoint_dir: str = 'checkpoints_v7',
        amp_dtype_str: str = 'auto',
        compile_model: bool = False,
        compile_mode: str = 'default',
        gen_every: int = 0,
        gen_prompt: str = 'The',
        log_interval: int = 50,
        save_every_steps: int = 0,
        start_epoch: int = 0,
        unitary_lambda: float = 0.0,
        run_label: str = 'V7',
        log_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        token_budget: Optional[int] = None,
        total_steps_override: Optional[int] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.max_epochs = max_epochs
        self.gradient_clip = gradient_clip
        self.unitary_lambda = unitary_lambda
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gen_every = gen_every
        self.gen_prompt = gen_prompt
        self.log_interval = log_interval
        self.save_every_steps = save_every_steps
        self.start_epoch = start_epoch
        self.shutdown_requested = False
        self.run_label = run_label
        self.log_path = log_path
        self.log_dir = log_dir
        self.token_budget = token_budget

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        param_groups = build_param_groups(model, weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=learning_rate, betas=(0.9, 0.95),
        )

        total_steps = total_steps_override
        if total_steps is None:
            try:
                total_steps = max(max_epochs * len(train_loader), 1)
            except TypeError:
                total_steps = max(warmup_steps * 10, 1)
        self.scheduler = build_lr_scheduler(
            self.optimizer, 'warmup_cosine', warmup_steps, total_steps,
        )

        self.amp_dtype = resolve_amp_dtype(amp_dtype_str)
        self.use_amp = self.amp_dtype is not None
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if self.use_amp and self.amp_dtype == torch.float16
            else None
        )

        if compile_model:
            print(f"Compiling model (mode={compile_mode})...")
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
            except Exception as e:
                print(f"torch.compile failed ({e}), continuing without")

        self.global_step = 0
        self.global_tokens = 0
        self.best_val_loss = float('inf')
        self.best_val_ppl = float('inf')

    @staticmethod
    def _masked_ce(logits, labels, loss_mask=None):
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        if loss_mask is None:
            return F.cross_entropy(flat_logits, flat_labels)
        flat_mask = loss_mask.view(-1).float()
        per_token = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        return (per_token * flat_mask).sum() / flat_mask.sum().clamp(min=1)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss_w = 0.0
        total_tokens = 0
        epoch_start = time.time()
        log_start = epoch_start
        log_tokens = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            loss_mask = batch.get('loss_mask')
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device, non_blocking=True)
            batch_tokens = input_ids.shape[0] * input_ids.shape[1]

            with torch.amp.autocast(
                self.device.type,
                enabled=self.use_amp,
                dtype=self.amp_dtype or torch.float16,
            ):
                logits, _, aux_loss = self.model(input_ids, labels=labels)
                main_loss = self._masked_ce(logits, labels, loss_mask)
                loss = main_loss
                if aux_loss.item() > 0:
                    m_cfg = self.model._orig_mod.config if hasattr(self.model, '_orig_mod') else self.model.config
                    aux_weight = getattr(m_cfg, 'aux_loss_weight', 1.0)
                    loss = loss + aux_weight * aux_loss
                if self.unitary_lambda > 0:
                    u_loss = torch.tensor(0.0, device=self.device)
                    for m in self.model.modules():
                        if isinstance(m, (V7ComplexLinear, LeanComplexLinear)):
                            WtW = (m.weight_real.T @ m.weight_real
                                   + m.weight_imag.T @ m.weight_imag)
                            eye = torch.eye(WtW.shape[0], device=WtW.device)
                            u_loss = u_loss + (WtW - eye).square().mean()
                    loss = loss + self.unitary_lambda * u_loss

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip,
                )
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1
            self.optimizer.zero_grad(set_to_none=True)

            self.global_tokens += batch_tokens
            total_tokens += batch_tokens
            log_tokens += batch_tokens
            main_loss_val = main_loss.item()
            total_loss_w += main_loss_val * batch_tokens

            if batch_idx % self.log_interval == 0:
                ppl = math.exp(min(main_loss_val, 20))
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                avg_tok_s = total_tokens / elapsed if elapsed > 0 else 0
                int_elapsed = time.time() - log_start
                inst_tok_s = log_tokens / int_elapsed if int_elapsed > 0 else 0

                try:
                    n_total = len(self.train_loader)
                    pct = 100.0 * (batch_idx + 1) / n_total
                    remaining = (
                        elapsed / (batch_idx + 1) * (n_total - batch_idx - 1)
                        if batch_idx > 0 else 0
                    )
                    eta_m, eta_s = divmod(int(remaining), 60)
                    progress = f"{batch_idx}/{n_total} ({pct:.0f}%)"
                    eta_str = f"ETA {eta_m}m{eta_s:02d}s"
                except TypeError:
                    progress = f"{batch_idx}"
                    eta_str = "ETA n/a"

                line = (
                    f"  [{epoch+1}] {progress} "
                    f"loss={main_loss_val:.4f} ppl={ppl:.1f} lr={lr:.2e} "
                    f"| {inst_tok_s:.0f} tok/s (avg {avg_tok_s:.0f}) "
                    f"{eta_str}"
                )
                if self.device.type == 'cuda':
                    mem = torch.cuda.memory_allocated() / 1e9
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    line += f" | GPU {mem:.1f}/{peak:.1f}GB"
                line += f" | gtok={self.global_tokens}"
                print(line)

                log_start = time.time()
                log_tokens = 0

            if (
                self.gen_every > 0
                and batch_idx > 0
                and batch_idx % self.gen_every == 0
                and self.tokenizer is not None
            ):
                try:
                    text = self._generate_sample(self.gen_prompt)
                    print(
                        f"  [mid-epoch sample @ batch {batch_idx}, "
                        f"{self.global_tokens:,} tok]"
                    )
                    print(f"  Prompt: {self.gen_prompt}")
                    print(f"  Generated: {text}")
                    ppl = math.exp(min(main_loss_val, 20))
                    lr = self.scheduler.get_last_lr()[0]
                    _gen_msg = (
                        f"**[{self.run_label} gen_every]** Epoch {epoch+1} batch {batch_idx} "
                        f"({self.global_tokens:,} tok)\n"
                        f"loss={main_loss_val:.4f} ppl={ppl:.1f} lr={lr:.2e} | "
                        f"{avg_tok_s:.0f} tok/s\n"
                        f"Prompt: {self.gen_prompt}\n"
                        f"Generated: {(text[:800] + '...') if len(text) > 800 else text}"
                    )
                    _notify_discord(_gen_msg)
                except Exception:
                    pass
                self.model.train()

            if (
                self.save_every_steps > 0
                and self.global_step > 0
                and self.global_step % self.save_every_steps == 0
            ):
                self._save_periodic_checkpoint(epoch)

            if self.shutdown_requested:
                print(
                    f"  Shutdown: saving latest.pt @ step {self.global_step}, "
                    f"{self.global_tokens:,} tok"
                )
                self.save_checkpoint('latest.pt', epoch)
                break

            if self.token_budget and self.global_tokens >= self.token_budget:
                print(
                    f"  Token budget reached: {self.global_tokens:,} "
                    f"/ {self.token_budget:,}"
                )
                break

        epoch_elapsed = time.time() - epoch_start
        avg_tok_s = total_tokens / epoch_elapsed if epoch_elapsed > 0 else 0
        avg_loss = total_loss_w / max(total_tokens, 1)
        return {
            'loss': avg_loss,
            'ppl': math.exp(min(avg_loss, 20)),
            'avg_tok_s': avg_tok_s,
            'epoch_tokens': total_tokens,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}
        self.model.eval()
        # Counted over masked (assistant) tokens when a loss_mask is present,
        # else over all tokens. This makes PPL in-distribution and adds a
        # next-token accuracy that is robust to single-token probability swings.
        total_loss_sum = 0.0
        total_correct = 0.0
        total_tokens = 0.0
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            loss_mask = batch.get('loss_mask')
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                self.device.type,
                enabled=self.use_amp,
                dtype=self.amp_dtype or torch.float16,
            ):
                logits, _, _aux = self.model(input_ids)

            flat_logits = logits.view(-1, logits.size(-1)).float()
            flat_labels = labels.view(-1)
            per_token = F.cross_entropy(flat_logits, flat_labels, reduction='none')
            correct = (flat_logits.argmax(dim=-1) == flat_labels).float()

            if loss_mask is not None:
                m = loss_mask.view(-1).float()
                total_loss_sum += (per_token * m).sum().item()
                total_correct += (correct * m).sum().item()
                total_tokens += m.sum().item()
            else:
                total_loss_sum += per_token.sum().item()
                total_correct += correct.sum().item()
                total_tokens += flat_labels.numel()

        if total_tokens == 0:
            return {}
        avg_loss = total_loss_sum / total_tokens
        return {
            'val_loss': avg_loss,
            'val_ppl': math.exp(min(avg_loss, 20)),
            'val_acc': total_correct / total_tokens,
        }

    @torch.no_grad()
    def _generate_sample(self, prompt: str = "The", max_tokens: int = 100) -> str:
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        generated = m.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        return self.tokenizer.decode(generated[0].tolist())

    def save_checkpoint(self, name: str, epoch: int):
        path = self.checkpoint_dir / name
        tmp_path = path.with_name(path.name + '.tmp')
        m = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        ckpt = {
            'model_state_dict': m.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'global_tokens': self.global_tokens,
            'best_val_loss': self.best_val_loss,
            'best_val_ppl': self.best_val_ppl,
            'epoch': epoch,
            'config': asdict(m.config),
        }
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
        print(f"Saved checkpoint: {path}")

    def _save_periodic_checkpoint(self, epoch: int):
        """Mid-epoch resume checkpoint only (no val — best_model is saved at epoch end)."""
        self.save_checkpoint('latest.pt', epoch)
        print(
            f"  [checkpoint @ step {self.global_step}, {self.global_tokens:,} tok]"
        )

    def _register_shutdown_handlers(self):
        def _handle_signal(signum, _frame):
            self.shutdown_requested = True
            print(
                f"\nSignal {signum} received — saving latest.pt at next batch boundary...",
                flush=True,
            )

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

    def train(self):
        training_start = time.time()
        self.shutdown_requested = False
        self._register_shutdown_handlers()
        print(f"\nTraining on {self.device}")
        m = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        params = m.count_parameters()
        print(f"Parameters: {params}")
        print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")
        try:
            n_batches = len(self.train_loader)
        except TypeError:
            n_batches = 'stream'
        print(
            f"Epochs: {self.start_epoch+1}..{self.max_epochs}, "
            f"Batches/epoch: {n_batches}"
        )
        if self.save_every_steps > 0:
            print(f"Periodic checkpoint: every {self.save_every_steps} steps -> latest.pt")
        print()

        for epoch in range(self.start_epoch, self.max_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.max_epochs}")
            print('=' * 60)

            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - training_start

            if self.shutdown_requested:
                print("\nTraining stopped by signal (latest.pt saved).")
                _notify_discord(
                    f"**{self.run_label} stopped by signal**\n"
                    f"Tokens: {self.global_tokens:,}\n"
                    f"Resume: --resume {self.checkpoint_dir / 'latest.pt'}"
                )
                return

            line = (
                f"Epoch {epoch+1}/{self.max_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"PPL: {train_metrics['ppl']:.2f} | "
                f"{train_metrics['avg_tok_s']:.0f} tok/s | "
                f"Time: {epoch_time:.1f}s ({self.global_tokens:,} tok)"
            )

            is_best = False
            if self.val_loader is not None and len(self.val_loader) > 0:
                val_metrics = self.validate()
                line += (
                    f" | Val Loss: {val_metrics['val_loss']:.4f} "
                    f"PPL: {val_metrics['val_ppl']:.2f}"
                )
                if 'val_acc' in val_metrics:
                    line += f" Acc: {val_metrics['val_acc']:.3f}"
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.best_val_ppl = val_metrics['val_ppl']
                    line += " *best*"
                    is_best = True
            print(line)

            if is_best:
                self.save_checkpoint('best_model.pt', epoch)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch)

            if self.token_budget and self.global_tokens >= self.token_budget:
                print(f"\nStopping: token budget {self.token_budget:,} reached.")
                self.save_checkpoint('final_model.pt', epoch)
                total_time = time.time() - training_start
                print(f"\nTraining complete (token budget)!")
                print(f"Total wall time: {total_time:.1f}s ({total_time/3600:.2f}h)")
                print(
                    f"Best Val Loss: {self.best_val_loss:.4f}, "
                    f"Best Val PPL: {self.best_val_ppl:.2f}"
                )
                _notify_discord(
                    f"**{self.run_label} Training complete (token budget)!**\n"
                    f"Tokens: {self.global_tokens:,}\n"
                    f"Best Val PPL: {self.best_val_ppl:.2f}"
                )
                return

            epoch_text = ""
            epoch_quality = None
            if self.tokenizer is not None:
                try:
                    text = self._generate_sample(self.gen_prompt)
                    epoch_text = text
                    print(f"\nPrompt: {self.gen_prompt}")
                    print(f"Generated: {text}")
                    qm = compute_text_quality(text)
                    epoch_quality = qm
                    print(
                        f"  Quality: rep3={qm['repeat_3gram']:.3f} "
                        f"rep4={qm['repeat_4gram']:.3f} "
                        f"restarts={qm['restart_frag']:.0f} "
                        f"uniq={qm['unique_word_ratio']:.3f}"
                    )
                except Exception as e:
                    print(f"(Sample generation failed: {e})")

            _ep_msg = (
                f"**{self.run_label} Epoch {epoch+1}/{self.max_epochs}**\n"
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"PPL: {train_metrics['ppl']:.2f} | "
                f"{train_metrics['avg_tok_s']:.0f} tok/s | "
                f"Time: {epoch_time:.1f}s"
            )
            if self.val_loader is not None and len(self.val_loader) > 0:
                _ep_msg += (
                    f"\nVal Loss: {val_metrics['val_loss']:.4f} "
                    f"PPL: {val_metrics['val_ppl']:.2f}"
                )
                if 'val_acc' in val_metrics:
                    _ep_msg += f" Acc: {val_metrics['val_acc']:.3f}"
                if is_best:
                    _ep_msg += " *best*"
            if epoch_text:
                _ep_msg += (
                    f"\nPrompt: {self.gen_prompt}\nGenerated: "
                    f"{(epoch_text[:600] + '...') if len(epoch_text) > 600 else epoch_text}"
                )
                if epoch_quality is not None:
                    _ep_msg += (
                        f"\nQuality: rep3={epoch_quality['repeat_3gram']:.3f} "
                        f"rep4={epoch_quality['repeat_4gram']:.3f} "
                        f"restarts={epoch_quality['restart_frag']:.0f} "
                        f"uniq={epoch_quality['unique_word_ratio']:.3f}"
                    )
            _notify_discord(_ep_msg)

        self.save_checkpoint('final_model.pt', self.max_epochs - 1)

        total_time = time.time() - training_start
        _done_msg = (
            f"**{self.run_label} Training complete!**\n"
            f"Wall time: {total_time:.1f}s ({total_time/3600:.2f}h)\n"
            f"Best Val Loss: {self.best_val_loss:.4f}, "
            f"Best Val PPL: {self.best_val_ppl:.2f}"
        )
        if self.log_path:
            _done_msg += f"\nLog: {self.log_path}"
        _done_msg += f"\nCheckpoint dir: {self.checkpoint_dir}"
        print(f"\nTraining complete!")
        print(f"Total wall time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        print(
            f"Best Val Loss: {self.best_val_loss:.4f}, "
            f"Best Val PPL: {self.best_val_ppl:.2f}"
        )
        _notify_discord(_done_msg)


def main():
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        try:
            for line in _env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip().strip("'\"")
                    if key == "DISCORD_HOOK" and value:
                        os.environ["DISCORD_HOOK"] = value
                        break
        except Exception as _e:
            print(f"[Discord] Could not read .env: {_e}", file=sys.stderr)
    if os.environ.get("DISCORD_HOOK"):
        print("[Discord] Webhook configured — notifications enabled", file=sys.stderr)
    else:
        print("[Discord] No webhook (set DISCORD_HOOK in .env to enable)", file=sys.stderr)

    parser = argparse.ArgumentParser(description='V7 PAM Language Model Training')
    parser.add_argument(
        '--model', type=str, default='v7',
        choices=['v7', 'lean'],
        help='Model variant: v7 (full V7LM) or lean (LeanPAMLM, stripped-down)',
    )
    all_presets = list(PRESETS.keys()) + list(LEAN_PRESETS.keys())
    parser.add_argument(
        '--preset', type=str, default=None,
        choices=all_presets,
    )
    parser.add_argument(
        '--dataset', type=str, default='wikitext103',
        choices=['wikitext103', 'tinystories'],
    )
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Override preset seq_len')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override preset dropout')
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--amp_dtype', type=str, default='auto',
                        choices=['auto', 'bf16', 'fp16'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=9999999)
    parser.add_argument('--gen_every', type=int, default=5000)
    parser.add_argument('--save_every_steps', type=int, default=0,
                        help='Save latest.pt every N optimizer steps (0=off; use 5000 for streaming pretrain)')
    parser.add_argument('--gen_prompt', type=str,
                        default='In 1923 , the University of')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v7')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Activation function
    parser.add_argument('--activation', type=str, default=None,
                        choices=['modrelu', 'swish', 'phase_mod'],
                        help='Override activation function (default: from preset)')

    # Ablation toggles
    parser.add_argument('--learned_pos', action='store_true',
                        help='Add learned absolute positional embeddings at input (token+pos)')
    parser.add_argument('--no_rope', action='store_true')
    parser.add_argument('--no_gsp', action='store_true')
    parser.add_argument('--no_fused_qkv', action='store_true')
    parser.add_argument('--qk_norm', action='store_true')
    parser.add_argument('--no_hierarchical_dt', action='store_true',
                        help='Disable hierarchical timescale (uniform dt_bias=-4.0)')
    parser.add_argument('--no_cross_level', action='store_true',
                        help='Disable cross-level drift conditioning')
    parser.add_argument('--no_grad_ckpt', action='store_true',
                        help='Disable gradient checkpointing (uses more VRAM, slightly faster)')
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Override PAM chunk size (0=full T^2, >0=chunked). Default: from preset (256)')
    parser.add_argument('--unitary_lambda', type=float, default=0.0,
                        help='Soft unitary regularization weight (0=disabled). Try 0.01.')
    parser.add_argument('--soft_state_norm', action='store_true',
                        help='[Lean] Enable soft state normalization S/(1+S_rms) in PAM.')
    parser.add_argument('--head_diversity_lambda', type=float, default=0.0,
                        help='[Lean] Inter-head phase diversity loss weight (0=disabled). Try 0.01.')

    # Multi-scale per-layer loss (Exp 7f)
    parser.add_argument('--multi_scale_loss', action='store_true',
                        help='Enable per-layer auxiliary prediction heads with temporal offsets')
    parser.add_argument('--aux_loss_weight', type=float, default=0.1,
                        help='Global weight for multi-scale auxiliary losses')
    parser.add_argument('--aux_layer_stride', type=int, default=3,
                        help='Place aux heads every N layers')
    parser.add_argument('--max_aux_offset', type=int, default=32,
                        help='Largest temporal offset (for the global-most aux layer)')

    # Reverse association (Exp 7g) -- enabled by default in config
    parser.add_argument('--no_reverse_assoc', action='store_true',
                        help='Disable reverse association (upper triangle reuse in PAM)')

    args = parser.parse_args()

    use_lean = args.model == 'lean'
    if args.preset is None:
        args.preset = 'lean_medium' if use_lean else 'medium_h6'
    model_label = 'Lean PAM' if use_lean else 'V7'

    # Logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_prefix = 'lean' if use_lean else 'v7'
    log_path = log_dir / f'{log_prefix}_{args.preset}_{args.dataset}.log'
    log_mode = 'a' if args.resume else 'w'
    tee = TeeLogger(log_path, mode=log_mode)
    sys.stdout = tee
    wall_clock_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Wall clock start: {wall_clock_start}")
    print("=" * 60)
    print(f"  {model_label}: Phase-Associative Memory Language Model")
    print(f"  Preset: {args.preset} | Dataset: {args.dataset}")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed_everything(args.seed)
    print(f"Seed: {args.seed}")

    # Config
    if use_lean:
        cfg = get_lean_config(args.preset)
        if args.seq_len is not None:
            cfg.max_seq_len = args.seq_len
        if args.dropout is not None:
            cfg.dropout = args.dropout
        if args.no_grad_ckpt:
            cfg.gradient_checkpointing = False
        if args.chunk_size is not None:
            cfg.chunk_size = args.chunk_size
        if args.soft_state_norm:
            cfg.soft_state_norm = True
        if args.head_diversity_lambda > 0:
            cfg.head_diversity_lambda = args.head_diversity_lambda
    else:
        cfg = get_config(args.preset)
        if args.seq_len is not None:
            cfg.max_seq_len = args.seq_len
        if args.dropout is not None:
            cfg.dropout = args.dropout
        if args.no_rope:
            cfg.use_rope = False
        if args.learned_pos:
            cfg.use_learned_pos = True
        if args.no_gsp:
            cfg.use_gsp = False
        if args.no_fused_qkv:
            cfg.fused_qkv = False
        if args.qk_norm:
            cfg.qk_norm = True
        if args.no_hierarchical_dt:
            cfg.hierarchical_dt = False
            cfg.dt_bias_schedule = None
        if args.no_cross_level:
            cfg.cross_level = False
        if args.no_grad_ckpt:
            cfg.gradient_checkpointing = False
        if args.activation is not None:
            cfg.activation = args.activation
        if args.chunk_size is not None:
            cfg.chunk_size = args.chunk_size
        if args.multi_scale_loss:
            cfg.multi_scale_loss = True
            cfg.aux_loss_weight = args.aux_loss_weight
            cfg.aux_layer_stride = args.aux_layer_stride
            cfg.max_aux_offset = args.max_aux_offset
        if args.no_reverse_assoc:
            cfg.use_reverse_assoc = False

    print(f"\nConfig: {asdict(cfg)}")
    print(
        f"Training: lr={args.lr}, warmup={args.warmup_steps}, wd={args.weight_decay}, "
        f"grad_clip={args.gradient_clip}"
    )
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"AMP: {args.amp_dtype}, Compile: {args.compile}")

    # Data
    seq_len = cfg.max_seq_len
    max_samples = args.max_samples if args.max_samples < 9999999 else None
    print(f"\nLoading {args.dataset} (seq_len={seq_len})...")

    if args.dataset == 'wikitext103':
        train_ds, val_ds, tokenizer = load_wikitext103(
            max_samples=max_samples, seq_len=seq_len,
        )
    else:
        train_ds, val_ds, tokenizer = load_tinystories(
            max_samples=max_samples or 20000, seq_len=seq_len,
        )

    use_cuda = torch.cuda.is_available()
    nw = args.num_workers if use_cuda else 0
    dl_kwargs = {}
    if nw > 0:
        dl_kwargs['persistent_workers'] = True
        dl_kwargs['prefetch_factor'] = 4

    dl_generator = torch.Generator()
    dl_generator.manual_seed(args.seed)

    _seed_val = args.seed
    def _worker_init_fn(worker_id: int) -> None:
        np.random.seed(_seed_val + worker_id)
        random.seed(_seed_val + worker_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=use_cuda,
        generator=dl_generator, worker_init_fn=_worker_init_fn,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=use_cuda,
        worker_init_fn=_worker_init_fn, **dl_kwargs,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = LeanPAMLM(cfg) if use_lean else V7LM(cfg)
    params = model.count_parameters()
    print(f"\nModel parameters: {params}")
    print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")

    # Resume
    start_epoch = 0
    checkpoint = None
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Trainer
    trainer = V7Trainer(
        model, train_loader, val_loader, tokenizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        max_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        amp_dtype_str=args.amp_dtype,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        gen_every=args.gen_every,
        gen_prompt=args.gen_prompt,
        log_interval=args.log_interval,
        save_every_steps=args.save_every_steps,
        start_epoch=start_epoch,
        unitary_lambda=args.unitary_lambda,
    )

    if checkpoint and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.global_tokens = checkpoint.get('global_tokens', 0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))

    _sep = "=" * 60
    _summary_lines: List[str] = []
    _sl = _summary_lines.append

    if args.resume:
        _sl(
            f"--- Resumed from {args.resume} at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
        )

    _sl(f"Wall clock start: {wall_clock_start}")
    _sl(_sep)
    _sl(f"{model_label}: Phase-Associative Memory Language Model")
    _sl(_sep)
    _sl(f"Host: {os.uname().nodename}")
    _sl(f"Preset: {args.preset} | Dataset: {args.dataset}")
    _sl(f"Complex dim: {cfg.dim} (= {cfg.dim * 2} real values/position)")
    _sl(f"Layers: {cfg.n_layers}")

    if use_lean:
        _sl("Architecture: Lean PAM (no CGU interleave; SimpleComplexFFN path)")
        _sl(
            f"PAM: heads={cfg.n_heads}, head_dim={cfg.head_dim} | "
            f"FFN expand={cfg.expand}"
        )
        _sl("RoPE: ENABLED | GSP: ENABLED | fused QKV: ENABLED (lean fixed)")
        _sl(f"Chunk size (PAM dual): {cfg.chunk_size}")
        _sl(
            f"Grad checkpointing: {'ENABLED' if cfg.gradient_checkpointing else 'DISABLED'}"
        )
        _sl(f"Soft state norm: {cfg.soft_state_norm}")
        _sl(f"Head diversity lambda: {cfg.head_diversity_lambda}")
    else:
        _sl(f"Feature: [CGU -> PAM] x{cfg.n_layers} (flat stack)")
        _sl(f"CGU expand: {cfg.expand}")
        _sl(f"PAM: heads={cfg.n_heads}, head_dim={cfg.head_dim}")
        _sl(f"Learned pos: {'ENABLED' if cfg.use_learned_pos else 'DISABLED'}")
        _sl(f"RoPE: {'ENABLED' if cfg.use_rope else 'DISABLED'}")
        _sl(f"GSP: {'ENABLED' if cfg.use_gsp else 'DISABLED'}")
        _sl(f"fused QKV: {'ENABLED' if cfg.fused_qkv else 'DISABLED'}")
        _sl(f"QK norm: {'ENABLED' if cfg.qk_norm else 'DISABLED'}")
        if cfg.chunk_size > 0:
            _sl(f"Chunk size (PAM dual): {cfg.chunk_size}")
        else:
            _sl("Chunk size (PAM dual): 0 (full T^2 dual form)")
        if cfg.hierarchical_dt:
            if cfg.dt_bias_schedule is not None:
                _sl(
                    f"Hierarchical dt: True "
                    f"(explicit schedule, len={len(cfg.dt_bias_schedule)})"
                )
            else:
                _sl("Hierarchical dt: True (auto-generated schedule)")
        else:
            _sl("Hierarchical dt: False (uniform dt_bias=-4.0)")
        _sl(f"Cross-level drift: {'ENABLED' if cfg.cross_level else 'DISABLED'}")
        _sl(
            f"Grad checkpointing: "
            f"{'ENABLED' if cfg.gradient_checkpointing else 'DISABLED'}"
        )
        _sl(f"Activation (CGU): {cfg.activation}")
        _sl(
            f"Reverse association (PAM dual): "
            f"{'ENABLED' if cfg.use_reverse_assoc else 'DISABLED'}"
        )
        if cfg.multi_scale_loss:
            _sl(
                "Multi-scale aux loss: ENABLED "
                f"(weight={cfg.aux_loss_weight}, stride={cfg.aux_layer_stride}, "
                f"max_aux_offset={cfg.max_aux_offset})"
            )
        else:
            _sl("Multi-scale aux loss: DISABLED")
        _sl(f"Tied embeddings: {cfg.tie_weights}")

    _sl(f"Epochs: {args.epochs}")
    _sl(
        f"LR: {args.lr} | warmup_steps={args.warmup_steps} | "
        f"wd={args.weight_decay} | grad_clip={args.gradient_clip} | "
        f"dropout={cfg.dropout}"
    )

    _tf32 = bool(torch.cuda.is_available())
    _compile_tail = (
        f" (mode={args.compile_mode})" if args.compile else ""
    )
    _sl(
        f"AMP: {args.amp_dtype}, TF32: {_tf32}, "
        f"Compile: {args.compile}{_compile_tail}"
    )
    _sl(f"Workers: {nw}, Pin memory: {use_cuda}")
    _sl(f"Batch log interval: {args.log_interval}")
    _sl(f"Gen every: {args.gen_every}")
    _sl(f"Log file: {log_path.resolve()}")
    _sl(f"Checkpoint dir: {Path(args.checkpoint_dir).resolve()}")
    if args.unitary_lambda > 0:
        _sl(f"Unitary lambda: {args.unitary_lambda}")
    _sl(_sep)
    _sl(f"Seed: {args.seed}")
    _sl(f"Params: {params['total']:,} ({params['total']/1e6:.1f}M)")
    _sl(
        f"Training on {trainer.device} | "
        f"Epochs: {trainer.start_epoch + 1}..{trainer.max_epochs} | "
        f"Batches/epoch: {len(train_loader)} | "
        f"Batch size: {args.batch_size}"
    )
    _sl(
        f"Val batches: {len(val_loader)} | "
        f"seq_len={cfg.max_seq_len}"
    )
    _discord_header = f"**{model_label} Training started**" if not args.resume else f"**{model_label} Training resumed**"
    _notify_discord_long(
        _discord_header + "\n```\n" + "\n".join(_summary_lines) + "\n```"
    )

    trainer.train()

    print(f"\nWall clock end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee._stdout
    tee.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        _notify_training_failure("stopped by user")
        raise
    except Exception as e:
        _notify_training_failure("OOM" if _is_oom_error(e) else "failed", e)
        raise
