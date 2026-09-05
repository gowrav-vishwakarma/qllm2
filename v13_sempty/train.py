"""Self-contained trainer for v13_sempty (no V7Trainer).

AdamW (betas 0.9/0.95), 2-D-only weight decay, warmup-cosine, grad-clip,
optional AMP, fused CE. Dataset/tokenizer loading is
reused from ``v7.data``; the step loop lives here.

Default device is CPU so a live GPU training run is not disturbed.
Use ``--device cuda`` explicitly if a free GPU is available.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from v13_sempty.model import LM, _retention_capture
from v13_sempty.config import PRESETS, get_config
from v13_sempty.triton_kernels import kernel_enabled, set_kernel_enabled

_NO_DECAY_SUFFIXES = {'dt_bias'}


def _git_hash() -> str:
    import subprocess
    try:
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=root,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ('-dirty' if dirty.stdout.strip() else '')
    except Exception:  # noqa: BLE001
        return 'unknown'


def _print_run_header(args, cfg, model, params, device, loader, val_loader,
                      tokenizer):
    """V13-style rich header at the top of every log.

    Records everything needed to reproduce/interpret the run later: wall clock,
    commit, full CLI args, full model config, geometry (batch/seq/tokens/steps/
    implied-epochs), dataset sizes, parameter breakdown, and the environment
    (device, GPU, AMP, kernel flags). Kept as plain prints so it lands in the
    tee'd .log verbatim.
    """
    bs, sl = args.batch_size, args.seq_len
    tok_per_step = bs * sl
    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = None
    epochs_implied = (args.steps / steps_per_epoch) if steps_per_epoch else None
    total_tokens = args.steps * tok_per_step

    bar = '=' * 72
    print(bar)
    print('  v13_sempty trainer  (real-arm PAM, O(1) memory)')
    print(f"  wall clock start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  commit           : {_git_hash()}")
    print(f"  preset / dataset : {args.preset} / {args.dataset}")
    print(bar)

    # --- environment --------------------------------------------------------
    gpu = 'cpu'
    if device.type == 'cuda' and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(device)
    print('[env]')
    print(f"  device={device} gpu={gpu} torch={torch.__version__}")
    print(f"  fused_pam={kernel_enabled()} fused_ce={args.fused_ce and not args.no_fused_ce} "
          f"ce_gemm_dtype={args.ce_gemm_dtype} amp={args.amp_dtype} "
          f"grad_ckpt={cfg.gradient_checkpointing}")

    # --- geometry -----------------------------------------------------------
    print('[geometry]')
    print(f"  batch_size={bs} seq_len={sl} tokens/step={tok_per_step:,}")
    _epstr = f"{epochs_implied:.2f}" if epochs_implied is not None else "n/a"
    print(f"  steps={args.steps:,} steps/epoch={steps_per_epoch} "
          f"epochs_implied={_epstr}")
    print(f"  planned_tokens={total_tokens:,}  "
          f"lr={args.lr} warmup={args.warmup_steps} wd={args.weight_decay} "
          f"grad_clip={args.gradient_clip}")
    print(f"  checkpoints: dir={args.checkpoint_dir} latest_every="
          f"{args.save_every_steps} keep_every={args.keep_every_steps} "
          f"keep_last={args.keep_last} best=on val_every={args.val_every} "
          f"resume={args.resume or 'off'}")

    # --- dataset ------------------------------------------------------------
    print('[data]')
    _tr = None
    try:
        _tr = len(loader.dataset)
    except (AttributeError, TypeError):
        pass
    _va = None
    if val_loader is not None:
        try:
            _va = len(val_loader.dataset)
        except (AttributeError, TypeError):
            pass
    print(f"  train_chunks={_tr} val_chunks={_va} "
          f"recall_frac={args.recall_frac} "
          f"vocab={cfg.vocab_size} tokenizer={'gpt2' if tokenizer else 'synthetic'}")

    # --- params -------------------------------------------------------------
    print('[params]')
    _pm = f"  total={params['total']:,} ({params['total']/1e6:.2f}M dense)"
    if 'cond_mem_table' in params:
        _pm += (f" + {params['cond_mem_table']:,} table "
                f"(total_with_table {params['total_with_table']/1e6:.2f}M)")
    print(_pm)

    # --- ladder (only non-default arch flags) -------------------------------
    _ladder = [k for k in ('n_states', 'vault', 'delta', 'cond_mem',
                           'chrono', 'out_gate')
               if getattr(cfg, k) not in (False, 1, 0.0)]
    if _ladder:
        print('[ladder] ' + " ".join(f"{k}={getattr(cfg, k)}" for k in _ladder))

    # --- full config + args (verbatim, for exact repro) ---------------------
    print('[config] ' + str(asdict(cfg)))
    print('[args] ' + str(vars(args)))
    print(bar, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_param_groups(model: nn.Module, weight_decay: float, learning_rate: float = 1e-4,
                       table_lr_mult: float = 5.0):
    """Split into decay/no-decay groups (same rules as v7.data.build_param_groups).

    A4 cond_mem lookup tables get their own group: no weight decay and lr x5
    (Engram convention). LambdaLR preserves the per-group lr ratio.
    """
    table_ids = model.cond_mem_table_param_ids() if hasattr(model, 'cond_mem_table_param_ids') else set()
    decay, no_decay, table = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in table_ids:
            table.append(param)
            continue
        suffix = name.split('.')[-1]
        if suffix in _NO_DECAY_SUFFIXES:
            no_decay.append(param)
        elif suffix in ('bias', 'bias_real', 'bias_imag'):
            no_decay.append(param)
        elif param.dim() >= 2 and suffix in ('weight', 'weight_real', 'weight_imag'):
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    if table:
        groups.append({'params': table, 'weight_decay': 0.0,
                       'lr': learning_rate * table_lr_mult})
    return groups


def build_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    total_steps = max(total_steps, 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_amp_dtype(amp_dtype_str: str, device: torch.device):
    if device.type != 'cuda':
        return None
    if amp_dtype_str == 'off':
        return None
    if amp_dtype_str == 'bf16':
        return torch.bfloat16
    if amp_dtype_str == 'fp16':
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class Trainer:
    def __init__(
        self,
        model: LM,
        train_loader,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 10,
        total_steps: int = 100,
        gradient_clip: float = 1.0,
        amp_dtype_str: str = 'off',
        fused_ce: bool = True,
        fused_ce_chunk: int = 4096,
        ce_gemm_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        log_interval: int = 1,
        val_loader=None,
        wiki_val_loader=None,
        is_streaming: bool = False,
        tokenizer=None,
        gen_every: int = 0,
        gen_prompt: str = 'The',
        gen_max_tokens: int = 80,
        save_every_steps: int = 0,
        keep_every_steps: int = 0,
        keep_last: int = 1,
        diag_every: int = 500,
        val_every: int = 0,
        max_val_batches: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        run_label: str = 'v13_sempty',
        data_cursor: Optional[dict] = None,
        run_args: Optional[dict] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.gradient_clip = gradient_clip
        self.fused_ce = fused_ce
        self.fused_ce_chunk = fused_ce_chunk
        # Training head GEMMs: None = autocast dtype (bf16). Validation always
        # runs the fp32 head so the reported NLL/PPL is exact.
        self.ce_gemm_dtype = ce_gemm_dtype
        self.log_interval = log_interval
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

        groups = build_param_groups(model, weight_decay, learning_rate=learning_rate)
        self.optimizer = torch.optim.AdamW(
            groups, lr=learning_rate, betas=(0.9, 0.95),
            fused=(self.device.type == 'cuda'),
        )
        self.scheduler = build_lr_scheduler(self.optimizer, warmup_steps, total_steps)
        self.amp_dtype = resolve_amp_dtype(amp_dtype_str, self.device)
        self.use_amp = self.amp_dtype is not None
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if self.use_amp and self.amp_dtype == torch.float16
            else None
        )
        self.global_step = 0
        self.global_tokens = 0
        self.val_loader = val_loader
        # Optional secondary eval: WikiText-103 val, the anchor comparable to
        # the 23.81 fair-run number (never mixed into training).
        self.wiki_val_loader = wiki_val_loader
        # Live IterableDataset stream => do not re-iterate across epochs.
        self.is_streaming = is_streaming
        self.tokenizer = tokenizer
        self.gen_every = gen_every
        self.gen_prompt = gen_prompt
        self.gen_max_tokens = gen_max_tokens
        self.save_every_steps = save_every_steps
        # Milestone copies (step_XXXXXX.pt) kept alongside latest/best so the
        # trajectory can be inspected later; 0 = off.
        self.keep_every_steps = keep_every_steps
        self.keep_last = keep_last
        # Live dicts owned by the streaming data pipeline ({'doc_counters':
        # {source: docs consumed}, 'token_counters': {source: tokens yielded}}).
        # Saved in every checkpoint = the stream position for --resume.
        self.data_cursor = data_cursor
        self.run_args = run_args
        # Steps done in THIS process (resume-aware ETA / tok/s).
        self.session_steps = 0
        self.resumed_from = None
        self.diag_every = diag_every
        self.max_val_batches = max_val_batches
        self.checkpoint_dir = checkpoint_dir
        self.run_label = run_label
        self.best_val_loss = float('inf')
        self.best_val_ppl = float('inf')
        self.val_every = val_every
        self._blocks = self.model.blocks
        self._set_capture(False)
        self._last_grad_norms = None

    def _step_loss(self, input_ids, labels, loss_mask=None):
        """One training step's loss: fused chunked CE, or plain CE fallback."""
        if self.fused_ce:
            lm, _aux_loss = self.model._hidden_to_lm(input_ids)
            return self.model.ce_from_lm(
                lm, labels, loss_mask=loss_mask, chunk=self.fused_ce_chunk,
                gemm_dtype=self.ce_gemm_dtype,
            )
        logits, _, _aux = self.model(input_ids, labels=labels)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    def step(self, batch) -> float:
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        loss_mask = batch.get('loss_mask')
        if loss_mask is not None:
            loss_mask = loss_mask.to(self.device)

        with torch.amp.autocast(
            self.device.type,
            enabled=self.use_amp,
            dtype=self.amp_dtype or torch.float16,
        ):
            loss = self._step_loss(input_ids, labels, loss_mask)

        if not torch.isfinite(loss).all():
            raise RuntimeError(f"non-finite loss at step {self.global_step}: {loss}")

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
        if self.diag_every > 0 and (self.global_step + 1) % self.diag_every == 0:
            self._last_grad_norms = self._block_grad_norms()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        self.session_steps += 1
        return float(loss.detach())

    # ── logging / diagnostics (mirrors v7.V7Trainer cadence) ────────────────

    def _set_capture(self, on: bool):
        """Toggle the realized-retention hook on every PAM layer."""
        for b in self._blocks:
            pam = getattr(b, 'pam', None)
            if pam is not None:
                pam.capture_decay = on

    @torch.no_grad()
    def _block_grad_norms(self) -> list:
        """Per-block grad L2, captured while grads are still live."""
        norms = []
        for b in self._blocks:
            g = 0.0
            for p in b.parameters():
                if p.grad is not None:
                    g += float(p.grad.detach().float().pow(2).sum())
            norms.append(g ** 0.5)
        return norms

    @torch.no_grad()
    def _val_loss(self, loader=None) -> Optional[float]:
        """Token-weighted val NLL over up to max_val_batches batches."""
        loader = loader if loader is not None else self.val_loader
        if loader is None or len(loader) == 0:
            return None
        self.model.eval()
        total, ntok = 0.0, 0
        for i, batch in enumerate(loader):
            if self.max_val_batches is not None and i >= self.max_val_batches:
                break
            x = batch['input_ids'].to(self.device)
            y = batch['labels'].to(self.device)
            lm, _ = self.model._hidden_to_lm(x)          # fp32, no autocast: exact val
            loss = self.model.ce_from_lm(
                lm, y, loss_mask=batch.get('loss_mask'),
                chunk=self.fused_ce_chunk, gemm_dtype=torch.float32,
            )
            total += float(loss.detach()) * x.numel()
            ntok += x.numel()
        self.model.train()
        return total / max(ntok, 1)

    @torch.no_grad()
    def _generate_sample(self, prompt: str, max_tokens: int) -> str:
        self.model.eval()
        try:
            ids = self.tokenizer.encode(prompt)
            x = torch.tensor([ids], device=self.device)
            out = self.model.generate(
                x, max_new_tokens=max_tokens, temperature=0.8, top_k=50,
                top_p=0.9, repetition_penalty=1.2,
            )
        finally:
            # A failing decode (e.g. chrono/delta: NotImplementedError) must
            # not leave the model in eval mode (dropout off) for the rest of
            # training -- the caller swallows the exception and keeps going.
            self.model.train()
        return self.tokenizer.decode(out[0].tolist())

    def _save_ckpt(self, name: str):
        if self.checkpoint_dir is None:
            return
        d = Path(self.checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        tmp = d / (name + '.tmp')
        ck = self.model.config
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': (self.scaler.state_dict()
                                  if self.scaler is not None else None),
            'global_step': self.global_step,
            'global_tokens': self.global_tokens,
            'best_val_loss': self.best_val_loss,
            'best_val_ppl': self.best_val_ppl,
            'config': asdict(ck),
            # --- everything --resume needs beyond the weights -----------------
            'data_cursor': ({k: dict(v) for k, v in self.data_cursor.items()}
                            if self.data_cursor else None),
            'rng': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
            'args': self.run_args,
            'saved_at': time.time(),
        }
        torch.save(ckpt, tmp)
        os.replace(tmp, path)
        print(f"  [checkpoint] step {self.global_step} -> {path}", flush=True)

    def _prune_milestones(self):
        """Rolling milestones: keep only the newest ``keep_last`` step_XXXXXX.pt.

        latest.pt (resume state, overwritten atomically) and best_model.pt are
        never touched. keep_last <= 0 disables pruning. Each copy is ~1.2 GB at
        102M params (weights + Adam moments), so the old keep-everything policy
        cost 12 GB per 3B-token run (2026-09-05 user rule: free stale ckpts).
        """
        if self.checkpoint_dir is None or self.keep_last <= 0:
            return
        files = sorted(Path(self.checkpoint_dir).glob('step_*.pt'))
        for old in files[:-self.keep_last]:
            try:
                old.unlink()
                print(f"  [checkpoint] pruned {old.name}", flush=True)
            except OSError as e:  # never let housekeeping kill the run
                print(f"  [checkpoint] prune failed {old.name}: {e}", flush=True)

    @staticmethod
    def peek_resume(path: Path) -> dict:
        """Load a checkpoint written by _save_ckpt (CPU) for --resume."""
        ck = torch.load(path, map_location='cpu', weights_only=False)
        for k in ('model_state_dict', 'optimizer_state_dict',
                  'scheduler_state_dict', 'global_step', 'global_tokens'):
            if k not in ck:
                raise ValueError(f"{path} is not a resumable checkpoint (missing {k})")
        return ck

    def load_resume(self, ck: dict, path) -> None:
        """Restore model, optimizer, LR schedule, AMP scaler, step/token
        counters, best-val bookkeeping and RNG streams. The data-stream cursor
        is restored by the caller when it builds the loader (it must exist
        before the Trainer does)."""
        self.model.load_state_dict(ck['model_state_dict'])
        self.optimizer.load_state_dict(ck['optimizer_state_dict'])
        self.scheduler.load_state_dict(ck['scheduler_state_dict'])
        if self.scaler is not None and ck.get('scaler_state_dict'):
            self.scaler.load_state_dict(ck['scaler_state_dict'])
        self.global_step = int(ck['global_step'])
        self.global_tokens = int(ck['global_tokens'])
        self.best_val_loss = float(ck.get('best_val_loss', float('inf')))
        self.best_val_ppl = float(ck.get('best_val_ppl', float('inf')))
        rng = ck.get('rng') or {}
        if rng.get('python') is not None:
            random.setstate(rng['python'])
        if rng.get('numpy') is not None:
            np.random.set_state(rng['numpy'])
        if rng.get('torch') is not None:
            torch.set_rng_state(rng['torch'])
        if rng.get('cuda') is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng['cuda'])
            except RuntimeError as e:  # different GPU count
                print(f"  [resume] cuda rng not restored: {e}")
        self.resumed_from = str(path)
        lr = self.optimizer.param_groups[0]['lr']
        print(f"[resume] {path}: step={self.global_step:,} "
              f"tokens={self.global_tokens:,} lr={lr:.3e} "
              f"best_val_ppl={self.best_val_ppl:.2f} "
              f"cursor={ck.get('data_cursor')}", flush=True)

    def _diagnostics(self) -> dict:
        """Per-layer learnable-scale + retention + grad/weight-norm snapshot.

        This is the 'which variable is helping' panel: cgu_scale (transform
        path), pam_scale (memory path), realized retention (how long the
        notebook holds), and per-block grad/weight L2 (where it's learning).
        """
        # realized retention: one no-grad forward with the capture hook on
        self._set_capture(True)
        _retention_capture.clear()
        try:
            it = iter(self.train_loader)
            try:
                b0 = next(it)
            except StopIteration:
                return {}
            x = b0['input_ids'].to(self.device)
            with torch.no_grad():
                _ = self.model(x)
        finally:
            self._set_capture(False)
        layer_ret = [float(r.mean()) for r in _retention_capture]

        g_norms = self._last_grad_norms
        if g_norms is None or len(g_norms) != len(self._blocks):
            g_norms = [0.0] * len(self._blocks)

        out = {
            'cgu_scale': [],
            'pam_scale': [],
            'dt_bias': [],
            'g_norm': [],
            'w_norm': [],
            'ret': [],
            'dt_bias_heads': None,   # per-head dt_bias, averaged over layers
        }
        # Per-head view (mean over layers): the per-layer head-MEAN in the
        # 'dtbias=' row hides any per-head structure (L1 run, 2026-09-05: a
        # -4..-12 ladder read as a flat -8), so keep both views.
        head_acc = None
        for i, b in enumerate(self._blocks):
            out['cgu_scale'].append(float(b.cgu_scale.detach()))
            out['pam_scale'].append(float(b.pam_scale.detach()))
            db = b.pam.dt_bias.detach().float()
            out['dt_bias'].append(float(db.mean()))
            head_acc = db.clone() if head_acc is None else head_acc + db
            out['g_norm'].append(g_norms[i])
            w = 0.0
            for p in b.parameters():
                w += float(p.detach().float().pow(2).sum())
            out['w_norm'].append(w ** 0.5)
            out['ret'].append(layer_ret[i] if i < len(layer_ret) else float('nan'))
        if head_acc is not None and head_acc.dim() >= 1:
            out['dt_bias_heads'] = (head_acc / len(self._blocks)).flatten().tolist()
        return out

    def _log_diag(self, d: dict):
        if not d:
            return
        def _row(key, fmt):
            return ' '.join(fmt.format(v) for v in d[key])
        print(
            f"  [diag] step {self.global_step} "
            f"cgu={_row('cgu_scale', '{:+.2f}')}  "
            f"pam={_row('pam_scale', '{:+.2f}')}",
            flush=True,
        )
        print(
            f"  [diag] dtbias={_row('dt_bias', '{:.2f}')}  "
            f"ret={_row('ret', '{:.2f}')}",
            flush=True,
        )
        if d.get('dt_bias_heads'):
            print(
                f"  [diag] dtbias/head(layer-mean)="
                f"{' '.join('{:.2f}'.format(v) for v in d['dt_bias_heads'])}",
                flush=True,
            )
        print(
            f"  [diag] gnorm={_row('g_norm', '{:.1e}')}  "
            f"wnorm={_row('w_norm', '{:.1f}')}",
            flush=True,
        )

    def train(self, max_steps: Optional[int] = None) -> list:
        self.model.train()
        losses = []
        train_start = time.time()
        log_start = time.time()
        log_tokens = 0
        self._session_tok0 = self.global_tokens
        if max_steps is not None and self.global_step >= max_steps:
            print(f"[resume] already at step {self.global_step} >= {max_steps}; "
                  f"nothing to do", flush=True)
            return losses
        try:
            steps_per_epoch = len(self.train_loader)
        except TypeError:
            steps_per_epoch = None
        # ETA/progress track the whole run when max_steps spans many epochs.
        n_batches = max_steps if max_steps is not None else steps_per_epoch
        # Epoch bookkeeping so the log reads like the V11/V13 per-epoch logs
        # even though we drive the loop off a single global step budget.
        total_epochs = (math.ceil(max_steps / steps_per_epoch)
                        if (max_steps and steps_per_epoch) else None)
        self._steps_per_epoch = steps_per_epoch
        self._total_epochs = total_epochs
        epoch_loss_sum, epoch_loss_n = 0.0, 0
        epoch_start = time.time()

        def _epoch_stream():
            """Re-iterate the loader across epochs (re-shuffles each pass) until
            max_steps; a single pass when max_steps is None or the loader is a
            live token stream (re-iterating an exhausted IterableDataset would
            busy-loop forever, so a streaming corpus runs exactly one pass and
            is stopped by its own token budget)."""
            while True:
                for b in self.train_loader:
                    yield b
                if max_steps is None or self.is_streaming:
                    return

        for batch_idx, batch in enumerate(_epoch_stream()):
            loss = self.step(batch)
            losses.append(loss)
            batch_tokens = batch['input_ids'].numel()
            self.global_tokens += batch_tokens
            log_tokens += batch_tokens
            epoch_loss_sum += loss
            epoch_loss_n += 1

            if self.log_interval and self.global_step % self.log_interval == 0:
                self._log_line(loss, batch_idx, n_batches, train_start,
                               log_start, log_tokens)
                log_start = time.time()
                log_tokens = 0

            if (
                self.gen_every > 0 and self.global_step > 0
                and self.global_step % self.gen_every == 0
                and self.tokenizer is not None
            ):
                try:
                    text = self._generate_sample(self.gen_prompt, self.gen_max_tokens)
                    print(f"  [gen @ step {self.global_step}, "
                          f"{self.global_tokens:,} tok] prompt: {self.gen_prompt}")
                    print(f"    {text[:600]}", flush=True)
                except Exception as e:
                    print(f"  [gen @ step {self.global_step}] failed: {e}", flush=True)

            if (
                self.val_every > 0 and self.global_step > 0
                and self.global_step % self.val_every == 0
                and self.val_loader is not None
            ):
                try:
                    vl = self._val_loss()
                except Exception as e:
                    vl = None
                    print(f"  [val @ step {self.global_step}] failed: {e}", flush=True)
                if vl is not None:
                    ppl = math.exp(min(vl, 20))
                    tag = ''
                    if vl < self.best_val_loss:
                        self.best_val_loss = vl
                        self.best_val_ppl = ppl
                        tag = ' *best*'
                        self._save_ckpt('best_model.pt')
                    print(
                        f"  [val @ step {self.global_step}] "
                        f"val_loss={vl:.4f} val_ppl={ppl:.2f}{tag} "
                        f"(best {self.best_val_ppl:.2f})",
                        flush=True,
                    )
                # Secondary anchor: WikiText-103 val (comparable to the 23.81
                # fair run). Eval-only; does not gate best_model checkpointing.
                if self.wiki_val_loader is not None:
                    try:
                        wvl = self._val_loss(self.wiki_val_loader)
                    except Exception as e:  # noqa: BLE001
                        wvl = None
                        print(f"  [wiki_val @ step {self.global_step}] failed: {e}",
                              flush=True)
                    if wvl is not None:
                        print(
                            f"  [wiki_val @ step {self.global_step}] "
                            f"val_loss={wvl:.4f} val_ppl={math.exp(min(wvl, 20)):.2f}",
                            flush=True,
                        )

            if (
                self.save_every_steps > 0 and self.global_step > 0
                and self.global_step % self.save_every_steps == 0
            ):
                self._save_ckpt('latest.pt')
            if (
                self.keep_every_steps > 0 and self.global_step > 0
                and self.global_step % self.keep_every_steps == 0
            ):
                self._save_ckpt(f'step_{self.global_step:06d}.pt')
                self._prune_milestones()

            if self.diag_every > 0 and self.global_step > 0 \
                    and self.global_step % self.diag_every == 0:
                try:
                    self._log_diag(self._diagnostics())
                except Exception as e:
                    print(f"  [diag @ step {self.global_step}] failed: {e}", flush=True)

            # Epoch-boundary banner (mirrors the V11/V13 "Epoch N/M" summary so
            # the logs align epoch-for-epoch with the reference tables).
            if steps_per_epoch and self.global_step % steps_per_epoch == 0:
                ep_done = self.global_step // steps_per_epoch
                ep_avg = epoch_loss_sum / max(epoch_loss_n, 1)
                ep_time = time.time() - epoch_start
                print(
                    f"=== Epoch {ep_done}/{total_epochs} done | step "
                    f"{self.global_step} | train_loss={ep_avg:.4f} "
                    f"ppl={math.exp(min(ep_avg, 20)):.2f} | "
                    f"gtok={self.global_tokens:,} | {ep_time/60:.1f} min | "
                    f"best_val_ppl={self.best_val_ppl:.2f} ===",
                    flush=True,
                )
                epoch_loss_sum, epoch_loss_n = 0.0, 0
                epoch_start = time.time()

            if max_steps is not None and self.global_step >= max_steps:
                break
        return losses

    def _log_line(self, loss, batch_idx, n_batches, train_start,
                  log_start, log_tokens):
        lr = self.optimizer.param_groups[0]['lr']
        ppl = math.exp(min(loss, 20))
        elapsed = time.time() - train_start
        # Rates from THIS process only (a resumed run has done global_step -
        # session_steps steps in earlier processes); progress from global_step.
        session_tokens = self.global_tokens - getattr(self, '_session_tok0', 0)
        avg_tok_s = session_tokens / elapsed if elapsed > 0 else 0
        inst_tok_s = log_tokens / max(time.time() - log_start, 1e-9)
        if n_batches:
            done = self.global_step
            pct = 100.0 * done / n_batches
            remaining = elapsed / max(self.session_steps, 1) * (n_batches - done)
            eta_m, eta_s = divmod(int(max(remaining, 0)), 60)
            eta_str = f"ETA {eta_m}m{eta_s:02d}s"
            prog = f"[{done}/{n_batches} {pct:3.0f}%]"
        else:
            eta_str, prog = "ETA n/a", f"[{self.global_step}]"
        # Which epoch this step falls in (1-indexed), matching the V11 logs.
        spe = getattr(self, '_steps_per_epoch', None)
        if spe:
            cur_ep = (self.global_step - 1) // spe + 1
            tot_ep = getattr(self, '_total_epochs', None)
            ep_str = f"ep{cur_ep}/{tot_ep} " if tot_ep else f"ep{cur_ep} "
        else:
            ep_str = ""
        line = (
            f"step {self.global_step} {ep_str}{prog}  loss={loss:.4f} ppl={ppl:.1f} "
            f"lr={lr:.2e} | {inst_tok_s:.0f} tok/s (avg {avg_tok_s:.0f}) "
            f"{eta_str} | gtok={self.global_tokens:,}"
        )
        if self.device.type == 'cuda':
            mem = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            line += f" | GPU {mem:.1f}/{peak:.1f}GB"
        print(line, flush=True)


def synthetic_loader(vocab_size: int, batch_size: int, seq_len: int, n_batches: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab_size, (n_batches * batch_size, seq_len), generator=g)
    labels = torch.randint(0, vocab_size, (n_batches * batch_size, seq_len), generator=g)
    ds = TensorDataset(ids, labels)

    def _collate(rows):
        x = torch.stack([r[0] for r in rows])
        y = torch.stack([r[1] for r in rows])
        return {'input_ids': x, 'labels': y}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)


def load_real_dataset(name: str, seq_len: int, max_samples: int):
    from v7.data import load_tinystories, load_wikitext103
    if name == 'tinystories':
        train_ds, val_ds, tok = load_tinystories(max_samples=max_samples, seq_len=seq_len)
    elif name == 'wikitext103':
        train_ds, val_ds, tok = load_wikitext103(max_samples=max_samples, seq_len=seq_len)
    else:
        raise ValueError(f"unknown dataset {name}")
    return train_ds, val_ds, tok


def build_argparser():
    p = argparse.ArgumentParser(description='v13_sempty self-contained trainer')
    p.add_argument('--preset', type=str, default='tiny', choices=list(PRESETS.keys()))
    p.add_argument('--dataset', type=str, default='synthetic',
                   choices=['synthetic', 'tinystories', 'wikitext103',
                            'dclm', 'fineweb', 'mix'],
                   help="dclm/fineweb/mix = streaming pretrain via "
                        "v7.data.load_pretrain_mix (use --target_tokens)")
    p.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--seq_len', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--dropout', type=float, default=None,
                   help='override the preset dropout (single-pass streaming '
                        'pretrain wants 0; the WikiText 10-epoch recipe uses the '
                        'preset 0.1)')
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--gradient_clip', type=float, default=1.0)
    p.add_argument('--amp_dtype', type=str, default='off',
                   choices=['off', 'auto', 'bf16', 'fp16'])
    p.add_argument('--fused_ce', action='store_true', default=True)
    p.add_argument('--no_fused_ce', action='store_true')
    p.add_argument('--fused_ce_chunk', type=int, default=4096)
    p.add_argument('--ce_gemm_dtype', type=str, default='auto', choices=['auto', 'fp32'],
                   help='head GEMM precision in training: auto = the autocast dtype '
                        '(bf16), fp32 = exact. Validation is always fp32.')
    p.add_argument('--fused_pam', dest='fused_pam', action='store_true', default=True,
                   help='Triton fused PAM scan for the real arm (default on)')
    p.add_argument('--no_fused_pam', dest='fused_pam', action='store_false',
                   help='use the plain-torch PAM scan instead of the Triton kernel')
    p.add_argument('--max_samples', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    # --- streaming pretrain (dclm/fineweb/mix) ---------------------------------
    p.add_argument('--target_tokens', type=int, default=0,
                   help='token budget for streaming pretrain (0=off). Derives '
                        '--steps = target_tokens // (batch_size*seq_len).')
    p.add_argument('--pretrain_sources', type=str, default='dclm,fineweb',
                   help="comma sources for --dataset mix (v7 SOURCE_REGISTRY: "
                        "dclm,fineweb,smoltalk2_mid,recall,reason)")
    p.add_argument('--pretrain_weights', type=str, default='',
                   help='comma interleave weights matching --pretrain_sources '
                        '(empty=equal)')
    p.add_argument('--edu_score_min', type=int, default=3,
                   help='edu-score filter for dclm/fineweb (>=)')
    p.add_argument('--fineweb_name', type=str, default='sample-10BT')
    p.add_argument('--holdout_pct', type=int, default=5,
                   help='%% of corpus reserved as the primary streaming val holdout')
    p.add_argument('--blend_warmup_tokens', type=int, default=0,
                   help='tokens to draw web-only before non-web sources enter the mix')
    p.add_argument('--answer_weight', type=float, default=1.0,
                   help='per-token CE weight on the ANSWER tokens of synthetic recall/'
                        'reason docs (v7.data.ANSWER_MARK); 1.0 = plain LM loss. The '
                        'micro-bench (EXPERIMENTS "Positive control") showed exact '
                        '8-way recall is learned only with concentrated answer signal. '
                        'Train loss becomes the weighted mean; val is unweighted.')
    p.add_argument('--num_workers', type=int, default=2,
                   help='DataLoader workers (forced 0 for a live stream)')
    p.add_argument('--no_wiki_val', action='store_true',
                   help='skip the secondary WikiText-103 val anchor')
    p.add_argument('--resume', type=str, default='',
                   help="resume a run: path to a checkpoint written by this "
                        "trainer, or 'auto' = <checkpoint_dir>/latest.pt if it "
                        "exists (else start fresh). Restores model, optimizer, "
                        "LR schedule, AMP scaler, step/token counters, best-val, "
                        "RNG streams and the data-stream cursor.")
    p.add_argument('--keep_every_steps', type=int, default=0,
                   help='also keep a milestone copy step_XXXXXX.pt every N steps')
    p.add_argument('--keep_last', type=int, default=1,
                   help='rolling milestones: keep only the newest N step_*.pt '
                        '(latest.pt/best_model.pt untouched; 0 = keep all)')
    p.add_argument('--no_chat_vocab', action='store_true',
                   help='streaming pretrain: use plain gpt2 (50257) instead of the '
                        'ChatML+reasoning tokenizer (50261). Default keeps the chat '
                        'tokens so the base can be SFT-ed later.')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_v13_sempty')
    p.add_argument('--gradient_checkpointing', action='store_true', default=False,
                   help='recompute blocks in backward; the memory lever for 16-layer runs')
    p.add_argument('--log_interval', type=int, default=50,
                   help='steps between training log lines (v7 default 50)')
    p.add_argument('--val_every', type=int, default=500,
                   help='steps between validation PPL evals (0=off)')
    p.add_argument('--gen_every', type=int, default=2000,
                   help='steps between in-loop generation samples (0=off)')
    p.add_argument('--gen_prompt', type=str, default='In 1923, the University of')
    p.add_argument('--gen_max_tokens', type=int, default=80)
    p.add_argument('--save_every_steps', type=int, default=2000,
                   help='steps between latest.pt checkpoints (0=only final)')
    p.add_argument('--diag_every', type=int, default=2000,
                   help='steps between per-layer diagnostic panels (0=off)')
    p.add_argument('--max_val_batches', type=int, default=128,
                   help='cap on val batches per eval (0=all)')
    p.add_argument('--recall_frac', type=float, default=0.0,
                   help='fraction of TRAIN samples replaced by synthetic recall '
                        'docs (v7._build_recall_doc). 0=off. Val is never mixed.')
    # Architecture-ladder overrides (real arm; see EXPERIMENTS_SEMPY). Each
    # toggles a cfg field on top of the chosen preset.
    p.add_argument('--n_states', type=int, default=None, help='A2: PAM states per head')
    p.add_argument('--vault', action='store_true', help='A2b: pinned state + protect gate')
    p.add_argument('--delta', action='store_true', help='A3: delta erase/write')
    p.add_argument('--cond_mem', action='store_true', help='A4: conditional n-gram memory')
    p.add_argument('--chrono', action='store_true',
                   help='N1: Chrono-PAM content-modulated rotary retention (real arm)')
    p.add_argument('--out_gate', action='store_true',
                   help='N4: per-head content-dependent read-out gate (real arm)')
    return p


def main():
    args = build_argparser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but not available; falling back to CPU')
        args.device = 'cpu'
    seed_everything(args.seed)
    cfg = get_config(args.preset)
    cfg.gradient_checkpointing = args.gradient_checkpointing
    if args.seq_len:
        cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    # Architecture-ladder overrides.
    if args.n_states is not None:
        cfg.n_states = args.n_states
    if args.vault:
        cfg.vault = True
    if args.delta:
        cfg.delta = True
    if args.cond_mem:
        cfg.cond_mem = True
    if args.chrono:
        cfg.chrono = True
    if args.out_gate:
        cfg.out_gate = True
    if args.dropout is not None:
        cfg.dropout = args.dropout
    device = torch.device(args.device)
    set_kernel_enabled(args.fused_pam)

    val_loader = None
    wiki_val_loader = None
    is_streaming = False
    data_cursor = None
    # --- resume: read the checkpoint BEFORE the data pipeline is built so the
    # stream can be re-opened at the saved cursor -------------------------------
    resume_ck, resume_path = None, None
    if args.resume:
        cand = (Path(args.checkpoint_dir) / 'latest.pt' if args.resume == 'auto'
                else Path(args.resume))
        if cand.exists():
            resume_path = cand
            resume_ck = Trainer.peek_resume(cand)
        elif args.resume != 'auto':
            raise FileNotFoundError(f"--resume {cand} does not exist")
        else:
            print(f"[resume] auto: no {cand}; starting fresh", flush=True)
    STREAM = {'dclm', 'fineweb', 'mix'}
    if args.dataset == 'synthetic':
        vocab = min(cfg.vocab_size, 256)
        cfg.vocab_size = vocab
        loader = synthetic_loader(vocab, args.batch_size, args.seq_len, args.steps, args.seed)
        tokenizer = None
    elif args.dataset in STREAM:
        # Streaming pretrain via the shared v7 pipeline (text iters -> weighted
        # interleave -> StreamingTokenChunkDataset -> {input_ids, labels}).
        from v7.data import load_pretrain_mix, load_wikitext103_val
        if args.dataset == 'dclm':
            sources = ('dclm',)
        elif args.dataset == 'fineweb':
            sources = ('fineweb',)
        else:
            sources = tuple(s.strip() for s in args.pretrain_sources.split(',')
                            if s.strip())
        weights = (tuple(float(w) for w in args.pretrain_weights.split(','))
                   if args.pretrain_weights else None)
        target = args.target_tokens if args.target_tokens > 0 else None
        # ChatML + reasoning tokens (<|im_start|> <|im_end|> <think> </think>,
        # vocab 50261) are needed by the later SFT, so the base is trained with
        # them from step 0. Default ON for the streaming pretrain; the implicit
        # triggers (smoltalk source / _chat preset) still force it on.
        use_chat_vocab = (not args.no_chat_vocab or cfg.vocab_size > 50257
                          or args.preset.endswith('_chat')
                          or any(s.startswith('smoltalk') for s in sources))
        # Stream cursor. `doc_counters` = docs each source has handed to the
        # tokenizer (post-filter), `token_counters` = tokens yielded per source
        # (also what the blend warmup measures itself against). Both live dicts
        # are mutated by the pipeline and saved in every checkpoint. On resume
        # every source skips its consumed docs, so no document is trained on
        # twice; the <=shuffle_buffer chunks that were read but not yet yielded
        # at save time are lost (<=20M tok per restart), not repeated.
        doc_counters: dict = {}
        token_counters: dict = {}
        skip_docs = None
        consumed_tokens = 0
        if resume_ck is not None:
            cur = resume_ck.get('data_cursor') or {}
            doc_counters.update(cur.get('doc_counters') or {})
            token_counters.update(cur.get('token_counters') or {})
            skip_docs = dict(doc_counters)
            consumed_tokens = int(resume_ck['global_tokens'])
        data_cursor = {'doc_counters': doc_counters, 'token_counters': token_counters}
        remaining_budget = (max(target - consumed_tokens, 1) if target is not None
                            else None)
        # use_cache=False: the token cache is keyed on the full budget and would
        # otherwise be consulted with a shifted (remaining) budget on resume.
        train_ds, val_ds, tokenizer = load_pretrain_mix(
            seq_len=args.seq_len, edu_score_min=args.edu_score_min,
            token_budget=remaining_budget, sources=sources, weights=weights,
            chat_vocab=use_chat_vocab, fineweb_name=args.fineweb_name,
            holdout_pct=args.holdout_pct, mix_seed=args.seed,
            blend_warmup_tokens=args.blend_warmup_tokens,
            skip_docs=skip_docs, token_counters=token_counters,
            doc_counters=doc_counters, use_cache=False,
            answer_weight=args.answer_weight,
        )
        cfg.vocab_size = len(tokenizer)
        is_streaming = not getattr(train_ds, 'pretrain_cached', False)
        # Streaming stops on the token budget; derive the step count so the LR
        # schedule and the header's planned_tokens stay honest.
        if target is not None:
            args.steps = max(1, target // (args.batch_size * args.seq_len))
        if resume_ck is not None:
            print(f"[resume] stream re-opened at docs={skip_docs} "
                  f"tokens_consumed={consumed_tokens:,} "
                  f"remaining_budget={remaining_budget:,}", flush=True)
        nw = 0 if is_streaming else args.num_workers
        loader = DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=not is_streaming, num_workers=nw,
                            drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        if not args.no_wiki_val:
            wv, _ = load_wikitext103_val(seq_len=args.seq_len)
            wiki_val_loader = DataLoader(wv, batch_size=args.batch_size,
                                         shuffle=False)
        print(f"streaming pretrain: sources={sources} weights={weights} "
              f"target_tokens={target} steps={args.steps} "
              f"cached={not is_streaming} chat_vocab={use_chat_vocab} "
              f"wiki_val={not args.no_wiki_val}")
    else:
        train_ds, val_ds, tokenizer = load_real_dataset(
            args.dataset, args.seq_len, args.max_samples,
        )
        tok_vocab = len(tokenizer)
        if tok_vocab != cfg.vocab_size:
            print(f"Adjusting vocab_size: {cfg.vocab_size} -> {tok_vocab}")
            cfg.vocab_size = tok_vocab
        if args.recall_frac > 0.0:
            from v13_sempty.data_mix import RecallMixDataset
            train_ds = RecallMixDataset(
                train_ds, frac=args.recall_frac, seq_len=args.seq_len,
                tokenizer=tokenizer, seed=args.seed)
            print(f"recall mix: {args.recall_frac:.1%} of train samples "
                  f"are synthetic recall docs (val unmixed)")
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = LM(cfg)
    params = model.count_parameters()
    _print_run_header(args, cfg, model, params, device, loader, val_loader,
                      tokenizer)

    fused = args.fused_ce and not args.no_fused_ce
    trainer = Trainer(
        model, loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        gradient_clip=args.gradient_clip,
        amp_dtype_str=args.amp_dtype,
        fused_ce=fused,
        fused_ce_chunk=args.fused_ce_chunk,
        ce_gemm_dtype=torch.float32 if args.ce_gemm_dtype == 'fp32' else None,
        device=device,
        log_interval=args.log_interval,
        val_loader=val_loader,
        wiki_val_loader=wiki_val_loader,
        is_streaming=is_streaming,
        tokenizer=tokenizer,
        gen_every=args.gen_every,
        gen_prompt=args.gen_prompt,
        gen_max_tokens=args.gen_max_tokens,
        save_every_steps=args.save_every_steps,
        keep_every_steps=args.keep_every_steps,
        keep_last=args.keep_last,
        val_every=args.val_every,
        diag_every=args.diag_every,
        max_val_batches=args.max_val_batches if args.max_val_batches > 0 else None,
        checkpoint_dir=Path(args.checkpoint_dir),
        run_label=args.preset,
        data_cursor=data_cursor,
        run_args=vars(args),
    )
    if resume_ck is not None:
        saved_args = resume_ck.get('args') or {}
        for k in ('preset', 'batch_size', 'seq_len', 'lr', 'warmup_steps',
                  'target_tokens', 'pretrain_sources', 'pretrain_weights', 'seed'):
            if k in saved_args and saved_args[k] != getattr(args, k):
                print(f"[resume] WARNING: --{k} differs from the checkpoint "
                      f"({saved_args[k]!r} -> {getattr(args, k)!r})", flush=True)
        trainer.load_resume(resume_ck, resume_path)
        del resume_ck
    losses = trainer.train(max_steps=args.steps)
    print(f"\nTraining complete. steps={trainer.global_step} "
          f"tokens={trainer.global_tokens:,} "
          f"best_val_ppl={trainer.best_val_ppl:.2f}")
    if len(losses) >= 2:
        print(f"train loss: {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"(delta {losses[-1] - losses[0]:+.4f})")
    trainer._save_ckpt('latest.pt')
    if is_streaming:
        # HF streaming (pyarrow/aiohttp worker threads) dies with
        # "Fatal Python error: PyGILState_Release" during interpreter teardown
        # AFTER everything is saved (smoke 2026-09-04: exit 134 on a complete
        # run). Flush and leave without running finalizers so the wrapper's
        # exit code reflects the training, not the teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
