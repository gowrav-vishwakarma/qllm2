"""V8 staged trainer.

Stages
------

* ``A``  -- train the QPAM backbone from scratch on WikiText-103. QLC is
            bypassed (passthrough). Output checkpoint feeds Stage B.
* ``B``  -- freeze the backbone, train QLC + EffectAlgebraBank + OrthoHalt
            with LM cross-entropy + ponder cost (+ optional InfoNCE on the
            synthetic entity cloze).
* ``C``  -- joint fine-tune backbone + QLC at low LR, with KL anchor to the
            Stage A logits to prevent grammar drift.

Usage
-----

::

    # Stage A from scratch on RTX 4090:
    uv run python -m v8.train --preset stageA_medium --epochs 10

    # Stage B with frozen backbone (assumes Stage A checkpoint exists):
    uv run python -m v8.train --preset stageB_T4 \\
        --backbone_ckpt v8/checkpoints/qpam_stageA.pt --epochs 5

    # TinyStories smoke gate (Stage A.5):
    uv run python -m v8.train --preset smoke_tiny_qlc_r4_T2 \\
        --dataset tinystories --epochs 3 --batch_size 16

See ``scripts/run_v8_*`` for fully configured launch commands.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import logging
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Suppress two harmless torch.compile / inductor warnings ────────────────
# 1) ``UserWarning: Online softmax is disabled on the fly ...`` -- inductor
#    perf hint, says it picked the regular split-reduction path because it
#    estimated that's faster for our shapes. The warning wraps across newlines
#    so ``filterwarnings`` matches against the FIRST line only; we anchor the
#    regex on the first line's text so the filter actually fires.
warnings.filterwarnings(
    "ignore",
    message=r"^\s*Online softmax is disabled on the fly.*",
    category=UserWarning,
)
# 2) ``W... _maybe_guard_rel() was called on non-relation expression
#    Eq(s29, 1) | Eq(s29, s11)`` -- internal symbolic-shape guard noise from
#    inductor's simplifier. Comes from the QLC's ``min(rank, k)`` branch in
#    ``_qr_basis``. The ``W0424...`` glog-style format is emitted by
#    PyTorch's ``torch._logging._init_logs()`` -- that init runs lazily on
#    first dynamo invocation and OVERRIDES any level set via plain
#    ``logging.getLogger(...).setLevel()``. We must use the official
#    ``torch._logging.set_logs(...)`` API instead, applied right after
#    ``import torch`` so it survives the lazy reinit. Belt-and-suspenders:
#    we ALSO add a logging filter that drops the specific noisy message,
#    in case a future PyTorch removes the ``symbolic_shapes`` registered
#    key.
class _DropMaybeGuardRel(logging.Filter):
    def filter(self, record):  # type: ignore[override]
        return "_maybe_guard_rel() was called on non-relation" not in record.getMessage()


_SYMSHAPE_LOGGER = logging.getLogger("torch.fx.experimental.symbolic_shapes")
_SYMSHAPE_FILTER = _DropMaybeGuardRel()


def _ensure_symshape_filter() -> None:
    """Idempotently attach the noise filter; safe to call multiple times."""
    if _SYMSHAPE_FILTER not in _SYMSHAPE_LOGGER.filters:
        _SYMSHAPE_LOGGER.addFilter(_SYMSHAPE_FILTER)


_ensure_symshape_filter()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Re-attach the symbolic_shapes filter AFTER importing torch. Python's
# ``logging`` module keeps filters attached to the logger object across
# handler reinitializations, so this survives PyTorch's lazy
# ``_init_logs()`` call on first dynamo invocation. We do NOT use
# ``torch._logging.set_logs(symbolic_shapes=...)`` because that key is not
# registered in every PyTorch version (raises ``TypeError`` if absent).
_ensure_symshape_filter()

# Enable TF32 on Ampere+ GPUs (RTX 4090 is Ada Lovelace -- supported). TF32
# trims float32 matmul mantissas from 23 to 10 bits and runs on the tensor
# cores, giving ~1.5-2x speedup with accuracy loss well below typical
# gradient noise. Affects float32 ops outside of AMP autocast: the optimizer
# step, the QLC ``_qr_basis`` (which runs eager, see effect_bank.py) and any
# other fp32-precision sites. AMP-managed bf16/fp16 ops are unaffected.
# Use the modern ``set_float32_matmul_precision`` API; ``'high'`` = TF32.
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

sys.path.insert(0, str(Path(__file__).parent.parent))

from v8.config import V8Config, get_config, PRESETS
from v8.model import V8LM, load_backbone_from_v7_checkpoint
from v8.data import (
    TeeLogger,
    load_wikitext103,
    load_tinystories,
    compute_text_quality,
    resolve_amp_dtype,
    build_lr_scheduler,
    build_param_groups,
    EntityClozeDataset,
    make_entity_cloze_loaders,
)


# ── Reproducibility ──────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Discord (optional) ───────────────────────────────────────────────────────


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
            headers={"Content-Type": "application/json", "User-Agent": "qllm2-v8/1.0"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[Discord] Webhook send failed: {e}", file=sys.stderr)


def _notify_discord_long(text: str, *, limit: int = 1900) -> None:
    """Send ``text`` to Discord, splitting into multiple messages at line
    boundaries so we never exceed the 2000-char webhook limit.
    """
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


# ── QLC runtime-safe schedule (single-run-v8-training plan v2) ───────────────


@dataclass
class QLCRuntimeSchedule:
    """Runtime-safe in-run schedule for the e2e single-run preset.

    Mutates ONLY attributes that are read fresh on every forward pass and
    that do not change tensor shapes or parameter identity:

    * ``model.qlc.t_max``           -- read by the loop's ``range(self.t_max)``
    * ``model.qlc_cfg.ponder_lambda`` -- multiplied into the aux loss in
      :meth:`V8LM.forward`
    * ``model.qlc.bank_temperature`` -- passed to ``bank.select_top_k`` each call

    Anything that would require rebuilding modules or resetting the
    optimizer (``halt_mode``, ``out_scale_learnable``, ``rank``, ``bank_size``,
    ``use_complex``) is intentionally NOT supported here; do soft warmup via
    ``out_scale_init`` in the preset instead.

    Phase boundaries are expressed as fractions of ``total_steps`` so the
    same schedule scales with run length.
    """

    enabled: bool = False
    warmup_end_frac: float = 1.0 / 3.0
    mid_end_frac: float = 2.0 / 3.0
    t_max_phases: Tuple[int, int, int] = (2, 3, 4)
    # v8.2: smaller absolute ponder_lambda values across all phases. The
    # original (0.0, 0.005, 0.01) ramp over-penalised the loop in the late
    # phase exactly when the alignment-aux + InfoNCE were finally letting
    # alpha/gamma carry signal, collapsing mean_iter back to ~1. The new
    # values keep the same warmup-zero shape but cap late-phase pressure.
    ponder_lambda_phases: Tuple[float, float, float] = (0.0, 0.002, 0.005)
    bank_temperature_phases: Optional[Tuple[float, float, float]] = None

    def phase_index(self, step: int, total_steps: int) -> int:
        if total_steps <= 0:
            return 2
        frac = step / float(total_steps)
        if frac < self.warmup_end_frac:
            return 0
        if frac < self.mid_end_frac:
            return 1
        return 2


# ── Trainer ──────────────────────────────────────────────────────────────────


class V8Trainer:
    """Stage-aware V8 trainer.

    Differences vs the v7 trainer:

    * Optimizer is built from ``model.parameters()`` *after* freezing, so
      Stage B only updates QLC params.
    * Adds ponder-cost into the loss (already weighted by ``ponder_lambda`` by
      the model itself).
    * Stage C: each batch also computes a KL term against the Stage A logits
      (recomputed with the *current* backbone in eval mode -- this anchors
      the joint update toward not drifting too far from Stage A).
    * Periodic QLC diagnostics logging (mean iter, alpha/beta/gamma).
    """

    def __init__(
        self,
        model: V8LM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        tokenizer,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        gradient_clip: float = 1.0,
        max_epochs: int = 10,
        checkpoint_dir: str = "v8/checkpoints",
        amp_dtype_str: str = "auto",
        compile_model: bool = False,
        compile_mode: str = "default",
        gen_every: int = 0,
        gen_prompt: str = "The",
        log_interval: int = 50,
        start_epoch: int = 0,
        kl_anchor_weight: float = 0.0,
        diag_every: int = 200,
        infonce_loader: Optional[DataLoader] = None,
        infonce_weight: float = 0.0,
        infonce_every: int = 1,
        qlc_schedule: Optional[QLCRuntimeSchedule] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.max_epochs = max_epochs
        self.gradient_clip = gradient_clip
        self.kl_anchor_weight = kl_anchor_weight
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gen_every = gen_every
        self.gen_prompt = gen_prompt
        self.log_interval = log_interval
        self.start_epoch = start_epoch
        self.diag_every = diag_every

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        param_groups = build_param_groups(model, weight_decay)
        adamw_kwargs = dict(lr=learning_rate, betas=(0.9, 0.95))
        if torch.cuda.is_available():
            adamw_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)

        total_steps = max(max_epochs * len(train_loader), 1)
        self.scheduler = build_lr_scheduler(
            self.optimizer, "warmup_cosine", warmup_steps, total_steps,
        )

        self.amp_dtype = resolve_amp_dtype(amp_dtype_str)
        self.use_amp = self.amp_dtype is not None
        self.scaler = (
            torch.amp.GradScaler("cuda")
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
        self.best_val_loss = float("inf")
        self.best_val_ppl = float("inf")

        self.qlc_schedule = qlc_schedule
        self._qlc_phase_current = -1
        self._qlc_total_steps = total_steps
        if (
            self.qlc_schedule is not None
            and self.qlc_schedule.enabled
            and self._orig().qlc is not None
        ):
            print(
                f"QLC runtime schedule: ENABLED "
                f"(warmup<{self.qlc_schedule.warmup_end_frac:.2f} "
                f"mid<{self.qlc_schedule.mid_end_frac:.2f}; "
                f"t_max={self.qlc_schedule.t_max_phases} "
                f"ponder_lambda={self.qlc_schedule.ponder_lambda_phases})"
            )
            # Apply phase 0 immediately so the very first batch sees the
            # warmup values rather than the preset defaults.
            self._apply_qlc_schedule(force=True)

        self.infonce_loader = infonce_loader
        self.infonce_weight = float(infonce_weight)
        self.infonce_every = max(1, int(infonce_every))
        self._infonce_iter = None
        self._infonce_loss_ema = 0.0
        if self.infonce_loader is not None and self.infonce_weight > 0:
            self._infonce_iter = self._infinite_iter(self.infonce_loader)
            print(
                f"InfoNCE auxiliary: weight={self.infonce_weight} "
                f"every={self.infonce_every} step(s) "
                f"({len(self.infonce_loader)} cloze batches/epoch)"
            )

    @staticmethod
    def _infinite_iter(loader: DataLoader):
        """Infinite cycler over a DataLoader (used for the InfoNCE aux)."""
        while True:
            for batch in loader:
                yield batch

    def _next_infonce_batch(self) -> Optional[dict]:
        if self._infonce_iter is None:
            return None
        try:
            return next(self._infonce_iter)
        except StopIteration:
            self._infonce_iter = self._infinite_iter(self.infonce_loader)
            return next(self._infonce_iter)

    def _compute_infonce_loss(self, batch: dict) -> torch.Tensor:
        """Run the bank's InfoNCE auxiliary on one entity-cloze batch.

        Picks the post-backbone hidden state at each sample's ``answer_pos``
        as the positive query, and uses circular-shifted in-batch samples
        as negatives. Gold effect index = ``entity_idx % bank_size``.
        """
        m = self._orig()
        if m.qlc is None:
            return torch.tensor(0.0, device=self.device)

        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        answer_pos = batch["answer_pos"].to(self.device, non_blocking=True)
        entity_idx = batch["entity_idx"].to(self.device, non_blocking=True)
        B = input_ids.shape[0]
        if B < 2:
            # Need at least one negative.
            return torch.tensor(0.0, device=self.device)

        psi = m.encode_backbone_state(input_ids)              # [B, T, d, 2]
        b_idx = torch.arange(B, device=self.device)
        psi_pos = psi[b_idx, answer_pos]                       # [B, d, 2]

        n_neg = min(B - 1, 8)
        # Circular-shift indices: for sample b, negatives are b+1, ..., b+n_neg
        shifts = (torch.arange(1, n_neg + 1, device=self.device)
                  .unsqueeze(0).expand(B, -1))                 # [B, n_neg]
        neg_b = (b_idx.unsqueeze(-1) + shifts) % B             # [B, n_neg]
        neg_pos = answer_pos[neg_b]                            # [B, n_neg]
        flat_b = neg_b.reshape(-1)
        flat_pos = neg_pos.reshape(-1)
        psi_neg_flat = psi[flat_b, flat_pos]                   # [B*n_neg, d, 2]
        psi_neg = psi_neg_flat.view(B, n_neg, *psi_neg_flat.shape[1:])

        gold = (entity_idx % m.qlc.bank.bank_size).long()
        return m.qlc.bank.infonce_loss(psi_pos, psi_neg, gold, temperature=0.1)

    # ── Train / eval loops ─────────────────────────────────────────────────

    def _orig(self) -> V8LM:
        return self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

    def _apply_qlc_schedule(self, force: bool = False) -> None:
        """Mutate runtime-safe QLC knobs based on the current global step.

        See :class:`QLCRuntimeSchedule` for the rules. Idempotent within a
        phase: only logs and writes when the phase index changes (or when
        ``force=True``, used during init to apply phase 0 before step 0).
        """
        sched = self.qlc_schedule
        if sched is None or not sched.enabled:
            return
        m = self._orig()
        if m.qlc is None:
            return
        phase = sched.phase_index(self.global_step, self._qlc_total_steps)
        if not force and phase == self._qlc_phase_current:
            return
        self._qlc_phase_current = phase
        new_t_max = max(1, int(sched.t_max_phases[phase]))
        new_lambda = float(sched.ponder_lambda_phases[phase])
        m.qlc.t_max = new_t_max
        m.qlc_cfg.ponder_lambda = new_lambda
        if sched.bank_temperature_phases is not None:
            new_temp = float(sched.bank_temperature_phases[phase])
            m.qlc.bank_temperature = new_temp
            temp_str = f" bank_temperature={new_temp:.3f}"
        else:
            temp_str = ""
        print(
            f"  [qlc-schedule] step={self.global_step}/"
            f"{self._qlc_total_steps} phase={phase} "
            f"t_max={new_t_max} ponder_lambda={new_lambda:.4f}{temp_str}"
        )

    def profile_one_shot(self, total_steps: int, output_dir: str | Path) -> None:
        """Capture a torch.profiler trace for the first ``total_steps``.

        Uses the standard wait/warmup/active schedule. The trace is written
        as Chrome JSON to ``output_dir`` and a textual top-30 summary
        (sorted by self CUDA time) is printed.

        Note: this exits the process after writing. It deliberately runs the
        *eager* model (no torch.compile) so the kernel breakdown is the
        unfused baseline we want to characterise.
        """
        from torch.profiler import (
            profile, record_function, schedule, ProfilerActivity, tensorboard_trace_handler,
        )
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        wait = 5
        warmup = 5
        active = max(1, total_steps - wait - warmup)
        sched = schedule(wait=wait, warmup=warmup, active=active, repeat=1)

        m = self._orig()
        m.train()

        loader_iter = iter(self.train_loader)
        print(
            f"\n[profile] capturing {total_steps} steps "
            f"(wait={wait}, warmup={warmup}, active={active}) -> {out_dir}"
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            schedule=sched,
            on_trace_ready=tensorboard_trace_handler(str(out_dir)),
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            for step in range(total_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                with torch.amp.autocast(
                    self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_dtype or torch.float16,
                ):
                    with record_function("v8_forward"):
                        logits, _, ponder_term = self.model(
                            input_ids, labels=labels, return_qlc_diag=False,
                        )
                        main_loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), labels.view(-1),
                        )
                        loss = main_loss + ponder_term

                with record_function("v8_backward"):
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                    else:
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.gradient_clip,
                    )

                with record_function("v8_optim"):
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                prof.step()

        torch.cuda.synchronize()
        print("\n[profile] top 30 ops by self CUDA time:")
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=30,
            )
        )
        print(f"\n[profile] trace written under {out_dir}")
        print(
            "[profile] open with: chrome://tracing or "
            "https://ui.perfetto.dev (drag the .json/.gz file)"
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        # When backbone is frozen, force its dropout/checkpoint modules to eval
        # to avoid stale gradients on the frozen weights.
        m = self._orig()
        if m.config.freeze_backbone:
            m.embed.eval()
            m.embed_norm.eval()
            for blk in m.blocks:
                blk.eval()
            m.output_norm.eval()
            if not m.config.unfreeze_lm_head:
                m.lm_head_proj.eval()
                m.lm_head_norm.eval()

        total_loss_w = 0.0
        total_ponder = 0.0
        total_tokens = 0
        epoch_start = time.time()
        log_start = epoch_start
        log_tokens = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            self._apply_qlc_schedule()

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            batch_tokens = input_ids.shape[0] * input_ids.shape[1]

            # Gate on ``batch_idx`` (resets per epoch) rather than
            # ``global_step`` (monotonic across epochs). The print below is
            # ALSO gated on ``batch_idx % self.log_interval == 0``; using
            # global_step here meant the two gates only co-aligned in
            # epoch 1 where ``global_step == batch_idx``. From epoch 2 on
            # the diag was computed at orphan steps (still paying the
            # ``return_qlc_diag=True`` forward cost) but never printed.
            # Assumes ``diag_every`` is a multiple of ``log_interval`` (the
            # launcher uses 100 / 50 by default).
            want_diag = (
                self.diag_every > 0
                and batch_idx % self.diag_every == 0
                and m.qlc is not None
            )

            with torch.amp.autocast(
                self.device.type,
                enabled=self.use_amp,
                dtype=self.amp_dtype or torch.float16,
            ):
                logits, _, ponder_term = self.model(
                    input_ids, labels=labels, return_qlc_diag=want_diag,
                )
                main_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1),
                )
                loss = main_loss + ponder_term

                infonce_loss_val = 0.0
                if (
                    self._infonce_iter is not None
                    and self.infonce_weight > 0
                    and self.global_step % self.infonce_every == 0
                ):
                    cloze_batch = self._next_infonce_batch()
                    if cloze_batch is not None:
                        infonce_loss = self._compute_infonce_loss(cloze_batch)
                        loss = loss + self.infonce_weight * infonce_loss
                        infonce_loss_val = float(infonce_loss.detach().item())
                        # Track an EMA so the periodic logger has something
                        # informative to print between InfoNCE steps.
                        self._infonce_loss_ema = (
                            0.9 * self._infonce_loss_ema + 0.1 * infonce_loss_val
                        )

                if self.kl_anchor_weight > 0:
                    with torch.no_grad():
                        target_logits = m.backbone_logits(input_ids)
                    kl = F.kl_div(
                        F.log_softmax(logits, dim=-1),
                        F.softmax(target_logits, dim=-1),
                        reduction="batchmean",
                    )
                    loss = loss + self.kl_anchor_weight * kl

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.gradient_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.gradient_clip,
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
            total_ponder += float(ponder_term.detach().item()) * batch_tokens

            if batch_idx % self.log_interval == 0:
                ppl = math.exp(min(main_loss_val, 20))
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                avg_tok_s = total_tokens / max(elapsed, 1e-6)
                inst_elapsed = time.time() - log_start
                inst_tok_s = log_tokens / max(inst_elapsed, 1e-6)
                pct = 100.0 * (batch_idx + 1) / len(self.train_loader)
                line = (
                    f"  [{epoch+1}] {batch_idx}/{len(self.train_loader)} "
                    f"({pct:.0f}%) loss={main_loss_val:.4f} ppl={ppl:.1f} "
                    f"lr={lr:.2e} | {inst_tok_s:.0f} tok/s (avg {avg_tok_s:.0f})"
                )
                if ponder_term.detach().item() > 0:
                    line += f" | ponder={ponder_term.detach().item():.3e}"
                if self._infonce_iter is not None and self.infonce_weight > 0:
                    line += f" | infonce={self._infonce_loss_ema:.3f}"
                if self.device.type == "cuda":
                    mem = torch.cuda.memory_allocated() / 1e9
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    line += f" | GPU {mem:.1f}/{peak:.1f}GB"
                print(line)
                log_start = time.time()
                log_tokens = 0

                if want_diag:
                    diag = m.last_qlc_diagnostics()
                    if diag is not None:
                        print(
                            f"    QLC: iter={diag.mean_iter:.2f} "
                            f"alpha={diag.mean_alpha:.3f} beta={diag.mean_beta:.3f} "
                            f"gamma={diag.mean_gamma:.3f} "
                            f"halt(yes/no/cont)="
                            f"{diag.halt_yes_rate:.2f}/{diag.halt_no_rate:.2f}/"
                            f"{diag.continue_rate:.2f} | "
                            f"align={diag.mean_amp:.4f} "
                            f"out_scale={diag.out_scale:.3f} "
                            f"psi_delta_l2={diag.psi_delta_l2:.3e}"
                        )

            if (
                self.gen_every > 0
                and batch_idx > 0
                and batch_idx % self.gen_every == 0
                and self.tokenizer is not None
            ):
                try:
                    text = self._generate_sample(self.gen_prompt)
                    avg_tok_s = total_tokens / max(time.time() - epoch_start, 1e-6)
                    print(
                        f"  [mid-epoch sample @ batch {batch_idx}, "
                        f"{self.global_tokens:,} tok]"
                    )
                    print(f"  Prompt: {self.gen_prompt}")
                    print(f"  Generated: {text}")
                    qm = compute_text_quality(text)
                    print(
                        f"  Quality: rep3={qm['repeat_3gram']:.3f} "
                        f"rep4={qm['repeat_4gram']:.3f} "
                        f"restarts={qm['restart_frag']:.0f} "
                        f"uniq={qm['unique_word_ratio']:.3f}"
                    )
                    inst_ppl = math.exp(min(main_loss_val, 20))
                    inst_lr = self.scheduler.get_last_lr()[0]
                    _notify_discord(
                        f"**V8 [{m.config.stage}] [gen_every]** "
                        f"Epoch {epoch+1} batch {batch_idx} "
                        f"({self.global_tokens:,} tok)\n"
                        f"loss={main_loss_val:.4f} ppl={inst_ppl:.1f} "
                        f"lr={inst_lr:.2e} | {avg_tok_s:.0f} tok/s\n"
                        f"Prompt: {self.gen_prompt}\n"
                        f"Generated: "
                        f"{(text[:800] + '...') if len(text) > 800 else text}"
                    )
                except Exception as e:
                    print(f"  (mid-epoch sample failed: {e})")

        epoch_elapsed = time.time() - epoch_start
        avg_tok_s = total_tokens / max(epoch_elapsed, 1e-6)
        avg_loss = total_loss_w / max(total_tokens, 1)
        avg_ponder = total_ponder / max(total_tokens, 1)
        return {
            "loss": avg_loss,
            "ppl": math.exp(min(avg_loss, 20)),
            "avg_tok_s": avg_tok_s,
            "epoch_tokens": total_tokens,
            "ponder": avg_ponder,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}
        self.model.eval()
        total_loss_w = 0.0
        total_tokens = 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            batch_tokens = input_ids.shape[0] * input_ids.shape[1]
            with torch.amp.autocast(
                self.device.type,
                enabled=self.use_amp,
                dtype=self.amp_dtype or torch.float16,
            ):
                logits, _, _ = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1),
                )
            total_loss_w += loss.item() * batch_tokens
            total_tokens += batch_tokens
        if total_tokens == 0:
            return {}
        avg_loss = total_loss_w / total_tokens
        return {"val_loss": avg_loss, "val_ppl": math.exp(min(avg_loss, 20))}

    @torch.no_grad()
    def _generate_sample(self, prompt: str = "The", max_tokens: int = 80) -> str:
        was_training = self.model.training
        self.model.eval()
        try:
            m = self._orig()
            ids = self.tokenizer.encode(prompt)
            t = torch.tensor([ids], device=self.device)
            out = m.generate(
                t, max_new_tokens=max_tokens,
                temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2,
            )
            return self.tokenizer.decode(out[0].tolist())
        finally:
            if was_training:
                self.model.train()

    def save_checkpoint(self, name: str, epoch: int) -> Path:
        path = self.checkpoint_dir / name
        m = self._orig()
        ckpt = {
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "global_tokens": self.global_tokens,
            "best_val_loss": self.best_val_loss,
            "best_val_ppl": self.best_val_ppl,
            "epoch": epoch,
            "v8_config": {
                "backbone": asdict(m.backbone_cfg),
                "qlc": asdict(m.qlc_cfg),
                "stage": m.config.stage,
                "freeze_backbone": m.config.freeze_backbone,
                "kl_anchor_weight": m.config.kl_anchor_weight,
            },
        }
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")
        return path

    def load_full_state(self, ckpt: dict) -> int:
        """Restore optimizer / scheduler / counters from a saved checkpoint.

        Assumes ``ckpt["model_state_dict"]`` was already loaded into the
        underlying model (so weights are present and on the right device
        before optimizer state is rehydrated). Restores everything
        :meth:`save_checkpoint` writes EXCEPT ``model_state_dict``:

        * ``optimizer_state_dict`` -- AdamW first/second moments (without
          this, resume effectively wipes momentum and spikes the loss).
        * ``scheduler_state_dict`` -- cosine LR position (without this, LR
          re-warms over ``warmup_steps`` from ~0 instead of continuing).
        * ``global_step`` / ``global_tokens`` -- so the runtime QLC
          schedule (:meth:`_apply_qlc_schedule`) and the InfoNCE cadence
          stay in their correct phase, and token counters keep counting.
        * ``best_val_loss`` / ``best_val_ppl`` -- so we don't trivially
          overwrite ``best_model.pt`` with a worse epoch right after
          resume.

        Returns the epoch index to start training from
        (``ckpt["epoch"] + 1``), or 0 if the checkpoint has no ``epoch``.

        All keys are optional: legacy weight-only checkpoints simply skip
        the missing fields.
        """
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"  [resume] optimizer state load failed: {e}")
        if "scheduler_state_dict" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"  [resume] scheduler state load failed: {e}")
        self.global_step = int(ckpt.get("global_step", 0))
        self.global_tokens = int(ckpt.get("global_tokens", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        self.best_val_ppl = float(ckpt.get("best_val_ppl", float("inf")))
        # Re-evaluate QLC phase against the restored step. Force=True
        # because phase index might be the SAME numerical value as the
        # init-time phase 0 we already applied, but the underlying
        # state needs to be re-asserted on the fresh optimizer/model.
        self._qlc_phase_current = -1
        self._apply_qlc_schedule(force=True)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.start_epoch = start_epoch
        return start_epoch

    def train(self) -> None:
        training_start = time.time()
        m = self._orig()
        # Per-epoch loop only -- the rich start-of-run summary (console +
        # Discord) is built and dispatched in main() so we don't duplicate it
        # here. We just emit the per-device confirmation locally.
        print(f"\nTraining on {self.device}\n")

        for epoch in range(self.start_epoch, self.max_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.max_epochs}")
            print("=" * 60)

            train_metrics = self.train_epoch(epoch)
            line = (
                f"Epoch {epoch+1}/{self.max_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"PPL: {train_metrics['ppl']:.2f} | "
                f"{train_metrics['avg_tok_s']:.0f} tok/s | "
                f"ponder={train_metrics['ponder']:.3e}"
            )

            is_best = False
            if self.val_loader is not None and len(self.val_loader) > 0:
                val_metrics = self.validate()
                line += (
                    f" | Val Loss: {val_metrics['val_loss']:.4f} "
                    f"PPL: {val_metrics['val_ppl']:.2f}"
                )
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.best_val_ppl = val_metrics["val_ppl"]
                    line += " *best*"
                    is_best = True
            print(line)

            if is_best:
                self.save_checkpoint("best_model.pt", epoch)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", epoch)

            if self.tokenizer is not None:
                try:
                    text = self._generate_sample(self.gen_prompt)
                    print(f"\nPrompt: {self.gen_prompt}")
                    print(f"Generated: {text}")
                    qm = compute_text_quality(text)
                    print(
                        f"  Quality: rep3={qm['repeat_3gram']:.3f} "
                        f"rep4={qm['repeat_4gram']:.3f} "
                        f"restarts={qm['restart_frag']:.0f} "
                        f"uniq={qm['unique_word_ratio']:.3f}"
                    )
                except Exception as e:
                    print(f"(Sample generation failed: {e})")

            _notify_discord(
                f"**V8 [{m.config.stage}] Epoch {epoch+1}/{self.max_epochs}**\n"
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"PPL: {train_metrics['ppl']:.2f}"
                + (
                    f" | Val PPL: {val_metrics['val_ppl']:.2f}"
                    if self.val_loader and len(self.val_loader) > 0 else ""
                )
                + (" *best*" if is_best else "")
            )

        self.save_checkpoint("final_model.pt", self.max_epochs - 1)
        total_time = time.time() - training_start
        print(f"\nTraining complete! Wall time: {total_time/3600:.2f}h")
        print(f"Best Val PPL: {self.best_val_ppl:.2f}")
        _notify_discord(
            f"**V8 [{m.config.stage}] DONE** wall {total_time/3600:.2f}h | "
            f"Best Val PPL: {self.best_val_ppl:.2f}"
        )


# ── CLI ──────────────────────────────────────────────────────────────────────


def _load_env_dotfile() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key == "DISCORD_HOOK" and value:
                    os.environ["DISCORD_HOOK"] = value
                    break
    except Exception as e:
        print(f"[Discord] Could not read .env: {e}", file=sys.stderr)


def main() -> None:
    _load_env_dotfile()
    if os.environ.get("DISCORD_HOOK"):
        print("[Discord] Webhook configured — notifications enabled",
              file=sys.stderr)
    else:
        print("[Discord] No webhook (set DISCORD_HOOK in .env to enable)",
              file=sys.stderr)

    parser = argparse.ArgumentParser(description="V8 Quantum-Logic Core trainer")
    parser.add_argument("--preset", type=str, required=True,
                        choices=sorted(PRESETS.keys()))
    parser.add_argument("--dataset", type=str, default="wikitext103",
                        choices=["wikitext103", "tinystories"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--amp_dtype", type=str, default="auto",
                        choices=["auto", "bf16", "fp16"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=9999999)
    parser.add_argument("--gen_every", type=int, default=0)
    parser.add_argument("--gen_prompt", type=str,
                        default="In 1923 , the University of")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--diag_every", type=int, default=200)
    parser.add_argument("--log_dir", type=str, default="logs/v8")
    parser.add_argument("--checkpoint_dir", type=str, default="v8/checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--backbone_ckpt", type=str, default=None,
                        help="Stage A checkpoint to load into the backbone "
                             "(use for Stage B / Stage C launches).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile_steps", type=int, default=0,
        help="If >0, run a one-shot torch.profiler over that many training "
             "steps, dump a Chrome trace under logs/v8/profiles/, and exit. "
             "Wait/warmup/active = 5/5/N-10. Compile is disabled internally "
             "for the profile run so the trace shows real eager kernels.",
    )
    parser.add_argument("--kl_anchor_weight", type=float, default=None,
                        help="Override Stage C KL anchor weight (default: from preset).")
    parser.add_argument("--infonce_weight", type=float, default=None,
                        help="Override the bank's InfoNCE weight (default: from preset).")
    parser.add_argument("--infonce_every", type=int, default=None,
                        help="Run InfoNCE every K steps (default: from preset).")
    parser.add_argument("--infonce_n_entities", type=int, default=64,
                        help="Synthetic entity-cloze: number of entities.")
    parser.add_argument("--infonce_examples_train", type=int, default=4096,
                        help="Synthetic entity-cloze: training examples.")
    parser.add_argument("--infonce_seq_len", type=int, default=32,
                        help="Synthetic entity-cloze: sequence length per example.")
    parser.add_argument("--infonce_batch_size", type=int, default=16,
                        help="Synthetic entity-cloze: batch size for the aux loader.")
    parser.add_argument(
        "--qlc_schedule", action="store_true",
        help="Enable the runtime-safe QLC schedule (e2e single-run plan v2). "
             "Mutates only model.qlc.t_max, model.qlc_cfg.ponder_lambda, and "
             "(optionally) model.qlc.bank_temperature based on global_step. "
             "Default: off (preset values are used unchanged).",
    )
    parser.add_argument("--qlc_warmup_end_frac", type=float, default=1.0/3.0,
                        help="QLC schedule: phase 0 -> 1 boundary as a "
                             "fraction of total steps.")
    parser.add_argument("--qlc_mid_end_frac", type=float, default=2.0/3.0,
                        help="QLC schedule: phase 1 -> 2 boundary as a "
                             "fraction of total steps.")
    parser.add_argument("--qlc_t_max_phases", type=int, nargs=3,
                        default=[2, 3, 4],
                        metavar=("P0", "P1", "P2"),
                        help="QLC schedule: t_max value for each of the 3 phases.")
    parser.add_argument("--qlc_ponder_lambda_phases", type=float, nargs=3,
                        default=[0.0, 0.002, 0.005],
                        metavar=("P0", "P1", "P2"),
                        help="QLC schedule: ponder_lambda for each phase. "
                             "v8.2 default lowered from (0,0.005,0.01) to keep "
                             "the late-phase cap from collapsing mean_iter "
                             "once the loop is finally engaged.")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"v8_{args.preset}_{args.dataset}.log"
    log_mode = "a" if args.resume else "w"
    tee = TeeLogger(log_path, mode=log_mode)
    sys.stdout = tee
    print(f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"  V8: Quantum-Logic Core over QPAM")
    print(f"  Preset: {args.preset} | Dataset: {args.dataset}")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed_everything(args.seed)

    cfg: V8Config = get_config(args.preset)
    if args.seq_len is not None:
        cfg.backbone.max_seq_len = args.seq_len
    if args.dropout is not None:
        cfg.backbone.dropout = args.dropout
    if args.kl_anchor_weight is not None:
        cfg.kl_anchor_weight = args.kl_anchor_weight
    if args.infonce_weight is not None:
        cfg.qlc.infonce_weight = args.infonce_weight
    if args.infonce_every is not None:
        cfg.qlc.infonce_every = args.infonce_every

    seq_len = cfg.backbone.max_seq_len
    max_samples = args.max_samples if args.max_samples < 9999999 else None
    print(f"\nLoading {args.dataset} (seq_len={seq_len})...")

    if args.dataset == "wikitext103":
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
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4

    dl_generator = torch.Generator(); dl_generator.manual_seed(args.seed)
    _seed_val = args.seed
    def _worker_init_fn(wid: int) -> None:
        np.random.seed(_seed_val + wid)
        random.seed(_seed_val + wid)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=use_cuda,
        generator=dl_generator, worker_init_fn=_worker_init_fn, **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=use_cuda,
        worker_init_fn=_worker_init_fn, **dl_kwargs,
    )

    model = V8LM(cfg)
    p = model.count_parameters()
    trainable = model.trainable_parameters()

    # Load Stage A backbone weights for Stage B / C launches.
    backbone_load_info = None
    if args.backbone_ckpt:
        print(f"\nLoading backbone from {args.backbone_ckpt}")
        backbone_load_info = load_backbone_from_v7_checkpoint(model, args.backbone_ckpt)
        print(f"  missing={len(backbone_load_info['missing'])} "
              f"unexpected={len(backbone_load_info['unexpected'])}")
        if backbone_load_info["missing"]:
            print(f"  example missing keys: {backbone_load_info['missing'][:5]}")

    start_epoch = 0
    resume_ckpt: Optional[dict] = None
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        resume_ckpt = torch.load(args.resume, weights_only=False, map_location="cpu")
        model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        # ``start_epoch`` is finalised AFTER the trainer is built, when we
        # call ``trainer.load_full_state`` to also restore optimizer /
        # scheduler / global_step / best_val_*.
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        print(f"Resumed model weights from epoch {start_epoch} "
              f"(optimizer/scheduler/step state will be restored "
              f"after trainer init)")

    infonce_loader = None
    if cfg.qlc.enabled and cfg.qlc.infonce_weight > 0:
        print(
            f"\nBuilding entity-cloze loader for InfoNCE auxiliary "
            f"(n_entities={args.infonce_n_entities}, "
            f"n_examples={args.infonce_examples_train}, "
            f"seq_len={args.infonce_seq_len})..."
        )
        cloze_train, _cloze_val = make_entity_cloze_loaders(
            tokenizer=tokenizer,
            n_entities=args.infonce_n_entities,
            n_examples_train=args.infonce_examples_train,
            n_examples_val=max(64, args.infonce_examples_train // 8),
            seq_len=args.infonce_seq_len,
            seed=args.seed,
        )
        infonce_loader = DataLoader(
            cloze_train, batch_size=args.infonce_batch_size, shuffle=True,
            num_workers=0, pin_memory=use_cuda,
        )

    # ── Build the rich startup summary (shared between console + Discord) ──
    _summary_lines: List[str] = []
    _sl = _summary_lines.append

    if args.resume:
        _sl(f"--- Resumed from {args.resume} at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    _sl(f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _sl("=" * 60)
    _sl(f"V8 Stage {cfg.stage}: Quantum-Logic Core over QPAM")
    _sl("=" * 60)
    _sl(f"Host: {os.uname().nodename}")
    _sl(f"Preset: {args.preset}")
    _sl(f"Dataset: {args.dataset} (seq_len={seq_len})")
    _sl(f"Seed: {args.seed}")

    bb = cfg.backbone
    _sl(f"Backbone: dim={bb.dim}, layers={getattr(bb, 'n_layers', '-')}, "
        f"n_heads={getattr(bb, 'n_heads', '-')}, "
        f"vocab={bb.vocab_size}, max_seq_len={bb.max_seq_len}")
    _bb_extras = []
    for _key in ("expand", "use_rope", "use_gsp", "fused_qkv", "head_dim"):
        if hasattr(bb, _key):
            _bb_extras.append(f"{_key}={getattr(bb, _key)}")
    if _bb_extras:
        _sl("  " + ", ".join(_bb_extras))
    _sl(f"  Dropout: {bb.dropout}")

    if cfg.qlc.enabled:
        q = cfg.qlc
        _sl(f"QLC: ENABLED (rank={q.rank}, bank_size={q.bank_size}, "
            f"top_k={q.top_k}, t_max={q.t_max})")
        _sl(f"  use_complex={q.use_complex}, "
            f"halt_mode={q.halt_mode}, "
            f"unsharp_target={q.unsharp_target}, "
            f"quantale_order_test={q.quantale_order_test}")
        _sl(f"  out_scale_init={q.out_scale_init}, "
            f"out_scale_learnable={q.out_scale_learnable}, "
            f"renormalize_psi={q.renormalize_psi}, "
            f"ponder_lambda={q.ponder_lambda}")
        _sl(f"  target_alignment_weight={q.target_alignment_weight} "
            f"(v8.2: pulls OrthoHalt u toward psi so |u^H psi|^2 leaves the "
            f"1/d noise floor)")
        if q.infonce_weight > 0:
            _sl(f"  InfoNCE: weight={q.infonce_weight}, every={q.infonce_every} "
                f"(entities={args.infonce_n_entities}, "
                f"examples={args.infonce_examples_train}, "
                f"batches/epoch={len(infonce_loader) if infonce_loader else 0})")
    else:
        _sl(f"QLC: DISABLED (passthrough / Stage A pretrain)")

    _sl(f"Stage: {cfg.stage}, freeze_backbone={cfg.freeze_backbone}, "
        f"unfreeze_lm_head={getattr(cfg, 'unfreeze_lm_head', False)}, "
        f"kl_anchor={cfg.kl_anchor_weight}")
    if args.qlc_schedule and cfg.qlc.enabled:
        _sl(
            f"QLC schedule: ENABLED "
            f"(warmup<{args.qlc_warmup_end_frac:.2f} "
            f"mid<{args.qlc_mid_end_frac:.2f}; "
            f"t_max={tuple(args.qlc_t_max_phases)} "
            f"ponder_lambda={tuple(args.qlc_ponder_lambda_phases)})"
        )
    elif cfg.qlc.enabled:
        _sl("QLC schedule: disabled (preset values used unchanged)")

    _sl(f"Optim: lr={args.lr}, warmup={args.warmup_steps}, "
        f"wd={args.weight_decay}, grad_clip={args.gradient_clip}")
    _sl(f"AMP: {args.amp_dtype}, Compile: {args.compile}"
        + (f" (mode={args.compile_mode})" if args.compile else ""))
    _sl(f"Workers: {nw}, pin_memory={use_cuda}, "
        f"log_interval={args.log_interval}, diag_every={args.diag_every}")

    if args.backbone_ckpt:
        miss = len(backbone_load_info["missing"]) if backbone_load_info else 0
        unex = len(backbone_load_info["unexpected"]) if backbone_load_info else 0
        _sl(f"Backbone ckpt: {args.backbone_ckpt} (missing={miss}, unexpected={unex})")

    _sl(f"Params: {p['total']:,} ({p['total']/1e6:.1f}M total) | "
        f"QLC: {p['qlc']:,} ({p['qlc']/1e6:.2f}M) | "
        f"Trainable: {trainable:,} ({trainable/1e6:.1f}M)")
    _sl(f"Training on {('cuda' if use_cuda else 'cpu')} | "
        f"Epochs: {start_epoch+1}..{args.epochs} | "
        f"Batch size: {args.batch_size} | "
        f"Train batches/epoch: {len(train_loader)} | "
        f"Val batches: {len(val_loader)}")
    _sl(f"Log file: {log_path}")
    _sl(f"Checkpoint dir: {args.checkpoint_dir}")
    _sl("=" * 60)

    print("")
    for _line in _summary_lines:
        print(_line)
    print("")

    _discord_header = "**V8 training resumed**" if args.resume else "**V8 training started**"
    _notify_discord_long(
        _discord_header + "\n```\n" + "\n".join(_summary_lines) + "\n```"
    )

    qlc_schedule = QLCRuntimeSchedule(
        enabled=bool(args.qlc_schedule),
        warmup_end_frac=float(args.qlc_warmup_end_frac),
        mid_end_frac=float(args.qlc_mid_end_frac),
        t_max_phases=tuple(int(x) for x in args.qlc_t_max_phases),  # type: ignore[arg-type]
        ponder_lambda_phases=tuple(float(x) for x in args.qlc_ponder_lambda_phases),  # type: ignore[arg-type]
    )

    trainer = V8Trainer(
        model, train_loader, val_loader, tokenizer,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, gradient_clip=args.gradient_clip,
        max_epochs=args.epochs, checkpoint_dir=args.checkpoint_dir,
        amp_dtype_str=args.amp_dtype,
        compile_model=(args.compile and args.profile_steps == 0),
        compile_mode=args.compile_mode, gen_every=args.gen_every,
        gen_prompt=args.gen_prompt, log_interval=args.log_interval,
        start_epoch=start_epoch, kl_anchor_weight=cfg.kl_anchor_weight,
        diag_every=args.diag_every,
        infonce_loader=infonce_loader,
        infonce_weight=cfg.qlc.infonce_weight,
        infonce_every=cfg.qlc.infonce_every,
        qlc_schedule=qlc_schedule,
    )

    if resume_ckpt is not None:
        resumed_start = trainer.load_full_state(resume_ckpt)
        best_ppl_str = (
            f"{trainer.best_val_ppl:.2f}"
            if trainer.best_val_ppl != float("inf") else "inf"
        )
        print(
            f"Resumed from {args.resume}: epoch={resumed_start} "
            f"start_step={trainer.global_step} "
            f"global_tokens={trainer.global_tokens:,} "
            f"best_val_loss={trainer.best_val_loss:.4f} "
            f"best_val_ppl={best_ppl_str}"
        )
        # Free the CPU-side checkpoint blob -- optimizer state on a 100M
        # param model is ~800MB and we don't need it any more.
        del resume_ckpt

    if args.profile_steps > 0:
        prof_dir = Path("logs/v8/profiles") / (
            f"{args.preset}_{args.dataset}_b{args.batch_size}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        trainer.profile_one_shot(args.profile_steps, prof_dir)
        return

    trainer.train()


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _notify_training_failure(reason: str, exc: Optional[BaseException] = None) -> None:
    parts = [
        f"**V8 training {reason}**",
        f"Host: {os.uname().nodename}",
        f"Command: {' '.join(sys.argv)}",
    ]
    if exc is not None:
        parts.append(f"Error: {type(exc).__name__}: {exc}")
    _notify_discord("\n".join(parts))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _notify_training_failure("stopped by user (Ctrl+C)")
        raise
    except Exception as e:
        _notify_training_failure("OOM" if _is_oom_error(e) else "failed", e)
        raise
