"""GPU diagnostic: replay the exact 500M V13 recipe (data/model/trainer math,
seed 42, B18/T2048, round-1 lr/warmup) and instrument the gate-surprisal aux +
PAM state growth + param/grad norms, STOPPING at the first non-finite value.

The 500M run died at step ~1551 (57.2M tok) on a device-side CUDA assert
(Loss.cu:91 target_val in [0,1]) inside F.binary_cross_entropy in the gate aux.
The per-step log only printed the MAIN loss (finite 5.39), so the gate-aux
internals + state magnitude are unobserved. This driver logs them and catches
non-finite in Python BEFORE the BCE kernel can assert (skips BCE that step).

Sync budget: ~8 tensor->cpu syncs/step; heavy param/state maxes only on log
steps (every 10). Run: .venv/bin/python -m v13.tmp.diag_gate_nan [--max_steps N]
"""
import sys, math, time, inspect, torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from v13.model import V13LM, get_config
from v7.data import (
    load_pretrain_mix, resolve_amp_dtype, build_lr_scheduler, build_param_groups,
)
from v7.train import seed_everything
from torch.utils.data import DataLoader


def main():
    max_steps = 2000
    if '--max_steps' in sys.argv:
        max_steps = int(sys.argv[sys.argv.index('--max_steps') + 1])
    log_every = 10
    seed = 42
    batch_size, seq_len = 18, 2048
    budget = 500_000_000
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(seed)

    cfg = get_config('v13_e3_k3_selective')
    cfg.max_seq_len = seq_len
    print(f"Config gate knobs: protect_gate_bias={cfg.protect_gate_bias} "
          f"gate_surprisal_lambda={cfg.gate_surprisal_lambda} "
          f"vault={cfg.vault_state} phase={cfg.write_phase_address} "
          f"erase_gate={cfg.delta_erase_gate} content_aware={cfg.gate_content_aware}", flush=True)

    print("Loading pretrain_mix (exact 500M recipe)...", flush=True)
    per_source_tokens, per_source_docs = {}, {}
    train_ds, val_ds, tokenizer = load_pretrain_mix(
        seq_len=seq_len, edu_score_min=3, token_budget=budget,
        sources=('dclm', 'fineweb', 'smoltalk2_mid'), weights=(48.0, 48.0, 4.0),
        chat_vocab=True, fineweb_name='sample-10BT', holdout_pct=5, mix_seed=seed,
        skip_docs={'dclm': 0, 'fineweb': 0, 'smoltalk2_mid': 0},
        blend_warmup_tokens=1_000_000_000,
        token_counters=per_source_tokens, doc_counters=per_source_docs,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = V13LM(cfg).cuda().train()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: {nparams:,} params, device cuda", flush=True)

    param_groups = build_param_groups(model, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95), fused=True)
    est_steps = budget // (batch_size * seq_len) + 500
    sched = build_lr_scheduler(opt, 'warmup_cosine', warmup_steps=500, total_steps=est_steps)
    amp_dtype = resolve_amp_dtype('auto')
    use_amp = amp_dtype is not None
    print(f"AMP={amp_dtype} use_amp={use_amp} est_steps={est_steps}", flush=True)

    raw = model
    _ce_emits_nll = 'return_nll' in inspect.signature(model.ce_from_lm).parameters
    tau = max(getattr(cfg, 'gate_surprisal_tau', 1.0), 1e-3)
    sign = getattr(cfg, 'gate_surprisal_sign', 1.0)
    gsl = cfg.gate_surprisal_lambda
    lam = cfg.aux_loss_weight

    # PAM state refs (detached) from the most recent forward; maxed on log steps.
    state_refs = {}
    def _make_state_hook(i, orig):
        def w(*a, **k):
            r = orig(*a, **k)
            st = r[1] if isinstance(r, tuple) else None
            if st is not None:
                state_refs[i] = st.detach()
            return r
        return w
    for i, blk in enumerate(model.blocks):
        blk.pam.forward = _make_state_hook(i, blk.pam.forward)

    all_params = [p for p in model.parameters() if p.is_floating_point()]
    gate_params = [p for n, p in model.named_parameters()
                   if 'protect_gate' in n or 'erase_beta' in n and p.is_floating_point()]

    def max_of(tensors):
        m = 0.0
        for t in tensors:
            v = float(t.abs().max())
            if v > m or math.isnan(v):
                m = v
        return m

    def state_max():
        return max_of(list(state_refs.values())) if state_refs else 0.0

    opt.zero_grad(set_to_none=True)
    it = iter(train_loader)
    t0 = time.time()
    step = 0
    gtok = 0
    while step < max_steps:
        td0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        input_ids = batch['input_ids'].cuda(non_blocking=True)
        labels = batch['labels'].cuda(non_blocking=True)
        loss_mask = batch.get('loss_mask')
        if loss_mask is not None:
            loss_mask = loss_mask.cuda(non_blocking=True)
        batch_tokens = input_ids.shape[0] * input_ids.shape[1]
        t_data = time.perf_counter() - td0

        tf0 = time.perf_counter()
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype or torch.float16):
            lm, aux_loss, gate_probs = model._hidden_to_lm(input_ids)
            main_loss, nll = raw.ce_from_lm(lm, labels, loss_mask=loss_mask,
                                            chunk=4096, return_nll=_ce_emits_nll)
        loss = main_loss
        if aux_loss.detach().abs().item() > 0:
            loss = loss + lam * aux_loss
        gate_loss_val = 0.0
        gate_skipped = False
        if gsl > 0 and gate_probs is not None:
            valid = labels != -100
            if loss_mask is not None:
                valid = valid & (loss_mask > 0)
            median_ce = nll[valid].median() if valid.any() else nll.median()
            target_p = torch.sigmoid(sign * (median_ce - nll) / tau).detach()
            gp_nan = not torch.isfinite(gate_probs).all().item()
            t_nan = not torch.isfinite(target_p).all().item()
            if gp_nan or t_nan:
                gate_skipped = True
            else:
                gp = gate_probs.float().clamp(1e-4, 1 - 1e-4)
                target = target_p.float().unsqueeze(0).expand_as(gp)
                vmask = valid.unsqueeze(0).expand_as(gp).to(gp.dtype)
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    bce = F.binary_cross_entropy(gp, target, reduction='none')
                gl = (bce * vmask).sum() / vmask.sum().clamp_min(1.0)
                loss = loss + gsl * gl
                gate_loss_val = float(gl.detach())
        t_fwd = time.perf_counter() - tf0
        # ---- diagnose BEFORE backward (crash site: BCE fwd) ----
        nll_max = float(nll.max())
        nll_nan = not torch.isfinite(nll).all().item()
        gp_nan2 = (gate_probs is not None) and not torch.isfinite(gate_probs).all().item()
        gp_max = float(gate_probs.max()) if gate_probs is not None else 0.0
        gp_min = float(gate_probs.min()) if gate_probs is not None else 0.0
        main_nan = not torch.isfinite(main_loss).all().item()
        loss_nan = not torch.isfinite(loss).all().item()
        if (nll_nan or gp_nan2 or main_nan or loss_nan) or gate_skipped:
            print(f"  <<< NON-FINITE at step {step} gtok={gtok:,} | "
                  f"nll_max={nll_max:.4g} nll_nan={nll_nan} gp_nan={gp_nan2 or gate_skipped} "
                  f"main_nan={main_nan} loss_nan={loss_nan} "
                  f"gp=[{gp_min:.4g},{gp_max:.4g}] state={state_max():.4g} "
                  f"main={float(main_loss):.4f} gate={gate_loss_val:.4g}", flush=True)
            for li, blk in enumerate(model.blocks):
                pg = blk.pam.protect_gate
                if not (torch.isfinite(pg.weight).all() and torch.isfinite(pg.bias).all()):
                    print(f"      protect_gate[li={li}] weight_nan="
                          f"{not torch.isfinite(pg.weight).all().item()} "
                          f"bias_nan={not torch.isfinite(pg.bias).all().item()}", flush=True)
            break
        tb0 = time.perf_counter()
        loss.backward()
        t_bwd = time.perf_counter() - tb0
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gnorm = float(gnorm)
        if not math.isfinite(gnorm):
            print(f"  <<< BAD GRAD at step {step} gtok={gtok:,} | gnorm={gnorm:.4g} "
                  f"main={float(main_loss):.3f} gate={gate_loss_val:.4g} "
                  f"nll_max={nll_max:.4g} state={state_max():.4g} "
                  f"gp=[{gp_min:.3f},{gp_max:.3f}]", flush=True)
            break
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        gtok += batch_tokens
        if step % log_every == 0:
            ppl = math.exp(min(float(main_loss), 20))
            lr = sched.get_last_lr()[0]
            tps = gtok / (time.time() - t0)
            smax = state_max()
            pmax = max_of(all_params)
            gmax = max_of([p.grad for p in all_params if p.grad is not None])
            gpmax_g = max_of(gate_params)
            print(f"  s{step:5d} gtok={gtok/1e6:6.1f}M main={float(main_loss):.4f} "
                  f"gate={gate_loss_val:.4g} nllmax={nll_max:.3f} "
                  f"gp=[{gp_min:.3f},{gp_max:.3f}] state={smax:.4g} "
                  f"pmax={pmax:.4g} gmax={gmax:.4g} gp_w={gpmax_g:.4g} "
                  f"gnorm={gnorm:.3f} ppl={ppl:.1f} lr={lr:.1e} {tps:.0f}t/s "
                  f"[data={t_data*1000:.0f} fwd={t_fwd*1000:.0f} bwd={t_bwd*1000:.0f} ms]", flush=True)
    print(f"DONE step={step} gtok={gtok:,} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
