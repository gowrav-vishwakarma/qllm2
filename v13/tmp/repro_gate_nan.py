"""Reproduce the 500M V13 gate NaN on CPU with the exact gate config.

Trains a micro V13 (6L/dim96) with the delta+GSP stack and logs, per step:
  main loss, gate aux loss, max|param|, max|grad|, gate_prob range, state max|.|
  and any-NaN flags. Detects the FIRST step a NaN appears and WHICH tensor.
Arms (argv[1]):
  auxon   : protect_gate_bias=-3.0, gate_surprisal_lambda=0.1  (the crash cfg)
  auxoff  : protect_gate_bias=-3.0, gate_surprisal_lambda=0.0
  relax   : protect_gate_bias=0.0,  gate_surprisal_lambda=0.1
"""
import sys, math, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from v13.model import V13LM, V13Config


def make_cfg(bias, lam):
    return V13Config(
        vocab_size=50261, dim=96, n_heads=3, head_dim=32, n_layers=6,
        expand=3, dropout=0.0, max_seq_len=512, use_rope=True, use_gsp=True,
        fused_qkv=True, tie_weights=True, gradient_checkpointing=False,
        activation='swish', chunk_size=256, decay_mode='head', write_mode='delta',
        delta_erase_gate=True, n_states=3, state_dt_spread=2.0, base_dt_bias=-4.0,
        gate_content_aware=True, protect_gate_bias=bias, routing_content_aware=False,
        state_compete=False, phase_init='zero', route_balance_lambda=0.0,
        aux_loss_weight=1.0, fused_e3=True, recompute_pam_chunks=False,
        gamma_floor=0.0, gate_surprisal_lambda=lam)


def gate_surprisal_loss(gate_probs, nll, lm, labels, loss_mask, cfg):
    # Exact replica of V7Trainer._gate_surprisal_loss (v7/train.py:354).
    B, T = labels.shape
    surprisal = nll
    valid = labels != -100
    if loss_mask is not None:
        valid = valid & (loss_mask > 0)
    median_ce = surprisal[valid].median() if valid.any() else surprisal.median()
    tau = max(getattr(cfg, 'gate_surprisal_tau', 1.0), 1e-3)
    sign = getattr(cfg, 'gate_surprisal_sign', 1.0)
    target_p = torch.sigmoid(sign * (median_ce - surprisal) / tau).detach()
    gp = gate_probs.float().clamp(1e-4, 1 - 1e-4)
    target = target_p.float().unsqueeze(0).expand_as(gp)
    vmask = valid.unsqueeze(0).expand_as(gp).to(gp.dtype)
    bce = F.binary_cross_entropy(gp, target, reduction='none')
    return (bce * vmask).sum() / vmask.sum().clamp_min(1.0)


def main():
    arm = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
    B, T = 8, 512
    if arm == 'auxon':
        bias, lam = -3.0, 0.1
    elif arm == 'auxoff':
        bias, lam = -3.0, 0.0
    elif arm == 'relax':
        bias, lam = 0.0, 0.1
    else:
        sys.exit(f"unknown arm {arm}")
    torch.manual_seed(0)
    cfg = make_cfg(bias, lam)
    m = V13LM(cfg).to('cpu').train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    x = torch.randint(0, 50261, (B, T))
    labels = torch.randint(0, 50261, (B, T))

    # capture PAM state magnitudes via forward hooks
    state_holder = {}
    def _hook(name, fn, *a, **k):
        r = fn(*a, **k)
        st = r[1] if isinstance(r, tuple) else None
        if st is not None:
            try:
                state_holder[name] = float(st.abs().max())
            except Exception:
                pass
        return r
    for i, blk in enumerate(m.blocks):
        blk.pam.forward = (lambda n, f, *a, _o=f, **k: _hook(n, _o, *a, **k))(i, blk.pam.forward)

    print(f"[{arm}] bias={bias} lam={lam} lr={lr} B{T} {steps} steps", flush=True)
    first_nan = None
    for i in range(steps):
        opt.zero_grad()
        state_holder.clear()
        lm, aux_loss, gate_probs = m._hidden_to_lm(x)
        main_loss, nll = m.ce_from_lm(lm, labels, chunk=4096, return_nll=True)
        loss = main_loss
        gate_loss_val = 0.0
        if lam > 0 and gate_probs is not None:
            gl = gate_surprisal_loss(gate_probs, nll, lm, labels, None, cfg)
            loss = loss + lam * gl
            gate_loss_val = float(gl.detach())
        loss.backward()
        pn = max(p.abs().max().item() for p in m.parameters() if p.is_floating_point())
        gn = max((p.grad.abs().max().item() for p in m.parameters()
                  if p.grad is not None and p.grad.is_floating_point()), default=0.0)
        gpn = float(gate_probs.max()) if gate_probs is not None else 0.0
        gpnmin = float(gate_probs.min()) if gate_probs is not None else 0.0
        smax = max(state_holder.values(), default=0.0)
        nan_param = math.isinf(pn) or math.isnan(pn)
        nan_grad = math.isinf(gn) or math.isnan(gn)
        nan_gp = math.isnan(gpn) or math.isinf(gpn) or math.isnan(gpnmin)
        nan_main = math.isnan(float(main_loss))
        nan_gate = math.isnan(gate_loss_val)
        nan_state = math.isinf(smax) or math.isnan(smax)
        if (nan_param or nan_grad or nan_gp or nan_main or nan_gate or nan_state) and first_nan is None:
            first_nan = (i, dict(param=nan_param, grad=nan_grad, gateprob=nan_gp,
                                 main=nan_main, gate=nan_gate, state=nan_state))
        if i % 20 == 0 or first_nan is not None:
            flag = ''
            if first_nan:
                d = first_nan[1]
                flag = '  <<<NaN param=%s grad=%s gp=%s main=%s gate=%s state=%s' % (
                    d['param'], d['grad'], d['gateprob'], d['main'], d['gate'], d['state'])
            print(f"  s{i:4d} main={float(main_loss):.3f} gate={gate_loss_val:.4f} "
                  f"wmax={pn:.3g} gmax={gn:.3g} gp=[{gpnmin:.3f},{gpn:.3f}] state={smax:.3g}{flag}", flush=True)
        if first_nan is not None:
            break
        opt.step()
    print(f"[{arm}] DONE first_nan={first_nan}", flush=True)


if __name__ == '__main__':
    main()
