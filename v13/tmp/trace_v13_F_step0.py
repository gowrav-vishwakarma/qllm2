#!/usr/bin/env python
"""Trace where the step-0 ON-vs-OFF divergence comes from (throwaway)."""
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13LM, get_config

dev = 'cuda'
cfg_off = get_config('v13_e3_k3_selective')
cfg_on = get_config('v13_e3_k3_selective')
cfg_on.ngram_read = True
cfg_on.ngram_size = 3
cfg_on.ngram_fusion = True

torch.manual_seed(123)
off = V13LM(cfg_off).eval()
torch.manual_seed(123)
on = V13LM(cfg_on).eval()
on.load_state_dict(off.state_dict(), strict=False)
off.to(dev)
on.to(dev)

# 1) Are the shared weights actually equal?
so, sn = off.state_dict(), on.state_dict()
bad = [k for k in so if not torch.equal(so[k], sn[k])]
print(f'shared keys: {len(so)}, mismatched after load: {len(bad)} {bad[:5]}')

# 2) key_proj zero?
kp = on.ngram_fusion.key_proj
print('key_proj max|w| =',
      max(p.detach().abs().max().item()
          for p in (kp.weight_real, kp.weight_imag,
                    kp.bias_real, kp.bias_imag)))

B, T = 2, 256
ids = torch.randint(0, cfg_on.vocab_size, (B, T), device=dev)
with torch.no_grad():
    # 3) ngram output itself
    on._ngram_ctx = None
    on._ngram_row_ctx = None
    ng = on._ngram_repr(ids, None)
    print(f'ngram out: max|.| = {ng.abs().max().item():.3e} '
          f'exact_zero={bool((ng == 0).all())} shape={tuple(ng.shape)}')

    # 4) conv output (pre key_proj)
    raw = torch.stack([on.embed.embed_real(ids), on.embed.embed_imag(ids)],
                      dim=-1)
    import torch.nn.functional as F
    ext = F.pad(raw, (0, 0, 0, 0, 2, 0))
    h = on.ngram_fusion.conv1d(ext.permute(0, 3, 1, 2).reshape(B, 2 * 384, T))
    print(f'conv out: max|.| = {h.abs().max().item():.3e} finite={bool(torch.isfinite(h).all())}')
    h2 = h.permute(0, 2, 1).reshape(B, T - 2 + 1, 384, 2)
    kout = on.ngram_fusion.key_proj(h2)
    print(f'key_proj out: max|.| = {kout.abs().max().item():.3e} '
          f'exact_zero={bool((kout == 0).all())}')
    nout = on.ngram_fusion.norm(kout)
    print(f'norm out: max|.| = {nout.abs().max().item():.3e} '
          f'exact_zero={bool((nout == 0).all())}')

    # 5) full forward diff, and a surgical one: force ON's ngram to exactly 0
    lg_off, _, _ = off(ids)
    on._ngram_ctx = None
    on._ngram_row_ctx = None
    lg_on, _, _ = on(ids)
    print(f'full fwd diff: {(lg_on - lg_off).abs().max().item():.3e}')

    on2 = on
    saved = on2.config.ngram_fusion
    on2.config.ngram_fusion = False  # ON model, fusion path OFF -> raw scaled
    on2._ngram_ctx = None
    lg_on_raw, _, _ = on2(ids)
    print(f'ON-nofuse(raw*0.5) vs OFF: {(lg_on_raw - lg_off).abs().max().item():.3e} '
          f'(sanity: should be ~E-scale diff, NOT 0)')
    on2.config.ngram_fusion = saved

    # 6) ON model with ngram path fully off (ngram_read False)
    on3 = V13LM(cfg_off).eval().to(dev)
    on3.load_state_dict(off.state_dict(), strict=False)
    on3._ngram_ctx = None
    lg_on3, _, _ = on3(ids)
    print(f'identical-config rebuild vs OFF: {(lg_on3 - lg_off).abs().max().item():.3e} '
          f'(determinism check: must be 0)')
