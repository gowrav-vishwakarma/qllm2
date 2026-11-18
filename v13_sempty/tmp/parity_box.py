"""Cross-box parity (Mind, 2026-11-18, Lead's gate (2)): one fixed batch (ids saved in a .pt, scp'd) through trained wio pam
(main stable_final) and wio gdn (decay latest) on this box's GPU; prints the mean CE in fp32 and in bf16 autocast.
    python v13_sempty/tmp/parity_box.py <batch.pt> <arch pam|gdn> <ckpt>"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ids = torch.load(sys.argv[1])
arch, ck = sys.argv[2], torch.load(sys.argv[3], map_location='cpu', weights_only=False)
if arch == 'gdn':
    from v13_sempty.baseline_gdn import GDNConfig, GDNLM
    m = GDNLM(GDNConfig(vocab_size=50261))
else:
    from v13_sempty.config import get_config
    from v13_sempty.model import LM
    cfg = get_config('baseline_real_pm')
    for k, v in ck['config'].items():
        if hasattr(cfg, k): setattr(cfg, k, v)
    m = LM(cfg)
missing, unexpected = m.load_state_dict(ck['model_state_dict'], strict=False)
assert not unexpected and not [k for k in missing if 'rope' not in k], (missing[:5], unexpected[:5])
m = m.cuda().eval()
def ce(dtype):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dtype == 'bf16')):
        outs = []
        for r in range(ids.size(0)):
            o = m(ids[r:r + 1].cuda())
            o = o[0] if isinstance(o, (tuple, list)) else o
            o = o.logits if hasattr(o, 'logits') else o
            outs.append(F.cross_entropy(o[:, :-1].float().reshape(-1, o.size(-1)), ids[r:r + 1, 1:].cuda().reshape(-1)).item())
        return sum(outs) / len(outs)
        out = m(ids.cuda())
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = out.logits if hasattr(out, 'logits') else out
        return F.cross_entropy(out[:, :-1].float().reshape(-1, out.size(-1)), ids[:, 1:].cuda().reshape(-1)).item()
print(f'{arch} step {ck.get("global_step")} fp32 CE {ce("fp32"):.6f} | bf16 CE {ce("bf16"):.6f} | {torch.cuda.get_device_name()}')
