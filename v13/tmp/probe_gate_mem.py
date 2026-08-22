"""One-off: find which sub-step of the gate-surprisal aux eats ~11.6GB at B14."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.nn.functional as F
from v13.model import V13LM, get_config

B, T = 14, 2048
cfg = get_config('v13_e3_k3_selective')
torch.manual_seed(0)
m = V13LM(cfg).cuda()
m.train()
ids = torch.randint(0, 50261, (B, T), device='cuda')
labels = torch.randint(0, 50261, (B, T), device='cuda')

def gb():
    return torch.cuda.max_memory_allocated() / 1e9

with torch.autocast('cuda', dtype=torch.bfloat16):
    lm, aux, gate_probs = m._hidden_to_lm(ids)
print(f"after _hidden_to_lm:          {gb():.2f} GB  gate_probs shape={tuple(gate_probs.shape)} dtype={gate_probs.dtype}")
loss = m.ce_from_lm(lm, labels, chunk=1024) + aux
print(f"after ce_from_lm+aux:         {gb():.2f} GB")

from v11.fused_ce import linear_ce_per_token
hc = torch.cat([lm[..., 0], lm[..., 1]], dim=-1).reshape(B * T, -1)
print(f"after hidden_concat:          {gb():.2f} GB  hc={tuple(hc.shape)}")
wc = torch.cat([m.embed.embed_real.weight, m.embed.embed_imag.weight], dim=-1)
print(f"after weight_concat:          {gb():.2f} GB  wc={tuple(wc.shape)}")
surp = linear_ce_per_token(hc.detach(), wc.detach(), labels.reshape(-1), chunk=1024).reshape(B, T)
print(f"after linear_ce_per_token:    {gb():.2f} GB  surp={tuple(surp.shape)}")
med = surp[labels != -100].median()
tp = torch.sigmoid(med - surp).detach()
print(f"after target_p:               {gb():.2f} GB")
gp = gate_probs.float().clamp(1e-4, 1 - 1e-4)
print(f"after gp=float(clamp):        {gb():.2f} GB  gp={tuple(gp.shape)}")
target = tp.float().unsqueeze(0).expand_as(gp)
print(f"after target expand:          {gb():.2f} GB")
vmask = (labels != -100).unsqueeze(0).expand_as(gp).to(gp.dtype)
print(f"after vmask to(dtype):        {gb():.2f} GB  vmask={tuple(vmask.shape)}")
with torch.amp.autocast(device_type='cuda', enabled=False):
    bce = F.binary_cross_entropy(gp, target, reduction='none')
print(f"after bce:                    {gb():.2f} GB")
gloss = (bce * vmask).sum() / vmask.sum().clamp_min(1.0)
print(f"after gate loss:              {gb():.2f} GB")
