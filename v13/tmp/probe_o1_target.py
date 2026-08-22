"""One-off: confirm the O(1) gate target  P = -logit[label]  is a valid surrogate
for the O(V) NLL target  S = logsumexp - logit[label], with the SAME protect
direction (filler=predictable=high logit[label] -> P low -> high protect, sign=+1).

Checks on random complex lm + embed (CPU, cheap):
  1. corr(P, S)  -- should be strongly positive (per-token variation shared).
  2. target ordering: high-logit (filler) tokens get HIGHER protect target than
     low-logit (content) tokens, under sign=+1.
"""
import torch

torch.manual_seed(0)
B, T, dim, V = 4, 64, 384, 50261
lr = torch.randn(B, T, dim)
li = torch.randn(B, T, dim)
Er = torch.randn(V, dim) * 0.02
Ei = torch.randn(V, dim) * 0.02
labels = torch.randint(0, V, (B, T))

# Full O(V) NLL (the current target source)
logits = torch.stack([lr, li], dim=1)                      # [B,T,2,dim]
logits = logits.permute(0, 1, 3, 2).reshape(B, T, dim * 2)  # [B,T,2dim]
W = torch.stack([Er, Ei], dim=1).reshape(V, dim * 2)        # [V,2dim]
all_logit = logits @ W.T                                     # [B,T,V]
nll = torch.logsumexp(all_logit, dim=-1) - all_logit.gather(-1, labels[..., None])[..., 0]  # S
S = nll

# O(1) proxy: P = -logit[label]  (single embed-row inner product, no V axis)
lr_f = lr.reshape(B * T, dim)
li_f = li.reshape(B * T, dim)
lab = labels.reshape(-1)
logit_label = (lr_f * Er[lab]).sum(-1) + (li_f * Ei[lab]).sum(-1)   # [B*T]
P = -logit_label.reshape(B, T)

# 1) correlation
def corr(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean())
print(f"corr(P, S) = {corr(P, S):+.3f}   (want strongly positive)")

# 2) direction: split tokens by logit_label rank; filler=high logit, content=low
logit_label_bt = logit_label.reshape(B, T)
k = B * T // 2
lo = torch.kthvalue(logit_label_bt.flatten(), k).values
hi = torch.kthvalue(logit_label_bt.flatten(), B * T - k).values
filler = logit_label_bt >= hi     # high logit -> predictable filler
content = logit_label_bt <= lo    # low logit -> surprising content
sign, tau = 1.0, 1.0

def target(x):
    med = x[torch.ones_like(x, dtype=bool)].median() if x.numel() else 0.0
    med = x.median()
    return torch.sigmoid(sign * (med - x) / tau)

tP_fill, tP_cont = target(P)[filler].mean(), target(P)[content].mean()
tS_fill, tS_cont = target(S)[filler].mean(), target(S)[content].mean()
print(f"O(1)  target  filler={tP_fill:.3f}  content={tP_cont:.3f}  gap={tP_fill-tP_cont:+.3f} (want >0: protect filler)")
print(f"NLL   target  filler={tS_fill:.3f}  content={tS_cont:.3f}  gap={tS_fill-tS_cont:+.3f} (same direction expected)")
ok = corr(P, S) > 0.5 and (tP_fill - tP_cont) > 0 and (tS_fill - tS_cont) > 0
print("PASS" if ok else "FAIL")
