# W1 window reader tests: chunked == full-mask reference; window + causality; zero-init identity on b2p2; grads flow.
import os, sys, torch
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, R); sys.path.insert(0, R + '/scripts')
from v13_sempty.model import WindowRead, LM
from v13_sempty.real_ops import named
from sempyt import Dim
torch.manual_seed(0)
class C: n_heads, head_dim, dim, swa_window = 6, 98, 588, 128
md = Dim('model_dim', 588); b, t = Dim('batch', 2), Dim('time', 600)
w = WindowRead(C, md, chunk=256); torch.nn.init.normal_(w.o_proj.weight, std=0.02)
x = torch.randn(2, 600, 588)
y = w(named(x, (b, t, md))).raw(b, t, md)
# reference: one full T x T banded mask
q, k, v = w.qkv(x).reshape(2, 600, 3, 6, 98).permute(2, 0, 3, 1, 4)
pos = torch.arange(600).float(); inv = 1.0 / (10000.0 ** (torch.arange(0, 98, 2).float() / 98)); ang = pos[:, None] * inv[None]
def rot(z):
    ze, zo = z.reshape(2, 6, 600, 49, 2).unbind(-1); return torch.stack((ze * ang.cos() - zo * ang.sin(), ze * ang.sin() + zo * ang.cos()), -1).reshape(2, 6, 600, 98)
i = torch.arange(600); m = (i[None] <= i[:, None]) & (i[None] > i[:, None] - 128)
ref = w.o_proj(torch.nn.functional.scaled_dot_product_attention(rot(q), rot(k), v, attn_mask=m).transpose(1, 2).reshape(2, 600, 588))
print(f'chunked vs full-mask max diff {float((y - ref).abs().max()):.2e}')
x2 = x.clone(); x2[:, 100] += 5.0                     # change token 100
y2 = w(named(x2, (b, t, md))).raw(b, t, md); d = (y2 - y).abs().amax(dim=(0, 2))
print(f'change at 100 -> max effect before 100 {float(d[:100].max()):.1e}, at 100..227 {float(d[100:228].min()):.1e} (min), from 228 on {float(d[228:].max()):.1e}')
# zero-init identity on b2p2
from run_memory_behavioral import _load_v13_sempty
ck = '/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt'
base, tok, _ = _load_v13_sempty(ck, 'baseline_real_pm', torch.device('cpu')); base.eval()
cfg = base.config; cfg2 = type(cfg)(**{**cfg.__dict__}); cfg2.swa_layers = (5, 9, 13); cfg2.swa_window = 128
m2 = LM(cfg2); missing, unexpected = m2.load_state_dict(base.state_dict(), strict=False); m2.eval()
print('new params:', sorted({k.split('.')[2] for k in missing}), 'unexpected:', unexpected, ' o_proj zero:', all(float(m2.blocks[i].window.o_proj.weight.abs().max()) == 0 for i in (5, 9, 13)))
ids = torch.randint(100, 50000, (1, 300))
with torch.no_grad():
    d = float((m2(ids)[0] - base(ids)[0]).abs().max())
print(f'b2p2 with W1 (zero-init) vs b2p2 logits max diff {d:.2e}')
m2.train(); m2.blocks[5].window.o_proj.weight.data.normal_(std=0.01)
loss = m2(ids)[0].float().logsumexp(-1).mean(); loss.backward()
print('grad on W1 qkv / o_proj:', float(m2.blocks[5].window.qkv.weight.grad.abs().sum()) > 0, float(m2.blocks[5].window.o_proj.weight.grad.abs().sum()) > 0)
print('params added:', sum(p.numel() for n, p in m2.named_parameters() if '.window.' in n))
