import sys, torch
sys.path.insert(0, '/home/gowrav/Development/qllm2'); sys.path.insert(0, '/home/gowrav/Development/qllm2/scripts')
from run_memory_behavioral import _load_v13_sempty
from v13_sempty.train import Trainer
dev = torch.device('cuda'); torch.manual_seed(0)
m, tok, _ = _load_v13_sempty(sys.argv[1], 'baseline_real_pm', dev); m.train()
B, T = 4, 512
x = torch.randint(100, 50000, (B, T + 1))
w = torch.ones(B, T); w[torch.rand(B, T) < 0.05] = 100.0; w[0, :200] = 100.0   # unequal weight sums per row
batch = {'input_ids': x[:, :-1], 'labels': x[:, 1:], 'loss_mask': w}
grads = {}
for n in (1, 2, 4):
    tr = Trainer(m, [batch], learning_rate=1e-4, warmup_steps=1, total_steps=10, amp_dtype_str='off', fused_ce=True, device=dev, log_interval=0)
    tr.grad_accum = n
    tr.optimizer.step = lambda *a, **k: None; tr.optimizer.zero_grad = lambda *a, **k: None
    tr.gradient_clip = 1e9
    for p in m.parameters(): p.grad = None
    loss = tr.step(batch)
    grads[n] = (loss, torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]).clone())
for n in (2, 4):
    d = (grads[n][1] - grads[1][1]).abs().max() / grads[1][1].abs().max()
    print(f'accum {n}: loss {grads[n][0]:.6f} vs {grads[1][0]:.6f}  max grad rel diff {float(d):.2e}')
