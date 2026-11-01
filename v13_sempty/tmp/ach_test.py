import sys, torch
sys.path.insert(0, '/home/gowrav/Development/qllm2'); sys.path.insert(0, '/home/gowrav/Development/qllm2/scripts')
from run_memory_behavioral import _load_v13_sempty
from v13_sempty.ach import AChBatcher, ach_loss, _answer_logits, TAG
torch.manual_seed(0)
m, tok, _ = _load_v13_sempty(sys.argv[1], 'baseline_real_pm', torch.device('cpu')); m.train()
window = torch.randint(100, 50000, (2049,))
B = AChBatcher(tok, seed=43, n_short=6, n_long=1, max_fill=256)
items = B.build(window)
for ev, no, n in items[:3] + items[-1:]:
    print('ev tail:', repr(tok.decode(ev[-(n + 12):])), '| answer:', repr(tok.decode(ev[-n:])), '| twin starts:', repr(tok.decode(no[:6])))
    assert ev[-(n + len(B.tag) + 0):] == no[-(n + len(B.tag)):], 'twin and evidence version must end the same'
# alignment: the logits at those positions, with the model in eval, predict the given answer tokens (teacher forcing)
m.eval()
with torch.no_grad():
    l, tgt = _answer_logits(m, [items[0][0]], [items[0][2]], torch.device('cpu'))
    full = m(torch.tensor([items[0][0]]))[0][0]
    n = items[0][2]; ref = full[len(items[0][0]) - 1 - n:len(items[0][0]) - 1]
print('answer-position logits vs model(x):', float((l - ref).abs().max()), '(scale', float(ref.abs().max()), ')')
m.train()
kl, acc, ok = ach_loss(m, items, 4.0, torch.device("cpu"))
kl.backward()
g = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
print(f'KL {float(kl):.4f}  answer acc {acc:.2f}  grad sum {g:.3e}  finite {torch.isfinite(kl).item()}')
