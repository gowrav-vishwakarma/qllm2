# kb1 wrapper test: batched loglikelihood == unbatched == a manual per-request computation (padding / sort / truncation safe).
import sys, torch
sys.path.insert(0, '/home/gowrav/Development/qllm2/scripts')
from lm_eval_bodies import BodyLM
class R:
    def __init__(self, c, k): self.args = (c, k)
reqs = [R('The capital of France is', ' Paris'), R('Question: What is 2 + 2?\nAnswer:', ' 4'), R('', 'Hello world'),
        R('A long story begins. ' * 30 + 'And then the cat sat on the', ' mat.'), R('She opened the door and ', 'saw a dog'),
        R('Numbers: 1, 2, 3, 4,', ' 5, 6')]
for ck, dev in (('/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt', 'cpu'),
                ('/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2t2_main_38e3c29/best_model.pt', 'cpu')):
    m = BodyLM(ck, dev, batch_size=4); a = m.loglikelihood(reqs); m.bs = 1; b = m.loglikelihood(reqs)
    man = []
    with torch.no_grad():
        for r in reqs:
            c, k = m._pair(*r.args); full = c + k; x = torch.tensor([full[:-1]])
            lg = m.m(x)[0][0, -len(k):, :m.V].float().log_softmax(-1)
            man.append(float(lg.gather(1, torch.tensor(k)[:, None]).sum()))
    print(type(m.m).__name__, 'params', m.n_params)
    for r, x, y, z in zip(reqs, a, b, man):
        print(f'  {r.args[1]!r:14s} batched {x[0]:9.4f} {x[1]!s:5s} unbatched {y[0]:9.4f} manual {z:9.4f}  max|diff| {max(abs(x[0] - y[0]), abs(x[0] - z)):.1e}')
