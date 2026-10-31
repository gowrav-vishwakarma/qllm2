# Stated rule, 0-2 shots, L = 0: trained shift values vs held-out (5, 9). Free = exact next token; letters = argmax among a-z.
import random, sys, torch
sys.path.insert(0, '/home/gowrav/Development/qllm2'); sys.path.insert(0, '/home/gowrav/Development/qllm2/scripts')
from run_memory_behavioral import _load_v13_sempty
from run_skillpack_probe import Body
L26 = 'abcdefghijklmnopqrstuvwxyz'
for ck in sys.argv[1:]:
    m, tok, _ = _load_v13_sempty(ck, 'baseline_real_pm', torch.device('cpu')); m.eval(); body = Body(m, torch.device('cpu'))
    lc = [tok.encode(' ' + c, add_special_tokens=False)[0] for c in L26]
    for name, ks in (('trained k', (1, 2, 3, 4, 6, 7, 8, 10, 11, 12)), ('held-out k', (5, 9))):
        for shots in (0, 1, 2):
            fr = le = n = 0; pred = {}
            for i in range(120):
                rng = random.Random(1000 * shots + i + (0 if name[0] == 't' else 77777)); k = rng.choice(ks)
                picks = rng.sample(L26[:26 - k], shots + 1); x = picks[-1]; y = L26[L26.index(x) + k]
                demo = ''.join(f'{a} -> {L26[L26.index(a) + k]}\n' for a in picks[:-1])
                ids = tok.encode(f' Rule: shift each letter forward by {k}.Letter rule.\n{demo}{x} ->', add_special_tokens=False)
                lg = body.last_logits(ids); ans = tok.encode(' ' + y, add_special_tokens=False)[0]
                fr += int(lg.argmax()) == ans; g = lc[int(lg[lc].argmax())]; le += g == ans; n += 1
                d = L26.index(tok.decode([g]).strip()) - L26.index(x) if tok.decode([g]).strip() in L26 else None
                pred[d] = pred.get(d, 0) + 1
            top = sorted(pred.items(), key=lambda t: -t[1])[:3]
            print(f"{ck.split('/')[-2][:18]:18s} {name:10s} shots {shots}: free {fr/n:.2f} letters {le/n:.2f}  predicted shift (top) {top}", flush=True)
