# Library's story check, the rule 2x + 1 (a trained affine value): demos (3 or 8) in the trained "Map rule." format, then L tokens
# of held-out web text, then "4 ->". b2p2, plain and contrast a = 4 (the twin = filler + query). 20 random demo sets per cell.
import sys, random, collections, torch
sys.path.insert(0, '/home/gowrav/Development/qllm2'); sys.path.insert(0, '/home/gowrav/Development/qllm2/scripts')
from memory_probes.spread import _SEP, filler_tokens
from run_memory_behavioral import _load_v13_sempty
from run_skillpack_probe import Body, filler_block
torch.set_num_threads(16); dev = torch.device('cpu')
model, tok, _ = _load_v13_sempty(sys.argv[1], 'baseline_real_pm', dev); model.eval(); body = Body(model, dev)
filler = filler_tokens('/home/gowrav/Development/qllm2/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')
enc = lambda s: tok.encode(s, add_special_tokens=False); sep = enc(_SEP)
res = collections.defaultdict(list); outs = collections.defaultdict(collections.Counter)
for n in (3, 8):
    for s in range(20):
        r = random.Random(9000 + s); xs = r.sample([x for x in range(1, 31) if x != 4], n)
        pack = enc('Map rule.\n' + ''.join(f'{x} -> {2 * x + 1}\n' for x in xs)); q = enc('4 ->')
        for L in (0, 2048):
            fill = sep + filler_block(filler, L, 9000 + s, L) + sep if L else []
            lp = body.last_logits(pack + fill + q).float(); ln = body.last_logits((fill or sep) + q).float()
            for name, lg in (('plain', lp), ('c4', lp + 4 * (lp - ln))):
                t = tok.decode([int(lg.argmax())]); res[(n, L, name)].append(t == ' 9'); outs[(n, L, name)][t] += 1
for k in sorted(res): print(f'demos {k[0]} L {k[1]:5d} {k[2]:5s} free {sum(res[k]) / len(res[k]):.2f}  top outputs {outs[k].most_common(4)}')
