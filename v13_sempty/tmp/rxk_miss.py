# Diagnostic: what does the mind write at R x K's 2k explicit cell when free decoding misses? (b2p2; plain and contrast a=4)
import sys, re, random, collections, torch
sys.path.insert(0, '/home/gowrav/Development/qllm2'); sys.path.insert(0, '/home/gowrav/Development/qllm2/scripts')
from memory_probes.spread import _SEP, filler_tokens, split_units
from run_memory_behavioral import _load_v13_sempty
from run_skillpack_probe import Body, filler_block
from v14.rules_ood import build_example as b2_example
from v7.data import _recall_key
torch.set_num_threads(16); dev = torch.device('cpu')
model, tok, _ = _load_v13_sempty(sys.argv[1], 'baseline_real_pm', dev); model.eval(); body = Body(model, dev)
filler = filler_tokens('/home/gowrav/Development/qllm2/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')
enc = lambda s: tok.encode(s, add_special_tokens=False); sep, nl = enc(_SEP), enc('\n')
kinds = collections.Counter(); ex_rows = []
for task in ('affine_map_new', 'shift_wide_new'):
    for seed in range(8800, 8850):
        ex = b2_example(tok, task, 8, seed); header, units = split_units(ex)
        pack = enc(header) + sum((enc(u) for u in units[:-1]), []); qtext = units[-1]
        K = _recall_key(random.Random(seed * 31 + 7)); m = re.match(r'^(\s*)(\S+)(\s*->.*)$', qtext, re.S); v = m.group(2)
        field = 'x' if task.startswith('affine') else 'letter'
        row = enc(f' Record: {K} | {field}: {v}.') + nl; q = enc(qtext)
        fill = sep + filler_block(filler, 2048, seed, 2048) + sep
        lp = body.last_logits(pack + fill + row + q).float(); ln = body.last_logits(fill + row + q).float()
        cands = {tok.decode([c]) for c in ex.candidate_token_ids}
        gold = tok.decode([ex.target_token_id])
        for name, lg in (('plain', lp), ('c4', lp + 4 * (lp - ln))):
            t = tok.decode([int(lg.argmax())])
            if t == gold: k = 'right'
            elif t.strip() == v: k = 'copies v'
            elif t in cands: k = 'wrong candidate'
            elif (t.strip().isdigit() if task.startswith('affine') else (len(t.strip()) == 1 and t.strip().isalpha())): k = 'right type, not a candidate'
            else: k = 'wrong type'
            kinds[(task[:6], name, k)] += 1
            if k != 'right' and len(ex_rows) < 12: ex_rows.append((task[:6], seed, name, repr(gold), repr(t), 'v=' + v))
for k in sorted(kinds): print(k, kinds[k])
for r in ex_rows: print(r)
