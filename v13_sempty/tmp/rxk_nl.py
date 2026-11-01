# rxk post-hoc control (not a bar cell): the L=0 mind stream and the ceiling glued the row onto the last demo line (no newline).
# Same docs (seeds 8800-8849 x 2 tasks), with "\n" between pack and row. Plain decode; free / 20-way.
import sys, re, json, random, torch
sys.path.insert(0, '/home/gowrav/Development/qllm2'); sys.path.insert(0, '/home/gowrav/Development/qllm2/scripts')
from memory_probes.rules import score_candidate_logits
from memory_probes.spread import _SEP, filler_tokens, split_units
from run_memory_behavioral import _load_v13_sempty
from run_skillpack_probe import Body, filler_block
from v14.rules_ood import build_example as b2_example
from v7.data import _recall_key
ck, out = sys.argv[1], sys.argv[2]; torch.set_num_threads(8); dev = torch.device('cpu')
model, tok, _ = _load_v13_sempty(ck, 'baseline_real_pm', dev); model.eval(); body = Body(model, dev)
filler = filler_tokens('/home/gowrav/Development/qllm2/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')
enc = lambda s: tok.encode(s, add_special_tokens=False); sep, nl = enc(_SEP), enc('\n')
f = open(out, 'w')
for task in ('affine_map_new', 'shift_wide_new'):
    for seed in range(8800, 8850):
        ex = b2_example(tok, task, 8, seed); header, units = split_units(ex)
        pack = enc(header) + sum((enc(u) for u in units[:-1]), []); qtext = units[-1]
        K = _recall_key(random.Random(seed * 31 + 7)); m = re.match(r'^(\s*)(\S+)(\s*->.*)$', qtext, re.S)
        field = 'x' if task.startswith('affine') else 'letter'
        row = enc(f' Record: {K} | {field}: {m.group(2)}.') + nl; q = enc(qtext)
        def j(ids):
            lg = body.last_logits(ids).float()
            return [int(int(lg.argmax()) == ex.target_token_id), int(score_candidate_logits(ex, lg[ex.candidate_token_ids].cpu().tolist())['correct'])]
        fill = sep + filler_block(filler, 2048, seed, 2048) + sep
        rec = {'ckpt': ck, 'sub': task, 'doc': seed,
               'L0_mind_nl': j(pack + nl + row + q), 'L0_pack_q': j(pack + q),
               'L2048_ceiling_nl': j(fill + pack + nl + row + q)}
        f.write(json.dumps(rec) + '\n'); f.flush()
