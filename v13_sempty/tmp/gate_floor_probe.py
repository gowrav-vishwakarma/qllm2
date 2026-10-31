# Throwaway: does zeroing PAM's SMALL write/erase gates at test time (no
# training) help label lookup across distance? If filler writes bury single
# items, a hard floor tau (gate < tau -> 0) should raise far label accuracy.
# argv: ckpt tau scope(echo|all) trials lengths(comma)
import sys, numpy as np, torch
torch.set_num_threads(8)
R = '/home/gowrav/Development/qllm2'
sys.path.insert(0, R); sys.path.insert(0, R + '/scripts')
from memory_probes.rules import aggregate, build_rule_suite, score_candidate_logits
from memory_probes.label_diag import exact_variant
from memory_probes.spread import filler_tokens, spread_suite
from run_memory_behavioral import _last_logits, _load_v13_sempty

ck, tau, scope, trials = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
lengths = tuple(int(x) for x in sys.argv[5].split(','))
model, tok, _ = _load_v13_sempty(ck, 'baseline_real_pm', torch.device('cpu')); model.eval()
H = model.config.n_heads
if tau > 0:
    def eh(mod, inp, out):
        dt, wr, er = out.split(H, dim=-1)
        wr = torch.where(torch.sigmoid(wr) < tau, torch.full_like(wr, -30.0), wr)
        er = torch.where(torch.sigmoid(er - 3.0) < tau, torch.full_like(er, -30.0), er)
        return torch.cat([dt, wr, er], dim=-1)
    model.echo_gates.register_forward_hook(eh)
    if scope == 'all':
        for _, m in model.named_modules():
            if hasattr(m, '_delta_gates') and getattr(m, 'delta', False):
                og = m._delta_gates
                def g(tokens, _og=og):
                    bw, be = _og(tokens)
                    return bw * (bw >= tau), be * (be >= tau)
                m._delta_gates = g
base = build_rule_suite(tok, tasks=('label_category',), shots=(8,), seeds=tuple(range(5000, 5000 + trials)))
examples = base + [exact_variant(ex, tok) for ex in base]
examples = spread_suite(examples, tok, lengths, filler_tokens(R + '/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
rows = []
with torch.no_grad():
    for ex in examples:
        logits = _last_logits('v13_sempty', model, torch.tensor([ex.prompt_ids]))[0]
        rows.append({**ex.to_dict(), **score_candidate_logits(ex, logits[ex.candidate_token_ids].float().tolist())})
agg = aggregate(rows)
print(ck.split('/')[-2], f'tau={tau} scope={scope} trials={trials}',
      {a['task']: round(a['accuracy'], 3) for a in agg}, flush=True)
