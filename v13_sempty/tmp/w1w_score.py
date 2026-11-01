# Score w1w against its pre-registration (EXPERIMENTS_SEMPY "w1w"): guards g1-g4 and the gain per new W, against W128's 2-seed mean.
import json, glob, re, collections
W = {'w1s_w0': '/home/gowrav/Development/qllm_w1s', 'w1s_w128': '/home/gowrav/Development/qllm_w1s',
     'w1r_w0': '/home/gowrav/Development/qllm_w1s', 'w1r_w128': '/home/gowrav/Development/qllm_w1s',
     'w1w_w512': '/home/gowrav/Development/qllm_w1w', 'w1w_w1024': '/home/gowrav/Development/qllm_w1w'}
HS = {'w1w_w512': 'f68ab64', 'w1w_w1024': 'f68ab64'}
def m(n):
    root, h = W[n], HS.get(n, '508aaff'); full = f'{n}_{h}'; out = {}
    log = sorted(glob.glob(f'{root}/logs/v13_sempty_{full}_2*.log'))[-1]
    out['hold'] = float(re.findall(r'\[val @ step \d+\] val_loss=([0-9.]+)', open(log).read())[-1])
    d = json.load(open(f'{root}/logs/learning_probes/v13_sempty_{full}_spread_s7000_t400.json'))['per_task_accuracy']
    out['s8D'] = sum(d[f'{k}@{L}'] for k in ('shift_wide_new', 'affine_map_new') for L in (2048, 4096)) / 4
    sp = [json.loads(l) for l in open(f'{root}/logs/diag/v13_sempty_skillpack_{full}.jsonl')]
    p8 = glob.glob(f'/home/gowrav/Development/qllm_w1w/logs/diag/v13_sempty_skillpack8k_{full}_f68ab64.jsonl')
    if p8: sp += [json.loads(l) for l in open(p8[0])]
    for L in (4096, 8192):
        x = [r['cand_a']['4.0'] for r in sp if r['task'] == 'contrast' and r['L'] == L]
        out[f'pack{L // 1024}k_c4'] = sum(x) / len(x) if x else float('nan')
    c = [json.loads(l) for l in open(f'{root}/logs/diag/v13_sempty_clk2_echo_{full}.jsonl')]
    out['clk2'] = sum(r['cand_s']['8.0+c4'] for r in c) / len(c)
    a = json.load(open(f'{root}/logs/memory_probes/v13_sempty_{full}_t40_behavioral.json'))['aggregates']
    far = [r['accuracy'] for r in a if r['target_position'] == 0.0 and r['context_tokens'] in (4096, 8192)]
    near = [r['accuracy'] for r in a if r['context_tokens'] in (512, 1024, 2048)]
    out['t40far'], out['t40near'] = sum(far) / len(far), sum(near) / len(near)
    t = []
    for k in ('synthetic', 'natural'):
        r = [json.loads(l) for l in open(glob.glob(f'{root}/logs/v16_textarm_{full}_32768_*_tool_{k}.jsonl')[0])]; r = [x for x in r if 'TP_free' in x]
        t.append(sum(x['TP_free'] for x in r) / len(r))
    out['tool'] = sum(t) / 2
    return out
R = {}
for n in W:
    try: R[n] = m(n)
    except Exception as e: print(n, 'missing:', e)
for n, v in R.items(): print(f'{n:10s} ' + '  '.join(f'{k} {x:.4f}' if k == 'hold' else f'{k} {x:.3f}' for k, x in v.items()))
mean = lambda ns, k: sum(R[n][k] for n in ns) / len(ns)
if all(n in R for n in ('w1s_w0', 'w1r_w0', 'w1s_w128', 'w1r_w128')):
    W0, W128 = ('w1s_w0', 'w1r_w0'), ('w1s_w128', 'w1r_w128')
    ref = {k: mean(W128, k) for k in ('pack4k_c4', 'clk2', 't40far', 's8D', 't40near')}
    gain_ref = (ref['s8D'] + ref['pack4k_c4'] + ref['clk2'] + ref['t40near']) / 4
    print(f"W128 mean: {ref}; W0 hold {mean(W0, 'hold'):.4f}, W0 t40far {mean(W0, 't40far'):.3f}, W128 pack8k_c4 {mean(W128, 'pack8k_c4'):.3f}")
    for n in ('w1w_w512', 'w1w_w1024'):
        if n not in R: continue
        v = R[n]
        g = {'g1 pack4k_c4 >= W128-0.05': v['pack4k_c4'] >= ref['pack4k_c4'] - 0.05, 'g2 clk2 >= W128-0.05': v['clk2'] >= ref['clk2'] - 0.05,
             'g3 t40far >= W128-0.05': v['t40far'] >= ref['t40far'] - 0.05, 'g4 hold <= W0+0.02': v['hold'] <= mean(W0, 'hold') + 0.02}
        gain = (v['s8D'] + v['pack4k_c4'] + v['clk2'] + v['t40near']) / 4 - gain_ref
        print(f"{n}: guards {g} -> {'PASS' if all(g.values()) else 'FAIL'}; gain {gain:+.3f} (pick needs >= +0.03); pack8k_c4 {v['pack8k_c4']:.3f}; tool {v['tool']:.3f}")
