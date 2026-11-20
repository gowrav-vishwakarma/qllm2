import json,sys,os
def load(n):
    fs=[f'logs/wio/v13_sempty_{n}_{s}spread_s7000_t400.json' for s in ['step_014500_','step_015000_','']]
    if not all(os.path.exists(f) for f in fs): return None
    out={}
    for f in fs:
        for a in json.load(open(f))['aggregates']: out.setdefault(a['task'],[]).append((a['accuracy'],a['mean_target_margin']))
    return {k:(sum(x for x,_ in v)/3,sum(y for _,y in v)/3) for k,v in out.items()}
for n in sys.argv[1:]:
    r=load(n)
    if r is None: print(n,'incomplete'); continue
    row=[]
    for t in ['shift_wide_new','affine_map_new']:
        m0,m4,m8=r[f'{t}@0'][1],r[f'{t}@4096'][1],r[f'{t}@8192'][1]
        row.append(f"{t[:5]} acc8k {r[f'{t}@8192'][0]:.3f} M0 {m0:+.2f} M4k {m4:+.2f} M8k {m8:+.2f} ratio8k/0 {m8/m0 if m0>0 else float('nan'):.2f}")
    print(n.ljust(26),' | '.join(row))
