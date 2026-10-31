# lg8r half-life summary: geometric mean of the filler half-lives of the 7 long-memory heads
# (echo h0-h5, notebook blocks.11 h4: filler half-life >= 500 in b2p2), from a gates_diag.py log.
import math, re, sys
def heads(path):
    out, blk = {}, None
    for line in open(path):
        if re.match(r'^(blocks\.\d+\.pam|echo)\s*$', line): blk = line.strip()
        m = re.match(r'^\s+h(\d+): hl\s+(\d+)/\s*(\d+)', line)
        if m and blk: out[(blk, int(m.group(1)))] = int(m.group(2))
    return out
LONG = [('echo', h) for h in range(6)] + [('blocks.11.pam', 4)]
for p in sys.argv[1:]:
    h = heads(p); v = [max(h[k], 1) for k in LONG]
    print(f"{p.split('/')[-1]}: long-head filler half-lives {v} geomean {math.exp(sum(map(math.log, v)) / len(v)):.0f}")
