# W1 streaming test (Library's package contract (1) / (2)). Usage:
#   (old code)  python w1_stream_test.py dump  <root>   -> writes reference logits from the code under <root>
#   (new code)  python w1_stream_test.py check <root>   -> compares with them, then runs the streaming parity tests
# Checks:
#   (a) no-window b2p2 and the trained W1 w1r_w128: one-pass forward logits and _hidden_to_lm == the old code (bitwise);
#   (b) W1 models: one pass == chunked prefill == prefill + token-by-token decode; deep-copied states continue identically;
#   (c) a 16nb + SWA body (r3c-like, random init, o_proj non-zero) passes (b) and keeps (z: [batch, time, model_dim], state)
#       per block, states = 16 block states + [window caches] + [echo].
import copy, sys, torch
mode, root = sys.argv[1], sys.argv[2]
sys.path.insert(0, root); sys.path.insert(0, root + '/scripts')
from run_memory_behavioral import _load_v13_sempty
torch.manual_seed(0); dev = torch.device('cpu')
CK = {'b2p2': '/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt',
      'w1r_w128': '/media/gowrav/d_drive/qllm2/checkpoints_v13_sempty/w1r_w128_508aaff/best_model.pt'}
ids = torch.randint(100, 50000, (1, 700), generator=torch.Generator().manual_seed(1))
REF = '/tmp/claude-1000/-home-gowrav-Development-qllm2/99371c3c-8382-42b3-a3d3-6861632abc19/scratchpad/w1_stream_ref.pt'
if mode == 'dump':
    out = {}
    for n, ck in CK.items():
        m, _, _ = _load_v13_sempty(ck, 'baseline_real_pm', dev); m.eval()
        with torch.no_grad():
            lg, st, _ = m(ids); lm, _ = m._hidden_to_lm(ids)
        out[n] = (lg, lm.raw(*lm.layout), len(st))
    torch.save(out, REF); print('dumped', {k: (v[0].shape, v[2]) for k, v in out.items()}); sys.exit()

ref = torch.load(REF)
def stream_parity(m, name, T=700, cuts=(250, 450), dec=40):
    with torch.no_grad():
        full = m(ids[:, :T])[0]
        st, outs, prev = None, [], 0
        for c in list(cuts) + [T]:
            lg, st, _ = m(ids[:, prev:c], states=st, step_offset=prev); outs.append(lg); prev = c
        chunked = torch.cat(outs, 1)
        lg, st2, _ = m(ids[:, :T - dec]); dl = [lg]
        st_copy = copy.deepcopy(st2)
        for t in range(T - dec, T):
            lg, st2, _ = m(ids[:, t:t + 1], states=st2, step_offset=t); dl.append(lg)
        decoded = torch.cat(dl, 1)
        lg_c, _, _ = m(ids[:, T - dec:T], states=st_copy, step_offset=T - dec)       # continue from a deep copy
    sc = float(full.abs().max())
    print(f'  {name}: chunked vs one pass {float((chunked - full).abs().max()) / sc:.1e}; decode vs one pass '
          f'{float((decoded - full).abs().max()) / sc:.1e}; deep-copy continuation vs one pass {float((lg_c - full[:, T - dec:]).abs().max()) / sc:.1e} (relative)')
    return st2
for n, ck in CK.items():
    m, _, _ = _load_v13_sempty(ck, 'baseline_real_pm', dev); m.eval()
    with torch.no_grad():
        lg, st, _ = m(ids); lm, _ = m._hidden_to_lm(ids)
    r = ref[n]
    print(f'{n}: forward == old code {torch.equal(lg, r[0])} (max {float((lg - r[0]).abs().max()):.1e}); _hidden_to_lm == old code '
          f'{torch.equal(lm.raw(*lm.layout), r[1])}; states {len(st)} (old {r[2]})')
    st = stream_parity(m, n)
    print(f'  {n}: carried states {len(st)} entries; window caches {[tuple(c[0].shape) for c in st[-2]] if any(b.window is not None for b in m.blocks) else "none"}')
# (c) 16nb + SWA (r3c-like body), random init with the window and memory paths switched on
from v13_sempty.model import LM
from v13_sempty.config import get_config
cfg = get_config('baseline_real_pm'); cfg.pam_layers = (); cfg.share_reads = False; cfg.echo = True; cfg.chrono = True
cfg.out_gate = True; cfg.delta = True; cfg.swa_layers = (5, 9, 13); cfg.swa_window = 128; cfg.dropout = 0.0
m = LM(cfg).eval()
for i in (5, 9, 13):
    torch.nn.init.normal_(m.blocks[i].window.o_proj.weight, std=0.02)
print(f'16nb + SWA: blocks {len(m.blocks)}, notebooks {sum(b.pam is not None for b in m.blocks)}, windows {[i for i, b in enumerate(m.blocks) if b.window is not None]}')
z, b_, t_ = m._stem(ids[:, :64], 0); outs = []
with torch.no_grad():
    for i, blk in enumerate(m.blocks):
        z, s = blk(z, pam_state=None, step_offset=0)
        outs.append((tuple(d.name for d in z.layout), type(s).__name__))
print('  per-block output (z layout, state type):', sorted(set(outs)))
st = stream_parity(m, '16nb + SWA', T=400, cuts=(130, 260), dec=20)
print(f'  states {len(st)} = 16 blocks + window caches + echo: {len(st) == 18}; echo last is a tuple: {isinstance(st[-1], tuple)}')
