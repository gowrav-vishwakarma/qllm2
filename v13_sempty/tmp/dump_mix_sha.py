"""Cross-box data check (Mind 2026-11-18): re-run a logged run's exact [args] with --dump_batches N (fresh cursor, CPU, no model
training) and print the sha of the first N micro-batches. Usage: python v13_sempty/tmp/dump_mix_sha.py <run.log> <N> [k=v overrides]"""
import ast, subprocess, sys, re
log, n = sys.argv[1], int(sys.argv[2])
line = next(l for l in open(log) if l.startswith('[args] '))
a = ast.literal_eval(line[len('[args] '):].strip())
a.update(device='cpu', resume='auto', continue_from='', init_from='', checkpoint_dir='/tmp/dump_mix_ckpt_' + str(__import__('os').getpid()), dump_batches=n)
for kv in sys.argv[3:]:
    k, v = kv.split('=', 1); a[k] = type(a[k])(v) if k in a and a[k] is not None and not isinstance(a[k], bool) else v
argv = []
for k, v in a.items():
    if v is None: continue
    if isinstance(v, bool):
        if v: argv.append('--' + k)
        continue
    argv += ['--' + k, str(v)]
sys.exit(subprocess.call([sys.executable, '-u', '-m', 'v13_sempty.train'] + argv))
