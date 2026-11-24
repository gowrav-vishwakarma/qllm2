"""Orphan-producer GPU hold (sft1 decay start OOM, 2026-11-24): python v13_sempty/tmp/test_producer_orphan.py
A parent allocates CUDA memory, forks a producer-like child that blocks on a full queue (as sft_mix producers do), then leaves via
os._exit(0) like train.py. Modes:
  control   : no fix              -> EXPECT the child to survive and the GPU memory to stay held (the bug reproduces)
  pdeathsig : child calls sft_mix._die_with_parent() first          -> child dead, memory freed
  terminate : parent runs train.py's active_children terminate block -> child dead, memory freed
PASS = control holds AND both fixes free."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOC_GB = 2

CHILD = r'''
import os, sys, time, multiprocessing as mp
sys.path.insert(0, %r)
import torch
from v13_sempty import sft_mix as sm
mode = sys.argv[1]
x = torch.empty(int(%d * 2**30), dtype=torch.uint8, device='cuda'); torch.cuda.synchronize()

def producer(q, fix):
    if fix:
        sm._die_with_parent()
    while True:
        q.put('x' * 1000)          # blocks once the queue is full, like a producer nobody reads

ctx = mp.get_context('fork')
q = ctx.Queue(maxsize=1)
p = ctx.Process(target=producer, args=(q, mode == 'pdeathsig'), daemon=True)
p.start()
print(p.pid, flush=True)
time.sleep(2)
if mode == 'terminate':
    for _p in mp.active_children():
        _p.terminate()
    for _p in mp.active_children():
        _p.join(timeout=5)
sys.stdout.flush(); os._exit(0)
''' % (str(ROOT), ALLOC_GB)


def gpu_used_mib() -> int:
    return int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).split()[0])


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    st = Path(f'/proc/{pid}/stat')
    return st.exists() and st.read_text().split()[2] != 'Z'


def run(mode: str):
    base = gpu_used_mib()
    pf = Path(f'/tmp/test_producer_orphan_{os.getpid()}_{mode}.out')
    with open(pf, 'w') as fo:                     # a file, not a pipe: a surviving child would keep a pipe open forever
        subprocess.run([sys.executable, '-c', CHILD, mode], stdout=fo, stderr=subprocess.STDOUT, timeout=120)
    pid = int(pf.read_text().split()[0])
    pf.unlink()
    time.sleep(4)
    held = gpu_used_mib() - base
    a = alive(pid)
    if a:
        os.kill(pid, 9)
        time.sleep(2)
    print(f'{mode:10s} child alive after parent exit: {a} | GPU held after exit: {held} MiB (alloc {ALLOC_GB * 1024})', flush=True)
    return a, held


ok = True
a, held = run('control')
ctl = a and held > ALLOC_GB * 1024 * 0.9
print('  control reproduces the hold:', ctl)
ok &= ctl
for m in ('pdeathsig', 'terminate'):
    a, held = run(m)
    freed = (not a) and held < 300
    print(f'  {m} frees the GPU:', freed)
    ok &= freed
print('ALL PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
