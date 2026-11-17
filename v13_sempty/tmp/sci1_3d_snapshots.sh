#!/bin/bash
# 3d powered readout (EXPERIMENTS_SEMPY "3d powered readout", Mind + Lead 2026-11-17): snapshot the 6000's rolling latest.pt at every
# 0.25B (3,815 steps) from 42.25B to 43.75B (the 42B / 43B / 44B milestones are kept by the run itself), copy each to d_drive for
# Library's TOLD-MARGIN on the 4090. The trainer is untouched: latest.pt is replaced by tmp + rename, so a hard link keeps the old inode.
#   tmux new-session -d -s mind3dsnap "bash v13_sempty/tmp/sci1_3d_snapshots.sh 2>&1 | tee -a logs/v13_sempty_3d_snapshots_<hash>_<stamp>.log"
cd "$(dirname "$0")/../.." || exit 1
R=ubuntu@34.131.203.207; RD=Development/qllm_sci1d/checkpoints_v13_sempty/sci1_long3d_cf9dee1
LD=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/sci1_long3d_cf9dee1
SO="-o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"   # a dead connection fails in ~1 min, not hangs
# step + the cumulative data counters at that save (chat passes, per-source tokens + docs; the window mix = the diff of two saves): Lead 2026-11-17, to test the two-state swing against a data cycle
mix_of() { ssh -o ConnectTimeout=20 $R "cd ~/$RD && ~/Development/qllm_sci1d/.venv/bin/python -c \"
import torch; c = torch.load('$1', map_location='cpu', weights_only=False, mmap=True); d = c.get('data_cursor') or {}
dc = d.get('doc_counters', {}); tc = d.get('token_counters', {}); T = sum(tc.values()) or 1
print('MIX step', c['global_step'], 'passes', {k: v for k, v in dc.items() if k.endswith('@pass')}, 'tok', dict(sorted(tc.items())), 'docs', {k: v for k, v in sorted(dc.items()) if '@' not in k and '#' not in k})\"" 2>/dev/null | tail -1; }
r_free_ok() { [ "$(ssh -o ConnectTimeout=20 $R "df --output=avail -BG ~/$RD | tail -1" | tr -dc 0-9)" -ge 30 ]; }   # 6000 floor 25 GB + one 4.6 GB snapshot
step_of() { ssh -o ConnectTimeout=20 $R "cd ~/$RD && ~/Development/qllm_sci1d/.venv/bin/python -c \"import torch; print(torch.load('$1', map_location='cpu', weights_only=False, mmap=True)['global_step'])\"" 2>/dev/null | tail -1; }
for T in 644693 648508 652322 659952 663767 667581; do   # 42.25 / 42.5 / 42.75 / 43.25 / 43.5 / 43.75B (milestone = 640,878 + k x 3,815)
  f=snap_${T}.pt; [ -f $LD/.snap_${T}.done ] && continue; [ -f $LD/.snap_${T}.skipped ] && { echo "=== target $T marked skipped"; continue; }   # 2026-11-17: .skipped was ignored -> an off-grid snap_668500
  until ssh $SO $R true 2>/dev/null; do echo "=== 6000 SSH not answering, retry in 2 min $(date)"; sleep 120; done   # decide resume only on a live link
  # resume (2026-11-17 SSH flap): a snapshot already moved aside on the 6000 for this target (logged "snapshot step S (target T)") is
  # copied again instead of taking a new one
  s=$(cat logs/v13_sempty_3d_snapshots_*.log 2>/dev/null | grep -E "=== snapshot step [0-9]+ \(target $T\)" | tail -1 | sed -E 's/.*snapshot step ([0-9]+) .*/\1/')
  if [ -n "$s" ] && ssh $SO $R "test -f ~/$RD/snap_${s}.pt"; then f=snap_${s}.pt; echo "=== resuming snapshot step $s (target $T) $(date)"
  else
  echo "=== waiting for latest.pt >= step $T $(date)"
  while true; do
    # a hard link only: latest.pt itself is never opened for writing, moved or removed (the trainer replaces it by tmp + rename)
    ssh -o ConnectTimeout=20 $R "cd ~/$RD && rm -f .probe.pt && ln latest.pt .probe.pt" 2>/dev/null
    s=$(step_of .probe.pt); [ -n "$s" ] && [ "$s" -ge "$T" ] && break
    ssh $R "rm -f ~/$RD/.probe.pt" 2>/dev/null; sleep 300
  done
  # next target already passed by > 1 save interval? then this one was missed (e.g. the box was down): take it anyway, step is logged
  if ! r_free_ok; then ssh $R "rm -f ~/$RD/.probe.pt"; echo "=== SKIP target $T: 6000 below 30 GB free (floor 25 + snapshot) $(date)"; touch $LD/.snap_${T}.skipped; continue; fi
  until ssh $SO $R "cd ~/$RD && { test -f snap_${s}.pt || mv .probe.pt snap_${s}.pt; }"; do echo "=== mv failed (connection?), retry in 5 min $(date)"; sleep 300; done
  f=snap_${s}.pt; echo "=== snapshot step $s (target $T) $(date)"; mix_of $f
  # weights-only (Lead 2026-11-17, d_drive at 11 GB): the readers need config + model_state_dict only (1.5 GB vs 4.6 GB)
  until ssh $SO $R "cd ~/$RD && ~/Development/qllm_sci1d/.venv/bin/python -c \"
import torch, os
f = torch.load('$f', map_location='cpu', weights_only=False, mmap=True)
if 'optimizer_state_dict' in f:
    torch.save({k: f[k] for k in ('config', 'model_state_dict', 'global_step', 'global_tokens', 'best_val_loss', 'args') if k in f}, '$f.slim')
    os.replace('$f.slim', '$f')
print('slim ok')\""; do echo "=== slim on the 6000 failed, retry in 5 min $(date)"; sleep 300; done
  fi
  # copy with retries; a delete on the 6000 ONLY after both shas are real 64-hex strings and equal (2026-11-17: an empty == empty
  # after a dropped connection would have counted as a match)
  for try in 1 2 3 4 5 6; do
    sz=$(ssh $SO $R "stat -c %s ~/$RD/$f" | tr -dc 0-9); [ -n "$sz" ] || { echo "=== stat failed (try $try) $(date)"; sleep 300; continue; }
    until bash scripts/require_free_gb.sh $LD 10 $sz; do echo "d_drive low, waiting $(date)"; sleep 600; done
    rm -f $LD/$f.part
    scp -q $SO $R:$RD/$f $LD/$f.part && [ "$(stat -c %s $LD/$f.part)" = "$sz" ] && mv $LD/$f.part $LD/$f && break
    echo "=== scp failed / short (try $try) $(date)"; sleep 300
  done
  a=$([ -s $LD/$f ] && sha256sum $LD/$f | cut -c1-64); b=$(ssh $SO $R "sha256sum ~/$RD/$f" | cut -c1-64)
  if [[ "$a" =~ ^[0-9a-f]{64}$ ]] && [ "$a" = "$b" ]; then ssh $SO $R "rm ~/$RD/$f" && touch $LD/.snap_${T}.done && echo "=== $f copied, sha ok, removed on the 6000 $(date)" \
      || echo "=== $f copied, sha ok, but the 6000 rm failed: both kept, target NOT marked done $(date)"
  else echo "=== $f not verified (local '$a' vs 6000 '$b'): kept on both, STOP $(date)"; exit 1; fi
done
echo "=== 3d snapshots DONE $(date)"
