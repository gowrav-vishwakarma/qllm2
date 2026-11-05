#!/bin/bash
# Move checkpoint files from the 6000 to d_drive: scp to a .part, rename, sha256 both sides, delete on the 6000 ONLY if they match.
#   bash v13_sempty/tmp/move_6000_to_d.sh <remote dir relative to ~/Development/qllm-private/checkpoints_v13_sempty> <file> [...]
R=ubuntu@34.131.203.207; RB=Development/qllm-private/checkpoints_v13_sempty; LB=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty
d=$1; shift
mkdir -p $LB/$d
for f in "$@"; do
  L=$LB/$d/$f
  [ -s $L ] || { scp -q $R:$RB/$d/$f $L.part && mv $L.part $L; } || { echo "COPY FAIL $d/$f"; continue; }
  a=$(sha256sum $L | cut -c1-64); b=$(ssh -n -o ConnectTimeout=20 $R "sha256sum $RB/$d/$f" | cut -c1-64)
  if [ -n "$a" ] && [ "$a" = "$b" ]; then ssh -n -o ConnectTimeout=20 $R "rm -f $RB/$d/$f" && echo "MOVED $d/$f sha ${a:0:16}"
  else echo "SHA MISMATCH $d/$f local ${a:0:16} remote ${b:0:16}: kept on the 6000"; fi
done
