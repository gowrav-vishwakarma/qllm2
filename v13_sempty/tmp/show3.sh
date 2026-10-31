#!/bin/bash
# 3 shown generations (MLB-1 A7) for one checkpoint: rule (greedy), recall (greedy), free text (sampled, seed 1)
CK=$1; PY=${PY:-.venv/bin/python}
echo "checkpoint: $CK"
g() { echo "--- prompt ($1): $(printf %q "$2")"; $PY -m v13_sempty.generate --checkpoint "$CK" --device cpu --seed 1 --prompt "$2" "${@:3}" 2>&1 | grep -v "Warning\|warn\|symbolic" | tail -n +1; }
g "rule, greedy" $'Map rule.\n3 -> 10\n5 -> 14\n8 -> 20\n11 ->' --temperature 0 --max_tokens 30
g "recall, greedy" $'Note for the team: the storage room code is 4172, and the backup key is with Priya in room 18.\nThe quarterly review covered hiring, the new office lease, travel budgets for the spring conference, and the plan to move the build servers to the new data centre before June.\nQuestion: who has the backup key? Answer:' --temperature 0 --max_tokens 12
g "free text, T 0.8 top-k 50 top-p 0.9 rep 1.2" 'The first printing presses in Europe' --max_tokens 60
