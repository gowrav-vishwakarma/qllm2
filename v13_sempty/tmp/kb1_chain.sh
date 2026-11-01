#!/bin/bash
# kb1 (EXPERIMENTS_SEMPY "kb1" pre-registration): the known zero-shot suite through lm-eval-harness 0.4.9.1 on the 4090, bf16,
# one model at a time, after w1w. Our bodies via scripts/lm_eval_bodies.py --ckpt; references via --hf (stock hf model).
cd "$(dirname "$0")/../.." || exit 1
export PATH=/usr/local/cuda/bin:$PATH
H=$(git rev-parse --short HEAD); P=/home/gowrav/.venvs/qllm311/bin/python; F=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty; C=/media/gowrav/d_drive/qllm2/checkpoints_v13_sempty
echo "kb1 chain: code $H, $(date -u +%FT%TZ)"
until ! tmux has-session -t '=w1w' 2>/dev/null; do sleep 60; done
mkdir -p logs/kb1
run() { O=logs/kb1/v13_sempty_kb1_$1_$H.json; [ -s $O ] && return; echo "=== $1 $(date -u +%FT%TZ)"; $P scripts/lm_eval_bodies.py $2 --device cuda --batch_size 32 --out $O 2>&1 | grep -vE 'Warn|symbolic|it/s\]|%\||Overwriting|stddev' | tail -8; }
run b2p2 "--ckpt $F/b2p2_main_38e3c29/best_model.pt"
run b2t2 "--ckpt $F/b2t2_main_38e3c29/best_model.pt"
run b2m2 "--ckpt $F/b2m2_main_38e3c29/best_model.pt"
run cu2a "--ckpt $C/cu2a_main_24b2969/best_model.pt"
run r3c "--ckpt $F/r3cb_main_38e3c29/best_model.pt"
run tpp3b "--ckpt $C/mix3b_transformer_answ100_e072a8f/best_model.pt"
run mamba3b "--ckpt $C/mix3b_mamba2_answ100_b36e3a5/best_model.pt"
run smollm2_135m "--hf HuggingFaceTB/SmolLM2-135M"
run pythia_160m "--hf EleutherAI/pythia-160m"
run gpt2_124m "--hf openai-community/gpt2"
echo "=== kb1 chain DONE $(date -u +%FT%TZ)"
