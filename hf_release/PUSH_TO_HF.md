# Pushing QLLM V11 to Hugging Face

Target repo: **https://huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat**

## Prerequisites

### 1. Hugging Face account

Create an account at [huggingface.co/join](https://huggingface.co/join) if you don't have one.

### 2. Create the model repo (one-time)

**Web UI:** Profile → **New model** → name it `qllm-pam-v11-e3k3-chat` under your namespace.

**CLI:**

```bash
hf repo create gowravvishwakarma/qllm-pam-v11-e3k3-chat --type model
```

### 3. Write access token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a token with **Write** scope
3. Login on this machine:

```bash
hf auth login
# or export HF_TOKEN=hf_...
```

## Continuous shipping (rounds + main)

Each `+2B` round ships to a Hugging Face **revision tag** (e.g. `round-2b-gate`,
`round-4b-gate`) **and** overwrites **`main`** so visitors who open the Files tab
see the latest round by default. Every round tag stays pullable for pinned downloads.
Training runs on GCP; publishing runs on the RTX4090 (which holds the HF token).

Server records each ship-ready round in `releases/server_manifest.json` (written
by `scripts/run_v11_round.sh export`). The RTX4090 pulls incrementally:

```bash
# On RTX4090 (has hf auth). One command does pull -> verify -> push:
ROUND_TAG=round-4b-gate ./scripts/run_v11_round.sh ship

# Or step by step:
./scripts/pull_v11_release.sh --list           # server vs local, no download
./scripts/pull_v11_release.sh --round round-4b-gate   # incremental, sha256-verified
cp releases/round-4b-gate/qllm_v11_e3k3_chat.pt hf_release/qllm_v11_e3k3_chat.pt
cd hf_release && bash verify.sh
uv run python scripts/push_qllm_hf.py --revision round-4b-gate --also-main
```

**Verification (required before any push):**

Run `verify.sh` on the RTX4090 against the round export staged in `hf_release/`:

| Script | Checkpoint | Expect |
|--------|------------|--------|
| `verify.sh` | `hf_release/qllm_v11_e3k3_chat.pt` (SFT) | Paris + `stopped_on_im_end=True` |
| `verify.sh` | `hf_release/qllm_v11_e3k3_pretrain.pt` (pretrain) | prefix `The capital of France is` → Paris |

```bash
cd hf_release
bash verify.sh          # current round (50261, content-aware gate)
```

Legacy ~10B weights live on the immutable HF tag **`v1-old-deprecated-10B-sft`** — no pre-ship check needed.

The smart pull uses per-machine state at `~/.qllm/v11_pull_state.json`, so a
second machine won't re-download a round it already has. Pull to a specific
revision later with:

```bash
huggingface-cli download gowravvishwakarma/qllm-pam-v11-e3k3-chat --revision round-2b-gate --local-dir .
```

## Build the release bundle (manual / one-off)

From the qllm2 repo root:

```bash
# 1. Export weights-only checkpoint + config.json (+ record round in manifest)
uv run python scripts/export_hf_release.py \
  --src checkpoints_v11_sft_chat_smoltalk_v2/best_model.pt \
  --round round-2b-gate --tag round-2b-gate \
  --pretrain_tokens_total 2000000000 --round_tokens 2000000000 --record-manifest

uv run python scripts/export_hf_release.py \
  --src checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt \
  --round round-6b-gate --tag round-6b-gate \
  --out-name qllm_v11_e3k3_pretrain.pt --config-name config_pretrain.json \
  --checkpoint-type pretrain \
  --pretrain_tokens_total 6000000000 --round_tokens 2000000000 --record-manifest

# 2. modeling_qllm.py, run_chat.py, requirements.txt live in hf_release/
#    (regenerate modeling if v11/model.py changed — see scripts/export_hf_release.py)

# 3. Verify locally BEFORE upload (required gate)
cd hf_release && bash verify.sh

# 4. Optional: regenerate sample Q&A for the model card
# ROUND_TAG=round-2b-gate ./scripts/run_v11_round.sh eval
```

Must print success (Paris + stop on im_end) before upload.

## One-time: promote round-2b to main (archive legacy) — done

Legacy ~10B was archived to tag **`v1-old-deprecated-10B-sft`**; `main` now tracks the latest
`round-*` bundle. Routine shipping uses `ship` with `--also-main`. No further action needed.

**Optional — refresh code on HF `main` only (weights unchanged):**

```bash
cd hf_release && bash verify.sh
./scripts/push_hf_main_code_only.sh
```

## Upload

```bash
# Publish to a round revision tag and update main (recommended):
uv run python scripts/push_qllm_hf.py --revision round-4b-gate --also-main
```

Or manually:

```bash
hf upload gowravvishwakarma/qllm-pam-v11-e3k3-chat hf_release/ \
  --repo-type model \
  --exclude ".git/*"
```

## What gets uploaded

| File | Purpose |
|------|---------|
| `qllm_v11_e3k3_chat.pt` | SFT chat weights-only checkpoint (~384 MB) |
| `qllm_v11_e3k3_pretrain.pt` | Pretrain base weights-only checkpoint (~384 MB) |
| `config.json` | SFT architecture + training metadata |
| `config_pretrain.json` | Pretrain architecture + training metadata |
| `modeling_qllm.py` | Self-contained model code (no qllm2 clone needed) |
| `run_chat.py` | Interactive / single-prompt SFT chat |
| `run_complete.py` | Prefix completion (pretrain base) |
| `eval_chat.py` | Batch SFT chat eval |
| `eval_compare.py` | Pretrain vs SFT comparison eval |
| `eval_prompts_round1.yaml` | SFT chat prompt suite |
| `eval_prompts_compare.yaml` | Paired prefix + chat questions |
| `SAMPLES_round-<tag>.md` | Full SFT chat sample log |
| `SAMPLES_round-<tag>-compare.md` | Pretrain vs SFT summary table |
| `requirements.txt` | `torch`, `transformers`, `PyYAML` |
| `README.md` | Model card |

## After upload

1. Open the model page and confirm files rendered
2. Test download + run from a clean directory:

```bash
mkdir /tmp/qllm-test && cd /tmp/qllm-test
huggingface-cli download gowravvishwakarma/qllm-pam-v11-e3k3-chat --local-dir .
pip install -r requirements.txt
python run_chat.py --prompt "What is the capital of France?"
```

3. Optional: link from the main GitHub README

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `401 Unauthorized` | Re-run `hf auth login` or check `HF_TOKEN` |
| `403 Forbidden` on commit | Token needs **Write** scope. Re-login: `hf auth login` with a write token, or `export HF_TOKEN=hf_...` (read tokens cannot push). If repo has branch protection, use `--create-pr` or disable protection in repo Settings. |
| `403` on xet-write-token | Set `HF_HUB_DISABLE_XET=1` before upload (already set in `push_qllm_hf.py`) |
| Upload timeout | Retry; ~400 MB should succeed in one shot |
| `verify.sh` fails | Do not upload — fix export or modeling first |
| CUDA OOM at inference | Use CPU or reduce `max_new_tokens` |
