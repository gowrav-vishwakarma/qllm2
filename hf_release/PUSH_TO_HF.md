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

## Build the release bundle

From the qllm2 repo root:

```bash
# 1. Export weights-only checkpoint + config.json
uv run python scripts/export_hf_release.py

# 2. modeling_qllm.py, run_chat.py, requirements.txt live in hf_release/
#    (regenerate modeling if v11/model.py changed — see scripts/export_hf_release.py)

# 3. Verify locally BEFORE upload (required gate)
cd hf_release && bash verify.sh
```

Verification must print `All checks passed` (Paris + stop on im_end).

## Upload

```bash
# From repo root, after verify.sh passes:
uv run python scripts/push_qllm_hf.py
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
| `qllm_v11_e3k3_chat.pt` | Weights-only checkpoint (~384 MB) |
| `config.json` | Architecture + training metadata |
| `modeling_qllm.py` | Self-contained model code (no qllm2 clone needed) |
| `run_chat.py` | Interactive / single-prompt chat |
| `requirements.txt` | `torch`, `transformers` |
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
