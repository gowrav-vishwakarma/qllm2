"""
Downstream evaluation for QPAM MLX model: HellaSwag and LAMBADA (zero-shot).

Usage:
    # Sanity check with random weights (no checkpoint):
    uv run python eval_downstream.py --no-checkpoint

    # With a trained checkpoint:
    uv run python eval_downstream.py --checkpoint checkpoints_qpam_mlx/best_model.npz

    # Run only one benchmark:
    uv run python eval_downstream.py --no-checkpoint --tasks hellaswag
    uv run python eval_downstream.py --no-checkpoint --tasks lambada

    # Limit examples for quick testing:
    uv run python eval_downstream.py --no-checkpoint --max-examples 100
"""

import sys, os, math, argparse, time
import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import QPAMModel


# ──────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────

def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


# ──────────────────────────────────────────────────────────────
# Log-likelihood computation
# ──────────────────────────────────────────────────────────────

def compute_log_likelihood(model, token_ids, max_ctx=1024):
    """
    Compute total log-likelihood of token_ids under the model.

    The model produces logits for positions 0..T-1 which predict tokens 1..T.
    We sum log P(token_t | tokens_<t) for t in [1, len(token_ids)-1].

    Returns (total_log_prob, num_scored_tokens).
    """
    if len(token_ids) < 2:
        return 0.0, 0

    # Truncate to max context length
    token_ids = token_ids[:max_ctx]

    tokens = mx.array(token_ids, dtype=mx.int32).reshape(1, -1)  # [1, T]
    logits = model(tokens)  # [1, T, vocab_size]
    mx.eval(logits)

    # Log softmax over vocab dimension
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)  # [1, T, V]

    # Score: for each position t, we want log P(token_{t+1} | prefix up to t)
    # logits at position t predict token t+1
    T = len(token_ids)
    targets = mx.array(token_ids[1:], dtype=mx.int32)  # [T-1]

    # Gather the log probs for each target token
    # log_probs shape: [1, T, V], we want positions 0..T-2 predicting tokens 1..T-1
    log_probs_flat = log_probs[0, :T - 1, :]  # [T-1, V]

    # Index into vocab dimension for each target
    target_log_probs = []
    for i in range(T - 1):
        target_log_probs.append(log_probs_flat[i, targets[i]])

    total = mx.sum(mx.array(target_log_probs))
    mx.eval(total)

    return total.item(), T - 1


def compute_continuation_log_likelihood(model, context_ids, continuation_ids, max_ctx=1024):
    """
    Compute log P(continuation | context) by running the model on [context + continuation]
    and summing log probs only over the continuation tokens.
    """
    full_ids = context_ids + continuation_ids
    if len(full_ids) > max_ctx:
        # Truncate context from the left to fit
        overflow = len(full_ids) - max_ctx
        full_ids = full_ids[overflow:]
        ctx_len = len(context_ids) - overflow
    else:
        ctx_len = len(context_ids)

    if ctx_len < 1 or len(continuation_ids) < 1:
        return -float("inf"), 0

    tokens = mx.array(full_ids, dtype=mx.int32).reshape(1, -1)
    logits = model(tokens)  # [1, T, V]
    mx.eval(logits)

    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # We only score the continuation part.
    # Position ctx_len-1 predicts token at ctx_len (first continuation token), etc.
    cont_len = len(continuation_ids)
    start = ctx_len - 1  # logit position that predicts first continuation token
    end = ctx_len - 1 + cont_len

    target_tokens = full_ids[ctx_len:ctx_len + cont_len]

    total_ll = 0.0
    for i in range(cont_len):
        pos = start + i
        tok = target_tokens[i]
        lp = log_probs[0, pos, tok]
        mx.eval(lp)
        total_ll += lp.item()

    return total_ll, cont_len


# ──────────────────────────────────────────────────────────────
# HellaSwag evaluation
# ──────────────────────────────────────────────────────────────

def preprocess_hellaswag_ending(text):
    """Clean up HellaSwag endings (they sometimes have artifacts like [header])."""
    import re
    text = text.strip()
    text = re.sub(r"\[.*?\]", "", text)  # remove bracketed items
    text = text.strip()
    return text


def eval_hellaswag(model, tokenizer, max_examples=None, max_ctx=1024):
    """
    HellaSwag zero-shot evaluation.

    For each example: compute log P(ending_i | context) for i in {0,1,2,3}.
    Pick the ending with highest log-likelihood. Compare to gold label.
    """
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("  HellaSwag (zero-shot)")
    print("=" * 60)

    ds = load_dataset("Rowan/hellaswag", split="validation")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = 0
    total = 0
    t0 = time.time()

    for idx, example in enumerate(ds):
        ctx_text = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        # Tokenize context
        ctx_ids = tokenizer.encode(ctx_text)

        # Score each ending
        scores = []
        for ending in endings:
            ending_clean = preprocess_hellaswag_ending(ending)
            # Prepend a space to the ending for proper BPE tokenization
            cont_ids = tokenizer.encode(" " + ending_clean)
            ll, n_tok = compute_continuation_log_likelihood(
                model, ctx_ids, cont_ids, max_ctx=max_ctx
            )
            # Length-normalize: average log prob per token
            scores.append(ll / max(n_tok, 1))

        pred = int(np.argmax(scores))
        if pred == label:
            correct += 1
        total += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            acc = correct / total * 100
            print(f"  [{idx + 1}/{len(ds)}] accuracy={acc:.2f}%  "
                  f"({elapsed:.0f}s, {total / elapsed:.1f} ex/s)")

    elapsed = time.time() - t0
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\n  HellaSwag results:")
    print(f"    Accuracy:  {accuracy:.2f}% ({correct}/{total})")
    print(f"    Time:      {elapsed:.1f}s ({total / elapsed:.1f} examples/sec)")
    print(f"    (Random baseline: ~25%)")
    return {"accuracy": accuracy, "correct": correct, "total": total}


# ──────────────────────────────────────────────────────────────
# LAMBADA evaluation
# ──────────────────────────────────────────────────────────────

def eval_lambada(model, tokenizer, max_examples=None, max_ctx=1024):
    """
    LAMBADA zero-shot evaluation.

    For each passage, the model must predict the final word.
    We measure:
      - Accuracy: whether argmax prediction matches the last word
      - Perplexity: exp(avg negative log-likelihood of last word tokens)
    """
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("  LAMBADA (zero-shot)")
    print("=" * 60)

    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = 0
    total = 0
    total_nll = 0.0
    total_last_word_tokens = 0
    t0 = time.time()

    for idx, example in enumerate(ds):
        text = example["text"]

        # Split into context and last word
        # The last word is the final whitespace-delimited token
        text = text.rstrip()
        last_space = text.rfind(" ")
        if last_space == -1:
            continue  # skip if no space found

        context = text[:last_space]
        last_word = text[last_space:]  # includes the leading space

        # Tokenize
        ctx_ids = tokenizer.encode(context)
        last_word_ids = tokenizer.encode(last_word)

        if len(last_word_ids) == 0 or len(ctx_ids) == 0:
            continue

        # Compute log-likelihood of the last word given context
        full_ids = ctx_ids + last_word_ids
        if len(full_ids) > max_ctx:
            overflow = len(full_ids) - max_ctx
            full_ids = full_ids[overflow:]
            ctx_len = len(ctx_ids) - overflow
        else:
            ctx_len = len(ctx_ids)

        if ctx_len < 1:
            continue

        tokens = mx.array(full_ids, dtype=mx.int32).reshape(1, -1)
        logits = model(tokens)  # [1, T, V]
        mx.eval(logits)

        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Check accuracy: does greedy continuation match the last word tokens?
        n_last = len(last_word_ids)
        match = True
        word_nll = 0.0

        for i in range(n_last):
            pos = ctx_len - 1 + i  # position whose logit predicts token at ctx_len + i
            target_tok = last_word_ids[i]

            # Greedy prediction
            pred_tok = mx.argmax(logits[0, pos], axis=-1)
            mx.eval(pred_tok)
            if pred_tok.item() != target_tok:
                match = False

            # NLL
            lp = log_probs[0, pos, target_tok]
            mx.eval(lp)
            word_nll -= lp.item()

        if match:
            correct += 1
        total += 1
        total_nll += word_nll
        total_last_word_tokens += n_last

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            acc = correct / total * 100
            ppl = math.exp(total_nll / max(total_last_word_tokens, 1))
            print(f"  [{idx + 1}/{len(ds)}] accuracy={acc:.2f}%  "
                  f"ppl={ppl:.2f}  ({elapsed:.0f}s, {total / elapsed:.1f} ex/s)")

    elapsed = time.time() - t0
    accuracy = correct / total * 100 if total > 0 else 0.0
    perplexity = math.exp(total_nll / max(total_last_word_tokens, 1))

    print(f"\n  LAMBADA results:")
    print(f"    Accuracy:   {accuracy:.2f}% ({correct}/{total})")
    print(f"    Perplexity: {perplexity:.2f}")
    print(f"    Time:       {elapsed:.1f}s ({total / elapsed:.1f} examples/sec)")
    print(f"    (Random baseline: ~0% accuracy, ~50257 perplexity)")
    return {"accuracy": accuracy, "perplexity": perplexity, "correct": correct, "total": total}


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QPAM MLX downstream evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.npz or .safetensors)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Run with randomly initialized model (sanity check)")
    parser.add_argument("--tasks", type=str, default="hellaswag,lambada",
                        help="Comma-separated list of tasks (hellaswag, lambada)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit number of examples per task (for debugging)")
    parser.add_argument("--max-ctx", type=int, default=1024,
                        help="Maximum context length in tokens")

    # Model config (must match the checkpoint)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--expand", type=int, default=3)

    args = parser.parse_args()

    if args.checkpoint is None and not args.no_checkpoint:
        parser.error("Provide --checkpoint or --no-checkpoint")

    tasks = [t.strip().lower() for t in args.tasks.split(",")]

    print("QPAM MLX Downstream Evaluation")
    print(f"  Device:     {mx.default_device()}")
    print(f"  Tasks:      {', '.join(tasks)}")
    print(f"  Checkpoint: {args.checkpoint or '(random init)'}")
    print(f"  Max ctx:    {args.max_ctx}")
    if args.max_examples:
        print(f"  Max examples per task: {args.max_examples}")
    print()

    # Build model
    print("Building model...")
    model = QPAMModel(
        vocab_size=50257,
        dim=args.dim,
        num_layers=args.layers,
        expand=args.expand,
        num_heads=args.heads,
        head_dim=args.head_dim,
    )

    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_weights(args.checkpoint)
        print("  Checkpoint loaded.")
    else:
        print("  Using random initialization (sanity check mode).")

    # Force eval of parameters
    mx.eval(model.parameters())

    # Count params
    def count_params(tree):
        if isinstance(tree, mx.array):
            return tree.size
        elif isinstance(tree, dict):
            return sum(count_params(v) for v in tree.values())
        elif isinstance(tree, list):
            return sum(count_params(v) for v in tree)
        return 0
    n_params = count_params(model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    print()

    # Tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = get_tokenizer()
    print()

    # Run evaluations
    results = {}

    if "hellaswag" in tasks:
        results["hellaswag"] = eval_hellaswag(
            model, tokenizer,
            max_examples=args.max_examples,
            max_ctx=args.max_ctx,
        )

    if "lambada" in tasks:
        results["lambada"] = eval_lambada(
            model, tokenizer,
            max_examples=args.max_examples,
            max_ctx=args.max_ctx,
        )

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if "hellaswag" in results:
        r = results["hellaswag"]
        print(f"  HellaSwag:  {r['accuracy']:.2f}% ({r['correct']}/{r['total']})")
    if "lambada" in results:
        r = results["lambada"]
        print(f"  LAMBADA:    {r['accuracy']:.2f}% acc, {r['perplexity']:.2f} ppl "
              f"({r['correct']}/{r['total']})")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
