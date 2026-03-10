"""
V6 Evaluation Probes -- beyond perplexity.

Tests entity tracking, fact persistence, bank specialization,
and working memory utilization on a trained v6 checkpoint.

Usage:
    uv run python scripts/v6_eval_probes.py --checkpoint checkpoints_v6/best_model.pt
    uv run python scripts/v6_eval_probes.py --checkpoint checkpoints_v6_wikitext103/best_model.pt --verbose
"""

import argparse
import json
import sys
import math
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from v6.model import PhaseFieldLM, create_model, ModelOutput
from v6.config import V6Config
from v6.core.complex import cabs


# ─────────────────────────────────────────────────────────────
# 1. Entity Co-reference Probe
# ─────────────────────────────────────────────────────────────

COREF_PROMPTS = [
    {
        "prompt": "Alice went to the store. She bought some milk. When she got home,",
        "expected_tokens": ["she", "Alice"],
        "wrong_tokens": ["he", "they", "it"],
        "description": "Female pronoun after female entity",
    },
    {
        "prompt": "Bob was a doctor. He worked at the hospital every day. His",
        "expected_tokens": ["patients", "office", "work", "job", "colleagues"],
        "wrong_tokens": ["her", "she"],
        "description": "Male pronoun continuity",
    },
    {
        "prompt": "The cat sat on the mat. The dog entered the room. It started barking because",
        "expected_tokens": ["it", "the", "he"],
        "wrong_tokens": [],
        "description": "Entity disambiguation (dog vs cat)",
    },
    {
        "prompt": "Paris is the capital of France. The city is known for",
        "expected_tokens": ["its", "the", "being"],
        "wrong_tokens": [],
        "description": "Entity-city co-reference",
    },
    {
        "prompt": "John and Mary went to the park. John played football while Mary",
        "expected_tokens": ["played", "sat", "watched", "read", "walked"],
        "wrong_tokens": [],
        "description": "Two-entity tracking",
    },
]


def run_coref_probe(model, tokenizer, device, verbose=False) -> Dict:
    """Test if model maintains correct entity references."""
    model.eval()
    results = []

    for case in COREF_PROMPTS:
        ids = tokenizer.encode(case["prompt"])
        input_ids = torch.tensor([ids], device=device)

        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)[0]

        expected_ids = []
        for tok in case["expected_tokens"]:
            encoded = tokenizer.encode(" " + tok)
            if encoded:
                expected_ids.append(encoded[0])

        wrong_ids = []
        for tok in case["wrong_tokens"]:
            encoded = tokenizer.encode(" " + tok)
            if encoded:
                wrong_ids.append(encoded[0])

        expected_prob = sum(probs[tid].item() for tid in expected_ids) if expected_ids else 0
        wrong_prob = sum(probs[tid].item() for tid in wrong_ids) if wrong_ids else 0

        top5_ids = probs.topk(5).indices.tolist()
        top5_tokens = [tokenizer.decode([t]).strip() for t in top5_ids]
        top5_probs = probs.topk(5).values.tolist()

        passed = expected_prob > wrong_prob if wrong_ids else expected_prob > 0.01
        results.append({
            "description": case["description"],
            "passed": passed,
            "expected_prob": expected_prob,
            "wrong_prob": wrong_prob,
            "top5": list(zip(top5_tokens, [f"{p:.4f}" for p in top5_probs])),
        })

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {case['description']}")
            print(f"    Expected prob: {expected_prob:.4f}, Wrong prob: {wrong_prob:.4f}")
            print(f"    Top 5: {results[-1]['top5']}")

    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    return {"probe": "coref", "pass_rate": pass_rate, "details": results}


# ─────────────────────────────────────────────────────────────
# 2. Fact Persistence Probe
# ─────────────────────────────────────────────────────────────

FACT_PROMPTS = [
    {
        "setup": "The president of Zandaland is named Marcus.",
        "filler_tokens": 50,
        "query": " The president of Zandaland is",
        "expected_tokens": ["Marcus", "named"],
        "description": "Recall name after filler (50 tokens)",
    },
    {
        "setup": "The color of the special gem is bright purple.",
        "filler_tokens": 100,
        "query": " The color of the special gem is",
        "expected_tokens": ["bright", "purple"],
        "description": "Recall attribute after filler (100 tokens)",
    },
    {
        "setup": "There are exactly seven dragons in the kingdom.",
        "filler_tokens": 30,
        "query": " The number of dragons in the kingdom is",
        "expected_tokens": ["seven", "7"],
        "description": "Recall number after filler (30 tokens)",
    },
]


def _generate_filler(model, tokenizer, device, num_tokens, seed_text="Meanwhile"):
    """Generate filler text to separate setup from query."""
    ids = tokenizer.encode(seed_text)
    input_ids = torch.tensor([ids], device=device)

    with torch.no_grad():
        generated = model.generate(
            input_ids, max_new_tokens=num_tokens,
            temperature=0.8, top_k=50, repetition_penalty=1.2,
        )
    return generated[0].tolist()


def run_fact_probe(model, tokenizer, device, verbose=False) -> Dict:
    """Test if model retains facts across intervening text."""
    model.eval()
    results = []

    for case in FACT_PROMPTS:
        setup_ids = tokenizer.encode(case["setup"])
        filler_ids = _generate_filler(
            model, tokenizer, device, case["filler_tokens"])
        query_ids = tokenizer.encode(case["query"])

        full_ids = setup_ids + filler_ids + query_ids
        input_ids = torch.tensor([full_ids], device=device)

        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)[0]

        expected_ids = []
        for tok in case["expected_tokens"]:
            encoded = tokenizer.encode(" " + tok)
            if encoded:
                expected_ids.append(encoded[0])

        expected_prob = sum(probs[tid].item() for tid in expected_ids) if expected_ids else 0

        top5_ids = probs.topk(5).indices.tolist()
        top5_tokens = [tokenizer.decode([t]).strip() for t in top5_ids]
        top5_probs = probs.topk(5).values.tolist()

        passed = expected_prob > 0.01
        results.append({
            "description": case["description"],
            "passed": passed,
            "expected_prob": expected_prob,
            "filler_tokens": case["filler_tokens"],
            "total_context": len(full_ids),
            "top5": list(zip(top5_tokens, [f"{p:.4f}" for p in top5_probs])),
        })

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {case['description']} (context={len(full_ids)} tokens)")
            print(f"    Expected prob: {expected_prob:.4f}")
            print(f"    Top 5: {results[-1]['top5']}")

    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    return {"probe": "fact_persistence", "pass_rate": pass_rate, "details": results}


# ─────────────────────────────────────────────────────────────
# 3. Bank Specialization Probe
# ─────────────────────────────────────────────────────────────

BANK_TEST_TEXTS = {
    "factual": [
        "Albert Einstein developed the theory of general relativity in 1915.",
        "The Amazon River is the largest river by discharge volume of water in the world.",
        "DNA carries the genetic instructions used in the growth and development of all organisms.",
    ],
    "narrative": [
        "Once upon a time, there was a little girl who loved to play in the garden.",
        "The old man walked slowly through the dark forest, his lantern flickering.",
        "She picked up the letter and began to read, her hands trembling with fear.",
    ],
    "conversational": [
        "Hey, did you see the game last night? It was absolutely incredible!",
        "I think we should probably go to the store before it closes.",
        "Wait, are you serious right now? That doesn't make any sense at all.",
    ],
}


def run_bank_probe(model, tokenizer, device, verbose=False) -> Dict:
    """Measure semantic vs context bank activation differences across text types."""
    model.eval()
    real_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    results = {}
    all_sem_mags = {}
    all_ctx_mags = {}

    for text_type, texts in BANK_TEST_TEXTS.items():
        sem_magnitudes = []
        ctx_magnitudes = []
        similarities = []

        for text in texts:
            ids = tokenizer.encode(text)
            input_ids = torch.tensor([ids], device=device)

            z = real_model.embed(input_ids)
            z = real_model.embed_norm(z)

            for layer_idx, bank_pair in enumerate(real_model.backbone.bank_pairs):
                with torch.no_grad():
                    sem_out, ctx_out = bank_pair(z)

                sem_mag = cabs(sem_out).mean().item()
                ctx_mag = cabs(ctx_out).mean().item()
                sem_magnitudes.append(sem_mag)
                ctx_magnitudes.append(ctx_mag)

                # Cosine similarity between bank outputs
                sem_flat = sem_out.reshape(-1, sem_out.shape[-2] * 2)
                ctx_flat = ctx_out.reshape(-1, ctx_out.shape[-2] * 2)
                cos_sim = F.cosine_similarity(sem_flat, ctx_flat, dim=-1).mean().item()
                similarities.append(cos_sim)

        avg_sem = sum(sem_magnitudes) / len(sem_magnitudes)
        avg_ctx = sum(ctx_magnitudes) / len(ctx_magnitudes)
        avg_sim = sum(similarities) / len(similarities)
        ratio = avg_sem / (avg_ctx + 1e-8)

        all_sem_mags[text_type] = avg_sem
        all_ctx_mags[text_type] = avg_ctx

        results[text_type] = {
            "semantic_mag": avg_sem,
            "context_mag": avg_ctx,
            "sem_ctx_ratio": ratio,
            "bank_similarity": avg_sim,
        }

        if verbose:
            print(f"  {text_type:15s} | sem={avg_sem:.4f} ctx={avg_ctx:.4f} "
                  f"ratio={ratio:.3f} sim={avg_sim:.4f}")

    # Are banks specializing differently across text types?
    sem_variance = _variance([all_sem_mags[t] for t in BANK_TEST_TEXTS])
    ctx_variance = _variance([all_ctx_mags[t] for t in BANK_TEST_TEXTS])
    specialization_score = sem_variance + ctx_variance

    if verbose:
        print(f"\n  Cross-type variance: sem={sem_variance:.6f} ctx={ctx_variance:.6f}")
        print(f"  Specialization score: {specialization_score:.6f} "
              f"(higher = banks respond differently to different text types)")

    return {
        "probe": "bank_specialization",
        "per_type": results,
        "specialization_score": specialization_score,
    }


def _variance(values):
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


# ─────────────────────────────────────────────────────────────
# 4. Working Memory Utilization Probe
# ─────────────────────────────────────────────────────────────

WM_TEST_TEXTS = [
    "Albert Einstein was born in Germany. He studied physics at ETH Zurich. Einstein later moved to America.",
    "The cat named Whiskers climbed the tree. A dog chased it. Whiskers meowed loudly from the branch.",
    "Tokyo is the capital of Japan. It has a population of over 13 million people. The city hosts the Olympics.",
]


def run_wm_probe(model, tokenizer, device, verbose=False) -> Dict:
    """Measure working memory write gate activations."""
    real_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    if real_model.backbone.working_memory is None:
        if verbose:
            print("  Working memory is disabled (0 slots) -- skipping probe")
        return {"probe": "working_memory", "status": "disabled"}

    model.eval()
    wm = real_model.backbone.working_memory
    results = []

    for text in WM_TEST_TEXTS:
        ids = tokenizer.encode(text)
        input_ids = torch.tensor([ids], device=device)

        with torch.no_grad():
            z = real_model.embed(input_ids)
            z = real_model.embed_norm(z)

            bank_z = z
            for bank_pair, coupler, scale in zip(
                real_model.backbone.bank_pairs,
                real_model.backbone.couplers,
                real_model.backbone.bank_scales,
            ):
                residual = bank_z
                sem_out, ctx_out = bank_pair(bank_z)
                coupled = coupler(sem_out, ctx_out)
                bank_z = residual + coupled * scale

            ssm_out, _ = real_model.backbone.ssm(bank_z, None)

            gate_raw = wm.write_gate_proj(ssm_out)
            write_gates = torch.sigmoid(cabs(gate_raw) + wm.gate_bias)

        gates = write_gates.squeeze(-1)[0].cpu().tolist()
        tokens = [tokenizer.decode([t]) for t in ids]

        # Find high-activation tokens
        threshold = 0.5
        hot_tokens = [(tokens[i], gates[i]) for i in range(len(tokens)) if gates[i] > threshold]

        avg_gate = sum(gates) / len(gates)
        max_gate = max(gates)
        sparsity = sum(1 for g in gates if g < 0.1) / len(gates)

        results.append({
            "text": text[:80] + "...",
            "avg_gate": avg_gate,
            "max_gate": max_gate,
            "sparsity": sparsity,
            "hot_tokens": hot_tokens[:10],
        })

        if verbose:
            print(f"  Text: {text[:60]}...")
            print(f"    avg_gate={avg_gate:.4f} max={max_gate:.4f} sparsity={sparsity:.2%}")
            if hot_tokens:
                hot_str = ", ".join(f"'{t}'({g:.3f})" for t, g in hot_tokens[:8])
                print(f"    Hot tokens (>{threshold}): {hot_str}")
            else:
                print(f"    No tokens above {threshold} threshold")

    return {"probe": "working_memory", "status": "active", "details": results}


# ─────────────────────────────────────────────────────────────
# 5. SSM Timescale Probe
# ─────────────────────────────────────────────────────────────

def run_ssm_probe(model, tokenizer, device, verbose=False) -> Dict:
    """Inspect learned SSM decay rates across timescale lanes."""
    real_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    ssm = real_model.backbone.ssm

    all_fast, all_med, all_slow = [], [], []

    with torch.no_grad():
        for layer_idx, layer in enumerate(ssm.layers):
            decays = torch.exp(layer.log_A_real).cpu()
            state_dim = decays.shape[0]
            fast_end = int(state_dim * 0.4)
            med_end = fast_end + int(state_dim * 0.3)

            fast_decay = decays[:fast_end].mean().item()
            med_decay = decays[fast_end:med_end].mean().item()
            slow_decay = decays[med_end:].mean().item()

            all_fast.append(fast_decay)
            all_med.append(med_decay)
            all_slow.append(slow_decay)

            if verbose:
                fast_hl = -1 / math.log(fast_decay + 1e-8) if fast_decay > 0 else float('inf')
                med_hl = -1 / math.log(med_decay + 1e-8) if med_decay > 0 else float('inf')
                slow_hl = -1 / math.log(slow_decay + 1e-8) if slow_decay > 0 else float('inf')
                print(f"  Layer {layer_idx:2d}: fast={fast_decay:.6f}({fast_hl:.0f}t) "
                      f"med={med_decay:.6f}({med_hl:.0f}t) "
                      f"slow={slow_decay:.6f}({slow_hl:.0f}t)")

    avg_fast = sum(all_fast) / len(all_fast)
    avg_med = sum(all_med) / len(all_med)
    avg_slow = sum(all_slow) / len(all_slow)

    avg_fast_hl = -1 / math.log(avg_fast + 1e-8) if avg_fast > 0 else float('inf')
    avg_med_hl = -1 / math.log(avg_med + 1e-8) if avg_med > 0 else float('inf')
    avg_slow_hl = -1 / math.log(avg_slow + 1e-8) if avg_slow > 0 else float('inf')

    result = {
        "probe": "ssm_timescales",
        "fast": {"decay": avg_fast, "half_life_tokens": avg_fast_hl},
        "medium": {"decay": avg_med, "half_life_tokens": avg_med_hl},
        "slow": {"decay": avg_slow, "half_life_tokens": avg_slow_hl},
        "separation": avg_slow - avg_fast,
    }

    if verbose:
        print(f"\n  Average across {len(ssm.layers)} layers:")
        print(f"    Fast lane   (40%): decay={avg_fast:.6f}  half-life={avg_fast_hl:.1f} tokens")
        print(f"    Medium lane (30%): decay={avg_med:.6f}  half-life={avg_med_hl:.1f} tokens")
        print(f"    Slow lane   (30%): decay={avg_slow:.6f}  half-life={avg_slow_hl:.1f} tokens")
        print(f"    Separation (slow - fast): {result['separation']:.6f}")

    return result


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V6 Evaluation Probes')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--probes', type=str, default='all',
                        help='Comma-separated: coref,fact,bank,wm,ssm (default: all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)
    config = V6Config(**checkpoint['config'])
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    probes_to_run = args.probes.split(',') if args.probes != 'all' else ['coref', 'fact', 'bank', 'wm', 'ssm']
    all_results = {}

    print(f"\nRunning probes: {', '.join(probes_to_run)}")
    print("=" * 60)

    if 'coref' in probes_to_run:
        print("\n--- Entity Co-reference Probe ---")
        all_results['coref'] = run_coref_probe(model, tokenizer, device, args.verbose)
        print(f"  Pass rate: {all_results['coref']['pass_rate']:.0%}")

    if 'fact' in probes_to_run:
        print("\n--- Fact Persistence Probe ---")
        all_results['fact'] = run_fact_probe(model, tokenizer, device, args.verbose)
        print(f"  Pass rate: {all_results['fact']['pass_rate']:.0%}")

    if 'bank' in probes_to_run:
        print("\n--- Bank Specialization Probe ---")
        all_results['bank'] = run_bank_probe(model, tokenizer, device, args.verbose)
        print(f"  Specialization score: {all_results['bank']['specialization_score']:.6f}")

    if 'wm' in probes_to_run:
        print("\n--- Working Memory Utilization Probe ---")
        all_results['wm'] = run_wm_probe(model, tokenizer, device, args.verbose)

    if 'ssm' in probes_to_run:
        print("\n--- SSM Timescale Probe ---")
        all_results['ssm'] = run_ssm_probe(model, tokenizer, device, args.verbose)

    print("\n" + "=" * 60)
    print("Summary:")
    if 'coref' in all_results:
        print(f"  Co-reference:     {all_results['coref']['pass_rate']:.0%}")
    if 'fact' in all_results:
        print(f"  Fact persistence: {all_results['fact']['pass_rate']:.0%}")
    if 'bank' in all_results:
        print(f"  Bank specialize:  {all_results['bank']['specialization_score']:.6f}")
    if 'ssm' in all_results:
        ssm = all_results['ssm']
        print(f"  SSM half-lives:   fast={ssm['fast']['half_life_tokens']:.0f} "
              f"med={ssm['medium']['half_life_tokens']:.0f} "
              f"slow={ssm['slow']['half_life_tokens']:.0f} tokens")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
