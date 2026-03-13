"""
V6 Bank Specialization Probe.

Loads a trained checkpoint and measures whether the semantic and context banks
have learned functionally different roles (entity-stable vs relation-varying).

Metrics:
1. Magnitude variance asymmetry: semantic bank should have lower variance
   across the sequence (entity-like stability) vs context bank.
2. Phase variance asymmetry: same logic for phase angle.
3. Cosine similarity between banks per layer: how different are their outputs?
4. Token-type analysis: compare bank activations on entity tokens (nouns/names)
   vs function/relation tokens (verbs, prepositions, articles).
5. Cross-layer consistency: does the pattern hold across all 12 layers?

Usage:
    python scripts/v6_bank_probe.py --checkpoint checkpoints_v6_reframe/run4_bank_role_0.0/best_model.pt
    python scripts/v6_bank_probe.py --checkpoint checkpoints_v6_reframe/run5_delayed_recall_episodic16/best_model.pt
"""

import argparse
import sys
import os
import math

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v6.config import V6Config
from v6.model import create_model
from v6.core.complex import cabs, creal_dot


PROBE_SENTENCES = [
    "Paris is the capital of France and has many famous landmarks .",
    "Delhi is the capital of India and is a very large city .",
    "London is the capital of England and sits on the River Thames .",
    "The cat sat on the mat and watched the birds outside .",
    "Albert Einstein developed the theory of relativity in 1905 .",
    "Marie Curie discovered radium and won two Nobel Prizes .",
    "The quick brown fox jumps over the lazy dog every morning .",
    "Tokyo is the largest city in Japan with millions of people .",
    "William Shakespeare wrote Hamlet and many other famous plays .",
    "The Amazon River flows through Brazil and into the Atlantic Ocean .",
    "Barack Obama served as the 44th President of the United States .",
    "The Great Wall of China was built over many centuries .",
    "Isaac Newton formulated the laws of motion and gravity .",
    "The Nile River is the longest river in Africa .",
    "Leonardo da Vinci painted the Mona Lisa in the 16th century .",
    "Mount Everest is the tallest mountain in the world .",
    "Charles Darwin proposed the theory of evolution by natural selection .",
    "The Pacific Ocean is the largest ocean on Earth .",
    "Wolfgang Amadeus Mozart composed his first symphony at age eight .",
    "The Sahara Desert covers most of northern Africa .",
]

ENTITY_TOKENS = {
    "Paris", "France", "Delhi", "India", "London", "England", "Thames",
    "Einstein", "Albert", "Curie", "Marie", "Tokyo", "Japan",
    "Shakespeare", "William", "Hamlet", "Amazon", "Brazil", "Atlantic",
    "Obama", "Barack", "China", "Newton", "Isaac", "Nile", "Africa",
    "Leonardo", "Lisa", "Mona", "Everest", "Darwin", "Charles",
    "Pacific", "Earth", "Mozart", "Wolfgang", "Amadeus", "Sahara",
    "Nobel", "River", "Ocean", "Mountain", "Desert", "Wall",
}

RELATION_TOKENS = {
    "is", "the", "of", "in", "and", "on", "was", "as", "by", "at",
    "has", "with", "to", "a", "an", "many", "over", "through", "into",
    "sat", "watched", "developed", "discovered", "wrote", "flows",
    "served", "built", "formulated", "painted", "proposed", "composed",
    "covers", "jumps", "sits", "won",
}


def load_model(checkpoint_path: str, device: str = "cpu"):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = V6Config(**ckpt['config'])
    model = create_model(config)
    model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_to_load.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    model.to(device)
    print(f"  Model: {config.dim}d, {config.num_layers} layers, {config.num_banks} banks")
    return model, config


def tokenize_sentences(sentences, device="cpu"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    results = []
    for sent in sentences:
        enc = tok(sent, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        tokens = [tok.decode([tid]) for tid in input_ids[0]]
        results.append((input_ids, tokens))
    return results, tok


def classify_tokens(tokens):
    """Classify each token position as entity, relation, or other."""
    labels = []
    for t in tokens:
        t_clean = t.strip().rstrip(".,!?;:")
        if t_clean in ENTITY_TOKENS or (len(t_clean) > 1 and t_clean[0].isupper()):
            labels.append("entity")
        elif t_clean.lower() in RELATION_TOKENS:
            labels.append("relation")
        else:
            labels.append("other")
    return labels


@torch.no_grad()
def extract_bank_outputs(model, input_ids):
    """Run forward pass with hooks to capture per-layer bank outputs."""
    layer_outputs = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            sem_out, ctx_out = out
            layer_outputs.append({
                "layer": layer_idx,
                "sem": sem_out.detach().cpu(),
                "ctx": ctx_out.detach().cpu(),
            })
        return hook_fn

    hooks = []
    for i, bp in enumerate(model.backbone.bank_pairs):
        h = bp.register_forward_hook(make_hook(i))
        hooks.append(h)

    _ = model(input_ids)

    for h in hooks:
        h.remove()

    return layer_outputs


def compute_metrics(layer_outputs, token_labels_list, all_tokens_list):
    """Compute specialization metrics across all sentences and layers."""
    num_layers = max(lo["layer"] for lo in layer_outputs) + 1
    num_sentences = len(token_labels_list)
    outputs_per_layer = [[] for _ in range(num_layers)]
    for lo in layer_outputs:
        outputs_per_layer[lo["layer"]].append(lo)

    results = {
        "per_layer": [],
        "entity_vs_relation": {"sem_entity_mag": [], "sem_relation_mag": [],
                               "ctx_entity_mag": [], "ctx_relation_mag": []},
    }

    for layer_idx in range(num_layers):
        layer_data = outputs_per_layer[layer_idx]

        all_sem_mag_var = []
        all_ctx_mag_var = []
        all_sem_phase_var = []
        all_ctx_phase_var = []
        all_cosine_sim = []

        for sent_idx, lo in enumerate(layer_data):
            sem = lo["sem"]  # [1, L, dim, 2]
            ctx = lo["ctx"]

            sem_mag = cabs(sem)  # [1, L, dim]
            ctx_mag = cabs(ctx)

            sem_mag_var_per_dim = sem_mag.var(dim=1).mean().item()
            ctx_mag_var_per_dim = ctx_mag.var(dim=1).mean().item()
            all_sem_mag_var.append(sem_mag_var_per_dim)
            all_ctx_mag_var.append(ctx_mag_var_per_dim)

            sem_phase = torch.atan2(sem[..., 1], sem[..., 0] + 1e-8)
            ctx_phase = torch.atan2(ctx[..., 1], ctx[..., 0] + 1e-8)
            all_sem_phase_var.append(sem_phase.var(dim=1).mean().item())
            all_ctx_phase_var.append(ctx_phase.var(dim=1).mean().item())

            sem_flat = sem.reshape(-1, sem.shape[-2], 2)
            ctx_flat = ctx.reshape(-1, ctx.shape[-2], 2)
            dot = creal_dot(sem_flat, ctx_flat)
            sem_norm = torch.sqrt(cabs(sem_flat).square().sum(dim=-1) + 1e-8)
            ctx_norm = torch.sqrt(cabs(ctx_flat).square().sum(dim=-1) + 1e-8)
            cos_sim = (dot / (sem_norm * ctx_norm + 1e-8)).abs().mean().item()
            all_cosine_sim.append(cos_sim)

            labels = token_labels_list[sent_idx]
            for pos, label in enumerate(labels):
                if pos >= sem_mag.shape[1]:
                    break
                s_m = sem_mag[0, pos].mean().item()
                c_m = ctx_mag[0, pos].mean().item()
                if label == "entity":
                    results["entity_vs_relation"]["sem_entity_mag"].append(s_m)
                    results["entity_vs_relation"]["ctx_entity_mag"].append(c_m)
                elif label == "relation":
                    results["entity_vs_relation"]["sem_relation_mag"].append(s_m)
                    results["entity_vs_relation"]["ctx_relation_mag"].append(c_m)

        layer_result = {
            "layer": layer_idx,
            "sem_mag_var": np.mean(all_sem_mag_var),
            "ctx_mag_var": np.mean(all_ctx_mag_var),
            "mag_var_ratio": np.mean(all_ctx_mag_var) / (np.mean(all_sem_mag_var) + 1e-10),
            "sem_phase_var": np.mean(all_sem_phase_var),
            "ctx_phase_var": np.mean(all_ctx_phase_var),
            "phase_var_ratio": np.mean(all_ctx_phase_var) / (np.mean(all_sem_phase_var) + 1e-10),
            "cosine_sim": np.mean(all_cosine_sim),
        }
        results["per_layer"].append(layer_result)

    return results


def print_results(results, checkpoint_name):
    print(f"\n{'='*80}")
    print(f"  Bank Specialization Probe: {checkpoint_name}")
    print(f"{'='*80}\n")

    print("Per-Layer Analysis:")
    print(f"{'Layer':>5} | {'Sem MagVar':>10} | {'Ctx MagVar':>10} | {'Ratio C/S':>9} | "
          f"{'Sem PhVar':>10} | {'Ctx PhVar':>10} | {'Ratio C/S':>9} | {'CosSim':>7}")
    print("-" * 95)

    mag_ratios = []
    phase_ratios = []
    cos_sims = []
    for lr in results["per_layer"]:
        print(f"{lr['layer']:>5} | {lr['sem_mag_var']:>10.6f} | {lr['ctx_mag_var']:>10.6f} | "
              f"{lr['mag_var_ratio']:>9.4f} | {lr['sem_phase_var']:>10.6f} | "
              f"{lr['ctx_phase_var']:>10.6f} | {lr['phase_var_ratio']:>9.4f} | "
              f"{lr['cosine_sim']:>7.4f}")
        mag_ratios.append(lr['mag_var_ratio'])
        phase_ratios.append(lr['phase_var_ratio'])
        cos_sims.append(lr['cosine_sim'])

    print(f"\nAggregated Metrics:")
    print(f"  Mean Mag Variance Ratio (ctx/sem):   {np.mean(mag_ratios):.4f}")
    print(f"  Mean Phase Variance Ratio (ctx/sem): {np.mean(phase_ratios):.4f}")
    print(f"  Mean Cosine Similarity:              {np.mean(cos_sims):.4f}")

    if mag_ratios:
        specializing_mag = sum(1 for r in mag_ratios if r > 1.1)
        specializing_phase = sum(1 for r in phase_ratios if r > 1.05)
        total = len(mag_ratios)
        print(f"\n  Layers with ctx_mag_var > 1.1x sem_mag_var: {specializing_mag}/{total}")
        print(f"  Layers with ctx_phase_var > 1.05x sem_phase_var: {specializing_phase}/{total}")

    evr = results["entity_vs_relation"]
    if evr["sem_entity_mag"] and evr["sem_relation_mag"]:
        sem_ent = np.mean(evr["sem_entity_mag"])
        sem_rel = np.mean(evr["sem_relation_mag"])
        ctx_ent = np.mean(evr["ctx_entity_mag"])
        ctx_rel = np.mean(evr["ctx_relation_mag"])

        print(f"\nEntity vs Relation Token Analysis (all layers):")
        print(f"  {'':>20} | {'Entity Tokens':>14} | {'Relation Tokens':>15} | {'Ratio E/R':>9}")
        print(f"  {'Semantic bank mag':>20} | {sem_ent:>14.6f} | {sem_rel:>15.6f} | {sem_ent/(sem_rel+1e-10):>9.4f}")
        print(f"  {'Context bank mag':>20} | {ctx_ent:>14.6f} | {ctx_rel:>15.6f} | {ctx_ent/(ctx_rel+1e-10):>9.4f}")

        sem_entity_bias = sem_ent / (sem_rel + 1e-10)
        ctx_entity_bias = ctx_ent / (ctx_rel + 1e-10)
        print(f"\n  Semantic bank entity bias (E/R ratio):  {sem_entity_bias:.4f}")
        print(f"  Context bank entity bias (E/R ratio):   {ctx_entity_bias:.4f}")

        if abs(sem_entity_bias - ctx_entity_bias) > 0.05:
            print(f"  --> Banks show DIFFERENT entity/relation sensitivity (delta={abs(sem_entity_bias-ctx_entity_bias):.4f})")
        else:
            print(f"  --> Banks show SIMILAR entity/relation sensitivity (delta={abs(sem_entity_bias-ctx_entity_bias):.4f})")

    print(f"\n{'='*80}")
    print("INTERPRETATION:")

    mean_cos = np.mean(cos_sims)
    mean_mag_ratio = np.mean(mag_ratios)
    mean_phase_ratio = np.mean(phase_ratios)

    if mean_cos > 0.8:
        print(f"  [COLLAPSED] Banks are near-identical (cosine sim {mean_cos:.3f} >> 0.5)")
        print(f"  --> Composite keys will NOT work: both banks produce same representation")
        print(f"  --> Need STRUCTURAL asymmetry (different architectures per bank)")
    elif mean_cos > 0.5:
        print(f"  [WEAK SPECIALIZATION] Banks are partially differentiated (cosine sim {mean_cos:.3f})")
        if mean_mag_ratio > 1.1 or mean_phase_ratio > 1.05:
            print(f"  --> Some role differentiation detected (mag ratio {mean_mag_ratio:.3f}, phase ratio {mean_phase_ratio:.3f})")
            print(f"  --> Composite keys may work but need stronger bank pressure")
        else:
            print(f"  --> No clear role differentiation despite moderate diversity")
            print(f"  --> Consider structural asymmetry or stronger role loss")
    else:
        print(f"  [SPECIALIZED] Banks are well-differentiated (cosine sim {mean_cos:.3f})")
        if mean_mag_ratio > 1.1:
            print(f"  --> Context bank varies more than semantic (ratio {mean_mag_ratio:.3f}) -- role specialization confirmed")
            print(f"  --> Composite keys using bank outputs are VIABLE")
        else:
            print(f"  --> Banks are different but not in the expected entity/relation pattern")
            print(f"  --> Need to investigate what roles they actually learned")

    print(f"{'='*80}\n")

    return {
        "mean_cosine_sim": mean_cos,
        "mean_mag_ratio": mean_mag_ratio,
        "mean_phase_ratio": mean_phase_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="V6 Bank Specialization Probe")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    model, config = load_model(args.checkpoint, args.device)
    tokenized, tokenizer = tokenize_sentences(PROBE_SENTENCES, args.device)

    all_layer_outputs = []
    all_labels = []
    all_tokens = []

    for input_ids, tokens in tokenized:
        layer_outputs = extract_bank_outputs(model, input_ids)
        all_layer_outputs.extend(layer_outputs)

        labels = classify_tokens(tokens)
        all_labels.append(labels)
        all_tokens.append(tokens)

    results = compute_metrics(all_layer_outputs, all_labels, all_tokens)
    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint))
    summary = print_results(results, ckpt_name)

    return summary


if __name__ == "__main__":
    main()
