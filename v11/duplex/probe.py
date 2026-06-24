"""Duplex mechanism probe: thinking-token retrieval under synthetic barge-in."""

import argparse

import numpy as np
import torch

from v11.duplex.config import get_duplex_config
from v11.duplex.interleave import ScenarioKind, build_block, block_to_token_lists
from v11.duplex.model import V11DuplexLM
from v11.duplex.thinking import VOCAB


def run_probe(model: V11DuplexLM, device: torch.device, n_trials: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    model.eval()
    correct = 0
    total = 0
    barge_listen = 0
    barge_total = 0

    with torch.no_grad():
        for _ in range(n_trials):
            block = build_block(ScenarioKind.BARGE_IN, rng, truncate_reply=int(rng.integers(1, 4)))
            inp, lab = block_to_token_lists(block)
            ids = torch.tensor([inp], dtype=torch.long, device=device)
            labels = torch.tensor([lab], dtype=torch.long, device=device)
            logits, _, _ = model(ids)
            pred = logits[:, :-1, :].argmax(dim=-1)
            shift_lab = labels[:, 1:]
            mask = shift_lab == VOCAB.listen
            if mask.any():
                barge_total += 1
                if (pred[mask] == VOCAB.listen).all():
                    barge_listen += 1
            acc, n = model.thinking_accuracy(logits, labels, VOCAB.thinking_ids)
            if n > 0:
                correct += acc * n
                total += n

    return {
        'think_acc': correct / max(1, total),
        'barge_listen_rate': barge_listen / max(1, barge_total),
        'n_trials': n_trials,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', default='duplex_5m')
    p.add_argument('--checkpoint', default='')
    p.add_argument('--n_trials', type=int, default=256)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = get_duplex_config(args.preset)
    model = V11DuplexLM(cfg).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])

    metrics = run_probe(model, device, n_trials=args.n_trials)
    random_baseline = 1.0 / len(VOCAB.thinking_ids)
    print(f"Preset: {args.preset}")
    print(f"Thinking accuracy: {metrics['think_acc']:.3f} (random={random_baseline:.3f})")
    print(f"Barge-in -> listen rate: {metrics['barge_listen_rate']:.3f}")
    return metrics


if __name__ == '__main__':
    main()
