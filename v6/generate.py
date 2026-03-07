"""
V6 Generation Script.

Usage:
    python -m v6.generate --checkpoint checkpoints_v6/best_model.pt --prompt "Once upon a time"
    python -m v6.generate --checkpoint checkpoints_v6/best_model.pt --prompt "The" --max_tokens 200
    python -m v6.generate --checkpoint ... --persistent_memory user_alice.pt
"""

import argparse
import torch
from pathlib import Path

from v6.model import PhaseFieldLM, create_model
from v6.config import V6Config
from v6.core.memory import PersistentMemoryStore


def main():
    parser = argparse.ArgumentParser(description='Generate text with V6')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, default='The quick brown')
    parser.add_argument('--max_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)
    parser.add_argument('--persistent_memory', type=str, default=None,
                        help='Path to persistent memory .pt file')
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False)
    config = V6Config(**ckpt['config'])
    model = create_model(config)
    model.load_state_dict(ckpt['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load persistent memory if provided
    pm_keys = pm_values = pm_mask = None
    if args.persistent_memory:
        pm_state = torch.load(args.persistent_memory, weights_only=False)
        pm_store = PersistentMemoryStore.load(pm_state, device)
        pm_keys, pm_values, pm_mask = pm_store.get_keys_values()
        print(f"Loaded persistent memory: {pm_store.num_slots} slots, {int(pm_mask.sum())} used")

    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)

    with torch.no_grad():
        generated = model.generate(
            prompt_tensor,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            persistent_keys=pm_keys,
            persistent_values=pm_values,
            persistent_mask=pm_mask,
        )

    text = tokenizer.decode(generated[0].tolist())
    print(text)


if __name__ == '__main__':
    main()
