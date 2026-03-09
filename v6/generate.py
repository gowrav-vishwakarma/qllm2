"""
V6 Generation Script.

Supports all three modes via checkpoint config:
  - autoregressive: token-by-token text generation
  - diffusion_text: iterative denoising to text
  - diffusion_image: iterative denoising to image grid

Usage:
    python -m v6.generate --checkpoint checkpoints_v6/best_model.pt --prompt "Once upon a time"
    python -m v6.generate --checkpoint checkpoints_v6/best_model.pt --num_samples 4  # diffusion
    python -m v6.generate --checkpoint ... --persistent_memory user_alice.pt
"""

import argparse
import torch
from pathlib import Path

from v6.model import create_model
from v6.config import V6Config
from v6.core.memory import PersistentMemoryStore


def generate_autoregressive(model, tokenizer, args, device):
    """Standard autoregressive text generation."""
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


def generate_diffusion_text(model, tokenizer, args, device, config):
    """Generate text via diffusion sampling."""
    seq_len = args.seq_len or config.max_seq_len
    num_samples = args.num_samples
    num_steps = args.diffusion_sample_steps or config.diffusion_steps

    print(f"\nGenerating {num_samples} text samples via {config.sampling_method}")
    print(f"Sequence length: {seq_len}, Steps: {num_steps}")
    print("-" * 40)

    tokens = model.sample(
        batch_size=num_samples, seq_len=seq_len, device=device,
        num_steps=num_steps,
    )

    for i in range(tokens.shape[0]):
        text = tokenizer.decode(tokens[i].tolist(), skip_special_tokens=True)
        print(f"\n[Sample {i+1}]")
        print(text[:500])


def generate_diffusion_image(model, args, device, config):
    """Generate images via diffusion sampling and save as grid."""
    num_samples = args.num_samples
    num_steps = args.diffusion_sample_steps or config.diffusion_steps

    if config.image_encoder == 'patch':
        seq_len = (config.image_size // config.patch_size) ** 2
    else:
        seq_len = config.image_size * config.image_size

    print(f"\nGenerating {num_samples} images via {config.sampling_method}")
    print(f"Image size: {config.image_size}x{config.image_size}, Steps: {num_steps}")
    print("-" * 40)

    images = model.sample(
        batch_size=num_samples, seq_len=seq_len, device=device,
        num_steps=num_steps,
    )

    out_path = Path(args.output or 'generated_images.png')
    if images.dim() == 4:
        images = (images.clamp(-1, 1) + 1) / 2
        try:
            from torchvision.utils import save_image
            save_image(images, out_path, nrow=int(num_samples ** 0.5) or 1)
            print(f"Saved {num_samples} images to {out_path}")
        except ImportError:
            torch.save(images, out_path.with_suffix('.pt'))
            print(f"Saved raw tensor to {out_path.with_suffix('.pt')} (torchvision not available)")
    else:
        torch.save(images, out_path.with_suffix('.pt'))
        print(f"Saved raw tensor to {out_path.with_suffix('.pt')}")


def main():
    parser = argparse.ArgumentParser(description='Generate with V6')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, default='The quick brown')
    parser.add_argument('--max_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)
    parser.add_argument('--persistent_memory', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--diffusion_sample_steps', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False)
    config = V6Config(**ckpt['config'])
    model = create_model(config)
    model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_to_load.load_state_dict(ckpt['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print(f"Mode: {config.mode}")

    if config.mode == 'autoregressive':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        generate_autoregressive(model, tokenizer, args, device)

    elif config.mode == 'diffusion_text':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        generate_diffusion_text(model, tokenizer, args, device, config)

    elif config.mode == 'diffusion_image':
        generate_diffusion_image(model, args, device, config)


if __name__ == '__main__':
    main()
