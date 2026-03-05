"""
V5 Standalone Generation Script.

Loads a trained checkpoint and generates text from a prompt.
Outputs only the generated continuation (prompt is not echoed).

Usage:
    python -m v5.generate --checkpoint checkpoints_v5/best_model.pt --prompt "The quick brown"
    python -m v5.generate --prompt "Once upon a time" --max_new_tokens 100 --temperature 0.8
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from v5.model import create_model
from v5.config import V5Config, get_config


def main():
    parser = argparse.ArgumentParser(
        description='Generate text from a trained V5 model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m v5.generate --checkpoint checkpoints_v5/best_model.pt --prompt "The quick brown"
  python -m v5.generate --prompt "Once upon a time" --max_new_tokens 200 --temperature 0.7
        """,
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints_v5/best_model.pt',
        help='Path to checkpoint (default: checkpoints_v5/best_model.pt)',
    )
    parser.add_argument(
        '--size',
        type=str,
        default='small',
        choices=['tiny', 'small', 'small-matched', 'medium', 'large'],
        help='Model size preset if config missing in checkpoint (default: small)',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='The quick brown',
        help='Prompt text to continue (default: The quick brown)',
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help='Maximum new tokens to generate (default: 100)',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)',
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k filtering, 0 to disable (default: 50)',
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p (nucleus) sampling, 0 to disable (default: 0.9)',
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.2,
        help='Repetition penalty (default: 1.2)',
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint or fall back to size preset
    if 'config' in checkpoint:
        config = V5Config(**checkpoint['config'])
    else:
        config = get_config(args.size)
        config.vocab_size = 50257  # GPT-2 vocab

    # Create model and load weights
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    # Load tokenizer (same as training)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    prompt_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    # Generate
    with torch.no_grad():
        generated = model.generate(
            prompt_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

    # Decode only the generated tokens (not the prompt)
    generated_ids = generated[0].tolist()
    new_token_ids = generated_ids[len(prompt_ids):]
    output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    print(output_text)


if __name__ == '__main__':
    main()
