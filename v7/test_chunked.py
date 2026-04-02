"""Verify chunked dual form matches full dual form numerically."""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from v7.model import V7Config, V7LM


def test_chunked_vs_full():
    """Compare chunked (chunk_size=256) vs full (chunk_size=0) dual form."""
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_cfg = dict(
        vocab_size=1000, dim=64, n_heads=2, head_dim=32,
        n_layers=2, expand=2, dropout=0.0, max_seq_len=512,
        hierarchical_dt=False, gradient_checkpointing=False,
    )

    cfg_full = V7Config(**base_cfg, chunk_size=0)
    cfg_chunked = V7Config(**base_cfg, chunk_size=64)

    model_full = V7LM(cfg_full).to(device).eval()
    model_chunked = V7LM(cfg_chunked).to(device).eval()

    # Copy weights from full to chunked
    model_chunked.load_state_dict(model_full.state_dict(), strict=False)

    # Test sequences of various lengths
    for T in [32, 64, 128, 200, 256, 512]:
        torch.manual_seed(123)
        ids = torch.randint(0, 1000, (2, T), device=device)

        with torch.no_grad():
            logits_full, states_full, _ = model_full(ids)
            logits_chunked, states_chunked, _ = model_chunked(ids)

        max_diff = (logits_full - logits_chunked).abs().max().item()
        mean_diff = (logits_full - logits_chunked).abs().mean().item()

        # State comparison
        state_diffs = []
        for sf, sc in zip(states_full, states_chunked):
            state_diffs.append((sf - sc).abs().max().item())
        max_state_diff = max(state_diffs)

        chunked_used = "chunked" if T > cfg_chunked.chunk_size else "full"
        status = "PASS" if max_diff < 1e-3 else "FAIL"
        print(
            f"T={T:4d} ({chunked_used:>7s}): "
            f"logit max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  "
            f"state max_diff={max_state_diff:.2e}  [{status}]"
        )
        if max_diff >= 1e-3:
            print(f"  *** FAIL: max_diff={max_diff:.2e} exceeds 1e-3 threshold")
            return False

    print("\nAll tests passed!")
    return True


def test_training_step():
    """Verify backward pass works and loss is reasonable."""
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = V7Config(
        vocab_size=1000, dim=64, n_heads=2, head_dim=32,
        n_layers=2, expand=2, dropout=0.1, max_seq_len=256,
        hierarchical_dt=False, gradient_checkpointing=False,
        chunk_size=64,
    )
    model = V7LM(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ids = torch.randint(0, 1000, (2, 128), device=device)
    target = ids[:, 1:]
    input_ids = ids[:, :-1]

    logits, _, _ = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1)
    )
    print(f"Training loss: {loss.item():.4f} (should be ~6.9 for vocab=1000)")

    loss.backward()
    optimizer.step()

    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()

    nonzero_grads = sum(1 for v in grad_norms.values() if v > 0)
    total_params = len(grad_norms)
    print(f"Gradients: {nonzero_grads}/{total_params} parameters have nonzero grad")

    if nonzero_grads == 0:
        print("*** FAIL: No gradients flowing!")
        return False

    print("Training step test passed!")
    return True


if __name__ == '__main__':
    ok1 = test_chunked_vs_full()
    print()
    ok2 = test_training_step()
    sys.exit(0 if ok1 and ok2 else 1)
