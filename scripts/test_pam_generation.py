#!/usr/bin/env python3
"""
Verify PAM inference bugs are fixed:
1. Dual form returns proper state (not empty)
2. State flows from prompt processing to generation
3. Generation runs without error
"""
import torch
from v6.config import get_config
from v6.model import create_model


def test_pam_state_flow():
    """Verify dual form returns state and it flows to next forward."""
    config = get_config("medium-pam")
    config.mode = "autoregressive"
    model = create_model(config)
    model.eval()

    # Simulate generation: prompt then single token
    prompt_ids = [1, 2, 3, 4, 5, 6, 7]  # 7 tokens
    prompt = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        # Step 1: Process prompt (state=None, T=7) -> dual form
        out1 = model(prompt, pam_state=None)
        assert out1.pam_state is not None, "Bug 1: pam_state should be returned after prompt"
        assert out1.pam_state.matrix.shape[0] == config.num_layers
        assert out1.pam_state.matrix.shape[2] == 6  # num_heads
        assert out1.pam_state.matrix.shape[3] == 64  # head_dim
        assert out1.pam_state.step == 7

        # Step 2: Process next token with carried state (T=1) -> recurrent form
        next_token = torch.tensor([[out1.logits[:, -1].argmax().item()]], dtype=torch.long)
        out2 = model(next_token, pam_state=out1.pam_state)
        assert out2.pam_state is not None
        assert out2.pam_state.step == 8

    print("PAM state flow: OK (state returned from dual form, flows to generation)")


def test_generation_runs():
    """Verify full generation loop runs without error."""
    config = get_config("medium-pam")
    config.mode = "autoregressive"
    model = create_model(config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(
            prompt, max_new_tokens=10,
            temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2,
        )
    assert generated.shape[1] == 5 + 10, "Should generate 10 new tokens"
    print("Generation loop: OK")


if __name__ == "__main__":
    print("Testing PAM inference fix...")
    test_pam_state_flow()
    test_generation_runs()
    print("All tests passed.")
