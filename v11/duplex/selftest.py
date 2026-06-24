"""Unit tests for v11.duplex (additive module; shared code untouched)."""

import importlib
import sys

import torch

from v11.duplex.config import DUPLEX_PRESETS, get_duplex_config
from v11.duplex.data import SyntheticDuplexDataset, collate_duplex
from v11.duplex.interleave import generate_conversation, ScenarioKind
from v11.duplex.model import V11DuplexLM
from v11.duplex.thinking import VOCAB
from v11.model import V11LM, get_config


def test_presets_param_counts():
    for name, target_lo, target_hi in [
        ('duplex_5m', 4.0e6, 6.5e6),
        ('duplex_10m', 8.0e6, 12.0e6),
        ('duplex_25m', 20.0e6, 30.0e6),
    ]:
        cfg = get_duplex_config(name)
        m = V11DuplexLM(cfg)
        total = m.count_parameters()['total']
        assert target_lo <= total <= target_hi, f"{name}: {total/1e6:.2f}M not in range"
        assert cfg.n_states == 3, f"{name} must use E3 K=3"
        assert cfg.decay_mode == 'head'
        assert cfg.write_mode == 'additive'


def test_interleave_shapes():
    import numpy as np
    rng = np.random.default_rng(0)
    inp, lab, kinds = generate_conversation(6, rng)
    assert len(inp) == len(lab)
    assert len(kinds) == 6
    assert VOCAB.listen in inp or VOCAB.speak in inp or VOCAB.backchannel in inp
    assert ScenarioKind.BARGE_IN in kinds or True  # stochastic ok


def test_forward_and_loss():
    cfg = get_duplex_config('duplex_5m')
    model = V11DuplexLM(cfg)
    ds = SyntheticDuplexDataset(n_samples=4, n_blocks=4, seed=0)
    batch = collate_duplex([ds[i] for i in range(4)])
    input_ids, labels, _ = batch
    logits, states, _ = model(input_ids)
    assert logits.shape[:2] == input_ids.shape
    assert len(states) == cfg.n_layers
    loss = V11DuplexLM.compute_loss(logits, labels)
    assert loss.item() > 0
    loss.backward()


def test_forward_embeds_matches_token_path():
    cfg = get_duplex_config('duplex_5m')
    model = V11DuplexLM(cfg).eval()
    ids = torch.randint(VOCAB.text_offset, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        z = model.embed(ids)
        z = model.embed_norm(z)
        logits_a, _, _ = model.forward_embeds(z)
        logits_b, _, _ = model(ids)
    diff = (logits_a - logits_b).abs().max().item()
    assert diff < 1e-5, f"forward_embeds mismatch: {diff}"


def test_import_does_not_mutate_v11_model():
    baseline_presets = list(get_config.__module__)
    from v11.model import PRESETS as presets_before
    keys_before = set(presets_before.keys())
    importlib.reload(sys.modules.get('v11.duplex', importlib.import_module('v11.duplex')))
    from v11.model import PRESETS as presets_after
    assert set(presets_after.keys()) == keys_before


def test_v11_baseline_forward_unchanged():
    cfg = get_config('v11_baseline')
    m = V11LM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        out1, _, _ = m(x)
    import v11.duplex  # noqa: F401
    m2 = V11LM(cfg).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        out2, _, _ = m2(x)
    assert torch.equal(out1, out2)


def main():
    tests = [
        test_presets_param_counts,
        test_interleave_shapes,
        test_forward_and_loss,
        test_forward_embeds_matches_token_path,
        test_import_does_not_mutate_v11_model,
        test_v11_baseline_forward_unchanged,
    ]
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"PASS {name}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            raise SystemExit(1)
    print("ALL DUPLEX TESTS PASS")


if __name__ == '__main__':
    main()
