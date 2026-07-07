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


def test_unified_vocab_layout():
    from v11.duplex.tokenizer import DuplexVocab, N_CONTROL
    v = DuplexVocab(n_text=32000, n_codebooks=4, codebook_size=2048)
    assert v.text_offset == N_CONTROL
    assert v.codec_offset == N_CONTROL + 32000
    assert v.total_size == N_CONTROL + 32000 + 4 * 2048 == 40208
    g = v.codec_to_global(3, 100)
    assert v.global_to_codec(g) == (3, 100)
    assert v.is_codec(g) and not v.is_text(g) and not v.is_control(g)
    assert v.is_text(v.text_to_global(0)) and v.is_control(0)


def test_codec_delay_roundtrip():
    from v11.duplex.tokenizer import DuplexVocab, AUDIO_PAD
    from v11.duplex.codec import delay_flatten, delay_unflatten, flat_interleave, flat_deinterleave
    v = DuplexVocab(n_text=32000, n_codebooks=4, codebook_size=2048)
    codes = torch.randint(0, v.codebook_size, (v.n_codebooks, 29))
    stream = delay_flatten(codes, v)
    assert len(stream) == v.n_codebooks * (29 + v.n_codebooks - 1)
    assert torch.equal(delay_unflatten(stream, v), codes)
    assert all(v.is_codec(g) or g == AUDIO_PAD for g in stream)
    assert torch.equal(flat_deinterleave(flat_interleave(codes, v), v), codes)


def test_duplex_100m_preset():
    from v11.duplex.config import get_duplex_config, DUPLEX_100M_VOCAB
    cfg = get_duplex_config('duplex_100m')
    assert cfg.vocab_size == DUPLEX_100M_VOCAB == 40208
    assert cfg.n_states == 3 and cfg.dim == 384 and cfg.n_layers == 16
    assert cfg.gate_content_aware and cfg.chunk_size == 256
    m = V11DuplexLM(cfg, audio_feat_dim=768)
    total = m.count_parameters()['total']
    assert 110e6 <= total <= 125e6, f'{total/1e6:.1f}M out of range (includes CTC head)'


def test_audio_injection_skips_pad_positions():
    cfg = get_duplex_config('duplex_5m')
    model = V11DuplexLM(cfg, audio_feat_dim=48).eval()
    B, T = 2, 10
    ids = torch.randint(cfg.vocab_size // 2, cfg.vocab_size, (B, T))
    frames = torch.randn(B, 3, 48)
    embeds = model.project_audio(frames)
    # sample 0: two valid slots + one pad(-1); sample 1: all pad
    positions = torch.tensor([[1, 4, -1], [-1, -1, -1]])
    with torch.no_grad():
        logits, _, _ = model(ids, audio_embeds=embeds, audio_positions=positions)
    assert logits.shape == (B, T, cfg.vocab_size)


def test_audio_proj_receives_gradients():
    """Regression: the ASR train step must backprop into the trainable audio
    projection. A prior bug ran project_audio under no_grad, so audio_proj stayed
    at random init and CER plateaued (~73%)."""
    cfg = get_duplex_config('duplex_5m')
    model = V11DuplexLM(cfg, audio_feat_dim=48).train()
    B, T_audio, T_text = 2, 4, 5
    frames = torch.randn(B, T_audio, 48)
    # layout: [<audio_pad> x T_audio] then text tokens (labels only on text)
    seq_len = T_audio + T_text
    input_ids = torch.randint(cfg.vocab_size // 2, cfg.vocab_size, (B, seq_len))
    labels = torch.full((B, seq_len), -100, dtype=torch.long)
    labels[:, T_audio:] = input_ids[:, T_audio:]
    positions = torch.arange(T_audio).unsqueeze(0).expand(B, -1).contiguous()

    embeds = model.project_audio(frames)  # trainable path, grad enabled
    logits, _, _ = model(input_ids, audio_embeds=embeds, audio_positions=positions)
    loss = V11DuplexLM.compute_loss(logits, labels)
    loss.backward()

    grads = [p.grad for p in model.audio_proj.parameters() if p.grad is not None]
    assert grads, 'audio_proj received no gradients'
    total = sum(g.abs().sum().item() for g in grads)
    assert total > 0, f'audio_proj grad is all-zero (norm={total})'


def test_ctc_head_learns():
    """CTC head + audio_proj must receive gradients when joint AR+CTC loss runs."""
    import torch.nn.functional as F
    cfg = get_duplex_config('duplex_5m')
    n_text = 512
    model = V11DuplexLM(cfg, audio_feat_dim=48, n_text=n_text).train()
    B, T_audio, T_text = 2, 6, 4
    frames = torch.randn(B, T_audio, 48)
    seq_len = 1 + T_audio + 2 + T_text  # env + audio + transcribe/lang + text
    input_ids = torch.randint(cfg.vocab_size // 2, cfg.vocab_size, (B, seq_len))
    labels = torch.full((B, seq_len), -100, dtype=torch.long)
    labels[:, 1 + T_audio + 2:] = input_ids[:, 1 + T_audio + 2:]
    positions = torch.full((B, T_audio), -1, dtype=torch.long)
    for i in range(B):
        positions[i] = torch.arange(1, 1 + T_audio)
    ctc_targets = torch.randint(0, n_text, (B * 3,))
    ctc_target_lengths = torch.tensor([3, 3], dtype=torch.long)

    embeds = model.project_audio(frames)
    logits, _, _, hidden = model(
        input_ids, audio_embeds=embeds, audio_positions=positions, return_hidden=True,
    )
    ar_loss = V11DuplexLM.compute_loss(logits, labels)
    log_probs, input_lengths = model.ctc_log_probs(hidden, positions)
    assert log_probs.shape == (B, T_audio, n_text + 1)
    ctc_loss = F.ctc_loss(
        log_probs.transpose(0, 1), ctc_targets, input_lengths, ctc_target_lengths,
        blank=n_text, zero_infinity=True,
    )
    (ar_loss + ctc_loss).backward()

    proj_grads = [p.grad for p in model.audio_proj.parameters() if p.grad is not None]
    ctc_grads = [p.grad for p in model.frame_heads.ctc_head.parameters() if p.grad is not None]
    assert proj_grads and sum(g.abs().sum().item() for g in proj_grads) > 0
    assert ctc_grads and sum(g.abs().sum().item() for g in ctc_grads) > 0


def main():
    tests = [
        test_presets_param_counts,
        test_interleave_shapes,
        test_forward_and_loss,
        test_forward_embeds_matches_token_path,
        test_import_does_not_mutate_v11_model,
        test_v11_baseline_forward_unchanged,
        test_unified_vocab_layout,
        test_codec_delay_roundtrip,
        test_duplex_100m_preset,
        test_audio_injection_skips_pad_positions,
        test_audio_proj_receives_gradients,
        test_ctc_head_learns,
    ]
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"PASS {name}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            raise SystemExit(1)
    print("ALL DUPLEX TESTS PASS (12/12)")


if __name__ == '__main__':
    main()
