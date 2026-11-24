#!/usr/bin/env python3
"""Parity test: the self-contained modeling_pam2.py vs the qllm2 repo model.

Needs the qllm2 repo (and sempyt) only for the reference side; the release
side imports nothing but modeling_pam2. Runs on CPU in fp32.

    cd <qllm2 repo>
    .venv/bin/python hf_release_sci1_chat/test_parity.py \
        --checkpoint hf_release_sci1_chat/sci1_chat_386m.pt

Checks (targets: max |logit diff| <= 1e-3, argmax agreement 1.0):
  tokenizer: release tokenizer ids == repo get_chat_tokenizer ids
  A. prefill, 3 inputs of 512 tokens (natural text; text with <|endoftext|> separators,
     which exercises reset_at_eos; random ids incl. 50256): repo LM.forward vs release forward
  B. decode: repo token-by-token forward with carried states vs release step() loop
  C. information only: chunked prefill vs per-token recurrence (the inherent fp32 / epsilon
     gap between the two algebras, present in the repo too); --fp64 proves they are one math
  D. state continuation: forward(prefix) then forward(rest, state) vs one forward
  E. greedy generation, 40 tokens, 3 prompts: identical token ids (repo LM.generate vs release)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Reference side: eager torch on CPU (no Triton, no torch.compile of the delta prep).
os.environ.setdefault('V13S_KERNEL', '0')
os.environ.setdefault('V13S_DELTA_COMPILE', '0')

import torch

sys.path.insert(0, str(HERE))
from modeling_pam2 import load_model, load_tokenizer  # noqa: E402

TEXT = (
    "The river that runs through the old town has changed its course three times in recorded history. "
    "The first change came after the great flood of the spring of 1740, when the water broke through "
    "the meadows on the eastern bank and cut a new channel past the mill. The miller, a man named "
    "Thomas Hale, lost his wheel and half of his barn, but he rebuilt both within a year, and the new "
    "mill stood for more than a century. The second change was slower. Over forty years the river "
    "carried sand and gravel from the hills and laid it down at the bend below the church, until the "
    "bend became a bar and the bar became an island. Children rowed out to it in summer, and a "
    "fisherman built a hut there that the town council later declared a public nuisance. The third "
    "change was made by people rather than by water. In 1911 the railway company straightened the "
    "river to protect its embankment, digging a deep cut through the island and lining the banks with "
    "stone. The old channel silted up and became a long, narrow pond, and the pond became a park. "
    "Today, if you stand on the bridge and look north, you can still see the line of willows that marks "
    "where the water used to flow. Scientists who study rivers say that every river is a kind of memory: "
    "its banks record the floods it has carried, the loads of sediment it has dropped, and the choices "
    "of the people who lived beside it. A river does not forget a flood; it simply writes the flood into "
    "the shape of the land. In the same way, the town remembers its river. The street that runs past "
    "the park is still called Mill Lane, though the mill burned down in 1893, and the school on the hill "
    "still holds a regatta every June, though the boats now race on the straight cut and not around the "
    "island. When the council proposed in 1987 to fill the pond and build a car park, more than two "
    "thousand residents signed a petition against it, and the plan was dropped. "
)
SECOND_DOC = (
    "Recipe for a simple loaf. Mix five hundred grams of flour with ten grams of salt and seven grams "
    "of dried yeast. Add three hundred and fifty millilitres of warm water and knead for ten minutes "
    "until the dough is smooth. Leave it to rise for an hour, shape it, let it rise again, and bake it "
    "at two hundred and twenty degrees for thirty-five minutes. "
)
PROMPTS = [
    "In 1923, the University of",
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
]


def compare(name, ref, out, rows):
    ref, out = ref.float(), out.float()
    diff = (ref - out).abs().max().item()
    agree = (ref.argmax(-1) == out.argmax(-1)).float().mean().item()
    rows.append((name, diff, agree, ref.abs().max().item()))
    ok = diff <= 1e-3 and agree == 1.0
    print(f'  {name:<58} max|dlogit| {diff:.3e}  argmax agree {agree:.4f}  '
          f'(max|logit| {ref.abs().max().item():.1f})  {"PASS" if ok else "FAIL"}', flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=str(HERE / 'sci1_chat_386m.pt'))
    ap.add_argument('--repo', default=str(HERE.parent), help='qllm2 repo root (reference model)')
    ap.add_argument('--threads', type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument('--gen_tokens', type=int, default=40)
    ap.add_argument('--fp64', action='store_true', help='also run the fp64 chunked-vs-recurrent identity check')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    sys.path.insert(0, args.repo)
    sys.path.insert(0, str(Path(args.repo) / 'scripts'))
    from run_memory_behavioral import _load_v13_sempty  # repo loader (v13_sempty LM + chat tokenizer)

    print(f'checkpoint: {args.checkpoint}')
    print(f'torch {torch.__version__}, CPU fp32, threads {args.threads}, '
          f'V13S_KERNEL={os.environ["V13S_KERNEL"]} V13S_DELTA_COMPILE={os.environ["V13S_DELTA_COMPILE"]}')
    t0 = time.time()
    ref, ref_tok, ref_cfg = _load_v13_sempty(Path(args.checkpoint), 'baseline_real_pm', torch.device('cpu'))
    ref.eval().float()
    print(f'repo model loaded ({type(ref).__module__}.{type(ref).__name__}, '
          f'{sum(p.numel() for p in ref.parameters()):,} params) in {time.time() - t0:.1f}s')
    t0 = time.time()
    rel = load_model(args.checkpoint, device='cpu', dtype=torch.float32)
    tok = load_tokenizer()
    print(f'release model loaded ({rel.count_parameters():,} params) in {time.time() - t0:.1f}s\n')
    all_ok = True

    print('[tokenizer]')
    probe = TEXT[:400] + '<|endoftext|>' + PROMPTS[1] + '<think>x</think>'
    same = tok.encode(probe) == ref_tok.encode(probe) and len(tok) == len(ref_tok) == rel.config.vocab_size
    specials = {t: tok.convert_tokens_to_ids(t) for t in
                ('<|endoftext|>', '<|im_start|>', '<|im_end|>', '<think>', '</think>')}
    print(f'  len {len(tok)} (repo {len(ref_tok)}), specials {specials}, ids identical: {same}')
    all_ok &= same

    # inputs
    ids_text = tok.encode(TEXT * 3)[:512]
    a = tok.encode(TEXT)[:300]
    b = tok.encode(SECOND_DOC)
    ids_eos = (a + [50256] + b + [50256] + tok.encode(TEXT))[:512]
    g = torch.Generator().manual_seed(1234)
    ids_rand = torch.randint(0, rel.config.vocab_size, (512,), generator=g)
    ids_rand[[100, 101, 300]] = 50256
    cases = [('512-token text', torch.tensor([ids_text])),
             (f'512 tokens with <|endoftext|> at {[i for i, t in enumerate(ids_eos) if t == 50256]}',
              torch.tensor([ids_eos])),
             ('512 random ids (50256 at 100, 101, 300)', ids_rand[None])]
    rows = []
    print('\n[A. prefill: repo LM.forward(x) vs release forward(x) (chunked closed form)]')
    ref_logits = {}
    for name, x in cases:
        with torch.no_grad():
            t0 = time.time(); lr, _, _ = ref(x); t_ref = time.time() - t0
            t0 = time.time(); lc, _ = rel(x); t_chunk = time.time() - t0
        ref_logits[name] = lr
        print(f' {name}  (CPU time: repo {t_ref:.2f}s, release {t_chunk:.2f}s)')
        all_ok &= compare('release forward (chunked)', lr, lc, rows)

    print('\n[B. decode: repo LM.forward one token at a time with carried states (the path its generate() '
          'uses) vs release step() loop]')
    for name, x in cases[:2]:
        with torch.no_grad():
            t0 = time.time()
            outs, st = [], None
            for t in range(x.shape[1]):
                lg, st, _ = ref(x[:, t:t + 1], states=st, step_offset=t)
                outs.append(lg)
            lrr = torch.cat(outs, 1); t_ref = time.time() - t0
            t0 = time.time(); lrec, _ = rel.forward_recurrent(x); t_rec = time.time() - t0
        print(f' {name}  (CPU time: repo {t_ref:.1f}s, release {t_rec:.1f}s)')
        seps = (x[0] == 50256).nonzero().flatten().tolist()
        if not seps:
            all_ok &= compare('release forward_recurrent (token loop)', lrr, lrec, rows)
        else:
            s0 = seps[0]
            all_ok &= compare(f'release token loop, positions < {s0} (before the first separator)',
                              lrr[:, :s0], lrec[:, :s0], rows)
            d_after = (lrr[:, s0:] - lrec[:, s0:]).abs().max().item()
            print(f'  positions >= {s0}: max|dlogit| {d_after:.3e}  (EXPECTED to differ: the repo decode path '
                  f'resets only the echo memory at <|endoftext|>, not the notebooks; the release step() resets '
                  f'both, as the repo prefill/training does. Release step vs release/repo prefill: see C)')

    print('\n[C. information: the two algebras inside the release (chunked prefill vs exact per-token '
          'recurrence), fp32]')
    for name, x in cases:
        with torch.no_grad():
            lrec, _ = rel.forward_recurrent(x)
        d = (ref_logits[name] - lrec).abs().max().item()
        ag = (ref_logits[name].argmax(-1) == lrec.argmax(-1)).float().mean().item()
        print(f'  {name}: repo prefill vs release token loop  max|dlogit| {d:.3e}  argmax agree {ag:.4f}')
    print('  (not a parity check: the chunked form uses log(retention + 1e-6) exactly like the training '
          'kernels, i.e. retention + 1e-6, while the per-token recurrence uses the exact retention; '
          'the repo has the same prefill/decode gap. --fp64 shows the two forms agree to ~1e-8 once that '
          'epsilon is removed.)')
    if args.fp64:
        import modeling_pam2 as M
        rel64 = load_model(args.checkpoint, device='cpu', dtype=torch.float64)
        x = cases[1][1]
        with torch.no_grad():
            lrec, _ = rel64.forward_recurrent(x)
            for eps in (1e-6, 1e-12):
                M._EPS = eps
                lc, _ = rel64(x)
                print(f'  fp64, chunked eps {eps:g} vs token loop ({cases[1][0][:30]}...): '
                      f'max|dlogit| {(lc - lrec).abs().max().item():.3e}')
            M._EPS = 1e-6
        del rel64

    print('\n[D. state continuation: forward(x[:200]) -> forward(x[200:], state) vs forward(x)]')
    x = cases[1][1]
    with torch.no_grad():
        lr, _, _ = ref(x)
        l1, st = rel(x[:, :200])
        l2, st = rel(x[:, 200:], st)
    all_ok &= compare('release, prefill split at 200 (state carried)', lr, torch.cat([l1, l2], 1), rows)
    print(f'  carried state: {st.nbytes() / 2**20:.1f} MiB (fixed, independent of context length)')

    print(f'\n[E. greedy generation, {args.gen_tokens} new tokens, no eos stop]')
    for p in PROMPTS:
        x = torch.tensor([tok.encode(p)])
        with torch.no_grad():
            t0 = time.time()
            out_ref = ref.generate(x, max_new_tokens=args.gen_tokens, temperature=0.0)
            t_ref = time.time() - t0
            t0 = time.time()
            out_rel = rel.generate(x, max_new_tokens=args.gen_tokens, temperature=0.0)
            t_rel = time.time() - t0
        same = out_ref.tolist() == out_rel.tolist()
        all_ok &= same
        new = out_rel[0, x.shape[1]:].tolist()
        n_same = sum(int(u == v) for u, v in zip(out_ref[0, x.shape[1]:].tolist(), new))
        print(f'  prompt {p!r}')
        print(f'    identical ids: {same} ({n_same}/{len(new)})  time repo {t_ref:.1f}s, release {t_rel:.1f}s')
        print(f'    output: {tok.decode(new)!r}')
        if 50256 in new:
            print('    NOTE: output contains <|endoftext|>; the repo decode path does not reset the '
                  'notebooks on it (release does), ids after it may differ')

    print('\nRESULT:', 'ALL PASS' if all_ok else 'SOME CHECKS FAILED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
