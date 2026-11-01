"""MLB-1 v1 mind-side cells: skill-pack persistence (R7) and replay integration.

Spec: v13_sempty/BENCHMARK_MIND_LIBRARY.md §3 (v1, fixed). One JSON line per
document in v16/bench.py's row vocabulary (arch, ckpt, task, L, doc, T, A_state,
O_reread, mem{...}, sec) so the §2 / §3 tables merge without translation.

Cells (examples from v14/rules_ood B2 builders, held-out values; memory_probes.rules
two_hop):

  skillpack       pack = the header + n_shots demonstration lines, read ONCE at t0;
                  then L tokens of held-out filler; then the query line.
                  A_state  = correct with the pack only at t0 (the mind carries it).
                  O_reread = correct with the pack placed right before the query
                             (= a re-fetch; for T++ this IS its "with re-fetch" arm,
                             its KV cost per query is mem.tpp_refetch_kv_bytes).
  replay_refresh  header at t0, L filler, then 2 replayed demonstration lines + query.
                  R_alone      = without the pack at t0 (replay only),
                  R_state_plus = with the pack at t0 as well.
                  Integration = R_state_plus - max(A_state, R_alone).
  replay_hop2     two_hop story: hop-1 facts ("O owns the P.") at t0, L filler, hop-2
                  facts ("The P is called M.") replayed before the query.
                  H_integrate = hop 1 in state + hop 2 replayed;
                  H_oracle    = both hops replayed before the query;
                  H_hop2only  = hop 2 replayed, hop 1 absent (floor).

In v1 the pack is PLACED at t0; fetching a pack by key from a library of packs is
bench.py's (Lead) and lands with Library's L1. Scored at the last position only
(no full [T, V] logits), so long L fits. MLB-1 A7: every cell also reports FREE
generation (`*_free`: greedy argmax over the whole vocabulary == the answer token; the
answers are single tokens), the primary number from A7 on; the 20-way score is secondary. T++ without re-fetch is the whole stream in
its window (run as far as memory allows; the limit goes in the table).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from memory_probes.rules import build_rule_example, score_candidate_logits  # noqa: E402
from memory_probes.spread import _SEP, filler_tokens, split_units  # noqa: E402
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from v14.rules_ood import build_example as b2_example  # noqa: E402

MAMBA2_STATE_BYTES = 13.4e6   # EXPERIMENT_INDEX scoreboard (102.6M Mamba-2: SSM + conv state)


class Body:
    """Last-position logits for PAM, Mamba-2 or T++ (all v13_sempty model types)."""

    def __init__(self, model, device):
        self.m, self.dev = model, device
        self.arch = 'pam' if hasattr(model, '_lm_vector') else (
            'mamba2' if hasattr(model, 'net') else 'transformer')

    @torch.no_grad()
    def last_logits(self, ids) -> torch.Tensor:
        x = torch.tensor([ids], device=self.dev)
        with torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == 'cuda'):
            if self.arch == 'pam':
                z, b, t = self.m._stem(x, 0)
                z, _ = self.m._run_blocks(z, b, t, None, 0, x)
                lm, _ = self.m._lm_vector(z, x, b, t)
                last = lm.raw(b, t, self.m.model_dim)[0, -1].float()
                return last @ self.m.embed.embed.weight.float().T
            hidden = self.m._hidden_to_lm(x)[0][0, -1].float()
        return hidden @ self.m._head_weight().float().T

    @torch.no_grad()
    def check(self, tok) -> float:
        """The last-position path must equal model(x) logits at the last position."""
        ids = tok('The quick brown fox jumps over the lazy dog. ' * 4, add_special_tokens=False)['input_ids']
        x = torch.tensor([ids], device=self.dev)
        with torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == 'cuda'):
            ref = self.m(x)[0][0, -1].float()
        return float((self.last_logits(ids) - ref).abs().max() / ref.abs().max())

    @torch.no_grad()
    def state_bytes(self) -> float:
        if self.arch == 'mamba2':
            return MAMBA2_STATE_BYTES
        if self.arch == 'transformer':
            return 0.0
        x = torch.zeros((1, 8), dtype=torch.long, device=self.dev)
        states = self.m(x)[1]
        total, stack = 0, [states]
        while stack:
            s = stack.pop()
            if torch.is_tensor(s):
                total += s.numel() * 4              # carried in fp32 at decode
            elif isinstance(s, (list, tuple)):
                stack.extend(s)
            elif s is not None and hasattr(s, 'data') and torch.is_tensor(s.data):
                total += s.data.numel() * 4
        return float(total)

    def kv_bytes(self, n_tokens: int) -> float:
        if self.arch != 'transformer':
            return 0.0
        dim = self.m.embed.weight.shape[1]
        return float(len(self.m.layers) * 2 * n_tokens * dim * 2)   # K and V, bf16


def filler_block(filler: np.ndarray, n: int, seed: int, L: int) -> list:
    """n held-out tokens, deterministic in (seed, L); wraps the 500k cache past its end."""
    if n <= 0:
        return []
    start = int(np.random.default_rng([seed, L, 31]).integers(0, len(filler)))
    idx = (start + np.arange(n)) % len(filler)
    return filler[idx].tolist()


def score(body, ex, ids):
    """(20-way correct, 20-way margin, free: greedy argmax over the whole vocab == answer)."""
    logits = body.last_logits(ids)
    r = score_candidate_logits(ex, logits[ex.candidate_token_ids].float().cpu().tolist())
    return int(r['correct']), float(r['target_margin']), int(int(logits.argmax()) == ex.target_token_id)


def run(args):
    dev = torch.device(args.device)
    if args.mem_frac and dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    model, tok, _ = _load_v13_sempty(args.checkpoint, 'baseline_real_pm', dev)
    model.eval(); body = Body(model, dev)
    err = body.check(tok)
    assert err < 2e-2, f'last-position path differs from model(x): rel {err:.3g}'
    print(f'{body.arch}: last-position check rel err {err:.2e}; state {body.state_bytes() / 1e6:.2f} MB', flush=True)
    filler = filler_tokens(args.filler)
    sep = tok.encode(_SEP, add_special_tokens=False)
    tasks = [t for t in args.tasks.split(',') if t]
    lengths = [int(x) for x in args.lengths.split(',') if x]
    variants = [v for v in args.variants.split(',') if v]
    alphas = [float(x) for x in args.alphas.split(',') if x]
    dilations = [float(x) for x in args.dilations.split(',') if x]
    tag_ids = tok.encode(' [answer]', add_special_tokens=False)
    out = open(args.out, 'a')
    S = body.state_bytes()
    for task in tasks + (['two_hop'] if 'replay_hop2' in variants else []):
        for seed in range(args.seed, args.seed + args.docs):
            if task == 'two_hop':
                ex = build_rule_example(tok, 'two_hop', args.hop_facts, seed)
            else:
                ex = b2_example(tok, task, args.shots, seed)
            header, units = split_units(ex)
            enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
            H, U = enc(header), [enc(u) for u in units]
            for L in lengths:
                t0 = time.time()
                fill = sep + filler_block(filler, L, seed, L) + sep if L else []
                row = {'arch': body.arch, 'ckpt': str(args.checkpoint), 'sub': task, 'L': L, 'doc': seed}
                if task != 'two_hop':
                    pack = H + sum(U[:-1], []); q = U[-1] + (tag_ids if args.tag else [])   # cu2b: ACh-1 answer tag
                    if 'skillpack' in variants:
                        ids = pack + fill + q; a, am, af = score(body, ex, ids)
                        refetch = pack + q; o, om, of = score(body, ex, refetch)
                        out.write(json.dumps({**row, 'task': 'skillpack', 'T': len(ids), 'A_state': a, 'A_margin': am,
                                              'A_free': af, 'O_reread': o, 'O_margin': om, 'O_free': of,
                                              'pack_tokens': len(pack),
                                              'mem': {'state_bytes': S, 'tpp_kv_bytes_full': body.kv_bytes(len(ids)),
                                                      'tpp_refetch_kv_bytes': body.kv_bytes(len(refetch))},
                                              'sec': round(time.time() - t0, 2)}) + '\n')
                    if 'contrast' in variants:           # cd1: l = l_pack + a * (l_pack - l_nopack)
                        lp_, ln_ = body.last_logits(pack + fill + q).float(), body.last_logits(fill + q).float()
                        fa, ca, ta = {}, {}, {}
                        for al in alphas:
                            lg = lp_ + al * (lp_ - ln_)
                            fa[str(al)] = int(int(lg.argmax()) == ex.target_token_id)
                            ta[str(al)] = int(lg.argmax())                       # what wins the free argmax
                            ca[str(al)] = int(score_candidate_logits(ex, lg[ex.candidate_token_ids].cpu().tolist())['correct'])
                        out.write(json.dumps({**row, 'task': 'contrast', 'T': len(pack + fill + q), 'free_a': fa,
                                              'cand_a': ca, 'tok_a': ta, 'target': ex.target_token_id, 'sec': round(time.time() - t0, 2)}) + '\n')
                    if 'dilate' in variants:             # clk1: retention ** (1/s) over the filler between read and query
                        fd, cd_ = {}, {}
                        for sc in dilations:
                            body.m._dilate = (len(pack), len(pack) + len(fill), sc, args.dilate_where) if fill else None
                            lg = body.last_logits(pack + fill + q).float()
                            fd[str(sc)] = int(int(lg.argmax()) == ex.target_token_id)
                            cd_[str(sc)] = int(score_candidate_logits(ex, lg[ex.candidate_token_ids].cpu().tolist())['correct'])
                            if sc > 1:                       # dilation + contrast (a = 4), both streams dilated over the filler
                                body.m._dilate = (0, len(fill), sc, args.dilate_where) if fill else None
                                ln_ = body.last_logits(fill + q).float(); lc = lg + 4.0 * (lg - ln_)
                                fd[f'{sc}+c4'] = int(int(lc.argmax()) == ex.target_token_id)
                                cd_[f'{sc}+c4'] = int(score_candidate_logits(ex, lc[ex.candidate_token_ids].cpu().tolist())['correct'])
                        body.m._dilate = None
                        out.write(json.dumps({**row, 'task': 'dilate', 'T': len(pack + fill + q), 'free_s': fd, 'cand_s': cd_,
                                              'sec': round(time.time() - t0, 2)}) + '\n')
                    if 'replay_refresh' in variants:
                        replay = sum(U[-3:-1], [])
                        ids_alone = H + fill + replay + q
                        ids_plus = pack + fill + replay + q
                        ra, _, raf = score(body, ex, ids_alone); rp, _, rpf = score(body, ex, ids_plus)
                        out.write(json.dumps({**row, 'task': 'replay_refresh', 'T': len(ids_plus),
                                              'R_alone': ra, 'R_state_plus': rp, 'R_alone_free': raf,
                                              'R_state_plus_free': rpf, 'replay_tokens': len(replay),
                                              'mem': {'state_bytes': S}, 'sec': round(time.time() - t0, 2)}) + '\n')
                else:
                    facts, q = U[:-1], U[-1]
                    texts = units[:-1]
                    hop1 = sum([f for f, s in zip(facts, texts) if ' owns the ' in s], [])
                    hop2 = sum([f for f, s in zip(facts, texts) if ' is called ' in s], [])
                    hi, _, hif = score(body, ex, H + hop1 + fill + hop2 + q)
                    ho, _, hof = score(body, ex, H + fill + hop1 + hop2 + q)
                    h2, _, h2f = score(body, ex, H + fill + hop2 + q)
                    out.write(json.dumps({**row, 'task': 'replay_hop2', 'T': len(H + hop1 + fill + hop2 + q),
                                          'H_integrate': hi, 'H_oracle': ho, 'H_hop2only': h2,
                                          'H_integrate_free': hif, 'H_oracle_free': hof, 'H_hop2only_free': h2f,
                                          'chance': ex.in_context_chance, 'mem': {'state_bytes': S},
                                          'sec': round(time.time() - t0, 2)}) + '\n')
                out.flush()
        print(f'  {task} done', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tasks', default='shift_wide_new,affine_map_new')
    p.add_argument('--variants', default='skillpack,replay_refresh,replay_hop2')
    p.add_argument('--shots', type=int, default=8)
    p.add_argument('--alphas', default='0,0.5,1,2,4', help='contrast variant (cd1): l_pack + a (l_pack - l_nopack)')
    p.add_argument('--dilations', default='1,2,4', help='dilate variant (clk1): retention ** (1/s) over the filler')
    p.add_argument('--dilate_where', default='all', choices=['all', 'echo', 'notebook'], help='clk2: which memory is dilated')
    p.add_argument('--tag', action='store_true', help="cu2b: append ACh-1's answer tag ' [answer]' after the query line")
    p.add_argument('--hop_facts', type=int, default=4, help='two_hop: owners (= pets = names) in the story')
    p.add_argument('--lengths', default='0,2048,8192,32768,131072')
    p.add_argument('--docs', type=int, default=100)
    p.add_argument('--seed', type=int, default=8000, help='probe seeds never used before (s8 used 7000-7399)')
    p.add_argument('--filler', default=str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--mem_frac', type=float, default=0.0)
    p.add_argument('--out', required=True)
    run(p.parse_args())
