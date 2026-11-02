"""Package v0 contract check (Library, v14/PACKAGE.md section 1): does a mind body keep the five interfaces that
v16/package.py uses, with no package code change?

    python v14/contract_check.py --ckpt <best_model.pt>                         # a trained mind
    python v14/contract_check.py --like <ref best_model.pt> --set dim=1024 n_layers=24 pam_layers=5,11,17,23
                                                                                # a RANDOM-INIT body of a new geometry

Checks (1) streaming: chunked prefill = one pass, token-by-token decode = one pass, a deep-copied state continues
identically; (2) model.blocks outputs (z [batch, time, D], state) with the address hooked after block n_blocks // 2;
(3) a per-position LM head; (4) the _dilate hook runs, changes the state and resets; (5) tokenizer stops.
Reports params, measured state bytes, the address bytes per token (int8 / sign bits on 1/16), read seconds, and a
package end-to-end toy (exact key -> text0; natural question -> F2). On a random-init body only the PATHS are
meaningful, not the answers.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from v16.package import Package, _nbytes  # noqa: E402

HOLDOUT = ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'


def _flat(st) -> torch.Tensor:
    """every tensor of a mind state (any nesting of lists / tuples / dicts), flattened into one vector."""
    out = []

    def walk(x):
        if torch.is_tensor(x):
            out.append(x.float().flatten())
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif torch.is_tensor(getattr(x, 'data', None)):
            out.append(x.data.float().flatten())
    walk(st)
    return torch.cat(out)


def _val(v: str):
    if ',' in v:
        return tuple(int(x) for x in v.split(',') if x != '')
    if v.lower() in ('true', 'false'):
        return v.lower() == 'true'
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def random_init_ckpt(like: str, sets: list[str], out_dir: str) -> tuple[str, dict]:
    """a checkpoint file holding a random-init body: the reference config with --set overrides (seed 0)."""
    from v14.hook import build_lm, config_from_raw
    raw = dict(torch.load(like, map_location='cpu', weights_only=False, mmap=True)['config'])
    for kv in sets:
        k, v = kv.split('=', 1)
        if k not in raw:
            sys.exit(f'--set {k}: not a config field ({sorted(raw)})')
        raw[k] = _val(v)
    torch.manual_seed(0)
    cfg = config_from_raw(raw); cfg.dropout = 0.0; cfg.gradient_checkpointing = False
    model = build_lm(cfg)
    path = os.path.join(out_dir, 'contract_random_init.pt')
    torch.save({'config': raw, 'model_state_dict': model.state_dict()}, path)
    return path, raw


@torch.no_grad()
def check(pkg: Package, n_tok: int = 4096, chunk: int = 1024) -> dict:
    m, dev, tok = pkg.mind.m, pkg.mind.dev, pkg.tok
    from memory_probes.spread import filler_tokens
    ids = filler_tokens(str(HOLDOUT))[:n_tok].tolist()
    x = torch.tensor([ids], device=dev)
    rel = lambda a, b: float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12))  # noqa: E731
    r = dict(params=sum(p.numel() for p in m.parameters()), n_blocks=len(m.blocks), address_layer=pkg.address_layer)
    # (2) + (3): one pass with the address hook
    caps, outs = [], []

    def hook(mod, i, o):
        outs.append(type(o).__name__ + (f'[{len(o)}]' if isinstance(o, tuple) else ''))
        caps.append((o[0] if isinstance(o, tuple) else o).data[0])
    h = m.blocks[pkg.address_layer].register_forward_hook(hook)
    t0 = time.time()
    try:
        full, st_full, _ = m(x, states=None, step_offset=0)
    finally:
        h.remove()
    r['one_pass_sec'] = round(time.time() - t0, 2)
    r['block_output'] = outs[0]
    r['address_shape'] = list(caps[0].shape)
    r['head_logits_shape'] = list(full.shape)
    r['head_ok'] = int(full.shape[:2] == (1, n_tok) and bool(torch.isfinite(full).all()))
    r['block_ok'] = int(outs[0].startswith('tuple') and caps[0].shape == (n_tok, caps[0].shape[-1]))
    # (1) streaming: chunked prefill, token decode, deep copy
    lg, st = [], None
    for c0 in range(0, n_tok, chunk):
        o, st, _ = m(x[:, c0:c0 + chunk], states=st, step_offset=c0)
        lg.append(o)
    r['chunked_vs_one_pass_rel'] = rel(full, torch.cat(lg, 1))
    r['state_entries'] = len(st) if isinstance(st, (list, tuple)) else 1
    k0 = n_tok - 16
    _, st_k, _ = m(x[:, :k0], states=None, step_offset=0)
    st_copy = copy.deepcopy(st_k)
    dec = []
    for t in range(k0, n_tok):
        o, st_k, _ = m(x[:, t:t + 1], states=st_k, step_offset=t)
        dec.append(o)
    r['decode_vs_one_pass_rel'] = rel(full[:, k0:], torch.cat(dec, 1))
    o2, _, _ = m(x[:, k0:], states=st_copy, step_offset=k0)
    r['deepcopy_continue_vs_one_pass_rel'] = rel(full[:, k0:], o2)
    r['streaming_ok'] = int(max(r['chunked_vs_one_pass_rel'], r['decode_vs_one_pass_rel'],
                                r['deepcopy_continue_vs_one_pass_rel']) < 1e-2)
    # (4) _dilate
    try:                                       # judged on the STATE: at random init the echo's read-out can be zero
        m._dilate = (n_tok // 4, 3 * n_tok // 4, 8.0, 'echo')
        dl, st_dl, _ = m(x, states=None, step_offset=0)
        m._dilate = None
        back, _, _ = m(x, states=None, step_offset=0)
        r['dilate_state_change_rel'] = rel(_flat(st_full), _flat(st_dl))
        r['dilate_logits_change_rel'] = rel(full[:, -1], dl[:, -1])
        r['dilate_reset_rel'] = rel(full, back)
        r['dilate_ok'] = int(r['dilate_state_change_rel'] > 0 and r['dilate_reset_rel'] < 1e-2)
    except Exception as e:                                      # noqa: BLE001
        m._dilate = None
        r['dilate_ok'] = 0; r['dilate_error'] = f'{type(e).__name__}: {e}'[:200]
    # (5) stops
    r['stops'] = [tok.decode([t]) for t in pkg.stops]
    r['stops_ok'] = int(len(pkg.stops) >= 2 and tok.decode([pkg.stops[0]]) == '.')
    # costs
    r['state_bytes'] = _nbytes(st_full)
    r['address_dim'] = int(caps[0].shape[-1])
    r['address_int8_B_per_token'] = pkg.address_bytes(1, 'int8')
    r['address_signbits_B_per_token'] = round(pkg.address_bytes(10 ** 6, 'sal1') / 10 ** 6, 2)
    return r


def toy(pkg: Package) -> list:
    filler = " The river ran past the old mill and the town grew around it over many years, and the people kept their old ways." * 60
    sess = ("Definition: wexlor shard refers to amber." + filler + " Ada Brann kept the spare key under the blue flowerpot by the door."
            + filler + " Result: job: tamrin; status: cedar; owner: plum; region: north; label: ivory." + filler)
    out = []
    t0 = time.time(); pkg.read(sess); out.append(dict(read_sec=round(time.time() - t0, 2), tokens=len(pkg._session['ids'])))
    for q in (" Query: wexlor shard refers to", " Query: job: tamrin; status:", " Where did Ada hide the spare key? Answer:"):
        a, info = pkg.answer(sess, q, mode='lenient')
        out.append(dict(q=q, answer=a, path=info['path'], row=(info.get('row_text') or '')[-70:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt'); ap.add_argument('--like'); ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--device', default='cpu'); ap.add_argument('--tokens', type=int, default=4096)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    tmp = None
    if a.like:
        tmp = tempfile.mkdtemp(prefix='contract_')
        ck, raw = random_init_ckpt(a.like, a.set, tmp)
    else:
        ck, raw = a.ckpt, None
    try:
        pkg = Package(ck, device=a.device)
        res = dict(ckpt=a.ckpt or f'random init like {a.like} with {a.set}', **check(pkg, a.tokens), toy=toy(pkg))
    finally:
        if tmp:
            for f in Path(tmp).iterdir():
                f.unlink()
            os.rmdir(tmp)
    res['contract_ok'] = int(all(res[k] for k in ('head_ok', 'block_ok', 'streaming_ok', 'dilate_ok', 'stops_ok')))
    print(json.dumps(res, indent=1, default=str))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
