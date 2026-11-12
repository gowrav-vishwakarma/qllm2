"""Stream decontamination index v2 = Mind's v1 (v7/decontam.py, .cache/decontam/v1.npy) + IFEval's 541 prompts (google/IFEval train),
hashed with Mind's own ngram_hashes (same 13-gram polynomial), so v7.data's DecontamFilter reads it unchanged. Deterministic:
run on each box and compare the printed sha256 (R5 audit 2026-11-12: smol-constraints / tulu-IF IFEval overlap; Mind's segment 3).
    python v14/decontam_v2.py [--v1 .cache/decontam/v1.npy] [--out .cache/decontam/v2.npy]
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--v1', default=str(ROOT / '.cache/decontam/v1.npy'))
    ap.add_argument('--out', default=str(ROOT / '.cache/decontam/v2.npy')); a = ap.parse_args()
    from datasets import load_dataset
    from v7.decontam import DecontamFilter, ngram_hashes
    v1 = np.load(a.v1); prompts = [r['prompt'] for r in load_dataset('google/IFEval', split='train')]
    hs = [ngram_hashes(p) for p in prompts]; ife = np.concatenate([h for h in hs if h.size])
    v2 = np.unique(np.concatenate([v1, ife]))
    np.save(a.out, v2); sha = hashlib.sha256(open(a.out, 'rb').read()).hexdigest()
    f = DecontamFilter(a.out); f1 = DecontamFilter(a.v1)
    ctl = sum(f.hit('The river ran past the old mill. ' + p) for p in prompts if len(p.split()) >= 13)
    n13 = sum(len(p.split()) >= 13 for p in prompts)
    meta = dict(path=a.out, sha256=sha, n_grams=int(v2.size), v1=a.v1, v1_sha256=hashlib.sha256(open(a.v1, 'rb').read()).hexdigest(),
                v1_grams=int(v1.size), ifeval_prompts=len(prompts), ifeval_grams=int(np.unique(ife).size),
                superset_of_v1=bool(np.isin(v1, v2).all()), ifeval_control=f'{ctl}/{n13}',
                v1_flags_ifeval=sum(f1.hit('The river ran past the old mill. ' + p) for p in prompts if len(p.split()) >= 13))
    json.dump(meta, open(a.out.replace('.npy', '.json'), 'w'), indent=1); print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
