"""BM25 over the ARC-LIB passage library (scipy sparse, no extra dependency). Term-major CSC weights:
w(d, t) = idf(t) * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl)), idf = log(1 + (N - df + 0.5) / (df + 0.5)).
A query scores each passage by the sum of its terms' columns (each query term once); top-k by argpartition."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

VEC = HashingVectorizer(n_features=2 ** 22, alternate_sign=False, norm=None, stop_words='english',
                        token_pattern=r'(?u)\b\w+\b', dtype=np.float32)


def build_bm25(texts, path, k1: float = 1.2, b: float = 0.75, chunk: int = 500_000):
    tf = sp.vstack([VEC.transform(texts[i:i + chunk]) for i in range(0, len(texts), chunk)]).tocsr()
    n = tf.shape[0]; dl = np.asarray(tf.sum(1)).ravel(); avg = float(dl.mean())
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(n), np.diff(tf.indptr)); t = tf.data
    tf.data = (idf[tf.indices] * t * (k1 + 1) / (t + k1 * (1 - b + b * dl[rows] / avg))).astype(np.float32)
    W = tf.tocsc(); sp.save_npz(path, W, compressed=False)
    return dict(n=n, nnz=int(W.nnz), avgdl=round(avg, 2), k1=k1, b=b, features=2 ** 22)


class BM25:
    def __init__(self, lib_dir):
        from pathlib import Path
        d = Path(lib_dir); self.W = sp.load_npz(d / 'bm25.npz').tocsc()
        self.texts = open(d / 'passages.txt').read().split('\n')[:-1]
        self.src = np.load(d / 'source.npy')

    def top(self, query: str, k: int = 3):
        cols = np.unique(VEC.transform([query]).indices)
        if cols.size == 0:
            return []
        s = np.asarray(self.W[:, cols].sum(1)).ravel()
        i = np.argpartition(-s, k)[:k]; i = i[np.argsort(-s[i])]
        return [(int(j), float(s[j]), self.texts[j]) for j in i if s[j] > 0]
