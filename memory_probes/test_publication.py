"""Tests for publication result semantics and matched-budget capacity."""

from __future__ import annotations

import json
import math
import re
import unittest

from memory_probes.behavioral import build_example, score_candidate_logits
from memory_probes.capacity import test_binding_matched_bytes
from memory_probes.publication import SCHEMA_VERSION, _mean_ci95, run_publication_sweep


class _WordTokenizer:
    def __init__(self) -> None:
        self.vocab = {}
        self.inverse = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for token in re.findall(r'\w+|[^\w\s]', text.lower()):
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.inverse[token_id] = token
            ids.append(self.vocab[token])
        return ids

    def decode(self, ids):
        return ' '.join(self.inverse[int(token_id)] for token_id in ids)


class PublicationTests(unittest.TestCase):
    def test_matched_budget_uses_equal_complex_scalars(self) -> None:
        result = test_binding_matched_bytes(
            matrix_d=4, ns=(1, 4), trials=1, seed=7,
        )
        self.assertEqual(result['hrr_d'], 16)
        self.assertEqual(result['state_complex_scalars'], 16)
        self.assertEqual(result['state_bytes'], 16 * 16)
        self.assertEqual(result['matrix_pam']['ns'], result['vector_hrr']['ns'])

    def test_confidence_interval_is_finite(self) -> None:
        stats = _mean_ci95([1.0, 2.0, 3.0])
        self.assertEqual(stats['n'], 3)
        self.assertEqual(stats['mean'], 2.0)
        self.assertTrue(math.isfinite(stats['ci95']))

    def test_smoke_result_is_strict_json(self) -> None:
        result = run_publication_sweep(
            seeds=(42,),
            dims=(4,),
            max_n=4,
            binding_trials=1,
            distances=(2,),
            rank_steps=2,
            matched_dims=(2,),
            matched_ns=(1, 2),
        )
        self.assertEqual(result['schema_version'], SCHEMA_VERSION)
        ns = result['records']['binding_equal_width'][0]['result']['matrix_pam']['ns']
        self.assertLessEqual(max(ns), 4)
        encoded = json.dumps(result, allow_nan=False)
        self.assertIn('binding_matched_bytes', encoded)

    def test_behavioral_prompt_has_exact_length_and_scores(self) -> None:
        example = build_example(
            _WordTokenizer(),
            context_tokens=128,
            target_position=0.25,
            associations=4,
            seed=42,
        )
        self.assertEqual(len(example.prompt_ids), 128)
        logits = [0.0] * len(example.candidate_token_ids)
        target_index = example.candidate_token_ids.index(example.target_token_id)
        logits[target_index] = 2.0
        score = score_candidate_logits(example, logits)
        self.assertTrue(score['correct'])
        self.assertEqual(score['target_rank'], 1)


if __name__ == '__main__':
    unittest.main()
