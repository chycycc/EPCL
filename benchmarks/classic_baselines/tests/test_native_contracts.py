import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from build_data_bundle import cem_view, kemp_vocabulary, validate_features
from native_generation import search, tokenize_response
from native_runtime import NativeRuntime, readonly_observation
from verify_cloud_bundle import verify
from audit_sources import sha256


class DataAdapterTests(unittest.TestCase):
    def test_kemp_roles_are_renamed_without_adding_output_entries(self):
        vocab = {"word2index": {"UNK": 0, "PAD": 1, "EOS": 2, "SOS": 3, "USR": 4, "SYS": 5, "hello": 6},
                 "word2count": {"USR": 1, "SYS": 1, "hello": 2},
                 "index2word": {0: "UNK", 1: "PAD", 2: "EOS", 3: "SOS", 4: "USR", 5: "SYS", 6: "hello"}, "n_words": 7}
        old = copy.deepcopy(vocab)
        converted = kemp_vocabulary(vocab)
        self.assertEqual(len(converted[0]), 7)
        self.assertEqual(converted[0]["[USR]"], 4)
        self.assertEqual(converted[0]["[SYS]"], 5)
        self.assertEqual(set(converted[0].values()), set(range(7)))
        self.assertEqual(vocab, old)

    def test_cem_relation_layout_keeps_named_order_and_input_tokens(self):
        relations = {name: [[name]] for name in ("xReact", "xWant", "xIntent", "xEffect", "xNeed")}
        data = {"context": [[["hello"]]], "target": [["hi"]], "emotion": ["joyful"], "situation": [["s"]],
                "emotion_context": [["hello"]], "utt_cs": [relations]}
        converted = cem_view(data)
        self.assertIs(converted["target"], data["target"])
        self.assertEqual([item[0][0] for item in converted["utt_cs"][0]], ["xIntent", "xNeed", "xWant", "xEffect", "xReact"])
        self.assertIsInstance(data["utt_cs"][0], dict)

    def test_invalid_concept_alignment_and_oov_are_rejected(self):
        cs = {name: [["hello"]] for name in ("xIntent", "xNeed", "xWant", "xEffect", "xReact")}
        cs["uttr"] = ["hello"]
        data = {"context": [[["hello"]]], "target": [["hello"]], "utt_cs": [cs],
                "concepts": [[[[["hello"]], [[[0.5, 0.0, 0.5]]], [[0.2]]]]],
                "vads": [[[[0.5, 0.0, 0.5]]]], "vad": [[[0.2]]]}
        vocabulary = {"word2index": {"hello": 0}}
        self.assertEqual(validate_features(data, vocabulary)["target_oov"], 0)
        data["target"][0] = ["unknown"]
        with self.assertRaises(ValueError):
            validate_features(data, vocabulary)
        data["target"][0] = ["hello"]
        data["concepts"][0][0][0] = []
        with self.assertRaises(ValueError):
            validate_features(data, vocabulary)


class NativeObservationTests(unittest.TestCase):
    def test_observation_restores_rng_and_auxiliary_frequency(self):
        import numpy as np
        import torch
        model = torch.nn.Linear(2, 2)
        model.word_freq = np.zeros(3)
        model.criterion = torch.nn.NLLLoss()
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        with readonly_observation(model):
            random.random()
            np.random.rand()
            torch.rand(3)
            model.word_freq += 5
            model.criterion.weight = torch.ones(3)
        self.assertEqual(random.getstate(), before[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertTrue(np.array_equal(model.word_freq, np.zeros(3)))
        self.assertIsNone(model.criterion.weight)

    def test_checkpoint_without_encoding_provenance_is_rejected_before_loading(self):
        import torch
        runtime = object.__new__(NativeRuntime)
        runtime.device = "cpu"
        runtime.model = torch.nn.Linear(2, 2)
        runtime.manifest = {"target_encoding_sha256": "expected"}
        runtime.provenance = {"source_commit": "commit"}
        before = runtime.model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            torch.save({"model": {"wrong": torch.zeros(1)}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "target-encoding"):
                runtime.load_checkpoint(checkpoint)
        self.assertTrue(torch.equal(before, runtime.model.weight))


class SearchTests(unittest.TestCase):
    def test_beam_keeps_finished_hypotheses_and_can_outperform_greedy(self):
        import torch
        class Distribution:
            def step(self, tokens):
                values = torch.full((7,), -100.0)
                scores = {(): {2: -1, 4: -0.2, 5: -0.4}, (4,): {2: -2, 6: -0.2},
                          (5,): {2: -0.1}, (4, 6): {2: -0.9}}
                for index, value in scores.get(tuple(tokens), {2: 0}).items():
                    values[index] = value
                return values
        self.assertEqual(search(Distribution(), 3, 1), [4, 6, 2])
        self.assertEqual(search(Distribution(), 3, 2), [5, 2])

    def test_greedy_ties_match_native_first_index_and_eos_counts_in_budget(self):
        import torch
        class Distribution:
            def step(self, tokens):
                return torch.tensor([-10.0, -10.0, 0.0, 0.0])
        self.assertEqual(search(Distribution(), 1, 1), [2])
        with self.assertRaises(ValueError):
            search(Distribution(), 0, 1)

    def test_common_tokenizer_requires_no_punkt_and_preserves_case(self):
        self.assertEqual(tokenize_response("  Hello, world!  "), ["Hello", ",", "world", "!"])


class BundleIntegrityTests(unittest.TestCase):
    def test_changed_bundle_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.py"
            path.write_text("original", encoding="utf-8")
            manifest = {"files_sha256": {"source.py": sha256(path)}}
            (root / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(verify(root), [])
            path.write_text("modified", encoding="utf-8")
            self.assertIn("Fingerprint mismatch", verify(root)[0])

    def test_manifest_paths_cannot_escape_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bundle_manifest.json").write_text(json.dumps({"files_sha256": {"../outside.py": "unused"}}), encoding="utf-8")
            self.assertIn("Invalid or missing file", verify(root)[0])


if __name__ == "__main__":
    unittest.main()
