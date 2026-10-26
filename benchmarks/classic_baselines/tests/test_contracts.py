"""CPU-only tests for isolation, source preparation and artifact integrity."""
import copy
import io
import math
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from audit_data import CacheReader, digest, sample
from audit_sources import parser_defaults
from cloud_preflight import validate
from prepare_workspaces import changes, patch_file
from score_records import align, diversity, likelihood, summarize


class ArtifactTests(unittest.TestCase):
    def test_sample_macro_and_token_micro_are_distinct(self):
        records = [{"nll_sum": math.log(2), "token_count": 1},
                   {"nll_sum": 3 * math.log(8), "token_count": 3}]
        result = likelihood(records)
        self.assertAlmostEqual(result["sample_macro_ppl"], 5)
        self.assertAlmostEqual(result["token_micro_ppl"], 2 ** 2.5)
        self.assertIsNone(result["emotion_acc_micro_pct"])

    def test_invalid_loss_and_padding_counts_rejected(self):
        for nll, length in [(float("nan"), 1), (float("inf"), 1), (-1, 1), (1, 0), (1, True), (True, 1)]:
            with self.subTest(nll=nll, length=length), self.assertRaises(ValueError):
                likelihood([{"nll_sum": nll, "token_count": length}])

    def test_micro_accuracy_and_missing_classifier(self):
        records = [{"nll_sum": 1, "token_count": 2, "emotion_pred": pred,
                    "emotion_gold": "sad", "emotion_ce": ce} for pred, ce in [("sad", 1), ("angry", 3)]]
        result = likelihood(records)
        self.assertEqual(result["emotion_acc_micro_pct"], 50)
        self.assertEqual(result["emotion_ce_sample_mean"], 2)
        records[1]["emotion_pred"] = None
        with self.assertRaises(ValueError):
            likelihood(records)

    def test_ids_and_gold_labels_cannot_be_reordered(self):
        canonical = [{"sample_id": "test:0", "emotion": "sad"}, {"sample_id": "test:1", "emotion": "joyful"}]
        records = [{"sample_id": row["sample_id"], "emotion_gold": row["emotion"]} for row in canonical]
        align(records, canonical)
        with self.assertRaises(ValueError):
            align(records[::-1], canonical)
        with self.assertRaises(ValueError):
            align(records[:1], canonical)
        records[1]["emotion_gold"] = "sad"
        with self.assertRaises(ValueError):
            align(records, canonical)

    def test_corpus_bigrams_do_not_cross_sentence_boundaries(self):
        result = diversity([["a", "b"], ["a", "b"], ["b", "c"]])
        self.assertAlmostEqual(result["dist_2_pct"], 200 / 3)
        self.assertAlmostEqual(result["unique_pct"], 200 / 3)
        self.assertEqual(diversity([[], []])["empty_n"], 2)
        with self.assertRaises(ValueError):
            diversity([["two words"]])

    def test_partial_generation_fails(self):
        canonical = [{"sample_id": str(i), "emotion": "sad"} for i in range(2)]
        records = [{"sample_id": str(i), "emotion_gold": "sad", "nll_sum": 1, "token_count": 2,
                    "greedy_tokens": ["hello"] if i == 0 else None} for i in range(2)]
        with self.assertRaises(ValueError):
            summarize(records, canonical)


class PreparationTests(unittest.TestCase):
    def test_parser_audit_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "config.py"
            source.write_text('raise RuntimeError("must not execute")\nparser.add_argument("--seed", type=int, default=13)\n', encoding="utf-8")
            self.assertEqual(parser_defaults(source)["seed"], 13)

    def test_cache_reader_rejects_executable_globals(self):
        with self.assertRaises(pickle.UnpicklingError):
            CacheReader(io.BytesIO(b"cos\nsystem\n.")).load()

    def test_tokenization_difference_changes_sample_fingerprint(self):
        one = {"context": [[["hello"]]], "target": [["good.."]], "emotion": ["sad"]}
        two = copy.deepcopy(one)
        two["target"][0] = ["good", ".."]
        self.assertNotEqual(digest(sample(one, 0)), digest(sample(two, 0)))

    def test_patch_count_guard_prevents_silent_wrong_source(self):
        with self.assertRaises(ValueError):
            patch_file("wrong", "expected", "new")
        with self.assertRaises(ValueError):
            patch_file("x x", "x", "y")

    def test_evaluation_weights_survive_initialization(self):
        source = '''def build(is_eval, model, config, xavier_uniform_):
    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)
'''
        namespace = {}
        exec(changes("cem", "main.py", source), namespace)
        class Parameter:
            def dim(self): return 2
        class Model:
            def named_parameters(self): return [("encoder.weight", Parameter())]
        class Config:
            pretrain_emb = True
        calls = []
        namespace["build"](True, Model(), Config(), calls.append)
        self.assertEqual(calls, [])
        namespace["build"](False, Model(), Config(), calls.append)
        self.assertEqual(len(calls), 1)

    def test_cloud_environment_check_rejects_local_runtime(self):
        import json
        target = json.loads((HERE / "cloud_target.json").read_text(encoding="utf-8"))
        good = {"os_release": {"ID": "ubuntu", "VERSION_ID": "22.04"}, "python": "3.10.14",
                "torch": "2.1.2+cu121", "torch_cuda_runtime": "12.1", "packages": {}, "cudnn": 8902,
                "gpus": [{"name": "NVIDIA GeForce RTX 4090D", "memory_bytes": 24 * 1024 ** 3}]}
        self.assertEqual(validate(good, target, copy.deepcopy(good)), [])
        spaced = copy.deepcopy(good)
        spaced["gpus"][0]["name"] = "NVIDIA GeForce RTX 4090 D"
        self.assertEqual(validate(spaced, target), [])
        for name, memory in [("NVIDIA GeForce RTX 4090", 24 * 1024 ** 3),
                             ("NVIDIA GeForce RTX 4090 D", 16 * 1024 ** 3)]:
            wrong_gpu = copy.deepcopy(spaced)
            wrong_gpu["gpus"][0].update(name=name, memory_bytes=memory)
            self.assertIn("Expected RTX 4090D 24GB", validate(wrong_gpu, target))
        wrong_torch = copy.deepcopy(good)
        wrong_torch["torch"] = "2.1.0+cu121"
        self.assertIn("Expected PyTorch 2.1.2", validate(wrong_torch, target))
        bad = copy.deepcopy(good)
        bad.update(python="3.8.20", torch="1.12.1+cu113")
        self.assertGreaterEqual(len(validate(bad, target)), 2)
        reference = copy.deepcopy(good)
        reference["packages"]["numpy"] = "different"
        self.assertIn("CASE-EPCL runtime differs: packages", validate(good, target, reference))


if __name__ == "__main__":
    unittest.main()
