"""双 Best 输出隔离、同状态复用和写入失败传播；推理函数用小型结果替代。"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

with contextlib.redirect_stdout(io.StringIO()):
    from src.scripts import eval_pipeline as pipeline


class DualBestEvaluationTests(unittest.TestCase):
    def run_pair(self, directory, same_state, fail_write=False):
        root = Path(directory)
        for role, value in (("composite", 1.0), ("ppl", 1.0 if same_state else 2.0)):
            torch.save({"iter": 9, "model": {"weight": torch.tensor([value])}},
                       root / f"CASE_9_35.0000_{role}_best")
        metrics = {"Test_PPL": 32.0, "Test_EMO_acc": 42.0, "Test_EMO_loss": 2.3,
                   "_scope": {"n_samples": 5255}}
        generation = {key: {"Dist-2": 18.0, "Unique": 60.0}
                      for key in ("Greedy", "Beam", "Sampling", "Sampling_0.8")}
        generation["_scope"] = {"n_samples": 500}
        test_set = SimpleNamespace(dataset=range(5255), batch_size=1)
        argv = ["eval_pipeline.py", "--model_path", str(root), "--eval_target", "both",
                "--skip_manifold", "--results_path", str(root / "eval.txt"),
                "--metrics_path", str(root / "metrics.json")]
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(pipeline, "load_v8_model", return_value=(None, None, test_set)))
            forward = stack.enter_context(patch.object(pipeline, "evaluate_test_set", return_value=metrics))
            decode = stack.enter_context(patch.object(pipeline, "evaluate_generation_diversity", return_value=generation))
            manifold = stack.enter_context(patch.object(pipeline, "evaluate_manifold_and_plot"))
            if fail_write:
                stack.enter_context(patch.object(pipeline.os, "replace", side_effect=OSError("disk full")))
            pipeline.main()
        self.assertEqual(forward.call_count, 1 if same_state else 2)
        self.assertEqual(decode.call_count, forward.call_count)
        manifold.assert_not_called()
        self.assertEqual(decode.call_args.kwargs["num_samples"], 500)
        return [json.loads((root / name).read_text(encoding="utf-8"))
                for name in ("metrics.json", "metrics_ppl.json")]

    def test_same_step_different_state_evaluated_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = self.run_pair(tmp, same_state=False)
            self.assertTrue(first["checkpoint"].endswith("composite_best"))
            self.assertTrue(second["checkpoint"].endswith("ppl_best"))
            self.assertIsNone(second["reused_from"])
            self.assertEqual(second["evaluation_protocol"]["test"]["evaluated_n"], 5255)
            for name, role in (("eval.txt", "composite_best"), ("eval_ppl.txt", "ppl_best")):
                self.assertIn(role, (Path(tmp) / name).read_text(encoding="utf-8"))

    def test_identical_state_reuses_fresh_results_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = self.run_pair(tmp, same_state=True)
            self.assertEqual(first["test_metrics"], second["test_metrics"])
            self.assertEqual(second["reused_from"]["checkpoint"], first["checkpoint"])
            self.assertIn("评测复用自", (Path(tmp) / "eval_ppl.txt").read_text(encoding="utf-8"))

    def test_json_write_failure_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OSError):
                self.run_pair(tmp, False, fail_write=True)

    def test_missing_best_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError):
                pipeline.evaluate_both(SimpleNamespace(model_path=tmp))


if __name__ == "__main__":
    unittest.main()
