"""Business regressions for selection, ordered evaluation and interrupted training."""
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_baseline as training
from training_state import capture
from test_training_smoke import Noam


def collate(rows):
    return {"target_batch": torch.tensor(rows, dtype=torch.float32).reshape(-1, 1)}


class SmallNative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.dropout = torch.nn.Dropout(0.2)
        self.optimizer = Noam(self)

    def train_one_batch(self, batch, iteration, train=True):
        x = batch["target_batch"]
        if not train:
            return (float(x.mean()),)
        self.optimizer.optimizer.zero_grad()
        loss = self.dropout(self.linear(x)).square().mean()
        loss.backward()
        self.optimizer.step()
        return (loss.item(),)


def runtime():
    return types.SimpleNamespace(model=SmallNative(), name="multi_trs", device="cpu",
                                 config=types.SimpleNamespace(pretrain=False), collate=collate,
                                 datasets={"train": [1, 2, 3, 4, 5], "valid": [1, 1, 3]},
                                 provenance={"source_commit": "test"})


class FormalTrainingTests(unittest.TestCase):
    def test_native_eligibility_and_three_misses_with_tie_reset(self):
        recipe = {"selection_start_step": 14000}
        p = {"best_ppl": float("inf"), "bad_validations": 0}
        self.assertFalse(training.select_best(p, 12000, 1, recipe))
        self.assertTrue(training.select_best(p, 14000, 4, recipe))
        self.assertFalse(training.select_best(p, 16000, 5, recipe))
        self.assertTrue(training.select_best(p, 18000, 4, recipe))
        self.assertEqual(p["bad_validations"], 0)
        for step in (20000, 22000, 24000):
            training.select_best(p, step, 5, recipe)
        self.assertEqual(p["bad_validations"], 3)
        self.assertEqual(p["best_step"], 18000)

    def test_shuffled_stream_restores_tail_and_next_epoch_without_rng_consumption(self):
        native = runtime()
        stream = training.TrainStream(native, 2, 13)
        state = torch.get_rng_state().clone()
        stream.next()
        saved = stream.state_dict()
        expected = [stream.next()["target_batch"] for _ in range(4)]
        resumed = training.TrainStream(native, 2, 13, saved)
        actual = [resumed.next()["target_batch"] for _ in range(4)]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_valid_keeps_native_batch_aggregation_and_restores_train_mode(self):
        native = runtime()
        metrics = training.valid_ppl(native, 2)
        self.assertEqual(metrics["n"], 3)
        self.assertEqual(metrics["batches"], 2)
        self.assertAlmostEqual(metrics["ppl"], math.exp(2))
        self.assertTrue(native.model.training)
        self.assertFalse(metrics["shuffle"])

    def test_full_training_resume_keeps_pretrain_noam_and_best_selection(self):
        recipe = {"pretrain_epochs": 1, "batch_size": 2, "seed": 13, "valid_batch_size": 2,
                  "valid_interval": 2, "selection_start_step": 4, "patience": 3, "max_main_steps": 12,
                  "checkpoint_interval": 2, "log_interval": 1}
        checkpoint_at_four = []
        original_save = training.save_checkpoint
        def save(path, value):
            if path.name == "last.pt" and value["progress"]["main_step"] == 4:
                checkpoint_at_four.append(value)
            original_save(path, value)
        def valid(native, size):
            # Three pretrain steps + main steps 2/4/6/8/10.
            score = {5: 9, 7: 4, 9: 5, 11: 6, 13: 7}[native.model.optimizer._step]
            return {"ppl": score}
        with tempfile.TemporaryDirectory() as directory:
            full_dir, resume_dir = Path(directory) / "full", Path(directory) / "resume"
            full_dir.mkdir()
            resume_dir.mkdir()
            torch.manual_seed(13)
            first = runtime()
            with patch.object(training, "valid_ppl", valid), patch.object(training, "save_checkpoint", save):
                result = training.fit(first, recipe, full_dir, "test")
            expected = capture(first, result["completed_updates"])
            second = runtime()
            with patch.object(training, "valid_ppl", valid):
                resumed = training.fit(second, recipe, resume_dir, "test", checkpoint_at_four[0])
            self.assertEqual(result, resumed)
            self.assertEqual(result["completed_updates"], 13)
            self.assertEqual(result["best_step"], 4)
            self.assertEqual(result["stop_reason"], "early_stopping")
            actual = capture(second, resumed["completed_updates"])
            torch.testing.assert_close(actual["model"], expected["model"], rtol=0, atol=0)
            torch.testing.assert_close(actual["optimizer"], expected["optimizer"], rtol=0, atol=0)
            self.assertEqual(actual["noam"], expected["noam"])
            best = torch.load(full_dir / "ppl_best.pt")
            self.assertEqual(best["progress"]["best_step"], 4)
            self.assertEqual(best["progress"]["best_ppl"], 4)


if __name__ == "__main__":
    unittest.main()
