import copy
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glove_audit import verify_file, verify_initialized
from training_state import capture, compare, fingerprint, protocol, restore


class Noam:
    def __init__(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98))
        self._step, self._rate = 0, 0
        self.warmup, self.factor, self.model_size = 8000, 1, 2

    def step(self):
        self._step += 1
        self._rate = self.model_size ** -0.5 * self._step * self.warmup ** -1.5
        for group in self.optimizer.param_groups:
            group['lr'] = self._rate
        self.optimizer.step()


def toy():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.3))
    model.criterion = torch.nn.NLLLoss()
    model.word_freq = np.zeros(2)
    model.optimizer = Noam(model)
    return types.SimpleNamespace(name='toy', model=model, device='cpu', config=types.SimpleNamespace(pretrain=False),
                                 provenance={'source_commit': 'commit', 'prepared_files_sha256': {'model.py': 'hash'}},
                                 manifest={'target_encoding_sha256': 'encoding', 'bundle_sha256': 'data'})


def update(runtime):
    model = runtime.model
    model.optimizer.optimizer.zero_grad()
    # Calling Sequential.forward would also invoke the dynamically added criterion.
    model[1](model[0](torch.rand(3, 2))).square().sum().backward()
    model.optimizer.step()
    model.word_freq += [random.random(), np.random.rand()]
    model.criterion.weight = torch.rand(2)


class TrainingStateTests(unittest.TestCase):
    def test_full_restore_replays_dropout_adam_noam_rng_and_dynamic_buffers(self):
        first = toy()
        update(first)
        saved = capture(first, 1)
        checkpoint = {'baseline_protocol': protocol(first, 'glove'), 'training_state': saved}
        update(first)
        expected = capture(first, 2)
        second = toy()
        self.assertEqual(restore(second, checkpoint, 'glove'), 1)
        self.assertEqual(fingerprint(capture(second, 1)), fingerprint(saved))
        update(second)
        result = compare(capture(second, 2), expected)
        self.assertTrue(result['passed'])
        self.assertTrue(result['bitwise_equal'])

    def test_checkpoint_provenance_mismatch_rejected_before_mutation(self):
        runtime = toy()
        update(runtime)
        state = capture(runtime, 1)
        for key in ('model_name', 'target_encoding_sha256', 'glove_sha256', 'shared_data_sha256', 'scope', 'source_commit'):
            metadata = protocol(runtime, 'glove')
            metadata[key] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'provenance'):
                restore(runtime, {'baseline_protocol': metadata, 'training_state': state}, 'glove')
            self.assertEqual(fingerprint(capture(runtime, 1)), fingerprint(state))

    def test_noam_or_rng_or_adam_corruption_is_visible(self):
        runtime = toy()
        update(runtime)
        state = capture(runtime, 1)
        for kind in ('noam', 'rng', 'optimizer'):
            damaged = copy.deepcopy(state)
            if kind == 'noam':
                damaged['noam']['_step'] = 0
            elif kind == 'rng':
                damaged['rng']['torch'][0] ^= 1
            else:
                next(iter(damaged['optimizer']['state'].values()))['exp_avg'].add_(1)
            self.assertFalse(compare(damaged, state)['passed'])

    def test_numeric_tolerance_cannot_hide_structural_nonfinite_or_large_errors(self):
        good = {'weights': torch.ones(2), 'counter': 1}
        tiny = {'weights': torch.ones(2) + 1e-7, 'counter': 1}
        self.assertTrue(compare(tiny, good)['passed'])
        self.assertFalse(compare(tiny, good)['bitwise_equal'])
        for bad in ({'weights': torch.ones(3), 'counter': 1},
                    {'weights': torch.ones(2), 'counter': 2},
                    {'weights': torch.full((2,), float('nan')), 'counter': 1},
                    {'weights': torch.ones(2) + 0.1, 'counter': 1}):
            self.assertFalse(compare(bad, good)['passed'])


class GloveTests(unittest.TestCase):
    def test_wrong_glove_file_fails_hash_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fake.txt'
            path.write_text('fake 0 0\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                verify_file(path)

    def test_all_matching_rows_checked_after_initializer(self):
        embedding = types.SimpleNamespace(lut=torch.nn.Embedding(3, 2))
        runtime = types.SimpleNamespace(name='toy', model=types.SimpleNamespace(embedding=embedding),
                                        vocabulary={'word2index': {'hi': 0, 'there': 1, 'UNK': 2}})
        with torch.no_grad():
            embedding.lut.weight[:2].copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vectors.txt'
            path.write_text('hi 0.1 0.2\nthere 0.3 0.4\n', encoding='utf-8')
            result = verify_initialized(runtime, path)
            self.assertEqual(result['matched_rows'], 2)
            self.assertEqual(result['native_random_rows'], 1)
            with torch.no_grad():
                embedding.lut.weight[1, 0] += 0.1
            with self.assertRaisesRegex(ValueError, 'exactly match'):
                verify_initialized(runtime, path)


if __name__ == '__main__':
    unittest.main()
