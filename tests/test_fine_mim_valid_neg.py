"""生产端细粒度 MIM 供体有效性掩码的纯 CPU 回归测试（无需加载数据集与完整模型）。"""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/models/CASE/model.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
case = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CASE")
names = {"build_react_slot_valid_mask", "fine_grained_infomax_score"}
methods = [n for n in case.body if isinstance(n, ast.FunctionDef) and n.name in names]
namespace = {"torch": torch, "config": SimpleNamespace(PAD_idx=0)}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
fine = namespace["fine_grained_infomax_score"]
slot_mask = namespace["build_react_slot_valid_mask"]


class FineMIMValidNegTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.model = SimpleNamespace(
            use_fine_mim_valid_neg=True,
            criterion_bce_fine=torch.nn.BCEWithLogitsLoss(reduction="none"),
        )
        self.c, self.cf, self.r, self.rf = [
            torch.randn(3, 4, 5, requires_grad=True) for _ in range(4)
        ]

    def loss(self, c_mask, r_mask):
        return fine(self.model, self.c, self.cf, self.r, self.rf, c_mask, r_mask)

    def test_slot_order_and_missing_clauses(self):
        batch = {
            "react_batch": torch.tensor([
                [[1, 0], [2, 0], [3, 0]],
                [[0, 0], [0, 0], [4, 0]],
            ]),
            "uttr_split_react_mask": torch.tensor([[1, 0], [1, 1]]),
            "max_uttr_cs_num": 1,
            "split_intent_num": 2, "split_need_num": 1,
            "split_want_num": 0, "split_effect_num": 1,
            "cs_mask": torch.ones(2, 9),
        }
        actual = slot_mask(self.model, batch)
        expected = torch.tensor([
            [1, 1, 1, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 1, 0, 1],
        ], dtype=torch.float32)
        self.assertTrue(torch.equal(actual, expected))
        batch["react_batch"] = batch["react_batch"][:, :1]
        batch["uttr_split_react_mask"] = torch.empty(2, 0, dtype=torch.long)
        batch["cs_mask"] = torch.ones(2, 1)
        self.assertEqual(tuple(slot_mask(self.model, batch).shape), (2, 1))

    def test_all_valid_matches_legacy_value_and_gradients(self):
        mask = torch.ones(3, 4)
        new = self.loss(mask, mask)
        g_new = torch.autograd.grad(new, (self.c, self.cf, self.r, self.rf), retain_graph=True)
        self.model.use_fine_mim_valid_neg = False
        old = self.loss(mask, None)
        g_old = torch.autograd.grad(old, (self.c, self.cf, self.r, self.rf))
        self.assertTrue(torch.equal(old, new))
        for a, b in zip(g_new, g_old):
            self.assertTrue(torch.equal(a, b))

    def test_each_direction_uses_its_own_donor_and_denominator(self):
        c_mask = torch.tensor([[1., 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0]])
        r_mask = torch.tensor([[1., 0, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]])
        actual = self.loss(c_mask, r_mask)
        # 标量参考实现：独立于向量化掩码枚举供体 j = (i - 1) mod B。
        direction1, direction2 = [], []
        for i in range(3):
            j = (i - 1) % 3
            for k in range(4):
                if not c_mask[i, k]:
                    continue
                pos = torch.sigmoid(torch.dot(self.c[i, k], self.r[i, k]))
                if r_mask[j, k]:
                    neg = torch.sigmoid(torch.dot(self.c[i, k], self.rf[i, k]))
                    direction1.append(F.softplus(-torch.sigmoid(pos - neg)))
                if c_mask[j, k]:
                    neg = torch.sigmoid(torch.dot(self.cf[i, k], self.r[i, k]))
                    direction2.append(F.softplus(-torch.sigmoid(pos - neg)))
        expected = torch.stack(direction1).mean() + torch.stack(direction2).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        m1 = c_mask * r_mask.roll(1, 0)
        m2 = c_mask * c_mask.roll(1, 0)
        self.assertTrue(torch.equal(self.rf.grad[m1 == 0], torch.zeros_like(self.rf.grad[m1 == 0])))
        self.assertTrue(torch.equal(self.cf.grad[m2 == 0], torch.zeros_like(self.cf.grad[m2 == 0])))

    def test_no_valid_pairs_return_differentiable_zero(self):
        # 仅第 0 行有正槽位，其错位供体第 2 行无有效常识与反应。
        c_mask = torch.zeros(3, 4)
        c_mask[0] = 1
        value = self.loss(c_mask, torch.zeros(3, 4))
        self.assertEqual(value.item(), 0.0)
        value.backward()
        for t in (self.c, self.cf, self.r, self.rf):
            self.assertTrue(torch.equal(t.grad, torch.zeros_like(t)))

    def test_one_empty_direction_preserves_other_direction(self):
        value = self.loss(torch.ones(3, 4), torch.zeros(3, 4))
        pos = torch.sigmoid((self.c * self.r).sum(-1))
        neg = torch.sigmoid((self.cf * self.r).sum(-1))
        expected = F.softplus(-torch.sigmoid(pos - neg)).mean()
        torch.testing.assert_close(value, expected)
        value.backward()
        self.assertTrue(torch.equal(self.rf.grad, torch.zeros_like(self.rf)))
        self.assertTrue(torch.isfinite(self.cf.grad).all())


if __name__ == "__main__":
    unittest.main()
