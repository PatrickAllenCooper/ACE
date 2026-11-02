"""Tiny numerical training checks. No pretrained model or scientific worlds."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from llm_intervention_learning import preference_loss
from llm_intervention_ranker import CandidateRanker, feature_tensor, fit_ranker, load_ranker, save_ranker, supervised_rank_loss


torch.set_num_threads(1)


class RankerTests(unittest.TestCase):
    def test_probability_and_preference_gradient_oracles(self):
        logits = torch.tensor([1., 2., 0.], requires_grad=True)
        self.assertAlmostEqual(float(supervised_rank_loss(logits, 0).detach()),
                               math.log(math.exp(1)+math.exp(2)+1)-1, places=6)
        logp = F.log_softmax(logits, dim=0)
        pair = logp[:2].unsqueeze(0)
        reference = pair.detach().clone().requires_grad_(True)
        loss = preference_loss(pair, reference, .5)
        self.assertAlmostEqual(float(loss.detach()), math.log(2), places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([-.25, .25, 0.]))
        self.assertIsNone(reference.grad)

    def test_seed_isolation_and_candidate_batch_order(self):
        torch.manual_seed(77)
        before = torch.get_rng_state().clone()
        model = CandidateRanker(3, hidden=8, seed=2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        x = torch.tensor([[1., 2., 3.], [-2., 1., 4.], [0., 0., 0.]])
        output = model(x)
        torch.testing.assert_close(model(x[[2, 0, 1]]), output[[2, 0, 1]])
        torch.testing.assert_close(model(x[:1]), output[:1])

    def test_fixed_training_reference_and_independent_checkpoint_pins(self):
        example = {"features": np.array([[1., 0.], [0., 1.], [0., 0.]]),
                   "chosen": 0, "pairs": ((0, 1),)}
        before = example["features"].copy()
        model, reference, receipt = fit_ranker((example,), scope="artificial", seed=3,
            hidden=8, sft_epochs=12, preference_epochs=4, learning_rate=.03, beta=.1)
        np.testing.assert_array_equal(example["features"], before)
        self.assertEqual(len(receipt["events"]), 16)
        self.assertTrue(all(math.isfinite(e["loss"]) and e["gradient_norm"] > 0 for e in receipt["events"]))
        self.assertLess(receipt["events"][11]["loss"], receipt["events"][0]["loss"])
        self.assertTrue(all(not p.requires_grad for p in reference.parameters()))
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(model.parameters(), reference.parameters())))
        original = {k: v.clone() for k, v in reference.state_dict().items()}
        with torch.no_grad():
            next(model.parameters()).add_(.01)
        self.assertTrue(all(torch.equal(v, reference.state_dict()[k]) for k, v in original.items()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"ranker.pt"
            pin = save_ranker(model, path, training_receipt_sha256="a"*64)
            restored = load_ranker(path, sha256=pin, training_receipt_sha256="a"*64, feature_width=2)
            x = feature_tensor(example["features"])
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
            for changes in ({"sha256": "b"*64}, {"training_receipt_sha256": "b"*64}, {"feature_width": 3}):
                with self.assertRaises(ValueError):
                    load_ranker(path, **{"sha256": pin, "training_receipt_sha256": "a"*64,
                                        "feature_width": 2, **changes})
            with self.assertRaises(FileExistsError):
                save_ranker(model, path, training_receipt_sha256="a"*64)

    def test_no_private_training_or_missing_preference_substitution(self):
        example = {"features": np.array([[0., 1.], [1., 0.]]), "chosen": 0, "pairs": ()}
        for scope in ("development", "confirmation"):
            with self.assertRaises(ValueError):
                fit_ranker((example,), scope=scope, seed=0)
        with self.assertRaises(ValueError):
            fit_ranker((example,), scope="artificial", seed=0, preference_epochs=1)
        for pairs in (((0, 0),), ((0, 1), (1, 0)), ((0, 1), (0, 1)), ((True, 1),)):
            bad = dict(example, pairs=pairs)
            with self.assertRaises(ValueError):
                fit_ranker((bad,), scope="artificial", seed=0)


if __name__ == "__main__":
    unittest.main()
