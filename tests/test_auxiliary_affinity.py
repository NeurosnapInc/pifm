"""Check auxiliary supervision without changing the matched classifier baseline."""

import ast
import copy
import math
from pathlib import Path
import unittest

import torch

from auxiliary_affinity import affinity_report, affinity_statistics, masked_affinity_loss, verify_classifier_cache
from model import MultiTaskGroupPairDataset, MultiTaskGroupPairModel, collate_multitask_batch


class AuxiliaryAffinityTests(unittest.TestCase):
  def split(self):
    ids = torch.tensor([1, 2, 3])
    labels = torch.tensor([[1.0], [0.0], [1.0]])
    return {
      "group1_input_ids": [[ids]] * 3, "group2_input_ids": [[ids]] * 3,
      "raw_labels": labels, "normalized_labels": labels.clone(),
      "label_mask": torch.ones(3, 1, dtype=torch.bool), "lengths": torch.tensor([6, 6, 6]),
      "sources": ["ppb_affinity", "negatome", "skempi"],
      "affinity_pkd": torch.tensor([5.0, 0.0, 9.0]),
      "affinity_mask": torch.tensor([True, False, True]),
    }

  def test_train_only_normalization_and_sparse_collation(self):
    split = self.split()
    stats = affinity_statistics(split)
    self.assertEqual((stats["mean"], stats["std"], stats["n"]), (7.0, 2.0, 2))
    self.assertEqual(stats["fit_split"], "train")
    batch = collate_multitask_batch(list(MultiTaskGroupPairDataset(split)), 0, include_sources=True)
    torch.testing.assert_close(batch[5][:, 0], split["raw_labels"][:, 0])
    torch.testing.assert_close(batch[7][:, 1], split["affinity_mask"])
    with self.assertRaisesRegex(ValueError, "Run tokenize_data.py"):
      affinity_statistics({})

  def test_masked_huber_ignores_missing_targets_and_backpropagates(self):
    stats = affinity_statistics(self.split())
    features = torch.tensor([0.0, 3.0, 0.0], requires_grad=True)
    predictions = features * 2
    targets = torch.tensor([5.0, float("nan"), 9.0])
    loss = masked_affinity_loss(predictions, targets, torch.tensor([True, False, True]), stats)
    self.assertEqual(loss.item(), 0.5)
    loss.backward()
    torch.testing.assert_close(features.grad, torch.tensor([1.0, 0.0, -1.0]))
    missing = torch.randn(3, requires_grad=True)
    zero = masked_affinity_loss(missing, targets, torch.zeros(3, dtype=torch.bool), stats)
    self.assertEqual(zero.item(), 0.0)
    zero.backward()
    torch.testing.assert_close(missing.grad, torch.zeros(3))

  def test_auxiliary_initialization_and_forward_preserve_classifier_rng(self):
    torch.manual_seed(68)
    baseline = MultiTaskGroupPairModel(None, ["interaction"], {"interaction": 2}, 8)
    state = torch.get_rng_state().clone()
    torch.manual_seed(68)
    auxiliary = MultiTaskGroupPairModel(None, ["interaction"], {"interaction": 2}, 8, auxiliary_affinity=True)
    self.assertTrue(torch.equal(state, torch.get_rng_state()))
    for key, tensor in baseline.state_dict().items():
      torch.testing.assert_close(tensor, auxiliary.state_dict()[key], rtol=0, atol=0)
    embeddings = torch.randn(4, 3, 8)
    inputs = (None, torch.ones(4, 3, dtype=torch.long), torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]), 2)
    torch.manual_seed(9)
    expected = baseline(*inputs, precomputed_embeddings=embeddings)["interaction"]
    state = torch.get_rng_state().clone()
    torch.manual_seed(9)
    outputs = auxiliary(*inputs, precomputed_embeddings=embeddings)
    torch.testing.assert_close(outputs["interaction"], expected, rtol=0, atol=0)
    self.assertTrue(torch.equal(state, torch.get_rng_state()))
    masked_affinity_loss(outputs["auxiliary_affinity"], torch.tensor([5.0, 9.0]), torch.ones(2, dtype=torch.bool),
                         affinity_statistics(self.split())).backward()
    self.assertGreater(auxiliary.pair_mlp[1].weight.grad.abs().sum().item(), 0)
    self.assertGreater(auxiliary.adapter.up_project.weight.grad.abs().sum().item(), 0)

  def test_cache_guard_preserves_classifier_examples(self):
    current = {"task_order": ["interaction"], "splits": {"train": self.split()}}
    previous = copy.deepcopy(current)
    del previous["splits"]["train"]["affinity_pkd"]
    del previous["splits"]["train"]["affinity_mask"]
    verify_classifier_cache(previous, current)
    changed = copy.deepcopy(current)
    changed["splits"]["train"]["group1_input_ids"][0][0][0] = 99
    with self.assertRaisesRegex(ValueError, "original cache was not overwritten"):
      verify_classifier_cache(previous, changed)
    changed = copy.deepcopy(current)
    changed["splits"]["train"]["raw_labels"][0, 0] = 0
    with self.assertRaises(ValueError):
      verify_classifier_cache(previous, changed)

  def test_tokenizer_preserves_existing_pkd_units_and_missingness(self):
    # Load just the builder, without triggering tokenizer downloads or DB reads.
    tree = ast.parse(Path("tokenize_data.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_tokenized_split")
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "tokenize_data.py", "exec"), namespace)
    records = [{"group1_sequences": ["A"], "group2_sequences": ["B"], "labels": [1.0], "mask": [True],
                "source": "ppb_affinity", "affinity_pkd": affinity} for affinity in (9.0, None, float("nan"))]
    result = namespace["_build_tokenized_split"](records, {"A": torch.tensor([1]), "B": torch.tensor([2])})
    torch.testing.assert_close(result["affinity_pkd"], torch.tensor([9.0, 0.0, 0.0]))
    torch.testing.assert_close(result["affinity_mask"], torch.tensor([True, False, False]))

  def test_diagnostics_are_in_pkd_and_source_specific(self):
    report = affinity_report([5.0, 9.0], [6.0, 7.0], ["ppb_affinity", "skempi"])
    self.assertEqual(report["n"], 2)
    self.assertEqual(report["mae"], 1.5)
    self.assertEqual(report["by_source"]["skempi"]["mae"], 2.0)
    self.assertIsNone(report["by_source"]["skempi"]["pearson"])
    self.assertIsNone(affinity_report([], [], [])["mae"])


if __name__ == "__main__":
  unittest.main()
