"""Exercise evaluation persistence with tiny cached embeddings and no network access."""

import contextlib
import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

import validate
from checkpoint_utils import checkpoint_path, save_checkpoint
from model import MultiTaskGroupPairModel, token_ids_key
from summarize_checkpoints import checkpoint_summary, summary_rows


class CheckpointWorkflowTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.path = Path(self.temporary.name) / "weights.pt"
    torch.manual_seed(1)
    self.meta = {"interaction": {"task_name": "interaction", "dtype": "bool", "num_classes": 2}}
    self.model = MultiTaskGroupPairModel(None, ["interaction"], {"interaction": 2}, 8, adapter_dim=4, dropout=0.0)
    ids = torch.tensor([1, 2, 3])
    self.embeddings = {token_ids_key(ids): torch.randn(3, 8)}

    def split(labels):
      n = len(labels)
      raw = torch.tensor(labels, dtype=torch.float).reshape(n, 1)
      return {
        "group1_input_ids": [[ids] for _ in labels],
        "group2_input_ids": [[ids, ids] for _ in labels],
        "lengths": [9] * n,
        "raw_labels": raw,
        "normalized_labels": raw.clone(),
        "label_mask": torch.ones((n, 1), dtype=torch.bool),
        "sources": ["positive" if label else "negative" for label in labels],
      }

    self.payload = {
      "task_order": ["interaction"], "task_metas": self.meta, "config": {"pad_token_id": 0},
      "splits": {"train": split([0, 1]), "validation": split([0, 1, 1]), "test": split([0, 0, 1, 1])},
    }
    self.checkpoint = {
      "adapter_state_dict": self.model.adapter.state_dict(),
      "residue_pool_state_dict": self.model.residue_pool.state_dict(),
      "group_pool_state_dict": self.model.group_pool.state_dict(),
      "pair_mlp_state_dict": self.model.pair_mlp.state_dict(),
      "head_state_dicts": {"interaction": self.model.heads["interaction"].state_dict()},
      "config": {
        "embed_dim": 8, "adapter_dim": 4, "dropout": 0.0,
        "calibration": {"source_split": "validation", "classification": {
          "interaction": {"threshold": 0.4, "calibration_size": 3},
        }},
      },
    }
    save_checkpoint(self.checkpoint, self.path)
    for name, value in (("DEVICE", torch.device("cpu")), ("AMP_ENABLED", False), ("PIN_MEMORY", False)):
      mock = patch.object(validate, name, value)
      mock.start()
      self.addCleanup(mock.stop)

  def evaluate(self, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), patch.object(validate, "tqdm", lambda iterable, **kw: iterable):
      return validate.evaluate_checkpoint(self.path, payload=self.payload, **kwargs)

  def test_both_splits_are_persisted_without_changing_weights(self):
    with patch.object(validate, "_load_embedding_cache", return_value=self.embeddings):
      self.evaluate()
    saved = torch.load(self.path, map_location="cpu")
    splits = saved["evaluation"]["splits"]
    self.assertEqual(set(splits), {"validation", "test"})
    for split, size in (("validation", 3), ("test", 4)):
      result = splits[split]
      self.assertEqual(result["dataset_size"], size)
      self.assertEqual(result["classification"]["interaction"]["n"], size)
      self.assertEqual(set(result["by_source"]), {"positive", "negative"})
      calibrated = result["calibrated_classification"]["interaction"]
      self.assertEqual(calibrated["threshold"], 0.4)
      self.assertEqual(calibrated["calibration_size"], 3)
      self.assertEqual(calibrated["calibration_split"], "validation")
    for key in ("adapter_state_dict", "residue_pool_state_dict", "group_pool_state_dict", "pair_mlp_state_dict"):
      for name, weight in self.checkpoint[key].items():
        self.assertTrue(torch.equal(weight, saved[key][name]))
    for name, weight in self.checkpoint["head_state_dicts"]["interaction"].items():
      self.assertTrue(torch.equal(weight, saved["head_state_dicts"]["interaction"][name]))
    self.assertEqual(len(list(summary_rows(checkpoint_summary(self.path)))), 4)

  def test_reused_model_and_single_split_preserve_other_results(self):
    self.checkpoint["evaluation"] = {"splits": {"test": {"sentinel": "keep"}}}
    save_checkpoint(self.checkpoint, self.path)
    with patch.object(validate, "_load_embedding_cache", side_effect=AssertionError("must reuse embeddings")):
      saved = self.evaluate(splits=("validation",), model=self.model, embedding_cache=self.embeddings)
    self.assertEqual(saved["evaluation"]["splits"]["test"], {"sentinel": "keep"})

  def test_second_split_failure_does_not_modify_checkpoint(self):
    original = self.path.read_bytes()
    with patch.object(validate, "_evaluate_split", side_effect=[{}, RuntimeError("test failed")]):
      with self.assertRaisesRegex(RuntimeError, "test failed"):
        self.evaluate(model=self.model, embedding_cache=self.embeddings)
    self.assertEqual(self.path.read_bytes(), original)

  def test_atomic_save_failure_preserves_checkpoint(self):
    original = self.path.read_bytes()
    with patch("checkpoint_utils.torch.save", side_effect=RuntimeError("disk error")):
      with self.assertRaisesRegex(RuntimeError, "disk error"):
        save_checkpoint(self.checkpoint, self.path)
    self.assertEqual(self.path.read_bytes(), original)
    self.assertEqual(list(self.path.parent.iterdir()), [self.path])

  def test_checkpoint_names_include_minutes_and_avoid_collisions(self):
    timestamp = datetime(2026, 10, 6, 14, 30, tzinfo=timezone.utc)
    first = checkpoint_path(self.path.parent, 1, timestamp)
    self.assertEqual(first.name, "prostt5_group_pair_adapter_best_2026-10-06_14-30_seed_1.pt")
    save_checkpoint({}, first)
    second = checkpoint_path(self.path.parent, 1, timestamp)
    self.assertEqual(second.stem, first.stem + "_2")

  def test_legacy_summary_does_not_invent_test_results(self):
    self.checkpoint["config"]["best_task_report"] = {"auroc": 0.8, "balanced_accuracy": 0.7, "tn": 1, "tp": 2}
    save_checkpoint(self.checkpoint, self.path)
    rows = list(summary_rows(checkpoint_summary(self.path)))
    self.assertEqual(rows[0][3], "training-best")
    self.assertEqual(rows[1][3], "not evaluated")

  def test_reevaluation_preserves_historical_reports_and_notes(self):
    archive = {"Regression Tasks": {"raw_table": "Original historical table", "rows": []}}
    notes = {"results_notes": "Original result notes"}
    self.checkpoint["evaluation"] = {
      "splits": {"validation": {"historical_reports": archive}}, "historical_train_log": notes,
    }
    save_checkpoint(self.checkpoint, self.path)
    saved = self.evaluate(model=self.model, embedding_cache=self.embeddings)
    self.assertEqual(saved["evaluation"]["splits"]["validation"]["historical_reports"], archive)
    self.assertEqual(saved["evaluation"]["historical_train_log"], notes)

  def test_historical_summary_preserves_missing_fields_and_test_size(self):
    self.checkpoint["evaluation"] = {"splits": {"test": {
      "dataset_size": 4,
      "historical_reports": {"Checkpoint Classification Calibration Applied": {"rows": [
        {"task": "interaction", "cal_n": 3, "thr": 0.4, "bal_acc": 0.7, "auroc": 0.8},
      ]}},
    }}}
    save_checkpoint(self.checkpoint, self.path)
    rows = list(summary_rows(checkpoint_summary(self.path)))
    self.assertEqual(rows[1][3], "historical-calibrated")
    self.assertEqual(rows[1][4], "4")
    self.assertEqual(rows[1][8], "0.7000")
    self.assertEqual(rows[1][9], "-")


if __name__ == "__main__":
  unittest.main()
