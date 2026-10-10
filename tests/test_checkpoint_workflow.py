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
from run_utils import create_run_directory, save_run, save_json, split_checkpoint
from migrate_checkpoints import migrate_checkpoint
from model import MultiTaskGroupPairModel, token_ids_key
from summarize_runs import run_summary, summary_rows


class CheckpointWorkflowTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.directory = Path(self.temporary.name) / "run"
    self.path = self.directory / "checkpoint.pt"
    self.metrics_path = self.directory / "metrics.json"
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
      "interaction_state_dict": self.model.interaction.state_dict(),
      "pair_mlp_state_dict": self.model.pair_mlp.state_dict(),
      "head_state_dicts": {"interaction": self.model.heads["interaction"].state_dict()},
      "config": {
        "embed_dim": 8, "adapter_dim": 4, "dropout": 0.0,
        "interaction_module": "bidirectional_chain_cross_attention",
        "interaction_hidden_dim": 256, "interaction_heads": 4,
        "calibration": {"source_split": "validation", "classification": {
          "interaction": {"threshold": 0.4, "calibration_size": 3},
        }},
      },
    }
    save_run(self.checkpoint, self.directory)
    for name, value in (("DEVICE", torch.device("cpu")), ("AMP_ENABLED", False), ("PIN_MEMORY", False)):
      mock = patch.object(validate, name, value)
      mock.start()
      self.addCleanup(mock.stop)

  def save_updated_metadata(self):
    _, config, metrics, _ = split_checkpoint(self.checkpoint)
    save_json(config, self.directory / "config.json")
    save_json(metrics, self.metrics_path)

  def evaluate(self, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), patch.object(validate, "tqdm", lambda iterable, **kw: iterable):
      return validate.evaluate_run(self.directory, payload=self.payload, **kwargs)

  def test_both_splits_are_persisted_without_changing_weights(self):
    original_bytes = self.path.read_bytes()
    with patch.object(validate, "_load_embedding_cache", return_value=self.embeddings):
      self.evaluate()
    self.assertEqual(self.path.read_bytes(), original_bytes)
    saved = torch.load(self.path, map_location="cpu")
    import json
    splits = json.loads(self.metrics_path.read_text())["splits"]
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
    for key in ("adapter_state_dict", "residue_pool_state_dict", "group_pool_state_dict", "interaction_state_dict", "pair_mlp_state_dict"):
      for name, weight in self.checkpoint[key].items():
        self.assertTrue(torch.equal(weight, saved[key][name]))
    for name, weight in self.checkpoint["head_state_dicts"]["interaction"].items():
      self.assertTrue(torch.equal(weight, saved["head_state_dicts"]["interaction"][name]))
    self.assertEqual(len(list(summary_rows(run_summary(self.directory)))), 4)

  def test_reused_model_and_single_split_preserve_other_results(self):
    self.checkpoint["evaluation"] = {"splits": {"test": {"sentinel": "keep"}}}
    self.save_updated_metadata()
    with patch.object(validate, "_load_embedding_cache", side_effect=AssertionError("must reuse embeddings")):
      saved = self.evaluate(splits=("validation",), model=self.model, embedding_cache=self.embeddings)
    self.assertEqual(saved["splits"]["test"], {"sentinel": "keep"})

  def test_second_split_failure_does_not_modify_checkpoint(self):
    original = self.metrics_path.read_bytes()
    with patch.object(validate, "_evaluate_split", side_effect=[{}, RuntimeError("test failed")]):
      with self.assertRaisesRegex(RuntimeError, "test failed"):
        self.evaluate(model=self.model, embedding_cache=self.embeddings)
    self.assertEqual(self.metrics_path.read_bytes(), original)

  def test_atomic_save_failure_preserves_metrics(self):
    original = self.metrics_path.read_bytes()
    with patch("run_utils.os.replace", side_effect=RuntimeError("disk error")):
      with self.assertRaisesRegex(RuntimeError, "disk error"):
        save_json({"new": "metrics"}, self.metrics_path)
    self.assertEqual(self.metrics_path.read_bytes(), original)
    self.assertFalse(list(self.directory.glob(".*.tmp")))

  def test_run_names_include_minutes_and_avoid_collisions(self):
    timestamp = datetime(2026, 10, 6, 14, 30, tzinfo=timezone.utc)
    first = create_run_directory(self.directory.parent, 1, timestamp)
    self.assertEqual(first.name, "2026-10-06_14-30_seed_1")
    second = create_run_directory(self.directory.parent, 1, timestamp)
    self.assertEqual(second.name, first.name + "_2")

  def test_legacy_summary_does_not_invent_test_results(self):
    self.checkpoint["config"]["best_task_report"] = {"auroc": 0.8, "balanced_accuracy": 0.7, "tn": 1, "tp": 2}
    self.save_updated_metadata()
    rows = list(summary_rows(run_summary(self.directory)))
    self.assertEqual(rows[0][3], "training-best")
    self.assertEqual(rows[1][3], "not evaluated")

  def test_reevaluation_preserves_historical_reports_and_notes(self):
    archive = {"Regression Tasks": {"raw_table": "Original historical table", "rows": []}}
    notes = {"source": "README.md", "notes_path": "notes.md"}
    self.checkpoint["evaluation"] = {
      "splits": {"validation": {"historical_reports": archive}}, "historical_train_log": notes,
    }
    self.save_updated_metadata()
    saved = self.evaluate(model=self.model, embedding_cache=self.embeddings)
    self.assertEqual(saved["splits"]["validation"]["historical_reports"], archive)
    self.assertEqual(saved["historical_train_log"], notes)

  def test_historical_summary_preserves_missing_fields_and_test_size(self):
    self.checkpoint["evaluation"] = {"splits": {"test": {
      "dataset_size": 4,
      "historical_reports": {"Checkpoint Classification Calibration Applied": {"rows": [
        {"task": "interaction", "cal_n": 3, "thr": 0.4, "bal_acc": 0.7, "auroc": 0.8},
      ]}},
    }}}
    self.save_updated_metadata()
    rows = list(summary_rows(run_summary(self.directory)))
    self.assertEqual(rows[1][3], "historical-calibrated")
    self.assertEqual(rows[1][4], "4")
    self.assertEqual(rows[1][8], "0.7000")
    self.assertEqual(rows[1][9], "-")

  def test_summary_does_not_load_weights(self):
    with patch("torch.load", side_effect=AssertionError("must not load weights")):
      summary = run_summary(self.directory)
    self.assertEqual(summary["run"], str(self.directory))
    self.assertNotIn("config", torch.load(self.path, map_location="cpu"))

  def test_migration_preserves_weights_metrics_and_archived_notes(self):
    source = Path(self.temporary.name) / "prostt5_group_pair_adapter_best_2026-07-13_seed_1.pt"
    self.checkpoint["evaluation"] = {
      "splits": {"validation": {"historical_reports": {"Original": {"raw_table": "Old table"}}}},
      "historical_train_log": {"source": "README.md", "results_notes": "Old conclusions"},
    }
    torch.save(self.checkpoint, source)
    destination = Path(self.temporary.name) / "runs"
    directory = migrate_checkpoint(source, destination, notes="Run notes")
    self.assertTrue(source.exists())
    self.assertEqual(directory.name, "2026-07-13_seed_1")
    self.assertIn("Old conclusions", (directory / "notes.md").read_text())
    self.assertEqual(run_summary(directory)["evaluation"]["splits"], self.checkpoint["evaluation"]["splits"])
    # A repeat verifies the existing run before removing the migrated original.
    self.assertEqual(migrate_checkpoint(source, destination, notes="Run notes", move=True), directory)
    self.assertFalse(source.exists())

  def test_migration_refuses_conflicting_existing_run(self):
    source = Path(self.temporary.name) / "prostt5_group_pair_adapter_best_2026-07-13_seed_1.pt"
    torch.save(self.checkpoint, source)
    directory = migrate_checkpoint(source, Path(self.temporary.name) / "runs")
    save_json({"changed": True}, directory / "config.json")
    with self.assertRaisesRegex(ValueError, "Metadata verification failed"):
      migrate_checkpoint(source, directory.parent, move=True)
    self.assertTrue(source.exists())


if __name__ == "__main__":
  unittest.main()
