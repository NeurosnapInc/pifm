"""Refresh incomplete MMseqs assignments without reusing or publishing stale work."""

import ast
import contextlib
import csv
import hashlib
import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch


class ClusterCacheTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    self.path = self.root / "clusters.tsv"
    self.env = {
      "csv": csv, "hashlib": hashlib, "shutil": shutil, "subprocess": subprocess,
      "tempfile": tempfile, "Path": Path, "List": List, "Dict": Dict,
      "MMSEQS_BINARY": "mmseqs", "CLUSTER_MIN_SEQ_ID": 0.5, "CLUSTER_COVERAGE": 0.8,
      "SEQUENCE_CLUSTER_FASTA_PATH": self.root / "sequences.fasta",
      "SEQUENCE_CLUSTER_TSV_PATH": self.path, "SEQUENCE_CLUSTER_WORK_DIR": self.root / "work",
    }
    names = {"_sequence_id", "_record_sequences", "_all_unique_sequences", "_write_sequence_fasta",
             "_read_cluster_assignments", "_run_mmseqs_clustering", "_load_sequence_clusters"}
    tree = ast.parse(Path("tokenize_data.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), "tokenize_data.py", "exec"), self.env)
    self.ids = {sequence: self.env["_sequence_id"](sequence) for sequence in ("AAAA", "CCCC")}
    self.records = [{"group1_sequences": ["AAAA"], "group2_sequences": ["CCCC"]}]

  def write_clusters(self, path, sequences):
    path.write_text("".join(f"{self.ids[s]}\t{self.ids[s]}\n" for s in sequences))

  def fake_mmseqs(self, command, check):
    self.assertTrue(check)
    self.assertEqual(command[:2], ["mmseqs", "easy-cluster"])
    self.assertEqual(command[5:], ["--min-seq-id", "0.5", "-c", "0.8", "--cov-mode", "0"])
    output = Path(command[3] + "_cluster.tsv")
    self.assertFalse(output.exists())
    self.assertTrue(Path(command[4]).is_dir())
    self.write_clusters(output, ["AAAA", "CCCC"])

  def load(self):
    with contextlib.redirect_stdout(io.StringIO()):
      return self.env["_load_sequence_clusters"](self.records)

  def test_complete_cache_is_reused_without_running_mmseqs(self):
    self.write_clusters(self.path, self.ids)
    before = self.path.read_bytes()
    with patch.dict(self.env, {"_run_mmseqs_clustering": lambda _: self.fail("complete cache must be reused")}):
      self.assertEqual(self.load(), self.ids)
    self.assertEqual(self.path.read_bytes(), before)

  def test_incomplete_cache_reclusters_all_sequences_in_fresh_directories(self):
    self.write_clusters(self.path, ["AAAA"])
    with patch.object(shutil, "which", return_value="/usr/bin/mmseqs"), \
         patch.object(subprocess, "run", side_effect=self.fake_mmseqs) as run:
      self.assertEqual(self.load(), self.ids)
    self.assertEqual(run.call_count, 1)
    fasta = self.env["SEQUENCE_CLUSTER_FASTA_PATH"].read_text()
    self.assertIn("AAAA", fasta)
    self.assertIn("CCCC", fasta)
    self.assertFalse(list(self.root.glob("clusters-*")))
    self.assertFalse(list((self.root / "work").iterdir()))

  def test_missing_cache_is_created(self):
    with patch.object(shutil, "which", return_value="/usr/bin/mmseqs"), \
         patch.object(subprocess, "run", side_effect=self.fake_mmseqs):
      self.assertEqual(self.load(), self.ids)

  def test_incomplete_output_does_not_replace_previous_assignments(self):
    self.write_clusters(self.path, ["AAAA"])
    before = self.path.read_bytes()

    def incomplete(command, check):
      self.write_clusters(Path(command[3] + "_cluster.tsv"), ["AAAA"])

    with patch.object(shutil, "which", return_value="/usr/bin/mmseqs"), \
         patch.object(subprocess, "run", side_effect=incomplete):
      with self.assertRaisesRegex(ValueError, "missing 1 sequences"):
        self.load()
    self.assertEqual(self.path.read_bytes(), before)

  def test_failed_process_preserves_cache_and_cleans_work(self):
    self.write_clusters(self.path, ["AAAA"])
    before = self.path.read_bytes()
    with patch.object(shutil, "which", return_value="/usr/bin/mmseqs"), \
         patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(1, "mmseqs")):
      with self.assertRaises(subprocess.CalledProcessError):
        self.load()
    self.assertEqual(self.path.read_bytes(), before)
    self.assertFalse(list(self.root.glob("clusters-*")))
    self.assertFalse(list((self.root / "work").iterdir()))


if __name__ == "__main__":
  unittest.main()
