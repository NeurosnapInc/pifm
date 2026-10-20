"""Conservative SNOOPPI filtering and existing-source duplicate precedence."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from aggregate_data import aggregate
from contract import InteractionEntry, SourceSpec
from sources import build_source_specs
from sources.snooppi import COLUMNS, _row_to_entry, iter_snooppi


def example(label="positive", **overrides):
  row = dict.fromkeys(COLUMNS)
  row.update({
    "SNOOPPI_final_label": label, "partner_A_sequence": "AAAA", "partner_B_sequence": "CCCC",
    "partner_A_mol_type": "protein", "partner_B_mol_type": "peptide",
    "seq_sort_origin_categories": "exact_seq_sort_seen_in_original_intact",
    "has_binding_site_relationship": "False",
    "exact_sequence_pair_intact_label_summary": label,
    "intact_interaction_ids": "EBI-123",
  })
  row.update(overrides)
  return row


class SnooppiTests(unittest.TestCase):
  def write_partition(self, directory, label, rows):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    with duckdb.connect() as con:
      con.from_df(frame).write_parquet(str(directory / f"{label}-00000-of-00001.parquet"))

  def test_plain_proteins_and_peptides_only(self):
    entry, reason = _row_to_entry(example(), "positive")
    self.assertIsNone(reason)
    self.assertEqual(entry.group1, ["AAAA"])
    self.assertTrue(entry.interaction_label)
    self.assertIsNone(entry.affinity_nm)
    for sequence in ("<S1+MI:0170(phosphorylated residue)>AAA", "AA:CC", "AA_CC", "AA-AA", "AA*", "", "unknown"):
      with self.subTest(sequence=sequence):
        self.assertEqual(_row_to_entry(example(partner_A_sequence=sequence), "positive")[1], "non_plain_sequence")
    self.assertEqual(_row_to_entry(example(partner_A_mol_type="RNA"), "positive")[1], "non_protein")
    self.assertIsNotNone(_row_to_entry(example(partner_A_mol_type="protein | peptide"), "positive")[0])
    self.assertEqual(_row_to_entry(example("unknown"), "positive")[1], "unexpected_label")
    self.assertEqual(_row_to_entry(example("unknown"), "unknown")[1], "unexpected_label")

  def test_negative_requires_direct_or_feature_evidence(self):
    self.assertIsNotNone(_row_to_entry(example("negative"), "negative")[0])
    for original_label in (None, "positive", "conflicting"):
      row = example("negative", exact_sequence_pair_intact_label_summary=original_label)
      self.assertEqual(_row_to_entry(row, "negative")[1], "unsupported_negative")
    mutation = example("negative", partner_A_sequence="AACA",
                       seq_sort_origin_categories="derived_from_mutation_feature",
                       exact_sequence_pair_intact_label_summary=None,
                       feature_relationship_types="mutation",
                       feature_relationship_summary="type=mutation ; feature=Gln3Cys ; modified_label=negative")
    entry, reason = _row_to_entry(mutation, "negative")
    self.assertIsNone(reason)
    self.assertFalse(entry.interaction_label)
    self.assertEqual(entry.group1, ["AACA"])
    mutation["feature_relationship_summary"] = "type=mutation ; wildtype_label=negative ; modified_label=positive"
    self.assertEqual(_row_to_entry(mutation, "negative")[1], "unsupported_negative")
    self.assertEqual(_row_to_entry(example("negative", intact_interaction_ids=None), "negative")[1], "unsupported_negative")

  def test_inferred_binding_negatives_excluded_even_with_other_evidence(self):
    for overrides in (
      {"has_binding_site_relationship": "True"},
      {"feature_relationship_types": "mutation | binding_site"},
      {"seq_sort_origin_categories": "exact_seq_sort_seen_in_original_intact | derived_from_binding_site_feature"},
    ):
      self.assertEqual(_row_to_entry(example("negative", **overrides), "negative")[1], "inferred_negative")

  def test_only_labeled_local_partitions_are_loaded(self):
    with tempfile.TemporaryDirectory() as temporary:
      directory = Path(temporary)
      self.write_partition(directory, "positive", [example()])
      self.write_partition(directory, "negative", [example("negative", partner_A_sequence="DDDD"),
                                                   example("negative", has_binding_site_relationship="True")])
      # No attempt may be made to inspect this invalid unknown-partition file.
      (directory / "unknown-00000-of-00005.parquet").touch()
      with contextlib.redirect_stdout(io.StringIO()) as output:
        entries = list(iter_snooppi(directory))
      self.assertEqual(len(entries), 2)
      self.assertEqual([entry.interaction_label for entry in entries], [True, False])
      self.assertIn("skipped_inferred_negative=1", output.getvalue())

  def test_missing_files_never_trigger_downloads(self):
    with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()) as output:
      self.assertEqual(list(iter_snooppi(Path(temporary))), [])
    self.assertIn("partition=positive skipped", output.getvalue())
    self.assertIn("partition=negative skipped", output.getvalue())

  def test_duplicates_retain_existing_source_and_labels(self):
    self.assertEqual(build_source_specs()[-1].name, "snooppi")
    with tempfile.TemporaryDirectory() as temporary:
      directory = Path(temporary)
      self.write_partition(directory, "positive", [example(partner_B_sequence="DDDD"),
                                                   example(partner_A_sequence="DDDD", partner_B_sequence="AAAA")])
      self.write_partition(directory, "negative", [example("negative", partner_A_sequence="CCCC", partner_B_sequence="AAAA")])
      prior = InteractionEntry(source="existing", group1=["AAAA"], group2=["CCCC"], interaction_label=True)
      specs = [SourceSpec("existing", lambda: [prior]), SourceSpec("snooppi", lambda: iter_snooppi(directory))]
      db = directory / "aggregated.duckdb"
      with contextlib.redirect_stdout(io.StringIO()):
        aggregate(specs, db)
      with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute("SELECT source, group1, group2, interaction_label FROM samples ORDER BY source").fetchall()
      self.assertEqual(rows, [("existing", "AAAA", "CCCC", 1.0), ("snooppi", "AAAA", "DDDD", 1.0)])


if __name__ == "__main__":
  unittest.main()
