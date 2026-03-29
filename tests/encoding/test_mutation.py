import unittest
import pandas as pd

from embkit.encoding.genome import vectorize_variant_count, chromosome_length


class TestVectorizeVariantCount(unittest.TestCase):

    def _make_df(self, rows, seq_col="chr", pos_col="pos"):
        return pd.DataFrame(rows, columns=[seq_col, pos_col])

    # ------------------------------------------------------------------
    # Basic structure / keys
    # ------------------------------------------------------------------

    def test_output_keys_cover_all_chromosomes(self):
        """Every chromosome bin should appear as a key even with an empty dataframe."""
        df = self._make_df([], seq_col="chr", pos_col="pos")
        result = vectorize_variant_count(df)
        # Each chromosome must have at least one bin
        for chrom in chromosome_length:
            self.assertTrue(
                any(k.startswith(f"{chrom}_") for k in result),
                f"No bin found for {chrom}",
            )

    def test_all_counts_zero_for_empty_dataframe(self):
        df = self._make_df([])
        result = vectorize_variant_count(df)
        self.assertTrue(all(v == 0 for v in result.values()))

    def test_bin_labels_are_unique(self):
        df = self._make_df([])
        result = vectorize_variant_count(df)
        keys = list(result.keys())
        self.assertEqual(len(keys), len(set(keys)))

    # ------------------------------------------------------------------
    # Correct bin assignment
    # ------------------------------------------------------------------

    def test_variant_in_first_bin_of_chr1(self):
        """A position in [1, 1_000_000] on chr1 should land in chr1_0."""
        df = self._make_df([("chr1", 500_000)])
        result = vectorize_variant_count(df)
        self.assertEqual(result["chr1_0"], 1)
        # all other bins must be zero
        other_counts = {k: v for k, v in result.items() if k != "chr1_0"}
        self.assertTrue(all(v == 0 for v in other_counts.values()))

    def test_variant_in_second_bin_of_chr1(self):
        """A position well inside the second 1 MB window belongs only to chr1_1."""
        df = self._make_df([("chr1", 1_000_002)])
        result = vectorize_variant_count(df)
        self.assertEqual(result["chr1_1"], 1)
        self.assertEqual(result["chr1_0"], 0)

    def test_bin_boundary_overlap(self):
        """The shared boundary point (start of bin N+1 == end of bin N) is counted
        in *both* adjacent bins because the implementation uses inclusive comparisons
        on both ends."""
        # bin 0: [1, 1_000_001], bin 1: [1_000_001, 2_000_001]
        df = self._make_df([("chr1", 1_000_001)])
        result = vectorize_variant_count(df)
        self.assertEqual(result["chr1_0"], 1)
        self.assertEqual(result["chr1_1"], 1)

    def test_bin_boundary_inclusive_start(self):
        """The start of each bin (i) is inclusive."""
        df = self._make_df([("chr1", 1)])  # first position in the genome
        result = vectorize_variant_count(df)
        self.assertEqual(result["chr1_0"], 1)

    def test_bin_boundary_inclusive_end(self):
        """The end of the first bin (1_000_000) must be counted in chr1_0."""
        df = self._make_df([("chr1", 1_000_000)])
        result = vectorize_variant_count(df)
        self.assertEqual(result["chr1_0"], 1)
        self.assertEqual(result["chr1_1"], 0)

    def test_multiple_variants_same_bin(self):
        df = self._make_df([("chr2", 100), ("chr2", 200), ("chr2", 999_999)])
        result = vectorize_variant_count(df)
        self.assertEqual(result["chr2_0"], 3)

    def test_variants_across_multiple_chromosomes(self):
        df = self._make_df([
            ("chr1", 500_000),
            ("chrX", 1_000),
            ("chrY", 50_000_000),
        ])
        result = vectorize_variant_count(df)
        self.assertEqual(result["chr1_0"], 1)
        self.assertEqual(result["chrX_0"], 1)
        # chrY position 50_000_000 is in bin index 49 (0-based, bin_size=1M)
        self.assertEqual(result["chrY_49"], 1)

    # ------------------------------------------------------------------
    # Custom bin_size
    # ------------------------------------------------------------------

    def test_custom_bin_size(self):
        """With bin_size=500_000 a position well inside the second window is in chr1_1."""
        df = self._make_df([("chr1", 500_002)])
        result = vectorize_variant_count(df, bin_size=500_000)
        self.assertEqual(result["chr1_1"], 1)
        self.assertEqual(result["chr1_0"], 0)

    # ------------------------------------------------------------------
    # Custom column names
    # ------------------------------------------------------------------

    def test_custom_column_names(self):
        df = pd.DataFrame({"chrom": ["chr1"], "position": [500_000]})
        result = vectorize_variant_count(df, seq_col="chrom", pos_col="position")
        self.assertEqual(result["chr1_0"], 1)

    # ------------------------------------------------------------------
    # Return type
    # ------------------------------------------------------------------

    def test_returns_dict(self):
        df = self._make_df([])
        result = vectorize_variant_count(df)
        self.assertIsInstance(result, dict)

    def test_values_are_integers(self):
        df = self._make_df([("chr1", 1)])
        result = vectorize_variant_count(df)
        for v in result.values():
            self.assertIsInstance(v, int)


if __name__ == "__main__":
    unittest.main()
