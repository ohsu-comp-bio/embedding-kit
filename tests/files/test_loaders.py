import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from embkit.files.loaders import load_gct, load_raw_hugo, load_gtex_hugo


class TestLoaders(unittest.TestCase):
    def test_load_gct_transposes_and_drops_description(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "x.gct"
            path.write_text(
                "#1.2\n"
                "2\t2\n"
                "Name\tDescription\tS1\tS2\n"
                "ENSG1\tGene1\t1\t2\n"
                "ENSG2\tGene2\t3\t4\n",
                encoding="utf-8",
            )

            df = load_gct(path)

            self.assertEqual(list(df.index), ["S1", "S2"])
            self.assertEqual(list(df.columns), ["ENSG1", "ENSG2"])
            self.assertEqual(float(df.loc["S1", "ENSG1"]), 1.0)
            self.assertEqual(float(df.loc["S2", "ENSG2"]), 4.0)

    def test_load_gct_with_nrows(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "x.gct"
            path.write_text(
                "#1.2\n"
                "2\t2\n"
                "Name\tDescription\tS1\tS2\n"
                "ENSG1\tGene1\t1\t2\n"
                "ENSG2\tGene2\t3\t4\n",
                encoding="utf-8",
            )

            df = load_gct(path, nrows=1)

            self.assertEqual(list(df.columns), ["ENSG1"])

    def test_load_raw_hugo_and_load_gtex_hugo(self):
        with TemporaryDirectory() as td:
            hugo_path = Path(td) / "hugo.tsv"
            hugo_path.write_text(
                "id\tlocus_group\tsymbol\n"
                "0\tprotein-coding gene\tTP53\n",
                encoding="utf-8",
            )
            gtex_hugo_path = Path(td) / "gtex.hugo.tsv"
            gtex_hugo_path.write_text(
                "sample\tTP53\tBRCA1\n"
                "s1\t1\t2\n"
                "s2\t3\t4\n",
                encoding="utf-8",
            )

            raw = load_raw_hugo(hugo_path)
            converted = load_gtex_hugo(gtex_hugo_path, nrows=1)

            self.assertIn("locus_group", raw.columns)
            self.assertEqual(list(converted.index), ["s1"])
            self.assertEqual(list(converted.columns), ["TP53", "BRCA1"])


if __name__ == "__main__":
    unittest.main()
