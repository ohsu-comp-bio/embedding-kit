import os
import tempfile
import unittest

import numpy as np

from embkit.files.read_csv import LargeCsvReader


class TestLargeCsvReader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "x.tsv")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("id\tA\tB\n")
            f.write("r1\t1\t2\n")
            f.write("r2\t3\t4\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_and_get_dict(self):
        reader = LargeCsvReader(self.path, sep="\t", index_column="id", skip_header=False, save_index=False)
        with reader:
            row = reader.get("r1")
            self.assertEqual(row, ["r1", "1", "2"])
            self.assertEqual(reader.get("missing"), None)
            d = reader.get_dict("r2")
            self.assertEqual(d["A"], "3")
            self.assertEqual(d["B"], "4")
            arr = list(reader.read(show_progress=False))
        self.assertEqual(len(arr), 2)
        self.assertTrue(np.allclose(arr[0], np.array([1.0, 2.0], dtype=np.float32)))

    def test_iter_requires_context(self):
        reader = LargeCsvReader(self.path, sep="\t", index_column=0, skip_header=False, save_index=False)
        with self.assertRaises(RuntimeError):
            list(reader)

    def test_load_existing_index(self):
        reader = LargeCsvReader(self.path, sep="\t", index_column=0, skip_header=False, save_index=True)
        self.assertTrue(os.path.exists(self.path + ".index"))
        reader2 = LargeCsvReader(self.path, sep="\t", index_column=0, skip_header=False, save_index=True)
        with reader2:
            self.assertEqual(reader2.get("r2"), ["r2", "3", "4"])


if __name__ == "__main__":
    unittest.main()
