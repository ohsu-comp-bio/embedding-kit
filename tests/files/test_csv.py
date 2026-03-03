import unittest
import os
import tempfile

from embkit.files import CsvReader


class TestCsvReader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        # CSV with header row
        self.csv_with_header = os.path.join(self.test_dir.name, "with_header.csv")
        with open(self.csv_with_header, 'w') as f:
            f.write("id,name,value\n")
            f.write("row1,Alice,10\n")
            f.write("row2,Bob,20\n")
            f.write("row3,Carol,30\n")

        # CSV without header row
        self.csv_no_header = os.path.join(self.test_dir.name, "no_header.csv")
        with open(self.csv_no_header, 'w') as f:
            f.write("row1,Alice,10\n")
            f.write("row2,Bob,20\n")
            f.write("row3,Carol,30\n")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_default_header_integer_index(self):
        reader = CsvReader(self.csv_with_header, index_column=0)
        rows = dict(reader)
        self.assertEqual(set(rows.keys()), {'row1', 'row2', 'row3'})
        self.assertEqual(rows['row1'], {'name': 'Alice', 'value': '10'})
        self.assertEqual(rows['row2'], {'name': 'Bob', 'value': '20'})

    def test_default_header_string_index(self):
        reader = CsvReader(self.csv_with_header, index_column='id')
        rows = dict(reader)
        self.assertEqual(set(rows.keys()), {'row1', 'row2', 'row3'})
        self.assertEqual(rows['row3'], {'name': 'Carol', 'value': '30'})

    def test_header_none_returns_list(self):
        reader = CsvReader(self.csv_no_header, index_column=0, header=None)
        rows = dict(reader)
        self.assertEqual(set(rows.keys()), {'row1', 'row2', 'row3'})
        self.assertIsInstance(rows['row1'], list)
        self.assertEqual(rows['row1'], ['Alice', '10'])

    def test_header_none_with_string_column_raises(self):
        reader = CsvReader(self.csv_no_header, index_column='id', header=None)
        with self.assertRaises(ValueError):
            list(reader)

    def test_string_column_not_in_header_raises(self):
        reader = CsvReader(self.csv_with_header, index_column='nonexistent')
        with self.assertRaises(ValueError):
            list(reader)

    def test_non_zero_index_column(self):
        reader = CsvReader(self.csv_with_header, index_column=1)
        rows = dict(reader)
        self.assertEqual(set(rows.keys()), {'Alice', 'Bob', 'Carol'})
        self.assertEqual(rows['Alice'], {'id': 'row1', 'value': '10'})

    def test_empty_file(self):
        empty_csv = os.path.join(self.test_dir.name, "empty.csv")
        with open(empty_csv, 'w') as f:
            pass
        reader = CsvReader(empty_csv, index_column=0)
        rows = list(reader)
        self.assertEqual(rows, [])

    def test_tab_separator(self):
        tsv_path = os.path.join(self.test_dir.name, "data.tsv")
        with open(tsv_path, 'w') as f:
            f.write("id\tname\tvalue\n")
            f.write("row1\tAlice\t10\n")
        reader = CsvReader(tsv_path, index_column='id', sep='\t')
        rows = dict(reader)
        self.assertEqual(rows['row1'], {'name': 'Alice', 'value': '10'})


if __name__ == '__main__':
    unittest.main()
