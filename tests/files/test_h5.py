import unittest
import os
import numpy as np
import torch
from embkit.files import H5Writer, H5Reader
import tempfile

class TestH5(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.filename = os.path.join(self.test_dir.name, "test.h5")
        self.group = "test_group"
        self.index = ["row1", "row2", "row3"]
        self.columns = ["col1", "col2"]
        self.data = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], dtype='f')

    def tearDown(self):
        self.test_dir.cleanup()

    def test_writer_and_reader(self):
        # Test writing
        writer = H5Writer(self.filename, self.group, self.index, self.columns)
        for i, row in enumerate(self.data):
            writer.set_irow(i, row)
        writer.close()

        # Test reading
        reader = H5Reader(self.filename, self.group)
        self.assertEqual(len(reader), 3)
        self.assertEqual(list(reader.index), self.index)
        self.assertEqual(list(reader.columns), self.columns)
        self.assertEqual(reader.shape, (3, 2))

        # Check data
        for i in range(len(reader)):
            row_tensor, *_ = reader[i]  # Unpack the tuple
            expected_tensor = torch.from_numpy(self.data[i])
            self.assertTrue(torch.allclose(row_tensor, expected_tensor))


    def test_writer_set_row_by_name(self):
        writer = H5Writer(self.filename, self.group, self.index, self.columns)
        writer.set_row("row2", self.data[1])
        writer.set_row("row1", self.data[0])
        writer.set_row("row3", self.data[2])
        writer.close()

        reader = H5Reader(self.filename, self.group)
        row_tensor, *_ = reader[0]
        self.assertTrue(torch.allclose(row_tensor, torch.from_numpy(self.data[0])))
        row_tensor, *_ = reader[1]
        self.assertTrue(torch.allclose(row_tensor, torch.from_numpy(self.data[1])))
        row_tensor, *_ = reader[2]
        self.assertTrue(torch.allclose(row_tensor, torch.from_numpy(self.data[2])))

if __name__ == "__main__":
    unittest.main()
