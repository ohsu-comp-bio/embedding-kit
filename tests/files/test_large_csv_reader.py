import os
import gzip
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np
from Bio import bgzf
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from embkit.files.pool import OpenFilePool
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

    def test_read_and_iterate(self):
        reader = LargeCsvReader(self.path, sep="\t", skip_header=False, save_index=False)

        self.assertEqual(list(reader), [["r1", "1", "2"], ["r2", "3", "4"]])
        arr = list(reader.read(show_progress=False))
        self.assertEqual(len(arr), 2)
        self.assertTrue(np.allclose(arr[0], np.array([1.0, 2.0], dtype=np.float32)))

    def test_load_existing_index(self):
        reader = LargeCsvReader(self.path, sep="\t", skip_header=False, save_index=True)
        self.assertTrue(os.path.exists(self.path + ".index"))
        reader2 = LargeCsvReader(self.path, sep="\t", skip_header=False, save_index=True)

        self.assertEqual(reader2[1], ["r2", "3", "4"])

    def test_dataset_line_indexing_offsets_header(self):
        reader = LargeCsvReader(self.path, sep="\t", skip_header=False, save_index=False)

        self.assertIsInstance(reader, Dataset)
        self.assertEqual(len(reader), 2)
        self.assertEqual(reader[0], ["r1", "1", "2"])
        self.assertEqual(reader[-1], ["r2", "3", "4"])
        with self.assertRaises(IndexError):
            reader[2]

    def test_bgzip_dataset_supports_random_access_and_dataloader(self):
        compressed_path = os.path.join(self.tmp.name, "x.tsv.bgz")
        with open(self.path, "rb") as source, bgzf.open(compressed_path, "wb") as target:
            target.write(source.read())

        reader = LargeCsvReader(
            compressed_path,
            sep="\t",
            skip_header=False,
            save_index=True,
        )

        with OpenFilePool(max_open=1) as pool:
            with mock.patch.object(reader, "_open", wraps=reader._open) as open_file:
                self.assertEqual(reader[1], ["r2", "3", "4"])
                batches = list(DataLoader(reader, batch_size=2, num_workers=0))
                self.assertEqual(open_file.call_count, 1)
            self.assertEqual(pool.open_count, 1)
        self.assertEqual(pool.open_count, 0)

        reloaded = LargeCsvReader(
            compressed_path,
            sep="\t",
            skip_header=False,
            save_index=True,
        )
        self.assertEqual(reloaded[0], ["r1", "1", "2"])

        self.assertEqual(
            [list(column) for column in batches[0]],
            [["r1", "r2"], ["1", "3"], ["2", "4"]],
        )

    def test_regular_gzip_with_bgz_suffix_supports_random_access(self):
        compressed_path = os.path.join(self.tmp.name, "x.tsv.bgz")
        with open(self.path, "rb") as source, gzip.open(compressed_path, "wb") as target:
            target.write(source.read())

        reader = LargeCsvReader(
            compressed_path, skip_header=False, save_index=False
        )

        self.assertFalse(reader._is_bgzip)
        self.assertTrue(reader._is_gzip)
        self.assertEqual(reader.sep, "\t")
        self.assertEqual(reader[1], ["r2", "3", "4"])
        self.assertEqual(reader[1]["A"], "3")
        self.assertEqual(reader[0], ["r1", "1", "2"])

    def test_dataset_without_header_starts_at_first_line(self):
        path = os.path.join(self.tmp.name, "without-header.tsv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("r1\t1\t2\n")
            f.write("r2\t3\t4\n")

        reader = LargeCsvReader(path, sep="\t", skip_header=True, save_index=False)

        self.assertEqual(reader[0], ["r1", "1", "2"])
        self.assertEqual(reader.shape, (2, 3))

    def test_sequential_getitem_reuses_handle_and_skips_seek(self):
        reader = LargeCsvReader(
            self.path, sep="\t", skip_header=True, save_index=False
        )
        original_open = reader._open
        handles = []

        class TrackingFile:
            def __init__(self, file):
                self.file = file
                self.seek_count = 0

            @property
            def closed(self):
                return self.file.closed

            def tell(self):
                return self.file.tell()

            def seek(self, offset):
                self.seek_count += 1
                return self.file.seek(offset)

            def readline(self):
                return self.file.readline()

            def close(self):
                self.file.close()

        def tracked_open():
            handle = TrackingFile(original_open())
            handles.append(handle)
            return handle

        reader._open = tracked_open
        reader[0]
        reader[1]

        self.assertEqual(len(handles), 1)
        self.assertEqual(handles[0].seek_count, 0)

        reader[0]
        self.assertEqual(len(handles), 1)
        self.assertEqual(handles[0].seek_count, 1)

        reader.close()
        self.assertTrue(handles[0].closed)

    def test_handle_lifecycle_is_context_and_worker_safe(self):
        reader = LargeCsvReader(
            self.path, sep="\t", skip_header=False, save_index=False
        )
        reader[0]
        inherited_file = reader._file

        with mock.patch(
            "embkit.files.read_csv.os.getpid",
            return_value=reader._file_pid + 1,
        ):
            reader[1]

        self.assertTrue(inherited_file.closed)
        self.assertIsNot(reader._file, inherited_file)

        restored = pickle.loads(pickle.dumps(reader))
        self.assertIsNone(restored._file)
        self.assertIsNone(restored._file_pid)

        with restored:
            restored[0]
            restored_file = restored._file
        self.assertTrue(restored_file.closed)
        self.assertIsNone(restored._file)

    def test_dataloader_workers_get_independent_handles(self):
        reader = LargeCsvReader(
            self.path, sep="\t", skip_header=False, save_index=False
        )
        self.assertEqual(reader[0], ["r1", "1", "2"])

        batches = list(DataLoader(reader, batch_size=1, num_workers=2))

        self.assertEqual(
            [[values[0] for values in batch] for batch in batches],
            [["r1", "1", "2"], ["r2", "3", "4"]],
        )

    def test_open_file_pool_uses_lru_eviction_and_closes_on_exit(self):
        readers = [
            LargeCsvReader(
                self.path, sep="\t", skip_header=False, save_index=False
            )
            for _ in range(3)
        ]

        with OpenFilePool(max_open=2) as pool:
            readers[0][0]
            readers[1][0]
            readers[0][1]
            readers[2][0]

            self.assertEqual(pool.open_count, 2)
            self.assertIsNotNone(readers[0]._file)
            self.assertIsNone(readers[1]._file)
            self.assertIsNotNone(readers[2]._file)

            opened_handles = [readers[0]._file, readers[2]._file]

        self.assertEqual(pool.open_count, 0)
        self.assertTrue(all(handle.closed for handle in opened_handles))
        self.assertTrue(all(reader._file is None for reader in readers))

    def test_open_file_pool_closes_after_exception(self):
        reader = LargeCsvReader(
            self.path, sep="\t", skip_header=False, save_index=False
        )

        with self.assertRaisesRegex(RuntimeError, "stop"):
            with OpenFilePool(max_open=1):
                reader[0]
                opened_handle = reader._file
                raise RuntimeError("stop")

        self.assertTrue(opened_handle.closed)
        self.assertIsNone(reader._file)

    def test_reader_returns_to_standalone_lifecycle_after_pool(self):
        reader = LargeCsvReader(
            self.path, sep="\t", skip_header=False, save_index=False
        )

        with OpenFilePool(max_open=1):
            self.assertEqual(reader[0], ["r1", "1", "2"])

        self.assertEqual(reader[1], ["r2", "3", "4"])
        standalone_handle = reader._file
        self.assertIsNotNone(standalone_handle)

        reader.close()
        self.assertTrue(standalone_handle.closed)

    def test_open_file_pool_is_process_safe(self):
        reader = LargeCsvReader(
            self.path, sep="\t", skip_header=False, save_index=False
        )

        with OpenFilePool(max_open=1) as pool:
            reader[0]
            inherited_file = reader._file
            inherited_pid = reader._file_pid

            with mock.patch(
                "embkit.files.read_csv.os.getpid",
                return_value=inherited_pid + 1,
            ):
                reader[1]
                self.assertEqual(pool.open_count, 1)

            self.assertTrue(inherited_file.closed)
            self.assertIsNot(reader._file, inherited_file)

    def test_open_file_pool_is_pickle_safe(self):
        reader = LargeCsvReader(
            self.path, sep="\t", skip_header=False, save_index=False
        )

        with OpenFilePool(max_open=1):
            reader[0]
            copied = pickle.loads(pickle.dumps(reader))

            self.assertIsNone(copied._file)

        self.assertIsNone(reader._file)
        self.assertTrue(copied._file_pool.active)
        self.assertEqual(copied[1], ["r2", "3", "4"])
        copied.close()
        self.assertIsNone(copied._file)

    def test_open_file_pool_survives_dataloader_worker_serialization(self):
        readers = [
            LargeCsvReader(
                self.path, sep="\t", skip_header=False, save_index=False
            )
            for _ in range(2)
        ]

        with OpenFilePool(max_open=1):
            dataset = ConcatDataset(readers)
            batches = list(DataLoader(dataset, batch_size=1, num_workers=2))

        self.assertEqual(
            [[values[0] for values in batch] for batch in batches],
            [["r1", "1", "2"], ["r2", "3", "4"]] * 2,
        )

    def test_open_file_pool_rejects_invalid_limits(self):
        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    OpenFilePool(max_open=invalid)

        with self.assertRaises(TypeError):
            OpenFilePool(max_open="2")


if __name__ == "__main__":
    unittest.main()
