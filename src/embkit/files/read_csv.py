import csv
import gzip
import json
import os
import operator
import numpy as np
import logging

from .pool import OpenFilePool

from Bio import bgzf
from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CsvRow(list):
    """A parsed CSV row that also supports lookup by column name."""
    def __init__(self, values, columns):
        super().__init__(values)
        self._columns = {name: idx for idx, name in enumerate(columns)}

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return super().__getitem__(self._columns[idx])
        return super().__getitem__(idx)


class CsvReader:
    """
    A simple CSV reader that yields key-value pairs based on a specified index column. 
    This class is designed to stream the CSV file one row at a time.
    
    Args:
        csv_path (str): The path to the CSV file.
        index_column (str or int): The name or 0-based index of the column to use as a key.
        sep (str, optional): The delimiter used in the CSV file. When omitted,
                             it is detected from the first line.
        header (int or None, optional): Row number to use as column names, similar to
            pandas.read_csv. If 0 (default), the first row is used as the header and
            rows are yielded as dicts. If None, there is no header row and rows are
            yielded as lists.
    """
    def __init__(self, csv_path, index_column, sep=',', header=0):
        self.csv_path = csv_path
        self.index_column = index_column
        self.sep = sep
        self.header = header
        self._header = None

    def __iter__(self):
        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=self.sep)

            if self.header is not None:
                # Read the first row as column names
                try:
                    self._header = next(reader)
                except StopIteration:
                    return
                # Resolve string column name to integer index
                if isinstance(self.index_column, str):
                    try:
                        key_index = self._header.index(self.index_column)
                    except ValueError:
                        raise ValueError(f"Column name '{self.index_column}' not found in CSV header.")
                else:
                    key_index = self.index_column
            else:
                self._header = None
                if isinstance(self.index_column, str):
                    raise ValueError("Cannot use a string column name when header=None.")
                key_index = self.index_column

            for row in reader:
                if key_index >= len(row):
                    continue
                key = row[key_index]
                values_without_key = [v for i, v in enumerate(row) if i != key_index]
                if self._header is not None:
                    header_without_key = [h for i, h in enumerate(self._header) if i != key_index]
                    yield (key, dict(zip(header_without_key, values_without_key)))
                else:
                    yield (key, values_without_key)
                    

class LargeCsvReader(Dataset):
    """
    A line-indexed PyTorch dataset for large CSV files.

    The reader builds byte offsets for integer access. BGZF-compressed files
    (the bgzip format) are detected by their header and use BGZF virtual
    offsets, allowing random access without decompressing the whole file for
    every sample.
    """
    def __init__(self, csv_path, index_path=None, sep=None, skip_header=False, save_index=True):
        """
        Initializes the reader. Creates an index file if one does not exist,
        or loads it if it does, based on the `save_index` flag.

        Args:
            csv_path (str): The path to the large CSV file.
            index_path (str, optional): The path to store the index file.
                                        Defaults to a .index JSON file next to the CSV.
            sep (str, optional): The delimiter used in the CSV file. When omitted,
                                 it is detected from the first line.
            skip_header (bool, optional): Whether to skip the first line as a header.
                                          Defaults to False.
            save_index (bool, optional): If True, saves the index to disk.
                                            If False, keeps it in memory only.
        """
        self.csv_path = csv_path
        self.sep = sep
        self.skip_header = skip_header
        self.save_index = save_index
        self.index_path = index_path or f"{csv_path}.index"

        self._line_offsets = []
        self._header = None
        self._file = None
        self._file_pid = None
        self._file_pool = OpenFilePool.current()
        self._is_bgzip = self._detect_bgzip()
        self._is_gzip = self._detect_gzip()

        # Check if a persistent index exists and we are allowed to use it
        if self.save_index and os.path.exists(self.index_path):
            self._load_index()
        else:
            self._generate_index()

        row_count = len(self._line_offsets)
        if row_count:
            with self._open() as f:
                f.seek(self._line_offsets[0])
                col_count = len(self._parse_row(f.readline()))
        else:
            col_count = len(self._header or [])
        self.shape = (row_count, col_count)

    def _detect_bgzip(self):
        with open(self.csv_path, "rb") as f:
            header = f.read(18)
        return (
            len(header) >= 18
            and header[:4] == b"\x1f\x8b\x08\x04"
            and header[10:12] == b"\x06\x00"
            and header[12:16] == b"BC\x02\x00"
        )

    def _detect_gzip(self):
        with open(self.csv_path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"

    def _open(self):
        if self._is_bgzip:
            return bgzf.open(self.csv_path, "rt")
        if self._is_gzip:
            return gzip.open(self.csv_path, "rt", newline="")
        return open(self.csv_path, "r", newline="")

    def _get_file(self):
        pid = os.getpid()
        if self._file is not None and self._file_pid != pid:
            self.close()

        pool = OpenFilePool.current()
        if pool is None and self._file_pool is not None and self._file_pool.active:
            pool = self._file_pool

        if pool is not None:
            if self._file_pool is not pool:
                self.close()
                self._file_pool = pool
            return pool.acquire(self)

        if self._file_pool is not None:
            self.close()
            self._file_pool = None

        if self._file is None:
            self._file = self._open()
            self._file_pid = pid
        return self._file

    def _adopt_pool_file(self, pool, f):
        self._file_pool = pool
        self._file = f
        self._file_pid = os.getpid()

    def _release_pool_file(self, f):
        if self._file is f:
            self._file = None
            self._file_pid = None

    def _configure_separator(self, f):
        if self.sep is not None:
            return

        position = f.tell()
        sample = f.readline()
        f.seek(position)
        if sample:
            try:
                self.sep = csv.Sniffer().sniff(
                    sample, delimiters=",\t;|"
                ).delimiter
                return
            except csv.Error:
                pass
        self.sep = ","

    def _read_header(self, f):
        """Read and store the header when one is expected."""
        self._configure_separator(f)
        if not self.skip_header:
            header_line = f.readline()
            self._header = next(csv.reader([header_line], delimiter=self.sep)) if header_line else []
        else:
            self._header = None

    def _parse_row(self, line):
        row = next(csv.reader([line], delimiter=self.sep))
        if self._header is not None:
            return CsvRow(row, self._header)
        return row

    def _generate_index(self):
        """Generates the byte-offset index from the CSV file."""
        logger.info("Generating index for '%s'...", self.csv_path)
        with self._open() as f:
            self._read_header(f)
            
            current_position = f.tell()
            while True:
                line = f.readline()
                if not line:
                    break
                self._line_offsets.append(current_position)
                current_position = f.tell()

        # Only save the index if persistence is enabled
        if self.save_index:
            self._save_index()
            logger.info("Index generation and saving complete.")
        else:
            logger.info("Index generated in memory only.")

    def _save_index(self):
        with open(self.index_path, 'w') as f:
            json.dump(self._line_offsets, f)

    def _load_index(self):
        """Loads a pre-existing index file into memory."""
        logger.info("Loading index from '%s'...", self.index_path)
        with open(self.index_path, 'r') as f:
            self._line_offsets = json.load(f)

        with self._open() as f:
            self._read_header(f)
        
        logger.info("Index loaded.")

    def __iter__(self):
        if not self._line_offsets:
            return
        with self._open() as f:
            f.seek(self._line_offsets[0])
            for _ in self._line_offsets:
                line = f.readline()
                yield self._parse_row(line)

    def __len__(self):
        return len(self._line_offsets)

    def __getitem__(self, idx):
        """Return the parsed CSV row at a data-line index."""
        try:
            idx = operator.index(idx)
        except TypeError:
            raise TypeError("CSV dataset indices must be integers") from None

        if idx < 0:
            idx += len(self._line_offsets)
        if idx < 0 or idx >= len(self._line_offsets):
            raise IndexError("CSV dataset index out of range")

        f = self._get_file()
        offset = self._line_offsets[idx]
        if f.tell() != offset:
            f.seek(offset)
        line = f.readline()
        return self._parse_row(line)

    def close(self):
        """Close the lazily opened dataset file, if any."""
        pool = self._file_pool
        if pool is not None:
            pool.release(self)
        if self._file is not None:
            self._file.close()
            self._file = None
            self._file_pid = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_file_pid"] = None
        return state

    def read(self, show_progress=False):
        rows = self
        if show_progress:
            rows = tqdm(
                self,
                total=len(self),
                disable=os.environ.get("GITHUB_ACTIONS") == "true",
            )
        for row in rows:
            yield np.array(row[1:], dtype=np.float32)
