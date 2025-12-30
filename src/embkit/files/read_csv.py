import csv
import json
import os
import functools
import numpy as np

from tqdm import tqdm


class LargeCsvReader:
    """
    A class for performing key-value random access on large CSV files
    by creating and using a byte-offset index. The index can be stored
    in a file for persistence or kept in memory.
    """
    def __init__(self, csv_path, index_column, index_path=None, sep=',', skip_header=False, save_index=True, cache_size=None):
        """
        Initializes the reader. Creates an index file if one does not exist,
        or loads it if it does, based on the `save_index` flag.

        Args:
            csv_path (str): The path to the large CSV file.
            index_column (str or int): The name or 0-based index of the column to use as a key.
            index_path (str, optional): The path to store the index file.
                                        Defaults to a .index JSON file next to the CSV.
            sep (str, optional): The delimiter used in the CSV file. Defaults to ','.
            skip_header (bool, optional): Whether to skip the first line as a header.
                                          Defaults to False.
            save_index (bool, optional): If True, saves the index to disk.
                                            If False, keeps it in memory only.
        """
        self.csv_path = csv_path
        self.index_column = index_column
        self.sep = sep
        self.skip_header = skip_header
        self.save_index = save_index
        self.index_path = index_path or f"{csv_path}.index"

        if cache_size is None:
            self.get = self._get
        else:
            self.get = functools.lru_cache(cache_size)(self._get)
        
        self._index = {}
        self._header = None
        self._file = None

        # Check if a persistent index exists and we are allowed to use it
        if self.save_index and os.path.exists(self.index_path):
            self._load_index()
        else:
            self._generate_index()

        row_count = len(self._index)
        # note: this assumes all rows are the same length
        with self:
            col_count = len(self.get( next(iter(self._index)) ) )
        self.shape = (row_count, col_count)


    def _get_index_column_and_header(self, f):
        """
        Reads the header (if skipping) and determines the key column index.
        """
        if not self.skip_header:
            header_line = f.readline()
            self._header = header_line.strip().split(self.sep)
        else:
            self._header = None # No header to store

        # Determine the key column index
        if isinstance(self.index_column, str):
            if not self._header:
                raise ValueError("Cannot specify a column name for indexing if `skip_header` is False.")
            try:
                key_column_index = self._header.index(self.index_column)
            except ValueError:
                raise ValueError(f"Key column '{self.index_column}' not found in CSV header.")
        else:
            key_column_index = self.index_column
        
        return key_column_index

    def _generate_index(self):
        """Generates the byte-offset index from the CSV file."""
        print(f"Generating index for '{self.csv_path}'...")
        with open(self.csv_path, 'r', newline='') as f:
            
            key_column_index = self._get_index_column_and_header(f)
            
            current_position = f.tell()
            while True:
                line = f.readline()
                if not line:
                    break
                row = line.strip().split(self.sep)
                if len(row) > key_column_index:
                    key = row[key_column_index]
                    self._index[key] = current_position
                
                current_position = f.tell()

        # Only save the index if persistence is enabled
        if self.save_index:
            with open(self.index_path, 'w') as f:
                json.dump(self._index, f)
            print("Index generation and saving complete.")
        else:
            print("Index generated in memory only.")

    def _load_index(self):
        """Loads a pre-existing index file into memory."""
        print(f"Loading index from '{self.index_path}'...")
        with open(self.index_path, 'r') as f:
            self._index = json.load(f)
        
        # We need the header for dictionary lookups, so re-read it.
        if not self.skip_header:
            with open(self.csv_path, 'r', newline='') as f:
                self._header = f.readline().strip().split(self.sep)
        
        print("Index loaded.")

    def __enter__(self):
        """Opens the CSV file for reading when entering a context manager."""
        self._file = open(self.csv_path, 'r', newline='')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the CSV file when exiting a context manager."""
        if self._file:
            self._file.close()

    def __iter__(self):
        if not self._file or self._file.closed:
            raise RuntimeError("File not open. Use 'with' statement or manually manage file lifecycle.")

        self._file.seek(0)            
        key_column_index = self._get_index_column_and_header(self._file)
        while True:
            line = self._file.readline()
            if not line:
                break
            reader = csv.reader([line], delimiter=self.sep)
            row = next(reader)
            if len(row) > key_column_index:
                key = row[key_column_index]
                if self._header:
                    yield (key, zip(self._header, row))
                else:
                    yield (key, row)

    def _get(self, key):
        """
        Retrieves a row from the CSV file based on the given key.

        Args:
            key: The key to look up.

        Returns:
            list or None: A list of the row's values, or None if the key is not found.
        """
        # Ensure key is always treated as string to match index
        key_str = str(key)
        if key_str not in self._index:
            return None

        if not self._file or self._file.closed:
            raise RuntimeError("File not open. Use 'with' statement or manually manage file lifecycle.")

        self._file.seek(self._index[key_str])
        line = self._file.readline()
        
        # Use a CSV reader for correct parsing
        reader = csv.reader([line], delimiter=self.sep)
        return next(reader)

    def get_dict(self, key):
        """
        Retrieves a row as a dictionary, with column headers as keys.

        Args:
            key: The key to look up.

        Returns:
            dict or None: A dictionary of the row's values, or None if the key is not found.
        """
        if not self._header:
            raise ValueError("Cannot return a dictionary if `skip_header` was False.")
            
        row_list = self.get(key)
        if row_list is None:
            return None
        
        return dict(zip(self._header, row_list))
    
    def read(self, show_progress=False):
        with self:
            if show_progress:
                for k, v in tqdm(self, total=self.shape[0]):
                    yield np.array(v[1:], dtype=np.float32)
            else:
                for k, v in self:
                    yield np.array(v[1:], dtype=np.float32)

# Example Usage
if __name__ == '__main__':
    dummy_csv_path = 'large_data.csv'
    if not os.path.exists(dummy_csv_path):
        with open(dummy_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'username', 'email'])
            for i in range(1000):
                writer.writerow([i, f'user_{i}', f'user_{i}@example.com'])
    
    # Example 1: Use persistent index (default behavior)
    print("--- Using persistent index ---")
    # This will generate and save the index the first time it runs.
    with LargeCsvReader(dummy_csv_path, index_column='user_id') as reader:
        row = reader.get('500')
        print(f"Persistent index lookup: {row}")

    # Subsequent runs will load the saved index instantly.
    with LargeCsvReader(dummy_csv_path, index_column='user_id') as reader:
        row = reader.get('501')
        print(f"Reloaded persistent index lookup: {row}")
    
    # Clean up the persistent index file for the next demo
    os.remove(dummy_csv_path + ".index.json")


    # Example 2: Use in-memory index only (pass save_index=False)
    print("\n--- Using in-memory index ---")
    with LargeCsvReader(dummy_csv_path, index_column='user_id', save_index=False) as reader:
        # This will regenerate the index every time the class is instantiated.
        # No file will be saved.
        row = reader.get('500')
        print(f"In-memory index lookup: {row}")

    with LargeCsvReader(dummy_csv_path, index_column='user_id', save_index=False) as reader:
        # Re-creating the object triggers another index generation.
        row = reader.get('501')
        print(f"Another in-memory index lookup: {row}")
        
    
    # check csv reader read function
    print("\n--- Reading all rows ---")
    with LargeCsvReader(dummy_csv_path, index_column='user_id', save_index=False) as reader:
        for arr in reader.read(show_progress=True):
            pass
        print("Completed reading all rows.")

