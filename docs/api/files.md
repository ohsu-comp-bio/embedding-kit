# Files & CSV API

Large-CSV readers and the open-file pool used to bound how many files stay open
while iterating over a dataset.

`LargeCsvReader` is a line-indexed random-access dataset (a
`torch.utils.data.Dataset`) over a CSV file. It builds a byte-offset index so a
row can be read by integer index without scanning the file, transparently handles
BGZF and gzip-compressed inputs, and — when used inside an
`OpenFilePool` context — shares a single open handle per reader and enforces a
limit on the number of simultaneously open files. This matters under a
`torch` `DataLoader` with multiple workers, where each worker process gets its
own pool.

## LargeCsvReader

::: embkit.files.read_csv.LargeCsvReader
    options:
      show_source: false
      members:
        - __init__
        - read
        - close
      merge_init_into_class: true
      filters:
        - "!^_"

### Indexing and iteration

`LargeCsvReader` implements the `torch` dataset protocol:

- `reader[i]` returns the parsed row at data-line index `i` (negative indices
  count from the end). With a header row present the row is a `CsvRow` and
  supports lookup by column name as well as by position; otherwise it is a plain
  list.
- `len(reader)` is the number of data rows, and `reader.shape` is
  `(rows, columns)`.
- Iterating yields rows in file order; `read(show_progress=True)` yields the
  numeric `float32` values of each row with the first column dropped.

The byte-offset index is persisted to `index_path` (a `.index` JSON file next to
the CSV by default) and reused on the next run. BGZF-compressed files use BGZF
virtual offsets, so random access works without decompressing the whole file per
sample.

## OpenFilePool

::: embkit.files.pool.OpenFilePool
    options:
      show_source: false
      members:
        - __init__
        - current
        - acquire
        - release
        - close
      merge_init_into_class: true
      filters:
        - "!^_"

```python
from embkit.files import OpenFilePool, LargeCsvReader

with OpenFilePool(max_open=32):
    readers = [LargeCsvReader(path) for path in paths]
    rows = [reader[0] for reader in readers]
# all pooled files are closed on exit
```

Readers created inside the context join the pool automatically; least-recently
used entries are evicted when the limit is reached. Each `DataLoader` worker
process maintains its own pool, so the process-wide open-file limit is
`max_open * num_workers`. The pool is fork-aware and picklable, which is what
lets it survive the worker fork used by `DataLoader`.

## CsvReader

A simple streaming reader that yields `(key, row)` pairs keyed on an index
column. Unlike `LargeCsvReader`, it does not build an index and reads the file
once. When `sep` is omitted it is auto-detected from the first line.

::: embkit.files.read_csv.CsvReader
    options:
      show_source: false
      members:
        - __init__
      merge_init_into_class: true
      filters:
        - "!^_"

## CsvRow

A parsed CSV row (a `list` subclass) that additionally supports lookup by column
name when the source had a header.

::: embkit.files.read_csv.CsvRow
    options:
      show_source: false
      filters:
        - "!^_"
