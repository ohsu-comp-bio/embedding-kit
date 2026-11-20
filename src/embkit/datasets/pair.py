"""
Pairwise co-occurrence datasets.

Builds a probabilistic, lazy dataset over a *population* of source datasets
(each of which is an iterable of rows). Two "items" (identified by a
user-supplied ``hash_item`` callable) are considered **co-occurring** when
both items' keys are present in the same source dataset. The
:class:`PairwiseDataset` exposes a fixed-size sample of such pairs, with the
co-occurrence percentage computed lazily on first access and cached.

Public classes:
- :class:`DatasetKeyBloom` — per-dataset key membership (Bloom filter).
- :class:`PairBloomFilter` — pairwise membership across all pairs of one dataset.
- :class:`PairwiseDataset` — PyTorch ``Dataset`` of (item_a, item_b, pct) samples.
"""

from hashlib import sha256
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

__all__ = ["DatasetKeyBloom", "PairBloomFilter", "PairwiseDataset"]


def hash_sha256(obj):
    """Fold any object into a stable signed 64-bit integer (first 16 bytes
    of its SHA-256 of ``str(obj)``). Used to give ``rbloom`` an integer
    hash of user keys of any hashable type."""
    h = sha256(str(obj).encode()).digest()
    return int.from_bytes(h[:16], "big", signed=True)


def _canonical_pair(a: Any, b: Any) -> Tuple[Any, Any]:
    """Return a canonical (order-insensitive) pair key from two items.

    Uses ``repr`` comparison as the ordering so it works on any hashable
    pair (str, int, bytes, …). For non-comparable types it falls back to
    id-based ordering; the result is still deterministic for a given (a, b).
    """
    if a == b:
        return (a, a)
    try:
        return (a, b) if a < b else (b, a)
    except TypeError:
        # non-comparable types: fall back to repr ordering
        ra, rb = repr(a), repr(b)
        return (a, b) if ra <= rb else (b, a)


class DatasetKeyBloom:
    """
    A per-dataset Bloom filter over the *keys* of rows in a source dataset.

    A row is any item in the dataset; the user-provided ``hash_item`` is
    called on each row to extract the key that is then inserted into the
    Bloom filter. This is the "pre-indexing" step: after construction,
    membership tests for a key are O(1) with no further I/O to the source
    dataset.

    Hashing convention:
      * ``hash_item`` is a **key extractor** ``(row) -> key``. It is applied
        only at add-time, when iterating the source dataset. At query time,
        the caller passes the *key* directly (i.e. the value they would get
        from ``hash_item(row)`` for some row).
      * The extracted key is folded to a stable integer via
        :func:`embkit.datasets.pair.hash_sha256` for the underlying
        ``rbloom.Bloom``, so any hashable key type (str, int, tuple, bytes)
        works.

    Args:
        dataset: An iterable of rows (e.g. a ``LargeCsvReader``). It will be
            fully iterated exactly once at construction.
        hash_item: A callable ``(row) -> key``. Defaults to the identity
            (rows treated as their own keys).
        bloom_capacity: Expected number of distinct keys in the dataset.
            Controls the Bloom filter size.
        bloom_error_rate: Target false-positive rate (default 0.01).
        bloom_file: Optional path to load a pre-existing filter from
            (skips the iteration). Must have been built with the same
            ``hash_item``.
    """

    def __init__(self, dataset, hash_item: Optional[Callable[[Any], Any]] = None,
                 bloom_capacity: int = 1000, bloom_error_rate: float = 0.01,
                 bloom_file: Optional[str] = None):
        from rbloom import Bloom  # local import so module import stays light
        self._hash_item = hash_item if hash_item is not None else (lambda x: x)
        # rbloom requires an integer hash; fold user keys through sha256.
        # Applied to the KEY (not the row) at both add-time and query-time.
        self._rbloom_hash = hash_sha256
        if bloom_file is not None:
            self._bloom = Bloom.load(bloom_file, self._rbloom_hash)
        else:
            self._bloom = Bloom(
                expected_items=bloom_capacity,
                false_positive_rate=bloom_error_rate,
                hash_func=self._rbloom_hash,
            )
            # Iterate fully so the filter is complete; the dataset may not
            # be re-iterable, so this single pass is authoritative.
            # Extract the key from each row and add the key (rbloom hashes
            # the key via hash_sha256 internally).
            for row in dataset:
                self._bloom.add(self._hash_item(row))
        self._size = self._bloom.size_in_bits

    def has(self, key: Any) -> bool:
        """True (possibly with a small false-positive rate) if the key was added."""
        return key in self._bloom

    def __contains__(self, key: Any) -> bool:
        return self.has(key)

    def save(self, bloom_file: str) -> None:
        self._bloom.save(bloom_file)

    def size_in_bits(self) -> int:
        return self._size

    def __len__(self) -> int:
        """Number of bits in the underlying Bloom filter (not items)."""
        return self._size


class PairBloomFilter:
    """
    A single Bloom filter over all unordered (a, b) pairs of *rows* in one
    source dataset.

    Useful as a negative-example generator: if a pair (a, b) is NOT in the
    filter (and the false-positive rate is low), then a and b very likely
    never co-occurred in the source dataset.

    Args:
        dataset: An iterable of rows. Each pair (row_i, row_j) with i < j
            contributes one entry to the filter.
        hash_item: A callable ``(row) -> key`` applied to each row before
            the pair is formed. Defaults to the identity (rows treated as
            their own keys). Should match the ``hash_item`` used on the
            datasets this filter is compared against.
        bloom_capacity: Upper bound on the number of distinct pairs. For a
            dataset of length L, the true count is L * (L - 1) // 2.
        bloom_error_rate: Target false-positive rate (default 0.01).
        bloom_file: Optional path to load a pre-existing filter.
    """

    def __init__(self, dataset, hash_item: Optional[Callable[[Any], Any]] = None,
                 bloom_capacity: Optional[int] = None, bloom_error_rate: float = 0.01,
                 bloom_file: Optional[str] = None):
        from rbloom import Bloom
        self._hash_item = hash_item if hash_item is not None else (lambda x: x)
        if bloom_file is not None:
            self._bloom = Bloom.load(bloom_file, self._pair_hash)
        else:
            # Default capacity: derive from dataset length if possible.
            if bloom_capacity is None:
                try:
                    n = len(dataset)
                except TypeError:
                    n = 1000
                bloom_capacity = max(1, n * (n - 1) // 2)
            self._bloom = Bloom(
                expected_items=bloom_capacity,
                false_positive_rate=bloom_error_rate,
                hash_func=self._pair_hash,
            )
            # Materialize all rows once (the dataset may not be re-iterable).
            rows = list(dataset)
            n = len(rows)
            for i in range(n):
                key_i = self._hash_item(rows[i])
                for j in range(i + 1, n):
                    key_j = self._hash_item(rows[j])
                    self._bloom.add(_canonical_pair(key_i, key_j))
        self._size = self._bloom.size_in_bits

    def _pair_hash(self, pair: Tuple[Any, Any]) -> int:
        # rbloom calls this with the item we .add()-ed, i.e. the canonical
        # (a, b) tuple. Keys were already passed through hash_item at add
        # time; here we fold the tuple into a single int for rbloom.
        from hashlib import sha256
        a, b = pair
        h = sha256(repr(a).encode() + b"\x00" + repr(b).encode()).digest()
        return int.from_bytes(h[:16], "big", signed=True)

    def has(self, key_a: Any, key_b: Any) -> bool:
        """True (possibly with a small false-positive rate) if (a, b) was added."""
        return _canonical_pair(key_a, key_b) in self._bloom

    def __contains__(self, key_a: Any) -> bool:
        raise TypeError("PairBloomFilter supports (a, b) membership via .has(a, b)")

    def save(self, bloom_file: str) -> None:
        self._bloom.save(bloom_file)

    def size_in_bits(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size


class PairwiseDataset(Dataset):
    """
    A PyTorch ``Dataset`` that yields (item_a, item_b, percentage) samples,
    where ``percentage`` is the co-occurrence rate of ``item_a`` and
    ``item_b`` across a population of source datasets.

    The dataset is built from:
      * a list of source datasets (each an iterable of rows, e.g. a
        ``LargeCsvReader``);
      * a user-supplied ``hash_item`` callable that maps a row to its key
        (e.g. ``lambda row: row['id']``);
      * a requested number of pair slots ``n_pairs``.

    Construction is O(sum(len(d) for d in datasets)) for the per-dataset
    Bloom pre-indexing, plus O(n_pairs) for the pair plan. Frequency
    computation is lazy: ``__getitem__(i)`` computes the percentage on
    first access and caches it.

    Args:
        datasets: A non-empty list of source datasets.
        n_pairs: Number of pair slots to expose (this is ``len(self)``).
        hash_item: A callable ``(row) -> key`` mapping a row to the key
            used for pairing. Defaults to the identity (rows treated as
            their own keys).
        seed: Base RNG seed for the pair plan. Defaults to 0.
        denominator: ``"any"`` (default) or ``"total"``.
            * ``"any"``: percentage = D_both / D_either (co-occurrence
              conditional on at least one key being present).
            * ``"total"``: percentage = D_both / N (raw corpus frequency).
        cache_file: Optional path to a pickled cache (plan + computed
            percentages + denominator mode). Loaded at construction if it
            exists; written on ``save_cache``.
        pairwise_bloom: Optional :class:`PairBloomFilter`. When provided,
            pairs NOT in the filter are treated as negative examples and
            their percentage is forced to 0.0. (Optional; default off.)
    """
    def __init__(self, datasets, n_pairs: int,
                 hash_item: Optional[Callable[[Any], Any]] = None,
                 seed: int = 0,
                 denominator: str = "any",
                 cache_file: Optional[str] = None,
                 pairwise_bloom: Optional["PairBloomFilter"] = None):
        # 1. Validate inputs.
        if not isinstance(datasets, (list, tuple)) or len(datasets) == 0:
            raise ValueError(
                "PairwiseDataset requires a non-empty list of source datasets."
            )
        if not isinstance(n_pairs, int) or n_pairs <= 0:
            raise ValueError(
                "PairwiseDataset requires a positive integer n_pairs."
            )
        if denominator not in ("any", "total"):
            raise ValueError(
                "denominator must be 'any' or 'total', got %r." % (denominator,)
            )
        for d in datasets:
            try:
                len(d)
            except TypeError:
                raise ValueError(
                    "Every source dataset must support len() and __getitem__."
                )

        self.datasets = list(datasets)
        self.n_pairs = n_pairs
        self._hash_item = hash_item if hash_item is not None else (lambda x: x)
        self.seed = seed
        self.denominator = denominator
        self._pair_bloom = pairwise_bloom

        # 2. Per-dataset key pre-indexing (Bloom over row keys). Datasets
        #    that already expose .has() (e.g. a DatasetKeyBloom) are reused.
        key_blooms = []
        for d in self.datasets:
            if callable(getattr(d, "has", None)):
                key_blooms.append(d)
            else:
                key_blooms.append(DatasetKeyBloom(d, hash_item=self._hash_item))
        self._key_blooms = key_blooms

        # 3. Deterministic pair plan.
        self._pair_plan = self._build_pair_plan()
        self._pair_cache = {}

        # 4. Optional pickle cache.
        if cache_file is not None:
            import os
            self._cache_file = cache_file
            if os.path.exists(cache_file):
                self.load_cache(cache_file)
        else:
            self._cache_file = None

    def _build_pair_plan(self) -> List[Tuple[int, int, int]]:
        """Build a deterministic, seeded list of (ds_idx, row_a, row_b) slots.

        Each slot picks a source dataset weighted by its length, then two
        distinct random row indices from that dataset. The resulting plan is
        reproducible from (seed, dataset lengths) alone.
        """
        import numpy as np
        lengths = [len(d) for d in self.datasets]
        if any(l < 2 for l in lengths):
            # Need at least 2 rows per dataset to form a pair.
            # Filter to datasets with >= 2 rows for sampling; if none, raise.
            usable = [(i, l) for i, l in enumerate(lengths) if l >= 2]
            if not usable:
                raise ValueError(
                    "PairwiseDataset requires at least one source dataset with "
                    ">= 2 rows to form pairs."
                )
        else:
            usable = list(enumerate(lengths))

        ds_indices = [i for i, _ in usable]
        ds_lengths = [l for _, l in usable]
        total = sum(ds_lengths)
        # Probability of picking dataset i is proportional to its length.
        probs = np.array(ds_lengths, dtype=np.float64) / total

        wi = torch.utils.data.get_worker_info()
        # main process: wid=0; DataLoader worker i: wid=i+1 — all distinct.
        wid = (wi.id + 1) if wi is not None else 0
        rng = np.random.default_rng(self.seed * 10_001 + wid)

        plan = []
        for _ in range(self.n_pairs):
            ds_local = int(rng.choice(len(ds_indices), p=probs))
            ds_idx = ds_indices[ds_local]
            ds_len = ds_lengths[ds_local]
            # Two distinct rows from this dataset.
            row_a = int(rng.integers(0, ds_len))
            row_b = int(rng.integers(0, ds_len))
            while row_b == row_a and ds_len > 1:
                row_b = int(rng.integers(0, ds_len))
            plan.append((ds_idx, row_a, row_b))
        return plan

    def _co_occurrence_pct(self, key_a: Any, key_b: Any) -> float:
        """Compute the co-occurrence percentage for (key_a, key_b).

        denominator == "any":
            pct = D_both / D_either   (0.0 if D_either == 0)
        denominator == "total":
            pct = D_both / N
        """
        D_both = 0
        D_either = 0
        for bloom in self._key_blooms:
            a = bloom.has(key_a)
            b = bloom.has(key_b)
            if a and b:
                D_both += 1
            if a or b:
                D_either += 1
        N = len(self._key_blooms)
        if self.denominator == "total":
            return D_both / N if N > 0 else 0.0
        return D_both / D_either if D_either > 0 else 0.0

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, idx: int):
        """Return (item_a, item_b, percentage) for the idx-th pair slot."""
        if idx < 0:
            idx += self.n_pairs
        if idx < 0 or idx >= self.n_pairs:
            raise IndexError(f"PairwiseDataset index {idx} out of range")
        if idx in self._pair_cache:
            return self._pair_cache[idx]
        ds_idx, row_a, row_b = self._pair_plan[idx]
        ds = self.datasets[ds_idx]
        item_a = ds[row_a]
        item_b = ds[row_b]
        key_a = self._hash_item(item_a)
        key_b = self._hash_item(item_b)
        if self._pair_bloom is not None and not self._pair_bloom.has(key_a, key_b):
            # Negative example: pair not in the source dataset's pair filter.
            result = (key_a, key_b, 0.0)
        else:
            pct = self._co_occurrence_pct(key_a, key_b)
            result = (key_a, key_b, pct)
        self._pair_cache[idx] = result
        return result

    def is_negative(self, idx: int) -> bool:
        """True if this slot is a negative example (pair not in pairwise_bloom)."""
        if self._pair_bloom is None:
            return False
        ds_idx, row_a, row_b = self._pair_plan[idx]
        ds = self.datasets[ds_idx]
        key_a = self._hash_item(ds[row_a])
        key_b = self._hash_item(ds[row_b])
        return not self._pair_bloom.has(key_a, key_b)

    def save_cache(self, file: str) -> None:
        import pickle
        payload = {
            "version": 1,
            "denominator": self.denominator,
            "seed": self.seed,
            "n_pairs": self.n_pairs,
            "plan": self._pair_plan,
            "cache": self._pair_cache,
        }
        with open(file, "wb") as f:
            pickle.dump(payload, f)

    def load_cache(self, file: str) -> None:
        import pickle
        with open(file, "rb") as f:
            payload = pickle.load(f)
        if payload.get("denominator") != self.denominator:
            raise ValueError(
                f"Cache denominator={payload.get('denominator')!r} does not match "
                f"current instance denominator={self.denominator!r}. "
                "Rebuild the dataset with the matching denominator."
            )
        if payload.get("n_pairs") != self.n_pairs:
            raise ValueError(
                f"Cache n_pairs={payload.get('n_pairs')} does not match "
                f"current instance n_pairs={self.n_pairs}. "
                "Rebuild the dataset with the matching n_pairs."
            )
        self._pair_plan = payload["plan"]
        self._pair_cache = payload["cache"]
