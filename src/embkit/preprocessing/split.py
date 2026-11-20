"""
Label-stratified train / validation / test splits.

A random split can leave a rare class entirely out of training or validation,
which makes the embedding look worse (or better) than it is. Splitting within
each label keeps every class represented in proportion, and capping the number
of samples per label stops a few large classes from dominating training.

- :func:`split_by_label` — deterministic, stratified index split with optional caps.

Ported from UniVII (``univii.preprocessing.split_by_label``).
"""
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def split_by_label(labels: Sequence, train_fraction: float = 0.8, val_fraction: float = 0.1,
                   train_cap: Optional[int] = None, val_cap: Optional[int] = None,
                   max_per_label: Optional[int] = None, seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Stratify samples by label into train / validation / test index arrays.

    Within each label the samples are shuffled, the first ``train_fraction``
    go to training, the next ``val_fraction`` to validation and the rest to
    test. Samples left over by a cap go to the test set.

    Args:
        labels: One label per sample.
        train_fraction: Fraction of each label's (possibly capped) pool used for training.
        val_fraction: Fraction used for validation. ``train_fraction + val_fraction`` must be < 1.
        train_cap: Optional maximum number of training samples per label.
        val_cap: Optional maximum number of validation samples per label.
        max_per_label: Optional cap on each label's pool before fractions are
            applied; overflow goes to test. Mutually exclusive with
            ``train_cap``/``val_cap``.
        seed: Random seed; the split is deterministic for a given seed.

    Returns:
        ``{"train": idx, "val": idx, "test": idx}`` with integer position arrays.

    Raises:
        ValueError: On missing labels, invalid fractions, or conflicting caps.
    """
    labels = np.asarray(labels)
    if max_per_label is not None and (train_cap is not None or val_cap is not None):
        raise ValueError("choose a pool cap (max_per_label) or separate train/validation caps, not both")
    if pd.isna(labels).any():
        raise ValueError("handle missing stratification labels explicitly before splitting")
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("require train_fraction > 0, val_fraction >= 0 and train_fraction + val_fraction < 1")
    rng = np.random.default_rng(seed)
    result = {key: [] for key in ("train", "val", "test")}
    for label in pd.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        used = len(idx) if max_per_label is None else min(len(idx), int(max_per_label))
        ntr = int(used * train_fraction)
        nva = int(used * val_fraction)
        if train_cap is not None:
            ntr = min(ntr, int(train_cap))
        if val_cap is not None:
            nva = min(nva, int(val_cap))
        result["train"].extend(idx[:ntr])
        result["val"].extend(idx[ntr:ntr + nva])
        result["test"].extend(idx[ntr + nva:])
    return {key: np.asarray(values, dtype=int) for key, values in result.items()}
