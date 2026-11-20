"""
Spatial utilities: neighbour graphs, niche aggregation, image tiles and local batches.

Spatial assays (Visium, Xenium, CosMx, imaging mass cytometry) attach a position
to every spot or cell. That position can feed an embedding model in several
ways, and these helpers cover the data-preparation side of each, on plain
arrays so they work with any reader:

- **Niche features** — :func:`neighborhood_aggregate` averages each spot's
  spatial neighbours into a second, "niche" feature matrix, so a model sees each
  spot together with its surroundings.
- **Image tiles** — :func:`extract_tiles` cuts a square tile around every spot
  from a tissue image, in the flat channel-first layout that
  :class:`~embkit.modules.conv.ConvBackbone` expects; :func:`unflatten_tiles`
  reshapes them back for plotting.
- **Spatially local batches** — :class:`SpatialBlockSampler` builds mini-batches
  from neighbouring spots, which spatial smoothness terms need.
- :func:`knn_graph`, :func:`neighborhood_weights` and
  :func:`median_neighbor_distance` are the shared building blocks.

Ported from UniVII (``univii.spatial``).
"""
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Sampler

Matrix = Union[np.ndarray, sp.spmatrix]


# ---------------------------------------------------------------------------
# neighbour graphs and aggregation
# ---------------------------------------------------------------------------

def knn_graph(coords: np.ndarray, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Indices and distances of each point's ``k`` nearest neighbours (self excluded).

    Robust to duplicate coordinates, where a point may not be its own first hit.

    Args:
        coords: ``(n, dims)`` coordinates.
        k: Neighbours per point, ``1 <= k < n``.

    Returns:
        Tuple ``(idx, dist)``, both ``(n, k)``.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2:
        raise ValueError("coords must be (n, dims)")
    if not 1 <= int(k) < coords.shape[0]:
        raise ValueError(f"k must be between 1 and n - 1 = {coords.shape[0] - 1}, got {k}")
    dist, idx = NearestNeighbors(n_neighbors=int(k) + 1).fit(coords).kneighbors(coords)
    rows = np.arange(coords.shape[0])[:, None]
    is_self = idx == rows
    keep = np.argsort(is_self, axis=1, kind="stable")[:, :int(k)]
    missing = ~is_self.any(axis=1)
    if missing.any():  # self absent from the k + 1 hits: keep the k closest
        keep[missing] = np.arange(int(k))[None, :]
    keep.sort(axis=1)
    return np.take_along_axis(idx, keep, 1), np.take_along_axis(dist, keep, 1)


def median_neighbor_distance(coords: np.ndarray, k: int = 1) -> float:
    """Median distance to the ``k``-th nearest neighbour: the typical spot spacing."""
    return float(np.median(knn_graph(coords, k)[1][:, -1]))


def neighborhood_weights(coords: np.ndarray, k: int = 6, kernel: str = "uniform", sigma: Optional[float] = None,
                         include_self: bool = False) -> sp.csr_matrix:
    """
    Row-normalized ``(n, n)`` sparse weights over each point's ``k`` nearest neighbours.

    Args:
        coords: ``(n, dims)`` coordinates.
        k: Neighbours per point.
        kernel: ``"uniform"`` weights neighbours equally; ``"gaussian"`` uses
            ``exp(-d^2 / (2 sigma^2))``.
        sigma: Gaussian width; defaults to the median nearest-neighbour distance.
        include_self: Add each point to its own neighbourhood.

    Returns:
        Sparse CSR matrix whose rows sum to 1.
    """
    idx, dist = knn_graph(coords, k)
    n = idx.shape[0]
    if kernel == "uniform":
        w = np.ones_like(dist)
    elif kernel == "gaussian":
        s = float(sigma) if sigma is not None else float(np.median(dist[:, 0]))
        if not s > 0:
            raise ValueError("sigma must be > 0 (duplicate coordinates give a zero default; pass sigma)")
        w = np.exp(-dist ** 2 / (2.0 * s ** 2))
    else:
        raise ValueError(f"kernel must be 'uniform' or 'gaussian', got {kernel!r}")
    W = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), idx.shape[1]), idx.ravel())), shape=(n, n))
    if include_self:
        W = W + sp.identity(n, format="csr")
    row_sum = np.asarray(W.sum(1)).ravel()
    row_sum[row_sum == 0] = 1.0
    return sp.diags(1.0 / row_sum) @ W


def neighborhood_aggregate(X: Matrix, coords: np.ndarray, k: int = 6, kernel: str = "uniform",
                           sigma: Optional[float] = None, include_self: bool = False,
                           groups: Optional[Sequence[Any]] = None) -> Matrix:
    """
    Weighted mean of each row's spatial neighbours (a "niche" feature matrix).

    Args:
        X: ``(n, features)`` dense array or sparse matrix.
        coords: ``(n, dims)`` coordinates of the rows of ``X``.
        k: Neighbours per row.
        kernel: ``"uniform"`` or ``"gaussian"`` (see :func:`neighborhood_weights`).
        sigma: Gaussian width (see :func:`neighborhood_weights`).
        include_self: Add each row to its own neighbourhood.
        groups: Optional group label per row (e.g. slide); neighbours are then
            searched within each group only, so slides do not mix.

    Returns:
        Matrix of the same shape as ``X`` (sparse in, sparse out).
    """
    coords = np.asarray(coords, dtype=float)
    if groups is None:
        out = neighborhood_weights(coords, k, kernel, sigma, include_self) @ X
        return out.tocsr() if sp.issparse(out) else np.asarray(out)
    groups = np.asarray(groups).astype(str)
    agg = sp.lil_matrix(X.shape) if sp.issparse(X) else np.zeros(X.shape, dtype=float)
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        agg[rows] = neighborhood_aggregate(X[rows], coords[rows], k, kernel, sigma, include_self)
    return agg.tocsr() if sp.issparse(X) else agg


# ---------------------------------------------------------------------------
# image tiles
# ---------------------------------------------------------------------------

def extract_tiles(image: np.ndarray, centers_xy: np.ndarray, tile_size: int, out_size: Optional[int] = None,
                  flatten: bool = True) -> np.ndarray:
    """
    Square tiles centred on pixel coordinates.

    Args:
        image: ``(H, W)`` or ``(H, W, C)`` array. Integer images are scaled to
            ``[0, 1]``; float images are used as they are.
        centers_xy: ``(n, 2)`` pixel coordinates as ``(x, y)`` = ``(column, row)``.
        tile_size: Side length in image pixels. Tiles overlapping the border are
            completed by reflecting the image.
        out_size: Resize tiles to this side length (area averaging when it
            divides ``tile_size``, otherwise antialiased bilinear); ``None``
            keeps ``tile_size``.
        flatten: Return ``(n, C * S * S)`` in channel-first order (the layout
            :class:`~embkit.modules.conv.ConvBackbone` expects); otherwise ``(n, C, S, S)``.

    Raises:
        ValueError: On malformed inputs or centres outside the image.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim != 3:
        raise ValueError("image must be (H, W) or (H, W, C)")
    tile_size = int(tile_size)
    if tile_size < 1:
        raise ValueError("tile_size must be >= 1")
    img = img.astype(np.float32) / 255.0 if np.issubdtype(img.dtype, np.integer) else img.astype(np.float32)
    centers = np.asarray(centers_xy, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_xy must be (n, 2) as (x, y)")
    H, W, C = img.shape
    if (centers[:, 0] < 0).any() or (centers[:, 0] > W).any() or (centers[:, 1] < 0).any() or (centers[:, 1] > H).any():
        raise ValueError("some centres lie outside the image; check the coordinate scale factor")
    pad = tile_size
    mode = "reflect" if min(H, W) > pad else "edge"
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode=mode)
    x0 = np.rint(centers[:, 0] - tile_size / 2.0).astype(int) + pad
    y0 = np.rint(centers[:, 1] - tile_size / 2.0).astype(int) + pad
    tiles = np.stack([padded[y:y + tile_size, x:x + tile_size] for x, y in zip(x0, y0)])  # (n, S, S, C)
    tiles = np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))
    if out_size is not None and int(out_size) != tile_size:
        t = torch.from_numpy(tiles)
        if tile_size % int(out_size) == 0:
            t = torch.nn.functional.avg_pool2d(t, tile_size // int(out_size))
        else:
            t = torch.nn.functional.interpolate(t, size=(int(out_size), int(out_size)), mode="bilinear",
                                                align_corners=False, antialias=True)
        tiles = t.numpy()
    return tiles.reshape(tiles.shape[0], -1) if flatten else tiles


def unflatten_tiles(X: Matrix, shape: Sequence[int]) -> np.ndarray:
    """Flat ``(n, C * H * W)`` matrix -> ``(n, C, H, W)``; ``shape`` is ``(C, H, W)``."""
    X = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
    return X.reshape(X.shape[0], *[int(v) for v in shape])


# ---------------------------------------------------------------------------
# spatially local batches
# ---------------------------------------------------------------------------

class SpatialBlockSampler(Sampler[List[int]]):
    """
    Batch sampler whose batches are spatial neighbourhoods.

    Each batch is a random unvisited seed spot plus its nearest spots, so most
    of a spot's neighbours share its batch. Batches are drawn until every spot
    has been seen; because neighbourhoods overlap, a spot can appear in more
    than one batch per epoch. With ``groups`` (e.g. a slide label) batches never
    span groups. Pass it to a DataLoader as ``batch_sampler=``.

    Args:
        coords: ``(n, dims)`` coordinates of the dataset rows.
        batch_size: Spots per batch (>= 2).
        groups: Optional group label per row.
        seed: Base seed; each epoch uses ``seed + epoch``.
    """

    def __init__(self, coords: np.ndarray, batch_size: int = 256, groups: Optional[Sequence[Any]] = None, seed: int = 0):
        self.coords = np.asarray(coords, dtype=float)
        self.batch_size = int(batch_size)
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        self.seed, self._epoch = int(seed), 0
        labels = (np.zeros(len(self.coords), dtype=int) if groups is None
                  else np.unique(np.asarray(groups).astype(str), return_inverse=True)[1])
        self._members = [np.flatnonzero(labels == g) for g in np.unique(labels)]
        self._trees = [NearestNeighbors().fit(self.coords[m]) for m in self._members]
        self._group_of = labels

    def __len__(self) -> int:
        return int(sum(np.ceil(len(m) / self.batch_size) for m in self._members))

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        unseen = np.ones(len(self.coords), dtype=bool)
        for seed_spot in rng.permutation(len(self.coords)):
            if not unseen[seed_spot]:
                continue
            g = self._group_of[seed_spot]
            members = self._members[g]
            k = min(self.batch_size, len(members))
            if k < 2:
                unseen[members] = False
                continue
            local = self._trees[g].kneighbors(self.coords[seed_spot][None, :], n_neighbors=k,
                                              return_distance=False)[0]
            batch = members[local]
            unseen[batch] = False
            yield batch.tolist()
