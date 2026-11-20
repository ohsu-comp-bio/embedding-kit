"""
Latent-space geometry: Euclidean, hyperbolic and product spaces.

Most embedding models place samples in a flat (Euclidean) latent space. When the
data has a branching, tree-like structure — cell lineages, tumour subclones, a
label taxonomy — a hyperbolic space can hold that hierarchy with far less
distortion, because the volume of a hyperbolic ball grows exponentially with its
radius. This module provides the pieces needed to use, measure and compare such
latent spaces without depending on any particular model.

Coordinate convention
---------------------
A latent vector is always a plain ``(n, latent_dim)`` array. The first
``euclidean_dim`` columns are ordinary Euclidean coordinates; the remaining
``hyperbolic_dim`` columns are *tangent coordinates at the origin* of a
hyperbolic space with curvature ``-c``. The point they denote is
``expmap0(sqrt(c) * t)`` on the unit hyperboloid. Consequences:

- The norm of the hyperbolic block is the geodesic distance from the origin
  (see :func:`latent_radius`).
- Euclidean distance between two hyperbolic blocks is *not* the geodesic
  distance; use :func:`latent_distance` / :func:`pairwise_latent_distance`.
- Posteriors on the hyperbolic block are wrapped normals (Nagano et al., 2019):
  :func:`wrapped_normal_rsample` and :func:`wrapped_normal_log_prob`.

``geometry`` is one of ``"euclidean"``, ``"hyperbolic"`` or ``"product"`` (see
:func:`split_latent_dims`). Hyperbolic computations run in float64 internally.

Provided here:

- Unit-curvature Lorentz primitives: :func:`expmap0`, :func:`logmap0`,
  :func:`lorentz_distance`, :func:`boost`, :func:`lorentz_centroid`,
  :func:`log_sinhc`.
- Wrapped-normal sampling / density / KL for hyperbolic VAEs.
- Array-level helpers: :func:`latent_distance`, :func:`pairwise_latent_distance`,
  :func:`latent_radius`, :func:`latent_centroid`, :func:`to_poincare`.
- Label trees: :func:`tree_distance_matrix` and :func:`hierarchy_metrics`.

Ported from UniVII (``univii.geometry``).
"""
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

GEOMETRIES = ("euclidean", "hyperbolic", "product")
_EPS = 1e-12
_LOG_2PI = math.log(2.0 * math.pi)

ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# dimension bookkeeping
# ---------------------------------------------------------------------------

def split_latent_dims(latent_dim: int, geometry: str = "euclidean",
                      hyperbolic_dim: Optional[int] = None) -> Tuple[int, int]:
    """
    Split a latent size into ``(euclidean_dim, hyperbolic_dim)``.

    Args:
        latent_dim: Total number of latent columns.
        geometry: ``"euclidean"`` (all Euclidean), ``"hyperbolic"`` (all
            hyperbolic) or ``"product"`` (the last ``hyperbolic_dim`` columns are
            hyperbolic; default: half, rounded up).
        hyperbolic_dim: Size of the hyperbolic block for ``"product"``.

    Returns:
        Tuple ``(euclidean_dim, hyperbolic_dim)``.

    Raises:
        ValueError: If ``geometry`` is unknown or ``hyperbolic_dim`` is
            inconsistent with it.
    """
    geometry = str(geometry).lower().strip()
    latent_dim = int(latent_dim)
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    if geometry == "euclidean":
        if hyperbolic_dim not in (None, 0):
            raise ValueError("hyperbolic_dim must be None or 0 when geometry='euclidean'")
        return latent_dim, 0
    if geometry == "hyperbolic":
        if hyperbolic_dim not in (None, latent_dim):
            raise ValueError("hyperbolic_dim must be None or latent_dim when geometry='hyperbolic'")
        return 0, latent_dim
    d_h = latent_dim - latent_dim // 2 if hyperbolic_dim is None else int(hyperbolic_dim)
    if not 0 < d_h < latent_dim:
        raise ValueError(
            f"geometry='product' needs 0 < hyperbolic_dim < latent_dim, got {d_h} with latent_dim={latent_dim}")
    return latent_dim - d_h, d_h


# ---------------------------------------------------------------------------
# unit-curvature primitives (Lorentz / hyperboloid model)
# ---------------------------------------------------------------------------

def expmap0(v: torch.Tensor) -> torch.Tensor:
    """
    Map tangent coordinates at the origin ``(..., d)`` onto the unit hyperboloid ``(..., d + 1)``.
    """
    r = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.cat([torch.cosh(r), torch.sinh(r) * v / r], dim=-1)


def logmap0(x: torch.Tensor) -> torch.Tensor:
    """
    Map hyperboloid points ``(..., d + 1)`` back to tangent coordinates at the origin ``(..., d)``.

    Inverse of :func:`expmap0`. Uses ``asinh`` of the spatial norm, which is
    better conditioned near the origin than ``acosh`` of the time coordinate.
    """
    xb = x[..., 1:]
    n = xb.norm(dim=-1, keepdim=True)
    return torch.asinh(n) * xb / n.clamp_min(_EPS)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance between points on the unit hyperboloid (broadcasts).

    Uses ``cosh(d) = 1 + <x - y, x - y>_L / 2``, which loses far less precision
    for nearby points than ``acosh(-<x, y>_L)``.
    """
    diff = x - y
    s = (-diff[..., 0] ** 2 + (diff[..., 1:] ** 2).sum(-1)).clamp_min(1e-24)
    return 2.0 * torch.asinh(0.5 * torch.sqrt(s))


def boost(center: torch.Tensor, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Apply the Lorentz boost that maps ``center`` to the origin (or its inverse).

    The boost is an isometry whose differential is parallel transport along the
    geodesic between ``center`` and the origin, so it carries wrapped normals
    centred at ``center`` to wrapped normals centred at the origin.
    """
    c0, cb = center[..., :1], center[..., 1:]
    if inverse:
        cb = -cb
    x0, xb = x[..., :1], x[..., 1:]
    dot = (cb * xb).sum(-1, keepdim=True)
    y0 = c0 * x0 - dot
    yb = xb - cb * x0 + cb * dot / (1.0 + c0)
    # re-project the time coordinate so rounding error does not accumulate
    y0 = torch.sqrt(1.0 + (yb ** 2).sum(-1, keepdim=True))
    return torch.cat([y0, yb], dim=-1)


def lorentz_centroid(x: torch.Tensor) -> torch.Tensor:
    """
    Lorentzian centroid of hyperboloid points ``(n, d + 1)``.

    Close to, but not exactly, the Frechet mean; it stays on the manifold.
    """
    s = x.sum(0)
    norm = torch.sqrt((s[0] ** 2 - (s[1:] ** 2).sum()).clamp_min(_EPS))
    return s / norm


def log_sinhc(r: torch.Tensor) -> torch.Tensor:
    """``log(sinh(r) / r)``, numerically stable for small and large ``r``."""
    small = r < 1e-3
    rs = torch.where(small, torch.ones_like(r), r)
    big = rs + torch.log1p(-torch.exp(-2.0 * rs)) - math.log(2.0) - torch.log(rs)
    return torch.where(small, r ** 2 / 6.0, big)


def _work_dtype(x: torch.Tensor) -> torch.dtype:
    # Apple MPS has no float64; everywhere else hyperbolic maths runs in float64.
    return x.dtype if x.device.type == "mps" else torch.float64


def _check_curvature(curvature: float) -> float:
    if not float(curvature) > 0.0:
        raise ValueError(f"curvature must be > 0, got {curvature}")
    return float(curvature)


# ---------------------------------------------------------------------------
# wrapped normal on a hyperbolic block (for hyperbolic VAEs)
# ---------------------------------------------------------------------------

def wrapped_normal_rsample(mu: torch.Tensor, logvar: torch.Tensor,
                           curvature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reparameterised sample from a wrapped normal, plus its log-density.

    ``mu`` is the mean in tangent coordinates at the origin and ``logvar`` the
    log-variance in the tangent space at the mean. Sampling draws a Gaussian in
    the tangent space at the origin, moves it to ``mu`` by parallel transport
    (a Lorentz boost), and returns the result in tangent coordinates at the
    origin — the same layout as ``mu`` — so a decoder can consume it directly.

    Args:
        mu: Posterior mean, ``(n, d)``.
        logvar: Posterior log-variance, ``(n, d)``.
        curvature: Curvature magnitude ``c`` (the space has curvature ``-c``).

    Returns:
        Tuple ``(t, log_q)``: the sample ``(n, d)`` and ``log q(t)`` per row ``(n,)``.
    """
    c = _check_curvature(curvature)
    out_dtype, wd = mu.dtype, _work_dtype(mu)
    m, lv = mu.to(wd), logvar.to(wd)
    sc = math.sqrt(c)
    v = torch.randn_like(lv) * torch.exp(0.5 * lv)
    z = boost(expmap0(sc * m), expmap0(sc * v), inverse=True)
    t = logmap0(z) / sc
    d = m.shape[-1]
    log_q = (-0.5 * (v ** 2 / lv.exp() + lv + _LOG_2PI)).sum(-1) - (d - 1) * log_sinhc(sc * v.norm(dim=-1))
    return t.to(out_dtype), log_q.to(out_dtype)


def wrapped_normal_log_prob(t: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                            curvature: float = 1.0) -> torch.Tensor:
    """
    Log-density of tangent coordinates ``t`` under a wrapped normal ``(mu, logvar)``.

    Args:
        t: Points in tangent coordinates at the origin, ``(n, d)``.
        mu: Wrapped-normal mean, ``(n, d)`` (broadcastable).
        logvar: Wrapped-normal log-variance, ``(n, d)`` (broadcastable).
        curvature: Curvature magnitude ``c``.

    Returns:
        Log-density per row, ``(n,)``.
    """
    c = _check_curvature(curvature)
    wd = _work_dtype(t)
    tt, m, lv = t.to(wd), mu.to(wd), logvar.to(wd)
    sc = math.sqrt(c)
    v = logmap0(boost(expmap0(sc * m), expmap0(sc * tt))) / sc
    d = tt.shape[-1]
    out = (-0.5 * (v ** 2 / lv.exp() + lv + _LOG_2PI)).sum(-1) - (d - 1) * log_sinhc(sc * v.norm(dim=-1))
    return out.to(t.dtype)


def wrapped_normal_kl(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                      mu_p: Optional[torch.Tensor] = None, logvar_p: Optional[torch.Tensor] = None,
                      curvature: float = 1.0) -> torch.Tensor:
    """
    One-sample Monte Carlo estimate of ``KL(q || p)`` between wrapped normals.

    There is no closed form on a hyperbolic space, so the KL is estimated as
    ``log q(t) - log p(t)`` with ``t ~ q``. The estimate is unbiased but noisy
    and can be slightly negative for a single row. The default prior ``p`` is
    the standard wrapped normal at the origin.

    Returns:
        KL estimate per row, ``(n,)``.
    """
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)
    t, log_q = wrapped_normal_rsample(mu_q, logvar_q, curvature)
    return log_q - wrapped_normal_log_prob(t, mu_p, logvar_p, curvature)


# ---------------------------------------------------------------------------
# geodesic distances on mixed coordinates
# ---------------------------------------------------------------------------

def _as_tensor(z: ArrayLike) -> torch.Tensor:
    return torch.as_tensor(np.asarray(z.detach().cpu() if torch.is_tensor(z) else z), dtype=torch.float64)


def _sqdist_rows(a: torch.Tensor, b: torch.Tensor, d_e: int, curvature: float) -> torch.Tensor:
    """Row-wise squared geodesic distance for float64 tensors with ``d_e`` Euclidean columns."""
    out = ((a[..., :d_e] - b[..., :d_e]) ** 2).sum(-1)
    if a.shape[-1] > d_e:
        sc = math.sqrt(curvature)
        d = lorentz_distance(expmap0(sc * a[..., d_e:]), expmap0(sc * b[..., d_e:])) / sc
        out = out + d ** 2
    return out


def _cdist_sq(a: torch.Tensor, b: torch.Tensor, d_e: int, curvature: float) -> torch.Tensor:
    """
    All-pairs ``(n, m)`` squared geodesic distance from inner products.

    Memory is ``O(n * m)`` rather than ``O(n * m * latent_dim)``.
    """
    out = torch.zeros(a.shape[0], b.shape[0], dtype=a.dtype)
    if d_e:
        ea, eb = a[:, :d_e], b[:, :d_e]
        out = out + ((ea ** 2).sum(-1, keepdim=True) + (eb ** 2).sum(-1).unsqueeze(0)
                     - 2.0 * ea @ eb.transpose(0, 1)).clamp_min(0.0)
    if a.shape[1] > d_e:
        sc = math.sqrt(curvature)
        x, y = expmap0(sc * a[:, d_e:]), expmap0(sc * b[:, d_e:])
        inner = x[:, 1:] @ y[:, 1:].transpose(0, 1) - x[:, :1] @ y[:, :1].transpose(0, 1)  # Lorentz inner product
        s = (-2.0 - 2.0 * inner).clamp_min(1e-24)                                           # <x - y, x - y>_L
        out = out + (2.0 * torch.asinh(0.5 * torch.sqrt(s)) / sc) ** 2
    return out


def _rows_per_chunk(n_columns: int, requested: int = 1024, budget: float = 5e7) -> int:
    """How many query rows to process at once so a ``rows x n_columns`` block stays near 400 MB."""
    return int(max(8, min(int(requested), budget // max(int(n_columns), 1))))


def latent_distance(a: ArrayLike, b: ArrayLike, geometry: str = "euclidean",
                    hyperbolic_dim: Optional[int] = None, curvature: float = 1.0) -> np.ndarray:
    """
    Row-wise geodesic distance between two ``(n, latent_dim)`` arrays.

    Args:
        a: First set of latent coordinates.
        b: Second set, same shape as ``a`` (broadcasts).
        geometry: ``"euclidean"``, ``"hyperbolic"`` or ``"product"``.
        hyperbolic_dim: Hyperbolic block size for ``"product"``.
        curvature: Curvature magnitude ``c`` of the hyperbolic block.

    Returns:
        Distances, ``(n,)``.
    """
    ta, tb = _as_tensor(a), _as_tensor(b)
    d_e, _ = split_latent_dims(ta.shape[-1], geometry, hyperbolic_dim)
    return torch.sqrt(_sqdist_rows(ta, tb, d_e, _check_curvature(curvature)) + _EPS).numpy()


def pairwise_latent_distance(a: ArrayLike, b: Optional[ArrayLike] = None, geometry: str = "euclidean",
                             hyperbolic_dim: Optional[int] = None, curvature: float = 1.0,
                             chunk: int = 1024) -> np.ndarray:
    """
    All-pairs ``(n, m)`` geodesic distance; ``b=None`` compares ``a`` with itself.

    Rows of ``a`` are processed in chunks of ``chunk`` so memory stays bounded.
    """
    c = _check_curvature(curvature)
    ta = _as_tensor(a)
    same = b is None
    tb = ta if same else _as_tensor(b)
    d_e, _ = split_latent_dims(ta.shape[-1], geometry, hyperbolic_dim)
    blocks = []
    for i in range(0, len(ta), int(chunk)):
        d2 = _cdist_sq(ta[i:i + chunk], tb, d_e, c)
        if same:
            # identical points have zero distance; inner-product cancellation can leave small errors
            rows = torch.arange(d2.shape[0])
            d2[rows, rows + i] = 0.0
        blocks.append(torch.sqrt(d2 + _EPS))
    return torch.cat(blocks).numpy()


def latent_radius(z: ArrayLike) -> np.ndarray:
    """
    Distance from the origin, per row.

    Valid in every geometry because the hyperbolic block is stored in tangent
    coordinates at the origin.
    """
    return np.linalg.norm(np.asarray(z.detach().cpu() if torch.is_tensor(z) else z), axis=-1)


def latent_centroid(z: ArrayLike, geometry: str = "euclidean", hyperbolic_dim: Optional[int] = None,
                    curvature: float = 1.0) -> np.ndarray:
    """
    Geometry-aware centroid of a set of latent points ``(n, latent_dim)``.

    The Euclidean block is averaged; the hyperbolic block uses the Lorentzian
    centroid, which stays on the manifold.

    Returns:
        The centroid in the same coordinates, ``(latent_dim,)``.
    """
    c = _check_curvature(curvature)
    t = _as_tensor(z)
    if t.ndim != 2 or t.shape[0] == 0:
        raise ValueError("expected a non-empty (n, latent_dim) array of latent coordinates")
    d_e, d_h = split_latent_dims(t.shape[1], geometry, hyperbolic_dim)
    parts = []
    if d_e:
        parts.append(t[:, :d_e].mean(0))
    if d_h:
        sc = math.sqrt(c)
        parts.append(logmap0(lorentz_centroid(expmap0(sc * t[:, d_e:]))) / sc)
    return torch.cat(parts).numpy()


def to_poincare(z: ArrayLike, geometry: str = "hyperbolic", hyperbolic_dim: Optional[int] = None,
                curvature: float = 1.0) -> np.ndarray:
    """
    Hyperbolic block as Poincare-ball coordinates inside the unit ball, for plotting.

    Raises:
        ValueError: If the geometry has no hyperbolic block.
    """
    c = _check_curvature(curvature)
    t = _as_tensor(z)
    d_e, d_h = split_latent_dims(t.shape[-1], geometry, hyperbolic_dim)
    if d_h == 0:
        raise ValueError("to_poincare needs a hyperbolic block; this latent space is Euclidean")
    x = expmap0(math.sqrt(c) * t[..., d_e:])
    return (x[..., 1:] / (1.0 + x[..., :1])).numpy()


# ---------------------------------------------------------------------------
# label trees
# ---------------------------------------------------------------------------

def tree_distance_matrix(parents: Mapping[str, Optional[str]],
                         lengths: Optional[Mapping[str, float]] = None,
                         order: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Path-length distances between the nodes of a rooted tree.

    Args:
        parents: ``{node: parent}`` with ``None`` for the single root.
        lengths: Optional ``{node: length of the edge to its parent}`` (default 1).
        order: Node order for rows/columns (default: the keys of ``parents``).

    Returns:
        Tuple ``(dist, names, depth)``: the ``(K, K)`` distance matrix, the node
        names in row order, and each node's distance from the root.

    Raises:
        ValueError: If the tree has zero or several roots, a cycle, or a
            parent that is not itself a node.
    """
    names = list(order) if order is not None else list(parents)
    if set(names) != set(parents):
        raise ValueError("order must contain exactly the nodes in parents")
    roots = [n for n in names if parents[n] is None]
    if len(roots) != 1:
        raise ValueError(f"tree needs exactly one root (parent None), found {len(roots)}")

    def path(n):
        out, seen = [], set()
        while n is not None:
            if n in seen:
                raise ValueError("parents contains a cycle")
            if n not in parents:
                raise ValueError(f"parent {n!r} is not itself a node")
            seen.add(n)
            out.append(n)
            n = parents[n]
        return out

    edge = {n: 0.0 if parents[n] is None else float((lengths or {}).get(n, 1.0)) for n in names}
    paths = {n: path(n) for n in names}
    depth = {n: sum(edge[a] for a in paths[n]) for n in names}
    dist = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        anc = set(paths[a])
        for j, b in enumerate(names):
            lca = next(n for n in paths[b] if n in anc)
            dist[i, j] = depth[a] + depth[b] - 2.0 * depth[lca]
    return dist, names, np.array([depth[n] for n in names])


def hierarchy_metrics(z: ArrayLike, labels: Sequence[int], dist: ArrayLike, depth: Optional[ArrayLike] = None,
                      geometry: str = "euclidean", hyperbolic_dim: Optional[int] = None, curvature: float = 1.0,
                      n_pairs: int = 20000, seed: int = 0) -> Dict[str, float]:
    """
    Score how well an embedding reproduces a label tree.

    Args:
        z: Latent coordinates ``(n, latent_dim)``.
        labels: Integer label per sample indexing rows of ``dist`` (negative = unlabelled).
        dist: ``(K, K)`` label-tree distance matrix (see :func:`tree_distance_matrix`).
        depth: Optional per-label depth; enables the ``radius_depth`` score.
        geometry: ``"euclidean"``, ``"hyperbolic"`` or ``"product"``.
        hyperbolic_dim: Hyperbolic block size for ``"product"``.
        curvature: Curvature magnitude ``c`` of the hyperbolic block.
        n_pairs: Number of random sample pairs used for the distance scores.
        seed: Random seed for pair selection.

    Returns:
        Dict with ``dist_spearman`` (rank agreement of latent vs tree distance),
        ``rel_distortion`` (mean relative error after the best single rescaling)
        and, if ``depth`` is given, ``radius_depth`` (rank agreement of latent
        radius with label depth).
    """
    from scipy.stats import spearmanr

    z_np = np.asarray(z.detach().cpu() if torch.is_tensor(z) else z)
    labels = np.asarray(labels)
    dist = np.asarray(dist, dtype=float)
    keep = np.flatnonzero((labels >= 0) & (labels < dist.shape[0]))
    rng = np.random.default_rng(seed)
    i, j = rng.choice(keep, n_pairs), rng.choice(keep, n_pairs)
    d_lat = latent_distance(z_np[i], z_np[j], geometry, hyperbolic_dim, curvature)
    d_tree = dist[labels[i], labels[j]]
    out = {"dist_spearman": float(spearmanr(d_lat, d_tree)[0])}
    ok = d_tree > 0
    s = (d_lat[ok] * d_tree[ok]).sum() / (d_tree[ok] ** 2).sum()
    out["rel_distortion"] = float(np.mean(np.abs(d_lat[ok] - s * d_tree[ok]) / (s * d_tree[ok])))
    if depth is not None:
        out["radius_depth"] = float(spearmanr(latent_radius(z_np[keep]), np.asarray(depth)[labels[keep]])[0])
    return out
