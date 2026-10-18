"""Downstream analysis of an embedding matrix: project, cluster, and score against labels.

This is the "test the model" half of the build-and-test loop.  Once you have an embedding
matrix (from ``train_vae.py`` or ``protein_embed.py``), this script shows whether the
learned representation actually encodes the biology:

    1. Reduce dimensionality with PCA and UMAP so the structure is visible in 2D.
    2. Cluster the samples with KMeans (unsupervised — no labels used here).
    3. Score how well the clusters recover the known labels, using:
         - Adjusted Rand Index (ARI)   — agreement vs. ground truth (0 = random, 1 = perfect)
         - Silhouette score            — internal cohesion/separation of the clusters
    4. Save a 2D scatter plot (PCA and UMAP) as PNG for visual inspection.

It works on any embedding TSV (rows = samples, columns = dims) plus a matching labels file
(rows = samples, one column of group labels).  By default it analyzes the VAE embedding
produced by ``train_vae.py``; point it at the protein embedding instead with ``--embedding``.

Uses scikit-learn (decomposition / cluster / metrics), umap-learn, and matplotlib.
``scanpy`` is intentionally NOT required.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VAE_EMB = os.path.join(DATA_DIR, "vae_embedding.tsv")
PROTEIN_EMB = os.path.join(DATA_DIR, "protein_embeddings.tsv")
VAE_LABELS = os.path.join(DATA_DIR, "labels.tsv")
PROTEIN_LABELS = os.path.join(DATA_DIR, "protein_labels.tsv")


def load_pair(emb_path: str, label_path: str):
    """Load an embedding matrix and its labels, aligned on the shared sample index.

    Tolerates both writer styles: a file with a header row (``train_vae.py`` writes
    ``z0..zN``) and one without (``protein_embed.py`` writes bare ``id \t v0 \t ...``).
    In either case the dimension columns are renamed to ``dim_0..dim_N`` — the names are
    irrelevant to the analysis, only the values matter.
    """
    emb = pd.read_csv(emb_path, sep="\t", index_col=0, header=None)
    emb.columns = [f"dim_{i}" for i in range(emb.shape[1])]
    emb.index.name = None
    emb = emb.astype("float32")

    labels = pd.read_csv(label_path, sep="\t", index_col=0)
    if labels.shape[1] > 1:
        labels = labels.iloc[:, 0].rename("label")
    else:
        labels = labels.rename(columns={labels.columns[0]: "label"})
        labels = labels["label"]

    common = emb.index.intersection(labels.index)
    if len(common) == 0:
        raise ValueError(f"No shared samples between {emb_path} and {label_path}")
    emb = emb.loc[common]
    labels = labels.loc[common]
    return emb, labels


def pca_2d(X: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    return PCA(n_components=2, random_state=0).fit_transform(Xs)


def umap_2d(X: np.ndarray) -> np.ndarray | None:
    try:
        import umap
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        return umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.1,
                         random_state=0).fit_transform(Xs)
    except Exception as e:
        print(f"  (UMAP unavailable: {e}; skipping)")
        return None


def kmeans_cluster(X: np.ndarray, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(Xs)


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    return float(silhouette_score(Xs, labels))


def ari_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(y_true, y_pred))


def plot_2d(x2d: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
            labels: pd.Series, out_path: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable: {e}; skipping plot)")
        return

    unique = np.unique(labels.values)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    for i, grp in enumerate(unique):
        m = labels.values == grp
        ax.scatter(x2d[m, 0], x2d[m, 1], c=[cmap(i % 10)], s=45,
                   edgecolors="black", linewidths=0.5, label=f"true: {grp}")
    # overlay kmeans cluster id as a second marker ring
    for i, c in enumerate(np.unique(y_pred)):
        m = y_pred == c
        ax.scatter(x2d[m, 0], x2d[m, 1], facecolors="none", s=90,
                   edgecolors=[cmap((i + len(unique)) % 10)], linewidths=1.2)
    ax.set_title(f"{title}\n(true label = fill, kmeans cluster = ring)")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  plot -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", default=VAE_EMB, help="Embedding TSV to analyze")
    parser.add_argument("--labels", default=None, help="Labels TSV (default: inferred)")
    parser.add_argument("--out-dir", default=DATA_DIR, help="Where to write plots")
    args = parser.parse_args()

    emb_path = args.embedding
    # infer labels file from the embedding path if the user did not pass one explicitly
    if args.labels is not None:
        label_path = args.labels
    elif os.path.abspath(emb_path) == os.path.abspath(PROTEIN_EMB):
        label_path = PROTEIN_LABELS
    else:
        label_path = VAE_LABELS

    # tag plots with the embedding's file stem so VAE and protein plots don't collide
    stem = os.path.splitext(os.path.basename(emb_path))[0]

    print(f"Embedding : {emb_path}")
    print(f"Labels    : {label_path}")
    emb, labels = load_pair(emb_path, label_path)
    X = emb.values
    y_true = labels.values
    n_clusters = int(np.unique(y_true).size)
    print(f"Samples   : {X.shape[0]} | dims: {X.shape[1]} | groups: {n_clusters}")
    print(f"Group sizes: {np.array([int((y_true == g).sum()) for g in np.unique(y_true)])}")

    # 1) projections
    print("\n[1] 2D projections")
    pca2 = pca_2d(X)
    umap2 = umap_2d(X)

    # 2) cluster (unsupervised)
    print("\n[2] KMeans clustering (n_clusters = %d)" % n_clusters)
    cluster = kmeans_cluster(X, n_clusters)

    # 3) scores
    print("\n[3] Quality scores")
    ari = ari_score(y_true, cluster)
    sil = silhouette_score(X, cluster)
    base_ari = ari_score(y_true, y_true)  # = 1.0 sanity
    print(f"  Adjusted Rand Index (ARI)  : {ari:.4f}   (0 = random, 1 = perfect)")
    print(f"  Silhouette score           : {sil:.4f}   (-1..1, higher = tighter clusters)")
    print(f"  Interpretation             : "
          f"{'clusters recover the labels well' if ari > 0.6 else 'clusters partially recover labels' if ari > 0.2 else 'clusters do NOT recover labels'}")

    # 4) plots
    print("\n[4] Plots")
    plot_2d(pca2, y_true, cluster, labels, os.path.join(args.out_dir, f"{stem}_pca_2d.png"), "PCA")
    if umap2 is not None:
        plot_2d(umap2, y_true, cluster, labels, os.path.join(args.out_dir, f"{stem}_umap_2d.png"), "UMAP")

    print("\nDone. If ARI is low, the embedding did not capture the label structure — "
          "try a larger model, more epochs, or a better loss.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
