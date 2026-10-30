"""Generate protein embeddings from a FASTA file using ESM2, and load them for downstream use.

This is the protein-embedding workflow: go from raw amino-acid sequences to fixed-length
numeric vectors that can be fed into any downstream analysis (clustering, classification,
VAE training, etc.) without a further model in the loop.

Primary path (default)
----------------------
Shells out to the ``embkit protein encode`` CLI, which wraps Meta's ESM2 language model
(`fair-esm`).  The bundled FASTA is small so even the default ``t33`` model runs in a few
seconds on CPU; switch to ``t6`` for the fastest option on a resource-constrained box.

    embkit protein encode sequences.fasta --model t33 --output protein_embeddings.tsv

Fallback path
-------------
If ``fair-esm`` is not installed, the GPU is unavailable, or the model download fails,
the script falls back to the bundled pre-computed embedding matrix
(``examples/data/protein_embeddings.tsv``) so you can still demonstrate the downstream
code path (normalisation, shape check, loading into a DataFrame).

Real-data variant
-----------------
Replace the bundled ``sequences.fasta`` with your own FASTA file.  The CLI handles any
number of sequences and will automatically pick the right device (CUDA if available).
For production-scale runs with thousands of sequences use ``--batch-size 32`` (or higher)
and ``--trim 4`` to round to 4 significant figures, reducing output file size.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FASTA = os.path.join(DATA_DIR, "sequences.fasta")
DEFAULT_OUT = os.path.join(DATA_DIR, "protein_embeddings.tsv")
FALLBACK_EMB = os.path.join(DATA_DIR, "protein_embeddings_fallback.tsv")


def run_embkit_protein_encode(fasta: str, model: str, output: str, batch_size: int = 128,
                              pool: str = "mean", trim: int | None = None) -> None:
    """Call the ``embkit protein encode`` CLI to generate an embedding TSV."""
    cmd = [
        sys.executable, "-m", "embkit",
        "protein", "encode", fasta,
        "--model", model,
        "--pool", pool,
        "--batch-size", str(batch_size),
        "--output", output,
    ]
    if trim is not None:
        cmd += ["--trim", str(trim)]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  -> {output}")


def load_embeddings(path: str) -> pd.DataFrame:
    """
    Load an embedding TSV.

    ``embkit protein encode`` writes one row per sequence:
        id \t dim0 \t dim1 \t ... \t dimN

    Returns a DataFrame indexed by sequence ID with one column per embedding dimension.
    """
    df = pd.read_csv(path, sep="\t", index_col=0, header=None)
    df.columns = [f"dim_{i}" for i in range(df.shape[1])]
    df.index.name = None  # header=None leaves index name = 0; clear it for clean display
    df = df.astype("float32")
    return df


def summarize(emb: pd.DataFrame) -> None:
    """Print a quick summary of the embedding matrix."""
    print(f"\nEmbedding shape : {emb.shape[0]} sequences x {emb.shape[1]} dims")
    print(f"Value range     : [{emb.values.min():.4f}, {emb.values.max():.4f}]")
    print(f"Mean            : {emb.values.mean():.4f}")
    print(f"Std             : {emb.values.std():.4f}")
    print(f"\nFirst 3 rows (first 6 dims):")
    print(emb.iloc[:3, :6].to_string())

    # Pairwise cosine similarity — quick sanity check that similar sequences cluster
    from sklearn.metrics.pairwise import cosine_similarity
    cos = cosine_similarity(emb.values)
    np.fill_diagonal(cos, np.nan)
    print(f"\nPairwise cosine similarity (off-diagonal):")
    print(f"  mean = {np.nanmean(cos):.4f}, min = {np.nanmin(cos):.4f}, max = {np.nanmax(cos):.4f}")

    # If labels are available, print mean similarity within vs. between groups
    labels_path = os.path.join(DATA_DIR, "protein_labels.tsv")
    if os.path.exists(labels_path):
        labels = pd.read_csv(labels_path, sep="\t", index_col=0)["label"].values
        within, between = [], []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    within.append(cos[i, j])
                else:
                    between.append(cos[i, j])
        print(f"\nWithin-group cosine (same label):  {np.mean(within):.4f}")
        print(f"Between-group cosine (diff label):  {np.mean(between):.4f}")
        print(f"Separation margin:                  {np.mean(within) - np.mean(between):+.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", default=FASTA, help="Input FASTA file")
    parser.add_argument("--model", default="t33",
                        choices=["t6", "t12", "t30", "t33", "t36", "t48"],
                        help="ESM2 model size (default: t33)")
    parser.add_argument("--output", default=None, help="Output TSV path")
    parser.add_argument("--pool", default="mean", choices=["mean", "sum", "none"],
                        help="Pooling strategy (default: mean)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trim", type=int, default=None, help="Round to N significant figures")
    parser.add_argument("--fallback", action="store_true",
                        help="Skip CLI; use bundled pre-computed embeddings")
    args = parser.parse_args()

    output_path = args.output or DEFAULT_OUT

    if args.fallback:
        print("Using bundled pre-computed embeddings (fallback mode).")
        output_path = args.output or FALLBACK_EMB
    else:
        try:
            run_embkit_protein_encode(
                fasta=args.fasta,
                model=args.model,
                output=output_path,
                batch_size=args.batch_size,
                pool=args.pool,
                trim=args.trim,
            )
        except Exception as e:
            print(f"\nCLI encoding failed ({e}).", file=sys.stderr)
            if os.path.exists(FALLBACK_EMB):
                print(f"Falling back to bundled embeddings: {FALLBACK_EMB}", file=sys.stderr)
                output_path = FALLBACK_EMB
            else:
                print("No fallback file available either. Aborting.", file=sys.stderr)
                return 1

    emb = load_embeddings(output_path)
    summarize(emb)
    print(f"\nDone. Embedding matrix ready at: {output_path}")
    print("Next step: feed it into analyze_embeddings.py for PCA / UMAP / clustering.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
