# Protein Encoding Example

This example shows how to generate protein or peptide embeddings from a FASTA file using Meta's ESM2 language models via `embkit protein encode`.

## Prerequisites

- `embkit` installed with the `fair-esm` optional dependency
- A FASTA file of protein or peptide sequences
- A GPU is strongly recommended for the larger ESM2 models (`t33`, `t48`)

---

## 1) Prepare sequences

Your input must be a standard FASTA file:

```
>seq001
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
>seq002
ACDEFGHIKLMNPQRSTVWY
```

---

## 2) Encode with mean pooling (default)

The default output is one row per sequence: a tab-separated line of `id \t dim1 \t dim2 \t ...`

```bash
embkit protein encode sequences.fasta \
    --model t33 \
    --pool mean \
    --output embeddings.tsv
```

This produces a TSV where each row is a fixed-length embedding for one sequence. Load it in Python:

```python
import pandas as pd

df = pd.read_csv("embeddings.tsv", sep="\t", index_col=0, header=None)
print(df.shape)  # (n_sequences, embedding_dim)
```

---

## 3) Choose a model

Larger models produce richer embeddings at the cost of memory and speed:

```bash
# Fastest — good for quick tests or CPU-only environments
embkit protein encode sequences.fasta --model t6 --output embeddings_t6.tsv

# Balanced — good general-purpose embeddings
embkit protein encode sequences.fasta --model t12 --output embeddings_t12.tsv

# Default — strong embeddings, requires ~3 GB GPU memory
embkit protein encode sequences.fasta --model t33 --output embeddings_t33.tsv
```

---

## 4) Filter sequences by ID

Use a regex to process only a subset of sequences:

```bash
embkit protein encode sequences.fasta \
    --filter "^HLA-A" \
    --model t33 \
    --output hla_a_embeddings.tsv
```

---

## 5) Per-residue embeddings (advanced)

With `--pool none`, each sequence produces a 2D tensor (sequence_length × embedding_dim). The output is one JSON array per line:

```bash
embkit protein encode sequences.fasta \
    --pool none \
    --fix-len 512 \
    --output residue_embeddings.jsonl
```

`--fix-len` pads or truncates all sequences to the same length, making it easier to stack them into a 3D array.

---

## 6) Use embeddings with a VAE

ESM embeddings are a fixed-size numeric representation. You can feed them directly into a `VAE` as features:

```python
import pandas as pd
from embkit import dataframe_loader
from embkit.models.vae import VAE
from embkit.factory.layers import Layer
from embkit.losses import bce_with_logits
from embkit import optimize

# Load embeddings produced by `embkit protein encode`
df = pd.read_csv("embeddings.tsv", sep="\t", index_col=0, header=None)
df = df.astype("float32")

# Normalize to [0, 1]
df = (df - df.min()) / (df.max() - df.min() + 1e-8)

loader = dataframe_loader(df, batch_size=128)

vae = VAE(
    features=list(df.columns),
    latent_dim=64,
    encoder_layers=[Layer(256, activation="relu"), Layer(128, activation="relu")],
    decoder_layers=[Layer(256, activation="relu")],
)

optimize.fit_vae(vae, X=loader, epochs=50, lr=1e-3, loss=bce_with_logits)
```

---

## 7) Reduce precision for large datasets

Use `--trim` to round values to a fixed number of significant figures, reducing file size:

```bash
embkit protein encode sequences.fasta --trim 5 --output embeddings_trimmed.tsv
```
