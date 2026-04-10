# CLI Reference

`embkit` is the command-line interface for Embedding Kit. It is organized into sub-command groups.

## Top-level help

```bash
embkit --help
```

---

## model

Commands for training and using VAE models.

### model train-vae

Train a VAE from a tab-separated matrix file (rows = samples, columns = features).

```bash
embkit model train-vae INPUT_PATH [OPTIONS]
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `INPUT_PATH` | Path to a `.tsv` or `.h5` input file |

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--group`, `-g` | — | HDF5 group name (only used with `.h5` input) |
| `--latent`, `-l` | `256` | Latent dimension size |
| `--epochs`, `-e` | `20` | Number of training epochs |
| `--batch-size`, `-b` | `256` | Batch size |
| `--encode-layers` | `400,200` | Comma-separated hidden layer sizes for the encoder |
| `--decode-layers` | `200,400` | Comma-separated hidden layer sizes for the decoder |
| `--normalize`, `-n` | `none` | Pre-normalization: `none`, `expMinMax`, `minMax` |
| `--final-activation` | `none` | Final decoder activation: `none`, `sigmoid`, `relu` |
| `--learning-rate`, `-r` | `0.0001` | Adam learning rate |
| `--out`, `-o` | auto-named | Output model file path |
| `--schedule`, `-s` | — | Beta KL schedule: `"20:0,20:0.1,40:0.3"` |
| `--loss` | `bce-logit` | Loss function: `mse`, `bce`, `bce-logit` |
| `--zero-mask` | — | Drop features that are zero in more than this fraction of samples |
| `--seed` | `42` | Random seed |
| `--bfloat16` | false | Use bfloat16 dtype for reduced memory usage |
| `--save-stats` | false | Save training statistics alongside the model |

For HDF5 input (`--group`), normalization must be `none`.

**Examples**

```bash
# Minimal
embkit model train-vae data/train.tsv --epochs 100 --latent 128 --out vae.model

# With beta schedule and custom layers
embkit model train-vae data/train.tsv \
    --epochs 120 \
    --latent 256 \
    --encode-layers 2048,1024,512 \
    --decode-layers 1024,2048 \
    --schedule "20:0,20:0.1,40:0.3,40:0.4" \
    --loss bce-logit \
    --normalize expMinMax \
    --out gtex.vae.model

# From HDF5
embkit model train-vae data/matrix.h5 --group rna --latent 64 --out rna.model
```

---

### model train-netvae

Train a pathway-constrained NetVAE. Requires a gene expression matrix and a pathway file in SIF format.

```bash
embkit model train-netvae INPUT_PATH PATHWAY_SIF [OPTIONS]
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `INPUT_PATH` | Path to a `.tsv` expression matrix |
| `PATHWAY_SIF` | Path to a pathway file in SIF (Simple Interaction Format) |

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs`, `-e` | `20` | Number of training epochs |
| `--normalize`, `-n` | `none` | Pre-normalization: `none`, `expMinMax` |
| `--learning-rate`, `-r` | `0.0001` | Adam learning rate |
| `--out`, `-o` | — | Output model file path |
| `--loss` | `bce-logit` | Loss function: `mse`, `bce`, `bce-logit` |
| `--group-layer-size` | `5,2,1` | Comma-separated per-group widths for masked NetVAE layers |
| `--save-stats` | false | Save training statistics |

`NetVAE` now accepts only `group_layer_size` in model configs/serialization. The legacy alias `group_layer_scaling` has been removed.

**Example**

```bash
embkit model train-netvae data/rna.tsv data/pathway.sif \
    --epochs 80 \
    --normalize expMinMax \
    --out netvae.model
```

---

### model encode

Encode samples from a matrix into latent space using a saved model.

```bash
embkit model encode INPUT_PATH MODEL_PATH [OPTIONS]
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `INPUT_PATH` | Path to a `.tsv` input matrix |
| `MODEL_PATH` | Path to a saved model file |

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--normalize`, `-n` | `none` | Pre-normalization to apply: `none`, `expMinMax` |
| `--out`, `-o` | `embedding.tsv` | Output TSV path |

**Example**

```bash
embkit model encode data/test.tsv vae.model --out test_embeddings.tsv
```

---

### model verify

Run integrity/sanity checks on a saved model artifact.

```bash
embkit model verify MODEL_PATH [OPTIONS]
```

This command is intended as a **sanity audit** (numerical health, mask leakage, architecture consistency), not cryptographic provenance validation.

**Arguments**

| Argument | Description |
|----------|-------------|
| `MODEL_PATH` | Path to a saved model file |

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | false | Emit machine-readable JSON report |
| `--ci` | false | CI mode (`--json` + fail on unhealthy) |
| `--fail-on-unhealthy` | false | Exit non-zero when report is unhealthy |
| `--strict` | false | Enable strict identity checks |
| `--expected-feature-count` | — | Required feature count in strict mode |
| `--expected-latent-dim` | — | Required latent dim in strict mode |
| `--expected-features-file` | — | Newline-delimited expected feature list in strict mode |

**Examples**

```bash
# Human-readable summary
embkit model verify netvae.model

# CI-safe machine output (fails on unhealthy)
embkit model verify netvae.model --ci

# Strict shape/feature identity checks
embkit model verify netvae.model \
  --strict \
  --expected-feature-count 15425 \
  --expected-latent-dim 2061 \
  --expected-features-file expected_features.txt \
  --fail-on-unhealthy
```

---

## matrix

Commands for working with feature matrices.

### matrix normalize

Quantile-normalize one or more TSV matrices and write the result to a single output file.

```bash
embkit matrix normalize SRCS... --out OUTPUT [OPTIONS]
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `SRCS` | One or more input `.tsv` files |

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--out`, `-o` | required | Output file path |
| `--features` | — | Path to a newline-delimited feature list; subset columns to this list |
| `--col-quantile` | false | Normalize per-column instead of per-row |
| `--quantile-max` | `0.9` | Quantile value used as the normalization maximum |
| `--precision`, `-p` | `5` | Decimal places in output |

**Example**

```bash
# Normalize multiple files and concatenate
embkit matrix normalize cohort1.tsv cohort2.tsv \
    --out combined.normalized.tsv \
    --quantile-max 0.9
```

### matrix pca

Run PCA on a TSV matrix and write principal components to TSV.

```bash
embkit matrix pca INPUT_PATH --pca-size N [OPTIONS]
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `INPUT_PATH` | Input `.tsv` matrix path |

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--pca-size` | required | Number of principal components |
| `--out`, `-o` | auto-named | Output TSV path (`<input_stem>.pca.tsv`) |

**Example**

```bash
embkit matrix pca data/rna.tsv --pca-size 64 --out rna.pca.tsv
```

---

## protein

Commands for encoding protein sequences.

### protein encode

Generate per-sequence or per-residue embeddings from a FASTA file using Meta's ESM2 models.

```bash
embkit protein encode FASTA [OPTIONS]
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `FASTA` | Path to a FASTA file |

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--filter` | — | Regex pattern to filter sequence IDs |
| `--batch-size` | `128` | Number of sequences per ESM batch |
| `--trim` | — | Round output values to this many significant figures |
| `--model`, `-m` | `t33` | ESM2 model size: `t6`, `t12`, `t30`, `t33`, `t36`, `t48` |
| `--pool`, `-p` | `mean` | Residue pooling: `mean`, `sum`, `none` (outputs per-residue vectors as JSON) |
| `--output`, `-o` | stdout | Output file path (default: standard output) |
| `--fix-len` | — | Pad/truncate sequences to this length (only used with `--pool none`) |

**Model size guide**

| Flag | Parameters | Notes |
|------|-----------|-------|
| `t6` | 8 M | Fastest; suitable for quick tests |
| `t12` | 35 M | Good balance |
| `t33` | 650 M | Default; strong general-purpose embeddings |
| `t48` | 3 B | Highest quality; requires significant GPU memory |

**Examples**

```bash
# Mean-pooled embeddings to stdout (TSV: id, dim1, dim2, ...)
embkit protein encode sequences.fasta --model t33 > peptide_embeddings.tsv

# Per-residue JSON vectors saved to file
embkit protein encode sequences.fasta --pool none --output residue_embeddings.jsonl

# Filter to sequences whose ID matches a pattern
embkit protein encode sequences.fasta --filter "^HLA" --output hla_embeddings.tsv
```

---

## align

Commands for aligning two embedding spaces.

### align pair

Given two embedding matrices, find an optimal one-to-one pairing of their rows based on Spearman rank correlation across shared columns.

```bash
embkit align pair MATRIX1 MATRIX2 [OPTIONS]
```

**Arguments**

| Argument | Description |
|----------|-------------|
| `MATRIX1` | Path to first embedding TSV (rows = samples) |
| `MATRIX2` | Path to second embedding TSV |

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--method`, `-m` | `linear` | Alignment method. Currently: `linear` (uses `linear_sum_assignment` on Spearman correlation) |
| `--cutoff`, `-c` | `0.5` | Minimum Spearman correlation to report a pair |

**Output**

Tab-separated lines to stdout: `sample_from_matrix1 \t sample_from_matrix2 \t spearman_score`

**Example**

```bash
embkit align pair embeddings_a.tsv embeddings_b.tsv \
    --cutoff 0.6 > alignment.tsv
```

---

## resources

Commands for downloading external biological datasets to `~/.embkit/`.

```bash
embkit resources --help
```

Datasets currently available via the CLI include GTEx gene TPM and transcript TPM. Files are cached locally after the first download.

---

## cbio

Commands for accessing cBioPortal data.

```bash
embkit cbio --help
```

Downloads molecular profile data from cBioPortal studies. Use `embkit cbio --help` to list available sub-commands and options.
