# Align (Python API)

Functions for aligning two embedding spaces. Backs the [`embkit align pair`](../cli.md#align) command.

| Name | Purpose |
|------|---------|
| `matrix_spearman_alignment_linear` | Optimal one-to-one row pairing by total Spearman correlation (uses `linear_sum_assignment`) |
| `procrustes` | Optimal rotation (det(R)=+1) aligning X to Y |
| `procrustes_scale` | Rotation + per-dimension scale factors |
| `procrustes_scale_centered` | Same as above, but mean-centers inputs first |
| `calc_rmsd` | Root-mean-square deviation between two arrays |

```python
from embkit.align import matrix_spearman_alignment_linear

a_ids, b_ids, scores = matrix_spearman_alignment_linear(df_a, df_b, cutoff=0.5)
pairs = list(zip(a_ids, b_ids, scores))
```

## API

::: embkit.align
    options:
      show_source: false
      filters:
        - "!^_"
