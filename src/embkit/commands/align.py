"""

Command line tools for aligning embedding spaces

"""

import click
import pandas as pd
from ..align import matrix_spearman_alignment_linear, matrix_spearman_alignment_hopkraft

align = click.Group(name="align", help="Embedding Alignment methods")

@align.command(name="pair")
@click.argument("matrix1")
@click.argument("matrix2")
@click.option("--method", "-m", required=False, type=str, default="linear")
@click.option("--cutoff", "-c", required=False, type=float, default=0.5)
def pair(matrix1, matrix2, method, cutoff):
    """
    Given two matrices, use feature rank correlation to
    create list of pairs
    """

    m1 = pd.read_csv(matrix1, sep="\t", index_col=0)
    m2 = pd.read_csv(matrix2, sep="\t", index_col=0)
    
    if method == "hopkraft":
        out = matrix_spearman_alignment_hopkraft(m1, m2)
        for k, v in out.items():
            print(f"{k}\t{v[0]}\t{v[1]}")

    elif method == "linear":
        out = matrix_spearman_alignment_linear(m1, m2, cutoff)
        for k, v in out.items():
            print(f"{k}\t{v[0]}\t{v[1]}")
