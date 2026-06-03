"""

Command line tools for aligning embedding spaces

"""

import click
import pandas as pd
from ..align import matrix_spearman_alignment_linear

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
    
    if method == "linear":
        out_a, out_b, out_score = matrix_spearman_alignment_linear(m1, m2, cutoff)
        for a_id, b_id, score in zip(out_a, out_b, out_score):
            click.echo(f"{a_id}\t{b_id}\t{score}")
