"""
Command lines for encoding protein data
"""
import re
import sys
from typing import List
import click
from Bio import SeqIO

from .. import get_device
from ..models.protein import ProteinEncoder


protein = click.Group(name="protein", help="Protein commands.")

def fasta_reader(path, filter=None):
    for record in SeqIO.parse(path, "fasta"):
        use = True
        if filter is not None:
            if not re.match(filter, record.id):
                use = False
        if use:
            yield (record.id, str(record.seq))

def stringify(l:List[float], trim=None) -> List[str]:
    out = []
    for i in l:
        if trim:
            out.append("%g" % (round(i,trim)))
        else:
            out.append("%f" % (i))
    return out

@protein.command()
@click.argument("fasta", type=str)
@click.option("--filter", type=str, default=None)
@click.option("--batch-size", type=int, default=128)
@click.option("--trim", type=int, default=None)
@click.option("--model", "-m", type=click.Choice(['t6', 't12', 't30', 't33', 't36', 't48']), default="t33")
@click.option("--pool", "-p", type=click.Choice(["mean", "sum"]), default="mean")
@click.option("--output", "-o", type=str, default=None)
def encode(fasta: str, filter:str, batch_size:int, model:str, trim:int, pool:str, output:str):
    pool_map = {
        "mean" : "mean-pool",
        "sum" : "sum-pool"
    }
    out = sys.stdout
    if output is not None:
        out = open(output, "wt")

    enc = ProteinEncoder(batch_size=batch_size, model=model)
    enc.to(get_device())
    for i, emb in enc.encode(fasta_reader(fasta, filter=filter), output=pool_map[pool]):
        out.write( f"{i}\t" + "\t".join(stringify(emb.tolist(), trim)))
        out.write("\n")
    out.close()