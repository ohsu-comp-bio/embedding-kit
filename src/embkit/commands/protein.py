"""
Command lines for encoding protein data
"""
import re
from typing import List
import click
from Bio import SeqIO

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

def encode(fasta: str, filter:str, batch_size:int, trim:int=None):
    enc = ProteinEncoder(batch_size)
    for i, emb in enc.encode(fasta_reader(fasta, filter=filter)):
        print( f"{i}\t" + "\t".join(stringify(emb.tolist(), trim)))
