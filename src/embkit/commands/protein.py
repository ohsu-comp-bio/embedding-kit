"""
Command lines for encoding protein data
"""
import re
import sys
from typing import List
import click
import json
from ..files.json import StringTrimEncoder
from Bio import SeqIO

import torch

from .. import get_device
from ..encoding.protein import ProteinEncoder


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
@click.option("--pool", "-p", type=click.Choice(["mean", "sum", "none"]), default="mean")
@click.option("--output", "-o", type=str, default=None)
@click.option("--fix-len", type=int, default=None)
def encode(fasta: str, filter:str, batch_size:int, model:str, trim:int, pool:str, output:str, fix_len):
    pool_map = {
        "mean" : "mean-pool",
        "sum" : "sum-pool"
    }
    out = sys.stdout
    if output is not None:
        out = open(output, "wt")

    enc = ProteinEncoder(batch_size=batch_size, model=model)
    enc.to(get_device())
    if pool == "none":
        for i, emb in enc.encode(fasta_reader(fasta, filter=filter), output="vector", fix_len=fix_len):
            out.write(f"{i}\t")
            if trim:
                out.write(json.dumps(emb, cls=StringTrimEncoder, trim=trim ))
            else:
                out.write(json.dumps(emb.tolist()))
            #out.write( f"{i}\t" + "\t".join(stringify(emb.tolist(), trim)))
            out.write("\n")
    else:
        for i, emb in enc.encode(fasta_reader(fasta, filter=filter), output=pool_map[pool]):
            out.write( f"{i}\t" + "\t".join(stringify(emb.tolist(), trim)))
            out.write("\n")
    out.close()