# HLA2Vec Example

This example demonstrates a peptide/HLA workflow using Embedding Kit utilities and a custom training script.

## Prerequisites

- BigMHC training/evaluation files available locally
- Embedding Kit installed with CLI access (`embkit`)

The commands below assume a directory named `BigMHC Training and Evaluation Data`.

## 1) Build a peptide FASTA file

```bash
cat ./BigMHC\ Training\ and\ Evaluation\ Data/el_train.csv \
  | grep -v mhc \
  | awk -F ',' '{print $2}' \
  | sort \
  | uniq \
  | awk '{print ">"$1 "\n" $1}' > el_train.fa
```

## 2) Encode peptide sequences

```bash
embkit protein encode el_train.fa --trim 5 --model t6 > el_train.fa.tsv
```

## 3) Run HLA2Vec training

```bash
python ./hla2vec.py \
  ./BigMHC\ Training\ and\ Evaluation\ Data/el_train.csv \
  ./BigMHC\ Training\ and\ Evaluation\ Data/pseudoseqs.csv \
  el_train.fa.tsv
```

## Reference training script (`hla2vec.py`)

```python
import sys
import math
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import IterableDataset, DataLoader, WeightedRandomSampler

import embkit
from embkit.models.hla2vec import HLA2Vec
from embkit.preprocessing.csv import LargeCsvReader

EMBEDDING_SIZE = 64

data_file = sys.argv[1]
hlaseq_file = sys.argv[2]
peptide_file = sys.argv[3]

bigmhc = pd.read_csv(data_file)
hlaseq = pd.read_csv(hlaseq_file, index_col=0)
hlaseq = hlaseq.astype('float32')

class PairBindingDataset(IterableDataset):
    def __init__(self, bigmhc, hlaseq, peptides):
        self.bigmhc = bigmhc
        self.peptides = peptides
        self.hlaseq = hlaseq

    def __iter__(self):
        g = bigmhc.groupby(["pep"])
        with self.peptides as peptides:
            for i in g:
                if i[1]["mhc"].count() > 1:
                    vec = peptides.get(i[0][0])
                    if vec is not None:
                        pep_vec = np.fromiter(vec[1:], dtype=np.float32)
                        for r1, r2 in itertools.permutations(i[1].values, 2):
                            yield (
                                self.hlaseq.loc[r1[0]].values,
                                self.hlaseq.loc[r2[0]].values,
                                pep_vec,
                                np.array([r1[2] == r2[2]], dtype=np.float32),
                            )
                    else:
                        print("Not found, ", i[0][0])

peptides = LargeCsvReader(peptide_file, sep="\t", index_column=0, skip_header=True, cache_size=128)

peptide_dim = peptides.shape[1] - 1
hla_dim = hlaseq.shape[1]

print(f"Pepdim: {peptide_dim} HLAdim {hla_dim} EmbDim:{EMBEDDING_SIZE}")

pair_bindings = PairBindingDataset(bigmhc, hlaseq, peptides)
loader = DataLoader(pair_bindings, batch_size=1024)

epochs: int = 4
lr: float = 1e-3
device = embkit.get_device()

model = HLA2Vec(hla_dim, peptide_dim, emb_dim=EMBEDDING_SIZE)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCEWithLogitsLoss()

steps_per_epoch = 2000
history = []

data_iter = iter(loader)
for e in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i in range(steps_per_epoch):
        try:
            hla1, hla2, context, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            hla1, hla2, context, y = next(data_iter)

        hla1 = hla1.to(device, non_blocking=True)
        hla2 = hla2.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(hla1, hla2, context)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        running_loss += loss.item() * bs
        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            total += bs

        if i and (i % 100 == 0 or i == steps_per_epoch - 1):
            print(f"[Epoch {e}/{epochs}] Step {i+1}/{steps_per_epoch} - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    history.append({"epoch": e, "loss": epoch_loss, "acc": epoch_acc})
```

## Notes

- `embkit protein encode` produces peptide embeddings from FASTA input.
- The script pairs HLA alleles per peptide and trains a binary comparison objective.
- Start with a smaller `steps_per_epoch` during local testing to validate your pipeline quickly.
