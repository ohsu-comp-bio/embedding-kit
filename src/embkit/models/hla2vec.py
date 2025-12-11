import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader

PEPTIDE_DIM_DEFAULT = 1280
EMBEDDING_DIM_DEFAULT = 64


# ----------------------------
# Datasets
# ----------------------------
class TableIterableDataset(IterableDataset):
    def __init__(self, table, label, allele_vec, embeddings, peptideMap, peptide_dim, mhc_dim):
        self.table = table
        self.label = np.array([label], dtype=np.float32)
        self.allele_vec = allele_vec
        self.embeddings = embeddings
        self.peptideMap = peptideMap
        self.peptide_dim = peptide_dim
        self.mhc_dim = mhc_dim

    def __iter__(self):
        for context, hla1, hla2 in self.table:
            hla1_name = hla1.decode("ascii")
            hla2_name = hla2.decode("ascii")
            hla1_seq = self.allele_vec[hla1_name]
            hla2_seq = self.allele_vec[hla2_name]
            pep_idx = self.peptideMap[context]
            context_seq = self.embeddings[pep_idx].astype(np.float32)
            yield (
                torch.from_numpy(hla1_seq),
                torch.from_numpy(hla2_seq),
                torch.from_numpy(context_seq),
                torch.from_numpy(self.label),
            )


class BalancedMixer(IterableDataset):
    def __init__(self, datasets, seed: int = 0):
        self.datasets = datasets
        self.seed = seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        iters = [iter(ds) for ds in self.datasets]
        while True:
            i = int(rng.integers(low=0, high=len(iters)))
            try:
                yield next(iters[i])
            except StopIteration:
                iters[i] = iter(self.datasets[i])
                yield next(iters[i])


def collate_batch(batch):
    hla1 = torch.stack([b[0] for b in batch], dim=0)
    hla2 = torch.stack([b[1] for b in batch], dim=0)
    context = torch.stack([b[2] for b in batch], dim=0)
    y = torch.stack([b[3] for b in batch], dim=0).view(-1, 1)
    return hla1, hla2, context, y


# ----------------------------
# Model
# ----------------------------
class HLA2Vec(nn.Module):
    def __init__(self, mhc_dim: int, peptide_dim: int, emb_dim: int = EMBEDDING_DIM_DEFAULT):
        super().__init__()
        self.hla_encoder = nn.Sequential(nn.Linear(mhc_dim, emb_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(emb_dim + emb_dim + peptide_dim, 1)

    def forward(self, hla1, hla2, context):
        h1 = self.hla_encoder(hla1)
        h2 = self.hla_encoder(hla2)  # shared weights
        x = torch.cat([h1, h2, context], dim=-1)
        return self.classifier(x)  # logits


# ----------------------------
# One-call training entrypoint (data-in → trained model out)
# ----------------------------
def fit_hla2vec(
        *,
        embeddings,
        pos_table,
        neg_table,
        allele_vec,
        peptide_map,
        peptide_dim,
        mhc_dim,
        emb_dim: int = EMBEDDING_DIM_DEFAULT,
        batch_size: int = 1024,
        epochs: int = 4,
        lr: float = 1e-3,
        seed: int = 42,
        num_workers: int = 0,
        device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_pos = len(pos_table)
    n_neg = len(neg_table)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Empty dataset: pos={n_pos}, neg={n_neg}. Provide at least 1 example in each table.")

    pos_ds = TableIterableDataset(pos_table, 1.0, allele_vec, embeddings, peptide_map, peptide_dim, mhc_dim)
    neg_ds = TableIterableDataset(neg_table, 0.0, allele_vec, embeddings, peptide_map, peptide_dim, mhc_dim)
    mixed_ds = BalancedMixer([pos_ds, neg_ds], seed=seed)

    kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    loader = DataLoader(mixed_ds, **kwargs)

    model = HLA2Vec(mhc_dim, peptide_dim, emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    steps_per_epoch = max(1, math.ceil((2 * min(n_pos, n_neg)) / batch_size))
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

        # Safe divide even if somehow total==0
        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        history.append({"epoch": e, "loss": epoch_loss, "acc": epoch_acc})

    return model, history


def load_bigmhc(h5_path: str):
    """
    Returns a dict of {embeddings, peptides, posTable, negTable, allele_vec, peptideMap, peptide_dim, mhc_dim}
    suitable to splat into fit_hla2vec(**data).
    """
    import h5py, pandas as pd  # local import to keep library light if unused

    dr = h5py.File(h5_path, "r")
    embeddings = dr["embeddings"]
    peptides = dr["peptides"]
    posTable = dr["positive"]
    negTable = dr["negative"]
    seqNames = dr["alleles"]
    seqValues = dr["allele_seqs"]

    data = {name.decode("ascii"): vec for name, vec in zip(seqNames, seqValues)}
    seqDF = pd.DataFrame(data).transpose()
    mhc_dim = seqDF.shape[1]
    peptide_dim = embeddings.shape[1]
    peptideMap = {n: i for i, n in enumerate(peptides)}
    allele_vec = {k: seqDF.loc[k].values.astype(np.float32) for k in seqDF.index}

    return {
        "embeddings": embeddings,
        "peptides": peptides,
        "posTable": posTable,
        "negTable": negTable,
        "allele_vec": allele_vec,
        "peptideMap": peptideMap,
        "peptide_dim": int(peptide_dim),
        "mhc_dim": int(mhc_dim),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("HLA2Vec demo")
    parser.add_argument("--h5", type=str, default="",
                        help="Path to BigMHC.hdf5 (optional). If omitted, uses synthetic data.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emb_dim", type=int, default=EMBEDDING_DIM_DEFAULT)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    if args.h5:
        data = load_bigmhc(args.h5)
    else:
        # ---- Synthetic data path: "data-in" only, no thresholds ----
        rng = np.random.default_rng(args.seed)

        n_peptides = 4000
        n_alleles = 60
        peptide_dim = PEPTIDE_DIM_DEFAULT
        mhc_dim = 256

        # target per class
        target_pos = 20000
        target_neg = 20000
        K = min(target_pos, target_neg)

        # peptides (bytes ids) + normalized embeddings
        peptides = [f"PEP_{i:06d}".encode("ascii") for i in range(n_peptides)]
        Xp = rng.normal(size=(n_peptides, peptide_dim)).astype(np.float32)
        Xp /= (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-9)

        # alleles (bytes ids) + vectors
        allele_names = [f"HLA-X*{i:02d}".encode("ascii") for i in range(n_alleles)]
        Xa = rng.normal(size=(n_alleles, mhc_dim)).astype(np.float32)

        # projection from allele space -> peptide space to define a learnable signal
        W = rng.normal(scale=0.25, size=(mhc_dim, peptide_dim)).astype(np.float32)

        # ---- vectorized candidate generation ----
        # make many random triples, then keep top-K & bottom-K by score
        C = max(5 * (target_pos + target_neg), 100_000)  # number of candidates
        p_idx = rng.integers(0, n_peptides, size=C, dtype=np.int32)
        i1_idx = rng.integers(0, n_alleles, size=C, dtype=np.int32)
        i2_idx = rng.integers(0, n_alleles, size=C, dtype=np.int32)

        # allele sum -> project -> normalize
        S = Xa[i1_idx] + Xa[i2_idx]  # [C, mhc_dim]
        S = S @ W  # [C, peptide_dim]
        S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)

        # cosine score with peptide embeddings
        scores = np.einsum("ij,ij->i", Xp[p_idx], S)  # [C]

        # pick top-K as positives, bottom-K as negatives
        order = np.argsort(scores)  # ascending
        neg_sel = order[:K]
        pos_sel = order[-K:]

        posTable = [(peptides[p_idx[i]], allele_names[i1_idx[i]], allele_names[i2_idx[i]]) for i in pos_sel]
        negTable = [(peptides[p_idx[i]], allele_names[i1_idx[i]], allele_names[i2_idx[i]]) for i in neg_sel]

        print(f"[demo] synthesized pos={len(posTable)}, neg={len(negTable)}")

        # data objects expected by fit_hla2vec
        data = {
            "embeddings": Xp,
            "pos_table": posTable,
            "neg_table": negTable,
            "allele_vec": {name.decode("ascii"): Xa[i] for i, name in enumerate(allele_names)},
            "peptide_map": {pid: i for i, pid in enumerate(peptides)},
            "peptide_dim": peptide_dim,
            "mhc_dim": mhc_dim,
        }

    # ---- Train via the library’s single entrypoint ----
    model, history = fit_hla2vec(
        **data,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    print("Final epoch:", history[-1])
    torch.save(model.state_dict(), "hla2vec_demo.pt")
    print("Saved weights -> hla2vec_demo.pt")
