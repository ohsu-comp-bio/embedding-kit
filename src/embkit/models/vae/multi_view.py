import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.manual_seed(7)


# Synthetic data (safer scales)
class SyntheticMixDataset(Dataset):
    def __init__(self, n_bulk=400, m_cells_per_bulk=8, n_genes=200, n_celltypes=6):
        self.number_of_genes = n_genes
        self.m = m_cells_per_bulk
        self.n_bulk = n_bulk
        rank = 8
        U = torch.randn(n_celltypes, rank) * 0.6
        V = torch.randn(rank, n_genes) * 0.6
        base_log_mu = U @ V
        base_log_mu = base_log_mu - base_log_mu.mean(dim=1, keepdim=True)
        base_mu = torch.exp(base_log_mu) + 0.2
        for t in range(n_celltypes):
            idx = torch.randperm(n_genes)[:10]
            base_mu[t, idx] += 1.5

        self.theta_sc = torch.full((n_genes,), 5.0)
        self.theta_bulk = torch.full((n_genes,), 12.0)

        def sample_nb(mu, theta):
            mu = mu.clamp(min=1e-8, max=5e5)
            theta = theta.clamp(min=1e-3, max=1e4)
            # Gamma-Poisson (avoid inf by working in rate space)
            gamma_shape = theta
            gamma_rate = theta / (mu + 1e-8)
            rate = torch.distributions.Gamma(gamma_shape, gamma_rate).sample()
            return torch.distributions.Poisson(rate).sample()

        self.items = []
        T = n_celltypes
        for _ in range(n_bulk):
            types = torch.randint(0, T, (self.m,))
            # smaller libraries (~3k–8k UMIs)
            lib_sc = torch.distributions.LogNormal(8.0, 0.25).sample((self.m, 1)) / 1e3
            cells = []
            for i, t in enumerate(types):
                # normalize base_mu to sum 1, then scale to library size
                mu_g = lib_sc[i] * (base_mu[t] / base_mu[t].sum()) * self.number_of_genes * 20.0
                xg = sample_nb(mu_g, self.theta_sc)
                cells.append(xg)
            cells = torch.stack(cells, dim=0)  # (m, number_of_genes)

            w = torch.distributions.Dirichlet(torch.ones(self.m)).sample()
            bulk_counts = cells.sum(dim=0)  # exact sum -> linear ground truth
            self.items.append({"cells": cells.to(torch.int64),
                               "weights": w, "bulk": bulk_counts.to(torch.int64)})
        self.n_genes = n_genes

    def __len__(self):
        return self.n_bulk

    def __getitem__(self, idx):
        it = self.items[idx]
        return it["cells"], it["weights"], it["bulk"]


# NB log-likelihood
def nb_loglik(x, mu, theta, eps=1e-8):
    # x: (..., number_of_genes) counts; mu, theta > 0
    x = x.float()  # ensure floating-point for log/ratio ops (counts may come in as ints)
    mu = mu.clamp(min=eps, max=1e7)  # clip mean away from 0/inf for numerical stability
    theta = theta.clamp(min=1e-3, max=1e5)  # clip dispersion away from 0/inf for stability

    # log NB with mean/disp: log C(x+θ-1, x) + x log(mu/(mu+θ)) + θ log(θ/(mu+θ))
    # Use log1p for stability: log(mu+θ) = logθ + log1p(mu/θ)

    # t1: log binomial coefficient term = lgamma(x+θ) - lgamma(θ) - lgamma(x+1)
    t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)

    log_theta = torch.log(theta)  # precompute log(θ) once (broadcasts over last dim)
    # log(μ+θ) computed stably as logθ + log1p(μ/θ)
    log_theta_plus_mu = log_theta + torch.log1p(mu / theta)

    # t2: x * [log(μ) - log(μ+θ)]  (success-count dependent part)
    t2 = x * (torch.log(mu) - log_theta_plus_mu)

    # t3: θ * [log(θ) - log(μ+θ)]  (dispersion-dependent part)
    t3 = theta * (log_theta - log_theta_plus_mu)

    # total log-likelihood per sample: sum over genes/features on the last dimension
    ll = (t1 + t2 + t3).sum(dim=-1)

    # replace any NaN/Inf outputs with a large negative sentinel to avoid breaking downstream code
    ll = torch.where(torch.isfinite(ll), ll, torch.full_like(ll, -1e6))

    return ll  # shape: input batch shape without the last (gene) dimension

#  Model
class Encoder(nn.Module):
    def __init__(self, number_of_genes, zdim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(number_of_genes, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, zdim)
        self.logvar = nn.Linear(hidden, zdim)

    def forward(self, x_norm):
        h = self.net(x_norm)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, number_of_genes, zdim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, number_of_genes)
        )
        self.bias = nn.Parameter(torch.zeros(number_of_genes))

    def forward(self, z, size_factor):
        # Predict per-gene rate on size-1 scale, then rescale by size_factor
        log_rate = self.net(z) + self.bias
        rate = F.softplus(log_rate) + 1e-4
        mu = rate * size_factor  # broadcast (B,1) or (B*m,1)
        mu = mu.clamp(max=1e7)
        return mu


class MultiViewVAE(nn.Module):
    def __init__(self, number_of_genes, zdim=12, hidden=256):
        super().__init__()
        self.enc_sc = Encoder(number_of_genes, zdim, hidden)
        self.dec_sc = Decoder(number_of_genes, zdim, hidden)
        self.enc_bulk = Encoder(number_of_genes, zdim, hidden)
        self.dec_bulk = Decoder(number_of_genes, zdim, hidden)
        self.theta_sc_unconstr = nn.Parameter(torch.full((number_of_genes,), 1.2))  # ~softplus->2.0
        self.theta_bulk_unconstr = nn.Parameter(torch.full((number_of_genes,), 1.7))  # -> ~3.0

    @property
    def theta_sc(self):   return F.softplus(self.theta_sc_unconstr) + 1e-3

    @property
    def theta_bulk(self): return F.softplus(self.theta_bulk_unconstr) + 1e-3

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar).clamp(max=50.0)
        return mu + torch.randn_like(std) * std

    def kl_normal(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    # helpers: size factors & encoder inputs
    def _prep_inputs_sc(self, x_cells):
        bulk_sample_groups, number_of_single_cells, number_of_genes = x_cells.shape
        x = x_cells.reshape(bulk_sample_groups * number_of_single_cells, number_of_genes).float()
        lib = x.sum(dim=1, keepdim=True).clamp(min=1.0)
        sf = lib / lib.median()  # (B*number_of_single_cells,1)
        x_norm = torch.log1p(x / sf)
        return x, sf, x_norm

    def _prep_inputs_bulk(self, x_bulk):
        x = x_bulk.float()
        lib = x.sum(dim=1, keepdim=True).clamp(min=1.0)
        sf = lib / lib.median()
        x_norm = torch.log1p(x / sf)
        return x, sf, x_norm

    def forward_sc(self, x_cells):
        bulk_sample_groups, number_of_single_cells, number_of_genes = x_cells.shape
        x, sf, x_norm = self._prep_inputs_sc(x_cells)
        mu_z, logvar_z = self.enc_sc(x_norm)
        z = self.reparam(mu_z, logvar_z)
        recon_mu = self.dec_sc(z, sf)
        kl = self.kl_normal(mu_z, logvar_z).reshape(bulk_sample_groups, number_of_single_cells)
        ll = nb_loglik(x, recon_mu, self.theta_sc).reshape(bulk_sample_groups, number_of_single_cells)
        return z.reshape(bulk_sample_groups, number_of_single_cells, -1), kl, ll

    def forward_bulk(self, x_bulk):
        x, sf, x_norm = self._prep_inputs_bulk(x_bulk)
        mu_z, logvar_z = self.enc_bulk(x_norm)
        z = self.reparam(mu_z, logvar_z)
        recon_mu = self.dec_bulk(z, sf)
        kl = self.kl_normal(mu_z, logvar_z)
        ll = nb_loglik(x, recon_mu, self.theta_bulk)
        return z, kl, ll


# ---------------- Train ----------------
def collate(batch):
    cells, weights, bulks = zip(*batch)
    return torch.stack(cells), torch.stack(weights), torch.stack(bulks)


device = "cuda" if torch.cuda.is_available() else "cpu"
number_of_genes = 200
dataset = SyntheticMixDataset(n_bulk=500, m_cells_per_bulk=8, n_genes=number_of_genes, n_celltypes=6)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)

model = MultiViewVAE(number_of_genes=number_of_genes, zdim=12, hidden=256).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

lambda_align = 10.0
lambda_rec_sc = 1.0
lambda_rec_bulk = 1.0


def train_epoch(epoch, total_epochs=20):
    model.train() # set train mode
    total_loss = 0.0
    beta_kl = min(1.0, epoch / max(5, total_epochs // 4)) # warmup over 5 epochs or 1/4 total epochs

    pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader),
                desc=f"Epoch {epoch:02d} β_KL={beta_kl:.2f}", leave=False)

    for step, (cells, weights, bulks) in pbar:
        cells, weights, bulks = cells.to(device), weights.to(device), bulks.to(device)
        z_sc, kl_sc, ll_sc = model.forward_sc(cells)
        z_bulk, kl_bulk, ll_bulk = model.forward_bulk(bulks)

        z_mix = torch.einsum('bm,bmz->bz', weights, z_sc)
        rec_sc = - ll_sc.mean(dim=1)
        rec_bulk = - ll_bulk
        kl_sc = kl_sc.mean(dim=1)
        align = F.mse_loss(z_bulk, z_mix)

        loss = (lambda_rec_sc * rec_sc
                + lambda_rec_bulk * rec_bulk
                + beta_kl * (kl_sc + kl_bulk)
                + lambda_align * align).mean()

        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        total_loss += loss.item()
        # live bar metrics (cheap to compute)
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "rec_sc": f"{rec_sc.mean().item():.1f}",
            "rec_bulk": f"{rec_bulk.mean().item():.1f}",
            "kl_sc": f"{kl_sc.mean().item():.1f}",
            "kl_bulk": f"{kl_bulk.mean().item():.1f}",
            "align": f"{align.item():.3f}",
        })

    print(f"Epoch {epoch:02d} | avg_loss={total_loss/len(train_loader):.3f} | β_KL={beta_kl:.2f}")

train_epochs = 20
for ep in range(1,train_epochs):
    train_epoch(ep, total_epochs=train_epochs)

# ---------------- Sanity check ----------------
model.eval()
with torch.no_grad():
    cells, weights, bulks = next(iter(train_loader))
    cells, weights, bulks = cells.to(device), weights.to(device), bulks.to(device)
    z_sc, _, _ = model.forward_sc(cells)
    z_bulk, _, _ = model.forward_bulk(bulks)
    z_mix = torch.einsum('bm,bmz->bz', weights, z_sc)
    mse = F.mse_loss(z_bulk, z_mix, reduction='none').mean(dim=1)
    cos = F.cosine_similarity(z_bulk, z_mix, dim=1)
    print("\nSanity check:")
    print("Mean MSE(z_bulk, sum w z_sc):", float(mse.mean()))
    print("Mean cosine similarity:", float(cos.mean()))
