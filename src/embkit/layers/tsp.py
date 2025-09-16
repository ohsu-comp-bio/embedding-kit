import time

import torch
import torch.nn as nn
from typing import Iterable, Optional


class TSPLayer(nn.Module):
    """
    Top-Scoring-Pair style layer.

    Args:
        pairs: iterable of (i, j) feature indices (0-based) to compare.
        beta: steepness for the smooth vote = sigmoid(beta * (x_i - x_j)).
              Use a large beta (e.g., 20-50) to approximate hard >.
        hard: if True, output hard 0/1 votes using (x_i > x_j).float()
              (non-differentiable). If False, output smooth votes in (0,1).
        learnable_weights: if True, learn a weight per pair (k-TSP-like).
        chunk_size: optionally compute votes in chunks to limit memory.
    """

    def __init__(
            self,
            pairs: Iterable[tuple],
            beta: float = 25.0,
            hard: bool = False,
            learnable_weights: bool = False,
            chunk_size: Optional[int] = None,
    ):
        super().__init__()
        pairs = torch.as_tensor(list(pairs), dtype=torch.long)  # [K, 2]
        assert pairs.ndim == 2 and pairs.size(1) == 2
        self.register_buffer("pairs", pairs)
        self.beta = beta
        self.hard = hard
        self.chunk_size = chunk_size

        if learnable_weights:
            self.weights = nn.Parameter(torch.ones(pairs.size(0)))
        else:
            self.register_parameter("weights", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] features (e.g., gene expression per sample)
        returns:
            votes: [B, K] (per-pair votes) if no weights
                   or [B] aggregated weighted vote if weights exist
        """
        B, D = x.shape
        K = self.pairs.size(0)

        def _compute_pairs(pairs_chunk):
            xi = x[:, pairs_chunk[:, 0]]  # [B, k]
            xj = x[:, pairs_chunk[:, 1]]  # [B, k]
            diff = xi - xj  # [B, k]
            if self.hard:
                votes = (diff > 0).float()  # non-differentiable step
            else:
                votes = torch.sigmoid(self.beta * diff)  # smooth surrogate
            return votes

        if self.chunk_size and K > self.chunk_size:
            outs = []
            for start in range(0, K, self.chunk_size):
                chunk = self.pairs[start:start + self.chunk_size]
                outs.append(_compute_pairs(chunk))
            votes = torch.cat(outs, dim=1)
        else:
            votes = _compute_pairs(self.pairs)

        if self.weights is not None:
            # k-TSP style aggregation (learnable weights)
            return (votes * self.weights).sum(dim=1)  # [B]
        return votes  # [B, K]


if __name__ == "__main__":
    start_time = time.time()
    # Config
    D = 20_000  # number of features
    K = 500  # number of pairs
    B = 200_000  # batch size

    # Create non-overlapping pairs within the 20k feature space.
    # This deterministically pairs features across the two halves of the vector.
    pairs = [(i, 10_000 + i) for i in range(K)]

    # Layer
    layer = TSPLayer(pairs=pairs, beta=10.0, hard=False, learnable_weights=True)

    # Create a [B, D] tensor
    torch.manual_seed(0)
    x = torch.randn(B, D, dtype=torch.float32)

    votes = layer(x)
    print("x shape:", x.shape)
    print("num pairs:", len(pairs))
    print("votes shape:", votes.shape)
    print(votes)

    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time:.2f} seconds.')
