import torch
import torch.nn as nn

class PairwiseComparison(nn.Module):
    """
    A layer that performs pairwise comparisons between features in the input tensor.
    The layer computes pairwise differences, ratios, or boolean comparisons (greater than, less than)
    between all unique pairs of features in the input tensor.
    The output tensor contains the results of these comparisons for each input sample.
    The number of output features is given by the formula: num_pairs = feature_dim * (feature_dim - 1) // 2,
    where feature_dim is the number of features in the input tensor.
    Supported comparison types:
    - "difference": Computes the difference between each pair of features (feature_i - feature_j).
    - "ratio": Computes the ratio between each pair of features (feature_i / feature_j).
    - "greater_than": Outputs 1 if feature_i > feature_j, else 0.
    - "less_than": Outputs 1 if feature_i < feature_j, else 0.
    Example:
        >>> layer = PairwiseComparison(comparison_type="difference")
        >>> input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> output_tensor = layer(input_tensor)
        >>> print(output_tensor)
        tensor([[-1., -2., -1.],
                [-1., -2., -1.]])
    """
    def __init__(self, comparison_type="difference"):
        super().__init__()
        if comparison_type not in {"difference", "ratio", "greater_than", "less_than"}:
            raise ValueError("Unsupported comparison_type")
        self.comparison_type = comparison_type
        self.register_buffer('i_idx', None)
        self.register_buffer('j_idx', None)

    def build_indices(self, feature_dim: int) -> None:
        """
        Build indices for pairwise comparisons of features.
        This creates two buffers: i_idx and j_idx, which contain the indices of the features
        to be compared. The pairs are generated such that i < j, covering all unique pairs
        of features in the input tensor.
        This is done only once per feature dimension.

        Args:
            feature_dim (int): The number of features in the input tensor.

        Returns:
            None
        """
        # Only build once
        if self.i_idx is not None and self.j_idx is not None:
            return
        pairs = [(i, j) for i in range(feature_dim) for j in range(i + 1, feature_dim)]
        i_idx, j_idx = zip(*pairs)
        self.register_buffer('i_idx', torch.tensor(i_idx, dtype=torch.long))
        self.register_buffer('j_idx', torch.tensor(j_idx, dtype=torch.long))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for pairwise comparison of features.

        Args:
            inputs (torch.Tensor): A 2D tensor of shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: A 2D tensor of shape (batch_size, num_pairs) where
                            num_pairs = feature_dim * (feature_dim - 1) // 2.

        """
        if inputs.ndim != 2:
            raise ValueError("Input must be 2D (batch_size, feature_dim)")
        batch_size, feature_dim = inputs.shape
        self.build_indices(feature_dim)

        feat_i = inputs[:, self.i_idx]  # shape: [batch_size, num_pairs]
        feat_j = inputs[:, self.j_idx]  # shape: [batch_size, num_pairs]

        if self.comparison_type == "difference":
            out = feat_i - feat_j
        elif self.comparison_type == "ratio":
            out = feat_i / (feat_j + 1e-7)
        elif self.comparison_type == "greater_than":
            out = (feat_i > feat_j).float()
        elif self.comparison_type == "less_than":
            out = (feat_i < feat_j).float()
        else:
            raise ValueError("Invalid comparison_type")

        return out

    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, feature_dim)
        n = input_shape[-1]
        return input_shape[:-1] + (n * (n - 1) // 2,)