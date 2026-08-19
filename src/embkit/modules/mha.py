
import torch
from torch import nn

from .. import factory

@factory.nn_module
class MHABlock(nn.Module):
    """
    Single multihead-attention block
    """
    def __init__(self, embed_dim, num_heads, out_dim, ffn_inner_dim=None, dropout=0.01, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dropout = dropout
        if ffn_inner_dim is None:
            ffn_inner_dim = embed_dim * 4
        self.ffn_inner_dim = ffn_inner_dim
        self.mha = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True, device=device, dtype=dtype)
        self.norm1 = nn.LayerNorm(embed_dim, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(out_dim, device=device, dtype=dtype)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Identity()
        if out_dim != embed_dim:
            self.residual_proj = nn.Linear(embed_dim, out_dim, device=device, dtype=dtype)
        #self.ffn = nn.Linear(embed_dim, out_dim, device=device)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.ffn_inner_dim, device=device, dtype=dtype), nn.GELU(), #nn.ReLU(),
            nn.Linear(self.ffn_inner_dim, out_dim, device=device, dtype=dtype),
        )

    def forward(self, x):
        attn_out, _ = self.mha(query=x, key=x, value=x)
        query = x + self.attn_dropout(attn_out)
        query = self.norm1(query)

        ffn_out = self.ffn(query)
        output = self.residual_proj(query) + self.ffn_dropout(ffn_out)
        output = self.norm2(output)
        return output

    def to_dict(self):
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "out_dim":self.out_dim,
            "ffn_inner_dim": self.ffn_inner_dim
        }

    @classmethod
    def from_dict(cls, params):
        return MHABlock(
            embed_dim=params["embed_dim"],
            num_heads=params["num_heads"],
            dropout=params["dropout"],
            out_dim=params["out_dim"],
            ffn_inner_dim=params["ffn_inner_dim"]
        )

@factory.nn_module
class MHAPooling(nn.Module):
    def __init__(self, embed_dim, num_heads=1, device=None, dtype=None):
        super().__init__()
        # The learnable query token that "probes the sequence
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim, device=device, dtype=dtype))
        
        # Multi-head attention layer acting as the pooling mechanism
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, device=device, dtype=dtype)
        
        # Optional layer norm to stabilize training
        self.ln = nn.LayerNorm(embed_dim, device=device, dtype=dtype)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Optional binary mask of shape (batch_size, seq_len) 
                              where True indicates values to be ignored.
        Returns:
            pooled_output: Shape (batch_size, embed_dim)
        """
        batch_size = x.size(0)
        
        # Expand the learnable query token to match the batch size
        # Shape becomes: (batch_size, 1, embed_dim)
        q = self.query_token.expand(batch_size, -1, -1)
        
        # Cross-attention: Query comes from our learnable parameter, 
        # Keys and Values come from the input sequence 'x'.
        # attn_output shape: (batch_size, 1, embed_dim)
        attn_output, _ = self.attn(
            query=q, 
            key=x, 
            value=x, 
            key_padding_mask=key_padding_mask
        )
        
        # Remove the singleton sequence dimension and apply norm
        # Output shape: (batch_size, embed_dim)
        pooled_output = self.ln(attn_output.squeeze(1))
        return pooled_output
    
    def to_dict(self):
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads
        }

    @classmethod
    def from_dict(cls, params):
        return MHAPooling(
            embed_dim=params["embed_dim"],
            num_heads=params["num_heads"]
        )
