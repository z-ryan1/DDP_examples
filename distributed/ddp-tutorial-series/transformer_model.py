"""
Shared decoder-only transformer and synthetic dataset for the DDP + Nsight Systems
profiling examples (single_gpu_nsys.py and multigpu_ddp_nsys.py).

Architecture mirrors a small GPT-style model:
  - Token + positional embeddings
  - N x (pre-norm attention + pre-norm MLP) blocks
  - Tied token embedding / output projection weights

Default config: vocab=8192, d_model=512, heads=8, layers=6, seq_len=128 (~25 M params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # PyTorch 2.x: dispatches to FlashAttention-2 when available
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SmallTransformer(nn.Module):
    """
    Decoder-only transformer for next-token prediction.

    Default config (~25 M params):
        vocab_size=8192, d_model=512, n_heads=8, n_layers=6, seq_len=128

    Scale up d_model / n_layers to stress the GPU more and make
    the DDP gradient-sync overlap easier to see in Nsight Systems.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        seq_len: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Standard weight tying: embedding and output projection share parameters
        self.tok_emb.weight = self.head.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.seq_len, f"Sequence length {T} exceeds model max {self.seq_len}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


class SyntheticTokenDataset(Dataset):
    """
    Synthetic next-token-prediction dataset.
    Generates random integer token sequences on CPU at construction time.
    Each item is an (input_tokens, target_tokens) pair of length seq_len.
    """

    def __init__(self, size: int, seq_len: int, vocab_size: int):
        self.tokens = torch.randint(0, vocab_size, (size, seq_len + 1))

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int):
        seq = self.tokens[idx]
        return seq[:-1], seq[1:]  # (input, target)
