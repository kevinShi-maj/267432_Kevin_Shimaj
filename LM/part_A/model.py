import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, L, d_model = x.size()

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        similarity = q @ k.transpose(-2, -1)
        similarity = similarity * (1 / torch.sqrt(torch.tensor(self.h_dim, dtype=torch.float)))
        similarity = similarity.masked_fill(mask == 0, float('-inf'))

        attn = self.attn_dropout(F.softmax(similarity, dim=-1))
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, L, d_model)
        y = self.proj_dropout(self.out_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.pos_emb_size = pos_emb_size

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        # Step 3 — weight tying: input embedding and output projection share the
        # same vocab_size x d_model matrix (same Parameter object, ~12.9M params saved)
        self.lm_head.weight = self.token_embed.weight

        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, idx):
        B, L = idx.shape
        assert L <= self.pos_emb_size

        pos = torch.arange(L, device=idx.device)
        x = self.emb_dropout(self.token_embed(idx) + self.pos_embed(pos))

        mask = self.mask[:, :, :L, :L]
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
