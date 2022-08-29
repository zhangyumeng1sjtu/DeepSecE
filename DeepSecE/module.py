from torch import nn, einsum
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_key=64, dim_value=64, dropout=0.):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)
        self.to_out = nn.Linear(dim_value * heads, dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, mask=None):
        n, h = x.shape[-2], self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        q = q * self.scale
        logits = einsum('b h i d, b h j d -> b h i j', q, k)

        if mask is not None:
            logits.masked_fill(mask == 0, -1e9)

        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn


class TransformerLayer(nn.Module):

    def __init__(self, hid_dim, heads, dropout_rate, att_dropout=0.05):
        super().__init__()
        self.attn = Attention(hid_dim, heads, hid_dim //
                              heads, hid_dim // heads, att_dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, hid_dim * 2),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Dropout(dropout_rate))
        self.layernorm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):

        residual = x
        x = self.layernorm(x)  # pre-LN
        x, attn = self.attn(x, mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.ffn(x)
        x = residual + x

        return x, attn


class MLPLayer(nn.Module):
    
    def __init__(self, in_dim, hid_dim, num_classes, dropout_rate=0.):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_rate),
                                nn.Linear(hid_dim, num_classes)
                            )
        
    def forward(self, x):
        return self.layer(x)
