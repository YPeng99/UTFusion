import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RMSNorm import RMSNorm
from models.DropPath import DropPath


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fn = fn

    def forward(self, x, *args):
        x = self.norm(x)
        return self.fn(x, *args)


class PatchAttention(nn.Module):
    def __init__(self, dim=192, n_head=12, dropout=0., bias=True):
        super().__init__()
        assert dim % n_head == 0, f"dim {dim} must be divisible by n_head {n_head}"

        self.dim = dim
        self.n_head = n_head
        self.head_size = self.dim // self.n_head
        self.scale = self.head_size ** -0.5
        self.proj_qkv = nn.Sequential(
            nn.Linear(dim, dim * 3, bias=bias),
            nn.Dropout(dropout),
        )
        self.proj_out = nn.Sequential(
            nn.Linear(dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        B, N, H, W, C = x.shape
        q, k, v = self.proj_qkv(x).chunk(3, -1)

        q = q.view(B, N, H, W, self.n_head, self.head_size).permute(0, 2, 3, 4, 1, 5)
        k = k.view(B, N, H, W, self.n_head, self.head_size).permute(0, 2, 3, 4, 5, 1)
        v = v.view(B, N, H, W, self.n_head, self.head_size).permute(0, 2, 3, 4, 1, 5)

        dots = torch.matmul(q, k) * self.scale

        if mask is not None:
            dots = dots.masked_fill(mask, -torch.inf)

        dots = F.softmax(dots, dim=-1)

        x = torch.matmul(dots, v).permute(0, 4, 1, 2, 3, 5).flatten(-2)
        x = self.proj_out(x)
        return x


class SwinGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., bias=True):
        super().__init__()
        self.w1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=bias),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=bias, groups=hidden_dim),
        )

        self.w2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=bias)
        self.w3 = nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, H, W, C = x.shape
        x = x.reshape(B * N, H, W, -1).permute(0, 3, 1, 2)
        x = self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        x = x.permute(0, 2, 3, 1).reshape(B, N, H, W, C)
        return x



class Former(nn.Module):
    def __init__(self, dim, n_head, hidden_dim, dropout=0.1, drop_path=0., bias=False):
        super().__init__()
        self.token_mixer = PreNorm(dim, PatchAttention(dim, n_head, dropout, bias))
        self.ffn = PreNorm(dim, SwinGLU(dim, hidden_dim, dropout, bias))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.token_mixer(x, mask))
        x = x + self.drop_path(self.ffn(x))
        return x


if __name__ == '__main__':
    x = torch.randn(8, 3, 3, 224, 224).cuda()
    x = torch.pixel_unshuffle(x, 8).permute(0, 1, 3, 4, 2)
    print(x.shape)
    model = Former(192, 12, 256).cuda()
    model.eval()
    x = model(x)
    print(x.shape)
