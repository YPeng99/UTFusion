import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MetaFormer import Former
from utils import get_parameter_number


class UTFusion(nn.Module):
    """
    introduction
    """

    def __init__(self, in_ch=1, n_head=8, patch_size=8,depth=6,mask=False,use_checkpoint = False):  # 8 8
        super().__init__()
        self.in_ch = in_ch
        self.n_head = n_head
        self.patch_size = patch_size
        self.mask = mask

        self.dim = int(in_ch * patch_size ** 2)
        self.fused_embedding = nn.Embedding(2, self.dim)

        self.in_proj = nn.Linear(self.dim,self.dim)

        self.formers = nn.ModuleList([Former(self.dim, self.n_head, self.dim * 2,use_checkpoint=use_checkpoint) for _ in range(depth)])

        self.out_proj = nn.Linear(self.dim, self.dim)

    def check_image_size(self, *input):
        H, W = input[0].shape[-2:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        outs = [F.pad(i, [0, pad_w, 0, pad_h], 'reflect') for i in input]
        outs = torch.stack(outs, 1)
        return outs, H, W

    def forward(self, *input):
        assert len(input[0]) >= 2, 'num of image must more than 2'
        outs, o_H, o_W = self.check_image_size(*list(input[0]))
        if len(input) ==1:
            fused_scheme = torch.ones(outs.shape[0],dtype=torch.int).to(outs.device)
        else:
            fused_scheme = input[1].repeat(outs.shape[0]) if len(input[1]) == 1 else input[1]

        patch = self.fused_embedding(fused_scheme)
        outs = torch.pixel_unshuffle(outs, self.patch_size).permute(0, 1, 3, 4, 2)
        patch = patch[:,None,None,None,:].expand(outs.shape[0], 1, outs.shape[2], outs.shape[3], outs.shape[4])
        outs = torch.cat([outs, patch], dim=1)

        mask = None
        if self.mask:
            mask = torch.zeros(outs.shape[1], outs.shape[1])
            mask[:-1, -1] = 1
            mask = mask.bool().to(outs.device)

        outs = self.in_proj(outs)
        for former in self.formers:
            outs = former(outs, mask=mask)

        out = outs[:,-1,...]
        out = self.out_proj(out).permute(0, 3, 1, 2)
        out = torch.pixel_shuffle(out, self.patch_size)
        return out[..., :o_H, :o_W]


if __name__ == '__main__':
    device = 'cuda'
    model = UTFusion().to(device)
    model.eval()
    x = torch.randn(10, 1, 520, 520).to(device)
    fuse_scheme = torch.randint(0, 2, (10,)).to(device)
    with torch.inference_mode():
        start = time.time()
        out = model([x, x],fuse_scheme)
        get_parameter_number(model)
    end = time.time()
    print(end - start)