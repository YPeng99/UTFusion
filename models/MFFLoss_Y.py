import torch
import torch.nn as nn
from kornia.losses import ssim_loss
import torch.nn.functional as F


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_Grad(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        ssim_loss_A = ssim_loss(image_A, image_fused, 11)
        ssim_loss_B = ssim_loss(image_B, image_fused, 11)
        return torch.mean(ssim_loss_A + ssim_loss_B)


class fusion_loss_mff(nn.Module):
    """无监督LOSS"""

    def __init__(self):
        super().__init__()
        self.L_Inten = nn.L1Loss(reduction='none')
        self.L_Grad = L_Grad()
        self.L_SSIM = L_SSIM()

    def forward(self, image_A, image_B, image_fused, fuse_scheme):
        g_loss = self.L_Grad(image_A, image_B, image_fused)
        l1_loss = self.L_Inten(image_fused, torch.max(image_A, image_B))
        mean_index = fuse_scheme == 0
        l1_loss[mean_index] = self.L_Inten(image_fused[mean_index], (image_A[mean_index] + image_B[mean_index]) / 2)
        fusion_loss = torch.mean(l1_loss) + g_loss * 2
        return fusion_loss


if __name__ == '__main__':
    S1 = torch.randn((8, 1, 224, 224))
    S2 = torch.randn((8, 1, 224, 224))
    fused = torch.randn((8, 1, 224, 224))
    fuse_scheme = torch.randint(0, 2, (8,))
    loss = fusion_loss_mff()
    l = loss(S1, S2, fused, fuse_scheme)

    print(l)
