import torch
import torch.nn as nn
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
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint, reduction='none')
        Loss_gradient = torch.mean(Loss_gradient, dim=[1, 2, 3])
        return Loss_gradient





class fusion_loss_mff(nn.Module):
    """无监督LOSS"""

    def __init__(self):
        super().__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = nn.L1Loss(reduction='none')

    def forward(self, image_A, image_B, image_fused, fuse_scheme):
        weight = torch.ones_like(fuse_scheme).float()
        weight[fuse_scheme == 1] = 2

        loss_l1 = self.L_Inten(image_fused, torch.max(image_A, image_B))
        mean_index = fuse_scheme == 1
        loss_l1[mean_index] = self.L_Inten(image_fused[mean_index],
                                           torch.mean(torch.stack([image_A[mean_index], image_B[mean_index]],
                                                                  dim=0), dim=0))
        loss_l1 = torch.mean(loss_l1, dim=[1, 2, 3]) * weight
        loss_gradient = self.L_Grad(image_A, image_B, image_fused) * weight

        loss_l1 = torch.mean(loss_l1)
        loss_gradient = torch.mean(loss_gradient)

        fusion_loss = 1 * loss_l1 + 2 * loss_gradient
        return fusion_loss, loss_l1, loss_gradient


if __name__ == '__main__':
    S1 = torch.randn((8, 1, 224, 224))
    S2 = torch.randn((8, 1, 224, 224))
    fused = torch.randn((8, 1, 224, 224))
    fuse_scheme = torch.randint(0, 2, (8,))
    loss = fusion_loss_mff()
    l = loss(S1, S2, fused, fuse_scheme)
    print(l)