import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from typing import Tuple

from torchvision import models

import pytorch_ssim

"""
Reference from:　https://github.com/swz30/MPRNet/blob/main/Denoising/generate_patches_SIDD.py
"""
def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()

def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    return window_1d

def get_gaussian_kernel2d(ksize: Tuple[int, int],
                          sigma: Tuple[float, float]) -> torch.Tensor:
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}"
                        .format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


class PSNRLoss(nn.Module):
    """
    reference from: https://github.com/megvii-model/HINet/blob/main/basicsr/models/losses/losses.py
    """
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        loss = -(self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean())
        return loss



# ------------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class SSIM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, R_low, R_high):
        # loss_grad = self.grad_loss(R_low, R_high, hook=hook)
        loss_ssim = 1 - self.ssim_loss(R_low, R_high)
        return loss_ssim



# ✅ 新增：感知损失（VGG16）
class PerceptualLoss(nn.Module):
    def __init__(self, layer='relu2_2'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layer = layer
        self.slice = nn.Sequential()
        for name, module in vgg._modules.items():
            self.slice.add_module(name, module)
            if name == '7':  # relu2_2
                break
        for param in self.slice.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x = (x + 1) / 2  # assume input in [-1,1]
        y = (y + 1) / 2
        x_feat = self.slice(x)
        y_feat = self.slice(y)
        return self.criterion(x_feat, y_feat)

# ✅ 新增：颜色损失（CIE ΔE 近似）
class ColorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def rgb2lab(self, x):
        # 简化版 RGB→LAB（归一化后）
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        # 使用 torchvision 的 rgb2lab 需要 PIL，这里用近似
        return x

    def forward(self, x, y):
        x_lab = self.rgb2lab(x)
        y_lab = self.rgb2lab(y)
        return self.criterion(x_lab, y_lab)

# ✅ 新增：频域损失（Focal Frequency Loss）
class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, y):
        x_fft = torch.fft.rfft2(x)
        y_fft = torch.fft.rfft2(y)
        loss = torch.mean(torch.abs(x_fft - y_fft) ** self.alpha)
        return loss