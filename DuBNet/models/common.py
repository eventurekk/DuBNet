import torch
import torch.nn as nn
import math


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, stride=1, norm=False, groups=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride, groups=groups)
        self.norm = norm

        if self.norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif self.norm == 'LN':
            self.norm_layer = nn.InstanceNorm2d(num_features=out_channels, momentum=0.3, affine=True, track_running_stats=True)

    def forward(self, x):
        if self.norm:
            return self.norm_layer(self.conv(x))
        else:
            return self.conv(x)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = [Conv(n_feat, n_feat, kernel_size, bias=bias), act, Conv(n_feat, n_feat, kernel_size, bias=bias)]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log2(channel) / gamma) + (b / gamma)))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            padding=(k - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class DualGuidedECALayer(nn.Module):
    """
    Dual-branch Guided ECA:
    current feature and guide feature generate separate channel responses,
    then the two responses are fused with a learnable guide scale.
    """
    def __init__(self, channel, gamma=2, b=1):
        super(DualGuidedECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log2(channel) / gamma) + (b / gamma)))
        k = t if t % 2 else t + 1
        self.guide_proj = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.guide_scale = nn.Parameter(torch.tensor(0.2))
        self.conv_x = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_g = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, guide):
        desc_x = self.avg_pool(x)
        desc_g = self.avg_pool(self.guide_proj(guide))

        desc_x = desc_x.squeeze(-1).transpose(-1, -2)
        desc_g = desc_g.squeeze(-1).transpose(-1, -2)

        att_x = self.conv_x(desc_x)
        att_g = self.conv_g(desc_g)
        scale = torch.sigmoid(self.guide_scale)
        y = att_x + scale * att_g
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class ECAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(ECAB, self).__init__()
        modules_body = [
            Conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            Conv(n_feat, n_feat, kernel_size, bias=bias),
        ]

        self.eca = ECALayer(n_feat)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.eca(res)
        res += x
        return res


class GuidedECAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(GuidedECAB, self).__init__()
        modules_body = [
            Conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            Conv(n_feat, n_feat, kernel_size, bias=bias),
        ]
        self.eca = DualGuidedECALayer(n_feat)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, guide):
        res = self.body(x)
        res = self.eca(res, guide)
        res += x
        return res


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        _, _, H, W = x.size()
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


class ECACoordCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(ECACoordCAB, self).__init__()
        self.body = nn.Sequential(
            Conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            Conv(n_feat, n_feat, kernel_size, bias=bias),
        )
        self.eca = ECALayer(n_feat)
        self.coord = CoordAtt(n_feat, reduction=reduction)

    def forward(self, x):
        res = self.body(x)
        res = self.eca(res)
        res = self.coord(res)
        res += x
        return res


class ECACoordAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, reduction=16, bias=False, act=nn.ReLU(True)):
        super(ECACoordAttention, self).__init__()
        self.eca = ECAB(in_channels, kernel_size=kernel_size, reduction=reduction, bias=bias, act=act)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x_eca = self.eca(x)
        avg_out = self.conv1(self.avg_pool(x_eca))
        max_out = self.conv2(self.max_pool(x_eca))
        out = self.sigmoid(avg_out + max_out)
        return x_eca * out
