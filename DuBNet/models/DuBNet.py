import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv, CAB, ECAB, ECACoordCAB, GuidedECAB
from torchvision.ops import StochasticDepth
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class BrightnessEmbed(nn.Module):               # 新增：亮度条件编码
    def __init__(self, dim, max_freq=10):
        super().__init__()
        self.dim=dim
        self.max_freq=max_freq
        half_dim = dim//2
        emb = math.log(max_freq) / (half_dim-1)
        self.register_buffer('freq', torch.exp(torch.arange(half_dim)*-emb))  # [dim//2]

    def forward(self, x):                       # x: [B,C,H,W]
        bright = x.mean(dim=(1,2,3), keepdim=True)          # [B,1,1,1]
        emb = bright*self.freq.view(1,-1,1,1)               # [B,dim//2,1,1]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)      # [B,dim,1,1]
        return emb                                          # 直接广播加到特征


class ConvTransformerBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size=3,
        window_size=8,
        drop_path=0.1,
        use_brightness=False,
        num_heads: int = None,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.win = window_size
        self.dim = out_c // 2  # 注意力通道数
        assert self.dim > 0

        # 多头数量（如果不指定，就按 dim/32 取整）
        if num_heads is None:
            self.num_heads = max(1, self.dim // 32)
        else:
            self.num_heads = num_heads
        assert self.dim % self.num_heads == 0, "dim 必须能被 num_heads 整除"
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # 亮度嵌入
        self.use_brightness = use_brightness
        if use_brightness:
            self.bright_emb = BrightnessEmbed(self.dim)

        # 主干局部卷积: DWConv + PWConv 把 in_c -> dim
        self.depth_conv = nn.Conv2d(
            in_c, in_c, kernel_size, padding=kernel_size // 2, groups=in_c
        )
        self.point_conv = nn.Conv2d(in_c, self.dim, 1)

        # QKV 投影，在 dim 维度上做
        self.qkv = nn.Conv2d(self.dim, self.dim * 3, 1)
        self.attn_drop = nn.Dropout(0.1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 简单的 Conv-MLP 做通道混合
        hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, self.dim, 1),
        )

        # CSP 残差 & 门控融合
        self.shortcut = nn.Conv2d(in_c, self.dim, 1)
        self.gate = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim * 2, 1), nn.Sigmoid())
        self.fuse = nn.Conv2d(self.dim, out_c, 1)

    # 把 [B,C,H,W] 切成窗口 [B*nW, L, C]
    def _window_partition(self, x):
        B, C, H, W = x.shape
        win = self.win
        pad_h = (win - H % win) % win
        pad_w = (win - W % win) % win
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        x = x.view(B, C, Hp // win, win, Wp // win, win)  # [B,C,Wh,win,Ww,win]
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()      # [B,Wh,Ww,win,win,C]
        windows = x.view(-1, win * win, C)                # [B*nW, L, C]
        meta = (B, Hp, Wp, pad_h, pad_w)
        return windows, meta

    # 把 [B*nW, L, C] 还原为 [B,C,H,W]
    def _window_reverse(self, windows, meta):
        B, Hp, Wp, pad_h, pad_w = meta
        win = self.win
        C = self.dim
        x = windows.view(B, Hp // win, Wp // win, win, win, C)  # [B,Wh,Ww,win,win,C]
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()            # [B,C,Wh,win,Ww,win]
        x = x.view(B, C, Hp, Wp)
        if pad_h != 0 or pad_w != 0:
            x = x[:, :, : Hp - pad_h, : Wp - pad_w]
        return x

    def window_attn(self, x):
        """
        x: [B, dim, H, W]
        实现“窗内多头 self-attention”：
        - 先做 QKV
        - 再按窗口切分，每个窗口内部做 [L,L] 的 Attention
        """
        B, C, H, W = x.shape
        assert C == self.dim

        # QKV: [B, 3*dim, H, W]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)  # 每个 [B, dim, H, W]

        # 切窗: 得到 [B*nW, L, C]
        q_windows, meta = self._window_partition(q)
        k_windows, _    = self._window_partition(k)
        v_windows, _    = self._window_partition(v)

        BnW, L, C = q_windows.shape
        h = self.num_heads
        d = self.head_dim

        # 拆成多头: [B*nW, h, L, d]
        q_windows = q_windows.view(BnW, L, h, d).permute(0, 2, 1, 3)
        k_windows = k_windows.view(BnW, L, h, d).permute(0, 2, 1, 3)
        v_windows = v_windows.view(BnW, L, h, d).permute(0, 2, 1, 3)

        # 窗内 self-attention
        attn = (q_windows @ k_windows.transpose(-2, -1)) * self.scale  # [B*nW,h,L,L]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v_windows  # [B*nW,h,L,d]

        # 合并多头: [B*nW, L, C]
        out = out.permute(0, 2, 1, 3).contiguous().view(BnW, L, C)

        # 还原回 [B, dim, H, W]
        out = self._window_reverse(out, meta)
        return out

    def forward(self, x):
        # 1) 局部卷积特征
        feat = self.point_conv(self.depth_conv(x))  # [B, dim, H, W]

        # 2) 亮度条件，只在亮度分支开启
        if self.use_brightness:
            bright = self.bright_emb(x)             # [B, dim, 1, 1]
            feat = feat + bright

        # 3) 窗内多头 self-attention 残差
        attn_out = self.window_attn(feat)
        feat = feat + self.drop_path(attn_out)

        # 4) Conv-MLP 残差
        mlp_out = self.mlp(feat)
        feat = feat + self.drop_path(mlp_out)

        # 5) CSP 残差 & 门控融合
        shortcut = self.shortcut(x)       # [B, dim, H, W]
        fuse_in = torch.cat([feat, shortcut], dim=1)   # [B, 2*dim, H, W]
        g1, g2 = self.gate(fuse_in).chunk(2, dim=1)    # 各 [B, dim, H, W]
        out = g1 * feat + g2 * shortcut                # [B, dim, H, W]

        # 6) fuse 到 out_c 通道
        out = self.fuse(out)                           # [B, out_c, H, W]
        return out


class Conv_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DeConv_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.deconv = Conv_Relu(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.deconv(x)

# 原rcab
# class RCAB(nn.Module):
#     def __init__(self, in_feat, out_feat, kernel_size, reduction, n_blocks, bias=False, act=nn.ReLU(True)):
#         super().__init__()
#         self.conv1 = Conv(in_feat, out_feat, 3)
#         self.cab = nn.Sequential(
#             *[CAB(out_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=act) for _ in range(n_blocks)])
#         self.conv2 = Conv(out_feat, out_feat, 3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x1 = self.cab(x) + x
#         x = self.conv2(x1)
#         return x

# eca结合坐标
class RCAB(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        kernel_size,
        reduction,
        n_blocks,
        bias=False,
        act=nn.ReLU(True),
        use_eca=True,      # 是否用 ECA 系列
        use_coord=True,   # 是否在 ECA 上叠加 Coordinate Attention
    ):
        super().__init__()
        self.conv1 = Conv(in_feat, out_feat, 3)

        # 选择 Block 类型
        if use_eca and use_coord:
            Block = ECACoordCAB      # ECA + 坐标注意力
        elif use_eca:
            Block = ECAB             # 只有 ECA 通道注意力
        else:
            Block = CAB              # 原始 SE-CAB

        self.cab = nn.Sequential(
            *[
                Block(
                    out_feat,
                    kernel_size=kernel_size,
                    reduction=reduction,
                    bias=bias,
                    act=act,
                )
                for _ in range(n_blocks)
            ]
        )

        self.conv2 = Conv(out_feat, out_feat, 3)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.cab(x) + x   # 外层残差不变
        x = self.conv2(x1)
        return x

# 改进eca
# class RCAB(nn.Module):
#     def __init__(self, in_feat, out_feat, kernel_size, reduction, n_blocks,
#                  bias=False, act=nn.ReLU(True), use_eca=False, use_guided_eca=False):
#         super(RCAB, self).__init__()
#         self.conv1 = Conv(in_feat, out_feat, 3)
#         self.use_guided_eca = use_guided_eca
#         if use_guided_eca:
#             self.cab = nn.Sequential(*[
#                 GuidedECAB(out_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=act)
#                 for _ in range(n_blocks)
#             ])
#         else:
#             Block = ECAB if use_eca else CAB
#             self.cab = nn.Sequential(*[
#                 Block(out_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=act)
#                 for _ in range(n_blocks)
#             ])
#         self.conv2 = Conv(out_feat, out_feat, 3)
#
#     def forward(self, x, guide=None):
#         x = self.conv1(x)
#         if self.use_guided_eca:
#             x1 = x
#             for blk in self.cab:
#                 x1 = blk(x1, guide)
#             x1 = x1 + x
#         else:
#             x1 = self.cab(x) + x
#         x = self.conv2(x1)
#         return x

class PCM(nn.Module):
    def __init__(self, channels, n_feat, kernel_size, bias):
        super().__init__()
        self.conv1 = Conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = Conv(n_feat, channels, kernel_size, bias=bias)
        self.conv3 = Conv(channels, n_feat, kernel_size, bias=bias)
        self.conv4 = Conv(n_feat, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = self.conv3(img)
        x3 = torch.sigmoid(self.conv4(x1))
        x2 = x3 * x2
        x1 = x + x2
        return x1, img


class PEN(nn.Module):
    def __init__(self, channel, n_feats, use_brightness=False):
        super().__init__()
        self.enc1 = nn.Sequential(
            ConvTransformerBlock(channel, n_feats, kernel_size=3, window_size=8, use_brightness=use_brightness),
            Conv(n_feats, n_feats, 3)
        )
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            ConvTransformerBlock(n_feats, n_feats * 2, kernel_size=3, window_size=8, use_brightness=use_brightness),
            Conv(n_feats * 2, n_feats * 2, 3)
        )
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvTransformerBlock(n_feats * 2, n_feats * 4, kernel_size=3, window_size=4, use_brightness=use_brightness)
        self.conv3 = Conv(n_feats * 4, n_feats * 4, 3)

        self.up32 = DeConv_Relu(n_feats * 4, n_feats * 2, 3, 1)
        self.dec2 = nn.Sequential(
            Conv_Relu(n_feats * 4, n_feats * 4, 3, 1),
            Conv(n_feats * 4, n_feats * 2, 3)
        )

        self.up21 = DeConv_Relu(n_feats * 2, n_feats, 3, 1)
        self.dec1 = nn.Sequential(
            Conv_Relu(n_feats * 2, n_feats * 2, 3, 1),
            Conv(n_feats * 2, n_feats, 3)
        )

        self.out2 = PCM(channel, n_feats, 3, bias=False)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pooling1(enc1))
        neck = self.enc3(self.pooling2(enc2))
        neck = self.conv3(neck)

        up2 = self.up32(neck)
        cat2 = torch.cat([up2, enc2], 1)
        dec2 = self.dec2(cat2)

        up1 = self.up21(dec2)
        cat1 = torch.cat([up1, enc1], 1)
        dec1 = self.dec1(cat1)

        feats, out = self.out2(dec1, x)
        out = torch.sigmoid(out)
        return out, feats


class PGB(nn.Module):
    def __init__(self, in_channel=3, f_channel=64, g_channel=1):
        super().__init__()
        self.conv1 = Conv(g_channel, 1, 1)
        self.conv2 = Conv(g_channel, f_channel, 1)
        self.conv3 = Conv(in_channel, f_channel, 3)
        self.conv4 = Conv(f_channel, f_channel, 3)
        self.conv5 = Conv(f_channel, f_channel, 3)

    def forward(self, img, guide_f):
        guide_mul = torch.sigmoid(self.conv1(guide_f))
        guide_add = self.conv2(guide_f)
        x = self.conv3(img)
        x = x * guide_mul
        x = self.conv4(x)
        x = x + guide_add
        x = self.conv5(x)
        return x


class IEN(nn.Module):
    def __init__(self, in_channel=3, f_channel=48, w_channel=48):
        super(IEN, self).__init__()

        self.layer0 = nn.Sequential(
                      Conv(in_channel, f_channel, 3),
                      Conv(f_channel, f_channel, 3)
        )

        self.para = torch.nn.Parameter(torch.ones(w_channel, 1, 1))

        self.guide1 = PGB(f_channel, f_channel, w_channel)
        self.layer1 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True), use_eca=True, use_coord=True)
        self.layer1 = RCAB(f_channel, f_channel, 3, 8, 3, bias=False, act=nn.ReLU(True), use_eca=True, use_coord=True)

        self.guide2 = PGB(f_channel, f_channel, w_channel)
        self.layer2 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True), use_eca=True, use_coord=True)
        self.guide3 = PGB(f_channel, f_channel, w_channel)
        self.layer3 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True), use_eca=True, use_coord=True)
        self.guide4 = PGB(f_channel, f_channel, w_channel)
        self.layer4 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True), use_eca=True, use_coord=True)
        self.guide5 = PGB(f_channel, f_channel, w_channel)
        self.layer5 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True), use_eca=True, use_coord=True)
        self.guide6 = PGB(f_channel, f_channel, w_channel)
        self.layer6 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True), use_eca=True, use_coord=True)
        self.out = Conv(f_channel, 3, 3)

    def forward(self, img, illu, rest):
        x = self.layer0(img)
        res_illu = illu-self.para*rest
        x = self.guide1(x, res_illu)
        x = self.layer1(x)
        x = self.guide2(x, rest)
        x = self.layer2(x)
        x = self.guide3(x, res_illu)
        x = self.layer3(x)
        x = self.guide4(x, rest)
        x = self.layer4(x)
        x = self.guide5(x, res_illu)
        x = self.layer5(x)
        x = self.guide6(x, rest)
        x = self.layer6(x)
        x = self.out(x)
        x = torch.sigmoid(x)

        return x
# class IEN(nn.Module):
#     def __init__(self, in_channel=3, f_channel=48, w_channel=48):
#         super(IEN, self).__init__()
#         self.layer0 = nn.Sequential(Conv(in_channel, f_channel, 3), Conv(f_channel, f_channel, 3))
#         self.para = torch.nn.Parameter(torch.ones(w_channel, 1, 1))
#         self.guide1 = PGB(f_channel, f_channel, w_channel)
#         self.layer1 = RCAB(f_channel, f_channel, 3, 8, 3, bias=False, act=nn.ReLU(True), use_guided_eca=True)
#         self.guide2 = PGB(f_channel, f_channel, w_channel)
#         self.layer2 = RCAB(f_channel, f_channel, 3, 8, 3, bias=False, act=nn.ReLU(True), use_guided_eca=True)
#         self.guide3 = PGB(f_channel, f_channel, w_channel)
#         self.layer3 = RCAB(f_channel, f_channel, 3, 8, 3, bias=False, act=nn.ReLU(True), use_guided_eca=True)
#         self.guide4 = PGB(f_channel, f_channel, w_channel)
#         self.layer4 = RCAB(f_channel, f_channel, 3, 8, 3, bias=False, act=nn.ReLU(True), use_guided_eca=True)
#         self.guide5 = PGB(f_channel, f_channel, w_channel)
#         self.layer5 = RCAB(f_channel, f_channel, 3, 8, 3, bias=False, act=nn.ReLU(True), use_guided_eca=True)
#         self.guide6 = PGB(f_channel, f_channel, w_channel)
#         self.layer6 = RCAB(f_channel, f_channel, 3, 8, 3, bias=False, act=nn.ReLU(True), use_guided_eca=True)
#         self.out = Conv(f_channel, 3, 3)
#
#     def forward(self, img, illu, rest):
#         x = self.layer0(img)
#         res_illu = illu - self.para * rest
#         x = self.guide1(x, res_illu)
#         x = self.layer1(x, res_illu)
#         x = self.guide2(x, rest)
#         x = self.layer2(x, rest)
#         x = self.guide3(x, res_illu)
#         x = self.layer3(x, res_illu)
#         x = self.guide4(x, rest)
#         x = self.layer4(x, rest)
#         x = self.guide5(x, res_illu)
#         x = self.layer5(x, res_illu)
#         x = self.guide6(x, rest)
#         x = self.layer6(x, rest)
#         x = self.out(x)
#         return x

class Illum_YCRCB_Denoise_IN(nn.Module):
    def __init__(self):
        super().__init__()
        self.illu = PEN(channel=1, n_feats=128, use_brightness=True)
        self.denoise = PEN(channel=2, n_feats=128, use_brightness=False)
        self.FuN = IEN(3, 128, 128)
        # self.eyuv = EYUV()

    def forward(self, yuv):
        # enhanced_yuv, _ = self.eyuv(yuv)
        y = yuv[:, 0:1, :, :]
        uv = yuv[:, 1:3, :, :]

        out_y, out_y_feats = self.illu(y)
        out_uv, out_uv_feats = self.denoise(uv)

        out = self.FuN(yuv, out_y_feats, out_uv_feats)
        return out_y, out_uv, out

class DirectConcatFuse(nn.Module):
    """
    Direct concatenation fusion baseline.
    It replaces Fus-Net with a simple concat + convolution head.
    """
    def __init__(self, in_channel=3, f_channel=128, guide_channel=128):
        super().__init__()
        self.fuse = nn.Sequential(
            Conv(guide_channel * 2 + in_channel, f_channel, 3),
            nn.ReLU(inplace=True),
            Conv(f_channel, f_channel, 3),
            nn.ReLU(inplace=True),
            Conv(f_channel, in_channel, 3),
        )

    def forward(self, img, illu, rest):
        x = torch.cat([img, illu, rest], dim=1)
        out = self.fuse(x)
        return out


class TwoBranch_DirectConcat_IN(nn.Module):
    """
    Two-branch direct-concat ablation.
    It keeps Lum-Net and Chr-Net unchanged, but replaces Fus-Net
    with direct concatenation followed by convolutional reconstruction.
    """
    def __init__(self):
        super().__init__()
        self.illu = PEN(channel=1, n_feats=64, use_brightness=True)
        self.denoise = PEN(channel=2, n_feats=64, use_brightness=False)
        self.FuN = DirectConcatFuse(in_channel=3, f_channel=64, guide_channel=128)

    def forward(self, yuv):
        y = yuv[:, 0:1, :, :]
        uv = yuv[:, 1:3, :, :]

        out_y, out_y_feats = self.illu(y)
        out_uv, out_uv_feats = self.denoise(uv)

        out = self.FuN(yuv, out_y_feats, out_uv_feats)
        return out_y, out_uv, out

class SingleBranch_YCRCB_IN(nn.Module):
    """
    Single-stream YCrCb baseline for ablation.
    It takes the complete YCrCb image as input and restores all three channels
    using one shared backbone, without explicit luminance/chrominance branches.
    """
    def __init__(self):
        super().__init__()
        self.single = PEN(channel=3, n_feats=128, use_brightness=True)

    def forward(self, yuv):
        out, feats = self.single(yuv)

        out_y = out[:, 0:1, :, :]
        out_uv = out[:, 1:3, :, :]

        return out_y, out_uv, out