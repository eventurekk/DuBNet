import torch


def yuv2rgb(yuv):
    # 标准 YUV601 → sRGB 矩阵，CUDA/CPU 通用
    Kr, Kg, Kb = 0.299, 0.587, 0.114
    mat = torch.tensor([[1,  0,  1.402],
                        [1, -0.344136, -0.714136],
                        [1,  1.772,  0]], dtype=yuv.dtype, device=yuv.device)
    # yuv 范围 y∈[0,1], uv∈[-1,1]
    y  = yuv[:, 0:1]
    u  = yuv[:, 1:2]
    v  = yuv[:, 2:3]
    rgb = torch.cat([y + 1.402*v,
                     y - 0.344136*u - 0.714136*v,
                     y + 1.772*u], dim=1)
    return rgb.clamp(0, 1)