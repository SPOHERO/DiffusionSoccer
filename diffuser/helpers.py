import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim=dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device) * - emb)
        emb = x[:, None] * emb[None,:]
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb
    
class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)
    
class Conv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding = k//2),
            nn.GroupNorm(groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)
    
def extract(a, t, x_shape, device=None):
    """
    a: buffer (len=T) on some device
    t: [B] long
    returns: [B, 1, ...] broadcastable to x_shape
    """
    b, *_ = t.shape
    if device is not None:
        out = a.gather(-1, t.to(device))
        return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(device)
    else:
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule (Nichol & Dhariwal 2021)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def apply_conditioning(x, conditions, cond_mask=None, action_dim=0, device=None,
                       pos_dim=None):
    """
    x : [B, T, D] — 전체 trajectory (pos + delta)
    conditions : dict[timestep -> tensor([B, D])]
    pos_dim : position feature dimension (예: 46)
    """
    B, T, D = x.shape
    if pos_dim is None:
        raise ValueError("pos_dim (position dimension) must be provided")

    # ✅ 추가: 텐서 입력도 자동 변환
    if isinstance(conditions, torch.Tensor):
        conditions = {t: conditions[:, t, :] for t in range(conditions.shape[1])}

    for t, val in conditions.items():
        if device is not None:
            val = val.to(device)
        # pos는 항상 condition에서 가져와 고정
        x[:, t, :pos_dim] = val[:, :pos_dim].clone()

        # delta 부분만 condition mask 적용
        if cond_mask is not None:
            if device is not None:
                cond_mask = cond_mask.to(device)
            delta_mask = cond_mask[:, t, pos_dim:]
            x[:, t, pos_dim:] = x[:, t, pos_dim:] * (1 - delta_mask) + \
                                val[:, pos_dim:] * delta_mask
        else:
            # mask 없으면, condition에서 제공된 delta만 overwrite
            nonzero_mask = (val[:, pos_dim:] != 0).float()
            x[:, t, pos_dim:] = x[:, t, pos_dim:] * (1 - nonzero_mask) + \
                                val[:, pos_dim:] * nonzero_mask

    if device is not None:
        x = x.to(device)
    return x

class WeightedL1(nn.Module):
    def __init__(self, weights, action_dim):
        """
        weights : [H, D] shape tensor (시점별, 차원별 가중치)
        action_dim : 행동/예측 구간의 시작 인덱스
        """
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, target):
        """
        pred, target : [B, H, D]
        """
        abs_err = (pred - target).abs()
        loss = (self.weights * abs_err).mean()
        return loss, {'weighted_l1': loss.item()}


Losses = {
    'l1': WeightedL1,
}