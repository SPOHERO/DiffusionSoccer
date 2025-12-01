import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)


class ResidualTemporalBlock(nn.Module):
    """
    1D residual block with time embedding.
    """

    def __init__(self, in_ch, out_ch, t_dim, k=5):
        super().__init__()
        self.c1 = Conv1dBlock(in_ch, out_ch, k)
        self.c2 = Conv1dBlock(out_ch, out_ch, k)
        self.tmlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(t_dim, out_ch),
        )
        self.proj = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        """
        x    : [B, C, T]
        t_emb: [B, t_dim]
        """
        h = self.c1(x)
        # broadcast time embedding to [B,C,T]
        t_term = self.tmlp(t_emb)[..., None]
        h = h + t_term
        h = self.c2(h)
        return h + self.proj(x)


class ConditionalTemporalUnet(nn.Module):
    """
    Temporal U-Net that conditions on:
        - current noisy trajectory x_t
        - full conditioning trajectory cond_tensor
        - observation mask cond_mask
        - diffusion timestep t

    Input shapes:
        x           : [B, H, D]
        cond_tensor : [B, H, D]
        cond_mask   : [B or 1, D, H]
        time        : [B]
    """

    def __init__(
        self,
        horizon,
        transition_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim

        # 🔥 우리가 원하는 구조: [x, cond_tensor, mask, x*mask] → 4D channels
        in_ch = transition_dim * 4

        dims = [in_ch, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)
        cur_horizon = horizon

        # ---------------- Down path ----------------
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, t_dim=time_dim),
                        ResidualTemporalBlock(dim_out, dim_out, t_dim=time_dim),
                        Residual(
                            PreNorm(dim_out, LinearAttention(dim_out))
                        )
                        if attention
                        else nn.Identity(),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                cur_horizon //= 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, t_dim=time_dim
        )
        self.mid_attn = (
            Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
            if attention
            else nn.Identity()
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, t_dim=time_dim
        )

        # ---------------- Up path ----------------
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2, dim_in, t_dim=time_dim
                        ),
                        ResidualTemporalBlock(
                            dim_in, dim_in, t_dim=time_dim
                        ),
                        Residual(
                            PreNorm(dim_in, LinearAttention(dim_in))
                        )
                        if attention
                        else nn.Identity(),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                cur_horizon *= 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, k=5),
            nn.Conv1d(dim, transition_dim, kernel_size=1),
        )

    def forward(self, x, cond_tensor, cond_mask, time):
        """
        x           : [B, H, D]
        cond_tensor : [B, H, D]
        cond_mask   : [B or 1, D, H]
        time        : [B]
        """
        B, H, D = x.shape

        # cond_mask broadcast
        if cond_mask.shape[0] == 1:
            cond_mask = cond_mask.expand(B, -1, -1)  # [B, D, H]

        # to channels-first
        x_ch = einops.rearrange(x, "b h d -> b d h")               # [B,D,H]
        cond_ch = einops.rearrange(cond_tensor, "b h d -> b d h")  # [B,D,H]
        m_ch = cond_mask                                           # [B,D,H]

        # 🔥 concat 4 inputs: x, cond, mask, x*mask
        x_aug = torch.cat([x_ch, cond_ch, m_ch, x_ch * m_ch], dim=1)

        # time embedding
        t_emb = self.time_mlp(time.float())  # [B, time_dim]

        # ----- Down path -----
        hs = []
        y = x_aug
        for res1, res2, attn, down in self.downs:
            y = res1(y, t_emb)
            y = res2(y, t_emb)
            y = attn(y)
            hs.append(y)
            y = down(y)

        # ----- Bottleneck -----
        y = self.mid_block1(y, t_emb)
        y = self.mid_attn(y)
        y = self.mid_block2(y, t_emb)

        # ----- Up path -----
        for res1, res2, attn, up in self.ups:
            y = torch.cat([y, hs.pop()], dim=1)
            y = res1(y, t_emb)
            y = res2(y, t_emb)
            y = attn(y)
            y = up(y)

        out = self.final_conv(y)                      # [B, D, H]
        out = einops.rearrange(out, "b d h -> b h d") # [B, H, D]
        return out
