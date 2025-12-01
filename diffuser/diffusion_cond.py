from collections import namedtuple
import numpy as np
import torch
from torch import nn

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
)

Sample = namedtuple("Sample", "trajectories values chains")


@torch.no_grad()
def default_sample_fn(model, x, cond_tensor, cond_mask, t):
    model_mean, _, model_log_variance = model.p_mean_variance(
        x=x,
        cond=(cond_tensor, cond_mask),
        t=t,
    )
    model_std = torch.exp(0.5 * model_log_variance)

    noise = torch.randn_like(x)
    noise[t == 0] = 0.0

    values = torch.zeros(x.size(0), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    return x[inds], values[inds]


def make_timesteps(batch_size, i, device):
    return torch.full((batch_size,), i, device=device, dtype=torch.long)


class GaussianDiffusion(nn.Module):

    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        n_timesteps=200,
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
        device="cuda",
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim     # pos_dim = 46
        self.action_dim = action_dim               # delta_dim = 46
        self.transition_dim = observation_dim + action_dim  # 92
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas.dtype), alphas_cumprod[:-1]], dim=0
        )

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1.0),
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

    # --------------------------------------------------------------
    # q(x_t | x_0)
    # --------------------------------------------------------------
    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = extract(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # --------------------------------------------------------------
    # model p(x_{t-1} | x_t)
    # --------------------------------------------------------------
    def p_mean_variance(self, x, cond, t):
        cond_tensor, cond_mask = cond

        eps_pred = self.model(x, cond_tensor, cond_mask, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=eps_pred)

        if self.clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)

        mp, pv, plv = self.q_posterior(
            x_start=x_recon,
            x_t=x,
            t=t,
        )
        return mp, pv, plv

    # --------------------------------------------------------------
    # sampling
    # --------------------------------------------------------------
    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        cond_mask,
        cond_tensor,
        verbose=True,
        return_chain=False,
        sample_fn=default_sample_fn,
        **sample_kwargs,
    ):
        device = self.betas.device
        B = shape[0]

        x = torch.randn(shape, device=device)

        x = apply_conditioning(
            x,
            cond,
            None,
            self.action_dim,
            device,
            pos_dim=self.observation_dim,
        )

        chain = [x] if return_chain else None
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()

        for i in reversed(range(self.n_timesteps)):
            t = make_timesteps(B, i, device)

            x, values = sample_fn(
                self,
                x,
                cond_tensor,
                cond_mask,
                t,
                **sample_kwargs,
            )

            x = apply_conditioning(
                x,
                cond,
                None,
                self.action_dim,
                device,
                pos_dim=self.observation_dim,
            )

            progress.update("")
            if return_chain:
                chain.append(x)

        x, values = sort_by_values(x, values)
        if return_chain:
            chain = torch.stack(chain, dim=1)

        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, cond_mask, horizon=None, **sample_kwargs):
        cond = cond.to(self.betas.device)
        cond_mask = cond_mask.to(self.betas.device)

        B, T, D = cond.shape
        horizon = horizon or self.horizon

        shape = (B, horizon, D)
        cond_dict = {t: cond[:, t, :] for t in range(T)}

        return self.p_sample_loop(
            shape=shape,
            cond=cond_dict,
            cond_mask=cond_mask,
            cond_tensor=cond,
            **sample_kwargs,
        )

    # --------------------------------------------------------------
    # TRAINING LOSS
    # --------------------------------------------------------------
    def p_losses(
        self,
        x_start,           
        cond,              
        cond_mask,         
        t,                 
        defender_pos_idx,  
        defender_delta_idx 
    ):

        B, T, D = x_start.shape
        pos_dim = D // 2   # =46

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        eps_pred = self.model(x_noisy, cond, cond_mask, t)

        # split
        noise_pos   = noise[:, :, :pos_dim]
        noise_delta = noise[:, :, pos_dim:]
        eps_pos     = eps_pred[:, :, :pos_dim]
        eps_delta   = eps_pred[:, :, pos_dim:]

        # cond_mask: [B, 92, T]
        pos_mask   = 1 - cond_mask[:, defender_pos_idx, :].permute(0, 2, 1)
        delta_mask = 1 - cond_mask[:, defender_delta_idx, :].permute(0, 2, 1)

        pos_loss = ((eps_pos[:, :, defender_pos_idx] - noise_pos[:, :, defender_pos_idx]) ** 2)
        pos_loss = (pos_loss * pos_mask).mean()

        delta_loss = ((eps_delta[:, :, defender_delta_idx] - noise_delta[:, :, defender_delta_idx]) ** 2)
        delta_loss = (delta_loss * delta_mask).mean()

        return pos_loss + delta_loss

    # --------------------------------------------------------------
    # loss() — Trainer가 호출하는 함수
    # --------------------------------------------------------------
    def loss(self, x, cond, cond_mask):
        """
        Trainer는 loss(x, cond, cond_mask) 형태로만 호출합니다.
        defender_pos_idx / delta_idx는 여기서 자동 계산합니다.
        """

        B, T, D = x.shape
        device = x.device

        pos_dim = D // 2  # 46

        # ---------------------------------------------
        # defender = teamB (항상 pos/delta 로컬 인덱스 11명*2=22 features)
        # 당신의 데이터 정렬 규칙에 따라 다음 인덱스 그대로 사용 가능
        # teamA_idx = 0~21 (11명)
        # teamB_idx = 22~43 (11명)
        # ball_idx = 44~45
        # ---------------------------------------------
        defender_pos_idx = torch.arange(22, 44).to(device)
        defender_delta_idx = defender_pos_idx.clone()  # delta도 동일한 로컬 인덱스

        t = torch.randint(0, self.n_timesteps, (B,), device=device).long()

        loss = self.p_losses(
            x_start=x,
            cond=cond,
            cond_mask=cond_mask,
            t=t,
            defender_pos_idx=defender_pos_idx,
            defender_delta_idx=defender_delta_idx
        )

        return loss, {"loss": loss.item()}

    # --------------------------------------------------------------
    def forward(self, cond, cond_mask, *args, **kwargs):
        return self.conditional_sample(cond, cond_mask, *args, **kwargs)
