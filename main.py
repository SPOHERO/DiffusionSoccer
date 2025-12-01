import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from temporal_cond import ConditionalTemporalUnet
from diffusion_cond import GaussianDiffusion


# =========================================================
# 1️⃣ Synthetic Moving-Dot Trajectory 생성
# =========================================================
def generate_synthetic_trajectories(batch_size=16, horizon=50, obs_dim=4, action_dim=2):
    """
    간단한 sin/cos 기반 움직이는 점 데이터 생성:
    - observation: 상대 움직임
    - action: 약간의 phase offset이 있는 우리 움직임
    """
    t = torch.linspace(0, 2 * np.pi, horizon)[None, :, None]

    obs = torch.cat([
        torch.sin(t) + 0.05 * torch.randn(1, horizon, 1),
        torch.cos(t) + 0.05 * torch.randn(1, horizon, 1),
        torch.sin(2 * t) * 0.5 + 0.05 * torch.randn(1, horizon, 1),
        torch.cos(2 * t) * 0.5 + 0.05 * torch.randn(1, horizon, 1),
    ], dim=-1)
    obs = obs.repeat(batch_size, 1, 1)

    offset = 0.4 * torch.sin(t + torch.rand(1) * np.pi)
    act = obs[:, :, :action_dim] + offset.repeat(batch_size, 1, action_dim)

    x = torch.cat([obs, act], dim=-1)
    return x


# =========================================================
# 2️⃣ 설정
# =========================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
horizon = 50
obs_dim = 4
action_dim = 2
transition_dim = obs_dim + action_dim
timesteps = 200

# =========================================================
# 3️⃣ 샘플 데이터 생성
# =========================================================
x_data = generate_synthetic_trajectories(batch_size, horizon, obs_dim, action_dim)
gt = x_data[0]  # 첫 번째 trajectory만 시각화
observed_frames = 40
conditions = {t: x_data[:, t, action_dim:].clone() for t in range(observed_frames)}

# =========================================================
# 4️⃣ 샘플 데이터셋 시각화 (무빙 dot)
# =========================================================
print("▶️ Showing sample dataset trajectories...")

fig1, ax1 = plt.subplots(figsize=(5, 5))
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_title("Sample Dataset: Moving Dot Trajectory")
obs_dot, = ax1.plot([], [], 'bo', label='Observation (relative)')
act_dot, = ax1.plot([], [], 'r*', label='Action (target)')
ax1.legend()

def init_sample():
    obs_dot.set_data([], [])
    act_dot.set_data([], [])
    return obs_dot, act_dot

def update_sample(frame):
    obs_dot.set_data(gt[frame, 0], gt[frame, 1])
    act_dot.set_data(gt[frame, 4], gt[frame, 5])
    return obs_dot, act_dot

ani1 = FuncAnimation(fig1, update_sample, frames=horizon,
                     init_func=init_sample, interval=150, blit=True)
plt.show()

# =========================================================
# 5️⃣ 모델 정의
# =========================================================
print("\nInitializing diffusion model...")
unet = ConditionalTemporalUnet(
    horizon=horizon,
    transition_dim=transition_dim,
    dim=32,
    dim_mults=(1, 2, 4),
    attention=True
).to(device)

diffusion = GaussianDiffusion(
    model=unet,
    horizon=horizon,
    observation_dim=obs_dim,
    action_dim=action_dim,
    n_timesteps=timesteps,
    loss_type='l1',
    clip_denoised=True,
    predict_epsilon=True,
    action_weight=1.0,
    loss_discount=0.98,
    device=device
).to(device)

# =========================================================
# 6️⃣ 간단한 학습 루프
# =========================================================
optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
x_train = x_data.to(device)
for epoch in range(3):
    optimizer.zero_grad()
    loss, info = diffusion.loss(x_train, conditions)
    loss.backward()
    optimizer.step()
    print(f"[Epoch {epoch+1}] Loss: {loss.item():.6f}")

# =========================================================
# 7️⃣ 샘플링 (예측)
# =========================================================
print("\nSampling predicted trajectory...")
sample = diffusion.conditional_sample(conditions, horizon=horizon, verbose=False)
x_pred = sample.trajectories.detach().cpu()
pred = x_pred[0]

# =========================================================
# 8️⃣ 예측 결과 시각화 (움직이는 dot)
# =========================================================
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.set_title("Predicted vs Ground Truth Moving Dot")
obs_dot, = ax2.plot([], [], 'bo', label='Observed')
gt_dot, = ax2.plot([], [], 'go', label='Ground Truth')
pred_dot, = ax2.plot([], [], 'r*', label='Predicted')
ax2.legend()

def init_pred():
    obs_dot.set_data([], [])
    gt_dot.set_data([], [])
    pred_dot.set_data([], [])
    return obs_dot, gt_dot, pred_dot

def update_pred(frame):
    if frame < observed_frames:
        obs_dot.set_data(gt[frame, 4], gt[frame, 5])
        gt_dot.set_data([], [])
        pred_dot.set_data([], [])
    else:
        obs_dot.set_data(gt[observed_frames-1, 4], gt[observed_frames-1, 5])
        gt_dot.set_data(gt[frame, 4], gt[frame, 5])
        pred_dot.set_data(pred[frame, 4], pred[frame, 5])
    return obs_dot, gt_dot, pred_dot

ani2 = FuncAnimation(fig2, update_pred, frames=horizon,
                     init_func=init_pred, interval=150, blit=True)
plt.show()
