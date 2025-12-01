import numpy as np
import torch

def follow_closest_attacker_reward(x_frame, teamA_idx, teamB_idx):
    """
    각 수비수가 가장 가까운 공격수를 따라가는 reward.
    모든 수비수의 (가장 가까운 공격수와의 거리^2) 평균에 음수 부호를 붙인 형태.
    """
    # pos only
    atk_pos = x_frame[teamA_idx].reshape(-1, 2)  # (11,2)
    def_pos = x_frame[teamB_idx].reshape(-1, 2)  # (11,2)

    # 모든 수비수 → 모든 공격수 거리행렬 (11,11)
    diffs = def_pos[:, None, :] - atk_pos[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)  # (11,11)

    # 각 수비수에 대해 가장 가까운 공격수 거리
    min_dist = dists.min(axis=1)  # (11,)

    # 평균 penalty (0에 가까울수록 좋음)
    reward = -np.mean(min_dist ** 2)

    return float(reward)
    
def infer_ball_possession(
    x_frame,
    teamA_idx,
    teamB_idx,
    ball_idx,
    feature_dim=46,
    r_possess=1.5  # 공 소유로 간주하는 거리 threshold (m 단위로 튜닝)
):
    """
    x_frame: (2F,) = [pos(46), delta(46)]
    teamA_idx, teamB_idx: pos index (len=22)
    ball_idx: pos index (len=2)

    return:
      possession_team: 'A', 'B', or None
      carrier_type: 'attacker', 'defender', or None
      carrier_player_local_idx: 팀 내부 index (0~10) or None
    """

    # pos 부분만 사용 (0~feature_dim-1)
    ball_pos = x_frame[ball_idx].reshape(1, 2)          # (1,2)

    atk_pos = x_frame[teamA_idx].reshape(-1, 2)         # (11,2)
    def_pos = x_frame[teamB_idx].reshape(-1, 2)         # (11,2)

    # 볼과의 거리 (공격/수비 각각)
    dist_atk = np.linalg.norm(atk_pos - ball_pos, axis=1)   # (11,)
    dist_def = np.linalg.norm(def_pos - ball_pos, axis=1)   # (11,)

    min_dist_atk = dist_atk.min()
    min_dist_def = dist_def.min()

    # 제일 가까운 선수 찾기
    atk_idx_local = int(np.argmin(dist_atk))  # 0~10
    def_idx_local = int(np.argmin(dist_def))  # 0~10

    # threshold 안에 있는 쪽이 소유팀
    if min_dist_atk < r_possess and min_dist_atk < min_dist_def:
        return 'A', 'attacker', atk_idx_local
    elif min_dist_def < r_possess and min_dist_def < min_dist_atk:
        return 'B', 'defender', def_idx_local
    else:
        return None, None, None

def possession_motion_reward_from_frame(
    x_frame_t,
    x_frame_tm1,
    teamA_idx,
    teamB_idx,
    ball_idx,
    feature_dim=46,
    r_possess=1.5,
    w_pos=1.0,
    w_vel=0.5
):
    """
    공 소유자(공격 or 수비)가 있을 때:
      - 공과 소유자 거리가 가까울수록 +
      - 공 속도와 소유자 속도가 비슷할수록 +

    return: reward (float)
    """
    # pos, vel 추출
    ball_pos_t = x_frame_t[ball_idx].reshape(1, 2)
    ball_pos_tm1 = x_frame_tm1[ball_idx].reshape(1, 2)
    ball_vel_t = ball_pos_t - ball_pos_tm1  # (1,2)

    # pos idx → vel idx (delta 부분)
    teamA_vel_idx = teamA_idx + feature_dim
    teamB_vel_idx = teamB_idx + feature_dim

    atk_pos_t = x_frame_t[teamA_idx].reshape(-1, 2)
    def_pos_t = x_frame_t[teamB_idx].reshape(-1, 2)

    atk_vel_t = x_frame_t[teamA_vel_idx].reshape(-1, 2)
    def_vel_t = x_frame_t[teamB_vel_idx].reshape(-1, 2)

    # 누가 공을 소유하는지 결정
    possession_team, carrier_type, carrier_local_idx = infer_ball_possession(
        x_frame_t, teamA_idx, teamB_idx, ball_idx,
        feature_dim=feature_dim,
        r_possess=r_possess
    )

    if possession_team is None:
        return 0.0

    if possession_team == 'A':
        carrier_pos = atk_pos_t[carrier_local_idx]   # (2,)
        carrier_vel = atk_vel_t[carrier_local_idx]   # (2,)
    else:
        carrier_pos = def_pos_t[carrier_local_idx]
        carrier_vel = def_vel_t[carrier_local_idx]

    # (1) 위치 근접 reward
    dist_pos = np.linalg.norm(carrier_pos - ball_pos_t[0])
    R_pos = - (dist_pos ** 2)

    # (2) 속도(방향) 일치 reward
    norm_ball = np.linalg.norm(ball_vel_t[0]) + 1e-6
    norm_car  = np.linalg.norm(carrier_vel) + 1e-6

    cos_sim = np.dot(ball_vel_t[0], carrier_vel) / (norm_ball * norm_car)
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))

    # 방향 비슷하면 +, 반대면 -
    R_vel = cos_sim

    return w_pos * R_pos + w_vel * R_vel

def defending_ball_carrier_reward_from_frame(
    x_frame_t,
    teamA_idx,
    teamB_idx,
    ball_idx,
    feature_dim=46,
    x_mid=0.0,
    d_def_star=2.5    # 공 가진 공격자와 수비수 이상적 거리
):
    """
    공을 가진 공격자가 있을 때,
    그 공격자 근처에 수비수 한 명이 적절한 거리로 붙어 있을수록 reward ↑
    """

    # pos 추출
    atk_pos = x_frame_t[teamA_idx].reshape(-1, 2)   # (11,2)
    def_pos = x_frame_t[teamB_idx].reshape(-1, 2)   # (11,2)

    # ball possession 판단
    possession_team, carrier_type, carrier_idx_local = infer_ball_possession(
        x_frame_t, teamA_idx, teamB_idx, ball_idx,
        feature_dim=feature_dim,
        r_possess=1.5
    )

    # 공격자가 공을 가지지 않은 경우 → 마킹 상황 아님
    if possession_team != 'A' or carrier_type != 'attacker':
        return 0.0

    ball_carrier_pos = atk_pos[carrier_idx_local]   # (2,)

    # 우리 진영 안의 수비수만 고려 (예: x <= x_mid)
    def_pos_our_half = def_pos[def_pos[:, 0] <= x_mid]
    if def_pos_our_half.shape[0] == 0:
        return -50.0  # 최악

    # ball carrier와 모든 수비수 거리
    dists = np.linalg.norm(def_pos_our_half - ball_carrier_pos[None, :], axis=1)
    min_dist = dists.min()

    # 이상적인 거리에서의 편차 제곱 → 음수 reward
    dev = min_dist - d_def_star
    R_def = - (dev ** 2)

    return R_def
    
def sep_reward_from_frame(x_frame, teamB_idx, feature_dim=46, x_mid=0.0, d_sep_star=6.0):
    """
    x_frame: (92,)
    teamB_idx: defender pos index (길이 22 = 11명×2)
    """
    # 1) 수비수 위치만 뽑기
    def_pos_idx = teamB_idx
    def_pos = x_frame[def_pos_idx].reshape(-1, 2)

    # 2) 자기 진영(half-line) 안에 있는 수비수만 사용
    active = def_pos[:, 0] <= x_mid
    def_pos = def_pos[active]

    if len(def_pos) < 2:
        return 0.0

    # 3) 모든 수비수 쌍의 거리 계산
    diffs = def_pos[:, None, :] - def_pos[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)

    # 4) 위쪽 삼각만 사용해서 중복 제거
    K = def_pos.shape[0]
    i_idx, j_idx = np.triu_indices(K, k=1)
    pair_dists = dists[i_idx, j_idx]

    # 5) 이상적인 거리 d_sep_star와의 편차 제곱합 → 음수 reward
    deviations = pair_dists - d_sep_star
    reward = -np.sum(deviations ** 2)

    return reward


def soft_boundary_reward(x_frame, teamB_idx,
                         x_min=-52.5, x_max=52.5,
                         y_min=-34.0, y_max=34.0,
                         lambda_bd=5.0):  # 경계 penalty 세기

    pos = x_frame[teamB_idx].reshape(-1, 2)

    # out-of-bound 양 계산
    dx = np.maximum(0, x_min - pos[:,0]) + np.maximum(0, pos[:,0] - x_max)
    dy = np.maximum(0, y_min - pos[:,1]) + np.maximum(0, pos[:,1] - y_max)

    d_out = dx + dy  # (11,)

    # Penalty = -lambda * (distance^2)
    boundary_penalty = -lambda_bd * np.sum(d_out ** 2)

    return boundary_penalty
    
def marking_reward_from_frame(
    x_frame,
    teamB_idx,   # 수비팀 pos 인덱스 (길이 22 = 11명 x,y)
    teamA_idx,   # 공격팀 pos 인덱스 (길이 22)
    feature_dim=46,
    x_mid=0.0,       # 중앙선 x좌표 (예: 0.0)
    d_mk_star=3.0    # 이상적인 마킹 거리 (미터)
):
    """
    한 프레임에서 수비 마킹 품질을 계산하는 보상 함수.
    
    Args:
        x_frame: np.ndarray, shape (92,)
            [pos(46), delta(46)] 가 들어있는 한 프레임 상태 벡터.
        teamB_idx: np.ndarray, shape (22,)
            수비수 pos 인덱스 (11명 x,y).
        teamA_idx: np.ndarray, shape (22,)
            공격수 pos 인덱스 (11명 x,y).
        feature_dim: int
            pos 차원 수 (기본 46).
        x_mid: float
            중앙선 x좌표. x <= x_mid 인 선수만 활성(우리 진영)으로 간주.
        d_mk_star: float
            이상적인 마킹 거리 (meters).
    
    Returns:
        reward: float
            마킹 보상 (<= 0, 0에 가까울수록 좋고 음수가 클수록 나쁨).
    """
    # 1) 수비, 공격 pos 추출 (각각 (11,2) 형태 → 실제 active 수에 따라 (Md,2), (Ma,2))
    def_pos = x_frame[teamB_idx].reshape(-1, 2)   # (11,2)
    atk_pos = x_frame[teamA_idx].reshape(-1, 2)   # (11,2)

    # 2) 우리 진영(half-line) 안에 있는 선수만 필터링
    #    예: x <= x_mid
    def_pos = def_pos[def_pos[:, 0] <= x_mid]     # (Md,2)
    atk_pos = atk_pos[atk_pos[:, 0] <= x_mid]     # (Ma,2)

    # 공격수가 우리 진영에 없으면 → marking을 논할 상황이 아님 → 보상 0
    if atk_pos.shape[0] == 0:
        return 0.0

    # 수비수가 우리 진영에 하나도 없으면 → 마킹 최악 → 큰 음수
    if def_pos.shape[0] == 0:
        return -100.0  # 상황에 맞게 조정 가능

    # 3) 각 공격수 → 모든 수비수까지 거리 행렬 계산
    # atk_pos: (Ma,2), def_pos: (Md,2)
    # → (Ma,1,2) - (1,Md,2) → (Ma,Md,2)
    atk_exp = atk_pos[:, None, :]      # (Ma, 1, 2)
    def_exp = def_pos[None, :, :]      # (1, Md, 2)

    dists = np.linalg.norm(atk_exp - def_exp, axis=-1)   # (Ma, Md)

    # 4) 각 공격수마다 가장 가까운 수비수 거리 m_i
    min_dists = dists.min(axis=1)      # (Ma,)

    # 5) 이상적 마킹 거리에서의 편차 제곱 평균 → 음수 reward
    #    R_mk = - mean_i (m_i - d_mk_star)^2
    deviations = min_dists - d_mk_star
    reward = - np.mean(deviations ** 2)

    return float(reward)

def compute_threat_sector_params(attacker_pos, attacker_vel, ball_pos,
                                 alpha=1.7, beta=np.deg2rad(40),
                                 v_max=8.0):
    """
    공격자의 Threat Sector 반경 R, 각도 θ를 자동 계산.
    R = alpha * dist(attacker, ball)
    θ = beta * (1 + ||v_attacker|| / v_max)
    """
    # 반경
    dist_ab = np.linalg.norm(attacker_pos - ball_pos)
    R = alpha * dist_ab

    # 각도
    va_norm = np.linalg.norm(attacker_vel)
    theta = beta * (1.0 + va_norm / v_max)

    return R, theta

def passing_lane_blocking_single(attacker_pos, attacker_vel, ball_pos,
                                 defender_positions,
                                 alpha=1.7, beta=np.deg2rad(40),
                                 v_max=8.0):
    """
    공격자 1명에 대해 Threat Sector를 만들고,
    수비수들이 이를 얼마나 잘 차단하는지 계산.
    
    RETURN: block_score (0~1 사이 값)
    """
    # --- Threat Sector Parameters ---
    R, theta = compute_threat_sector_params(
        attacker_pos, attacker_vel, ball_pos, alpha, beta, v_max
    )
    if R < 1e-3:
        return 0.0

    # --- Threat 방향 정의 ---
    direction = attacker_vel
    if np.linalg.norm(direction) < 1e-6:  # 속도 없으면 공 기준 fallback
        direction = attacker_pos - ball_pos
    direction = direction / np.linalg.norm(direction)

    best_block = 0.0

    # --- 수비수별 차단 점수 계산 ---
    for dpos in defender_positions:
        rel = dpos - attacker_pos
        d = np.linalg.norm(rel)
        if d < 1e-6:
            continue

        # 거리 기반 감쇠 (멀수록 score ↓)
        dist_score = np.exp(-d / R)

        # 각도 기반 감쇠 (각도 차이가 클수록 score ↓)
        rel_unit = rel / d
        angle = np.arccos(np.clip(np.dot(direction, rel_unit), -1.0, 1.0))
        angle_score = np.exp(-(angle / theta)**2)

        # 최종 score
        block_score = dist_score * angle_score
        best_block = max(best_block, block_score)

    return best_block

def passing_lane_reward_from_frame(x_frame,
                                   teamA_idx, teamB_idx, ball_idx,
                                   feature_dim=46,
                                   x_mid=0.0):
    """
    한 프레임에서 Passing-Lane Blocking Reward 계산.
    x_frame: shape (92,)  # pos(46) + delta(46)

    teamA_idx: 공격수 pos index (len=22)
    teamB_idx: 수비수 pos index (len=22)
    ball_idx: 볼 pos index (len=2)
    """
    # --------------------------
    # 1) POS & VELOCITY 추출
    # --------------------------
    atk_pos = x_frame[teamA_idx].reshape(-1, 2)                          # (11,2)
    atk_vel = x_frame[teamA_idx + feature_dim].reshape(-1, 2)            # (11,2)

    def_pos = x_frame[teamB_idx].reshape(-1, 2)                          # (11,2)
    ball_pos = x_frame[ball_idx].reshape(1, 2)[0]                        # (2,)


    # --------------------------
    # 2) half-line 필터링
    #    x <= x_mid 범위의 공격수/수비수만 고려
    # --------------------------
    atk_mask = atk_pos[:, 0] <= x_mid
    def_mask = def_pos[:, 0] <= x_mid

    atk_pos = atk_pos[atk_mask]
    atk_vel = atk_vel[atk_mask]     # pos와 개수 맞추기

    def_pos = def_pos[def_mask]

    # 공격자/수비수가 없으면 score = 0
    if len(atk_pos) == 0 or len(def_pos) == 0:
        return 0.0

    # --------------------------
    # 3) 공격자별 blocking score 계산
    # --------------------------
    total = 0.0
    N = len(atk_pos)

    for i in range(N):
        total += passing_lane_blocking_single(
            attacker_pos = atk_pos[i],
            attacker_vel = atk_vel[i],
            ball_pos     = ball_pos,
            defender_positions = def_pos
        )

    # 공격자 평균
    return total / N


#============================================================
# Smoothness Reward (움직임 부드러움 보상)
#============================================================
def smoothness_reward_from_frame(x_frame_t, x_frame_tm1,
                                 teamB_idx, feature_dim=46,
                                 lambda_pos=0.1):
    """
    x_frame_t:    현재 프레임 (92,)
    x_frame_tm1:  이전 프레임 (92,)
    teamB_idx:    수비수 pos 인덱스 (11명 x,y → len=22)
    """

    # --- 수비수 pos(t), pos(t-1) 추출 ---
    pos_t   = x_frame_t[teamB_idx].reshape(-1, 2)     # (11,2)
    pos_tm1 = x_frame_tm1[teamB_idx].reshape(-1, 2)   # (11,2)

    # Δpos
    diff = pos_t - pos_tm1

    # L2 제곱합
    sq = np.sum(diff**2)

    # Penalty 형태 → 음수 reward
    reward = -lambda_pos * sq

    return reward

#============================================================
# Velocity / Acceleration Penalty 포함 Total Reward
#============================================================
def vel_acc_reward_from_frame(x_frame_t, x_frame_tm1,
                              teamB_idx, feature_dim=46,
                              lambda_vel=0.01,
                              lambda_acc=0.01):
    """
    x_frame_t:    현재 프레임 (92,)
    x_frame_tm1:  이전 프레임 (92,)
    teamB_idx:    수비수 pos 인덱스
    """

    # --- Velocity: v(t), v(t-1) ---
    vel_t   = x_frame_t[teamB_idx + feature_dim].reshape(-1, 2)
    vel_tm1 = x_frame_tm1[teamB_idx + feature_dim].reshape(-1, 2)

    # Velocity penalty
    vel_sq = np.sum(vel_t**2)
    R_vel = -lambda_vel * vel_sq

    # Acceleration penalty
    acc = vel_t - vel_tm1
    acc_sq = np.sum(acc**2)
    R_acc = -lambda_acc * acc_sq

    return R_vel + R_acc
# -------------------------------------------------
# 4. Total Reward
# -------------------------------------------------
def total_reward_from_frame(
    x_frame_t, x_frame_tm1,
    teamA_idx, teamB_idx, ball_idx,
    feature_dim=46,
    x_mid=0.0,
    w_sep=1.0, w_mk=1.0, w_pl=1.0,
    lambda_pos=0,
    lambda_vel=0,
    lambda_acc=0,
    lambda_bd=0,
    w_possess=1.0,          # 🔥 새로 추가
    w_defend_ball=1.5       # 🔥 새로 추가
):
    # 1) 기존 리워드
    r_sep = sep_reward_from_frame(x_frame_t, teamB_idx, feature_dim, x_mid)
    r_mk  = marking_reward_from_frame(x_frame_t, teamB_idx, teamA_idx, feature_dim, x_mid)
    r_pl  = passing_lane_reward_from_frame(x_frame_t, teamA_idx, teamB_idx, ball_idx,
                                           feature_dim, x_mid)

    r_smooth = smoothness_reward_from_frame(
        x_frame_t, x_frame_tm1, teamB_idx,
        feature_dim=feature_dim, lambda_pos=lambda_pos)

    r_dyn = vel_acc_reward_from_frame(
        x_frame_t, x_frame_tm1, teamB_idx,
        feature_dim=feature_dim,
        lambda_vel=lambda_vel, lambda_acc=lambda_acc)

    r_bd = soft_boundary_reward(
        x_frame_t, teamB_idx,
        x_min=-52.5, x_max=52.5,
        y_min=-34.0, y_max=34.0,
        lambda_bd=lambda_bd
    )

    # 2) 🔥 새 리워드: 공 소유자 & 공 함께 움직이기
    r_possess = possession_motion_reward_from_frame(
        x_frame_t, x_frame_tm1,
        teamA_idx, teamB_idx, ball_idx,
        feature_dim=feature_dim,
        r_possess=1.5,
        w_pos=1.0,
        w_vel=0.5
    )

    # 3) 🔥 새 리워드: 공 가진 공격자에 대한 집요한 수비
    r_def_ball = defending_ball_carrier_reward_from_frame(
        x_frame_t,
        teamA_idx, teamB_idx, ball_idx,
        feature_dim=feature_dim,
        x_mid=x_mid,
        d_def_star=2.5
    )

    total = (
        w_sep * r_sep +
        w_mk  * r_mk +
        w_pl  * r_pl +
        r_smooth +
        r_dyn +
        r_bd +
        w_possess * r_possess +      # 🔥 추가
        w_defend_ball * r_def_ball   # 🔥 추가
    )

    return total, {
        "sep": r_sep,
        "mk": r_mk,
        "pl": r_pl,
        "smooth": r_smooth,
        "dyn": r_dyn,
        "bd": r_bd,
        "possess": r_possess,
        "def_ball": r_def_ball,
    }
    
def compute_all_rewards_with_smoothness(
    x_combined,
    teamA_idx,
    teamB_idx,
    ball_idx,
    feature_dim=46,
    x_mid=0.0,
    lambda_pos=0,
    lambda_vel=0,
    lambda_acc=0,
    lambda_bd=0
):
    """
    전체 trajectory에 대해 프레임별 reward 계산.
    total_reward_from_frame 안의 모든 reward:
        - sep
        - mk
        - pl
        - smooth
        - dyn
        - boundary
        - possess  (🔥 추가)
        - def_ball (🔥 추가)
    """

    # numpy 변환
    if isinstance(x_combined, torch.Tensor):
        x_np = x_combined.detach().cpu().numpy()
    else:
        x_np = x_combined

    B, T, D = x_np.shape

    # frame-level outputs
    R_total   = np.zeros((B, T), dtype=np.float32)
    R_sep     = np.zeros((B, T), dtype=np.float32)
    R_mk      = np.zeros((B, T), dtype=np.float32)
    R_pl      = np.zeros((B, T), dtype=np.float32)
    R_smooth  = np.zeros((B, T), dtype=np.float32)
    R_dyn     = np.zeros((B, T), dtype=np.float32)
    R_bd      = np.zeros((B, T), dtype=np.float32)
    R_possess = np.zeros((B, T), dtype=np.float32)      # 🔥 추가
    R_defball = np.zeros((B, T), dtype=np.float32)      # 🔥 추가

    # index to numpy
    teamA_idx = np.array(teamA_idx)
    teamB_idx = np.array(teamB_idx)
    ball_idx  = np.array(ball_idx)

    for b in range(B):
        for t in range(T):

            x_frame_t   = x_np[b, t]
            x_frame_tm1 = x_np[b, t-1] if t > 0 else x_frame_t

            total, parts = total_reward_from_frame(
                x_frame_t, x_frame_tm1,
                teamA_idx, teamB_idx, ball_idx,
                feature_dim=feature_dim,
                x_mid=x_mid,
                lambda_pos=lambda_pos,
                lambda_vel=lambda_vel,
                lambda_acc=lambda_acc,
                lambda_bd=lambda_bd
            )

            R_total[b, t]   = total
            R_sep[b, t]     = parts["sep"]
            R_mk[b, t]      = parts["mk"]
            R_pl[b, t]      = parts["pl"]
            R_smooth[b, t]  = parts["smooth"]
            R_dyn[b, t]     = parts["dyn"]
            R_bd[b, t]      = parts["bd"]
            R_possess[b, t] = parts["possess"]      # 🔥 추가
            R_defball[b, t] = parts["def_ball"]     # 🔥 추가

    return (
        R_total,
        R_sep,
        R_mk,
        R_pl,
        R_smooth,
        R_dyn,
        R_bd,
        R_possess,      # 🔥 반환 포함
        R_defball       # 🔥 반환 포함
    )

def auto_calibrate_rewards(
    R_sep, R_mk, R_pl, R_smooth, R_dyn, R_bd,
    R_possess, R_defball,      # 🔥 추가
    w_sep=1.0, w_mk=1.0, w_pl=1.0,
    w_smooth=0, w_dyn=0, w_bd=0,
    w_possess=1.0, w_defball=1.0,  # 🔥 추가
    temp=1.0,
    eps=1e-8
):
    """
    모든 reward term을 표준화 + 가중치 적용 + tanh로 정규화.
    """

    components = {
        "sep":    R_sep,
        "mk":     R_mk,
        "pl":     R_pl,
        "smooth": R_smooth,
        "dyn":    R_dyn,
        "bd":     R_bd,
        "possess": R_possess,   # 🔥 추가
        "defball": R_defball    # 🔥 추가
    }

    weights = {
        "sep": w_sep,
        "mk":  w_mk,
        "pl":  w_pl,
        "smooth": w_smooth,
        "dyn": w_dyn,
        "bd":  w_bd,
        "possess": w_possess,   # 🔥 추가
        "defball": w_defball    # 🔥 추가
    }

    stats = {}
    R_lin = np.zeros_like(R_sep, dtype=np.float32)

    for name, R in components.items():
        mu = R.mean()
        std = R.std()
        stats[name] = {"mean": float(mu), "std": float(std)}

        R_norm = (R - mu) / (std + eps)
        R_lin += weights[name] * R_norm

    R_auto = np.tanh(R_lin / temp)

    return R_auto, stats