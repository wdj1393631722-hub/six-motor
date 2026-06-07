#!/usr/bin/env python3
"""平地行走 RL 奖励 — 支持跟踪速度 / 尽量跑快两种模式。"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# 跟踪模式用的目标速度
TRACK_VX_MPS = 0.06
# 竞速模式：步态基准指令上限（RL 残差可再往上推）
MAX_SPEED_VX_MPS = 0.14


def compute_walk_reward(
    v_forward: float,
    v_lateral: float,
    wz: float,
    z_err: float,
    roll: float,
    pitch: float,
    action: np.ndarray,
    last_action: np.ndarray,
    *,
    mode: str = "track",
    vx_target: float = TRACK_VX_MPS,
) -> Tuple[float, Dict[str, float]]:
    """
    mode:
      track     — 跟踪固定目标速度（稳）
      max_speed — 越快奖励越高，仍惩罚侧滑/偏航/倾倒
    """
    if mode == "max_speed":
        vf = float(v_forward)
        r_fwd = 220.0 * max(vf, 0.0)
        r_bonus = 120.0 * max(0.0, vf) ** 2
        r_backward = -90.0 * max(0.0, -vf)
        r_vel_track = 0.0
        r_act_w = 0.012
        r_smooth_w = 0.02
    else:
        vf = float(v_forward)
        r_fwd = 80.0 * vf
        r_bonus = 0.0
        r_backward = 0.0
        r_vel_track = -40.0 * (vf - vx_target) ** 2
        r_act_w = 0.03
        r_smooth_w = 0.05

    r_lat = -10.0 * abs(v_lateral)
    r_yaw = -3.0 * abs(wz)
    r_z = -35.0 * z_err * z_err
    r_rp = -3.0 * (roll**2 + pitch**2)
    r_act = -r_act_w * float(np.sum(action * action))
    r_smooth = -r_smooth_w * float(np.sum((action - last_action) ** 2))
    r_alive = 0.12

    reward = (
        r_fwd
        + r_bonus
        + r_backward
        + r_vel_track
        + r_lat
        + r_yaw
        + r_z
        + r_rp
        + r_act
        + r_smooth
        + r_alive
    )
    return reward, {
        "v_forward": vf,
        "v_lateral": float(v_lateral),
        "reward_fwd": r_fwd + r_bonus + r_backward,
        "reward_mode": mode,
    }
