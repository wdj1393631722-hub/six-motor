#!/usr/bin/env python3
"""平地行走 RL 奖励 — 支持跟踪速度 / 尽量跑快两种模式，含 IMU 水平与平稳项。"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# 跟踪模式用的目标速度
TRACK_VX_MPS = 0.06
# 竞速模式：步态基准指令上限（RL 残差可再往上推）
MAX_SPEED_VX_MPS = 0.14

# IMU 水平目标：|roll|,|pitch| 小于约 8.6° 时给满额奖励
LEVEL_TARGET_RAD = 0.15


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
    gyro: np.ndarray | None = None,
    acc: np.ndarray | None = None,
) -> Tuple[float, Dict[str, float]]:
    """
    mode:
      track     — 跟踪固定目标速度（稳）
      max_speed — 越快奖励越高，仍惩罚侧滑/偏航/倾倒

    gyro / acc: 机体 IMU 角速度(rad/s) 与线加速度(m/s²)，用于平稳奖励。
    """
    if gyro is None:
        wx = wy = 0.0
    else:
        wx, wy = float(gyro[0]), float(gyro[1])
    if acc is None:
        az = 9.81
    else:
        az = float(acc[2])

    if mode == "max_speed":
        vf = float(v_forward)
        r_fwd = 220.0 * max(vf, 0.0)
        r_bonus = 120.0 * max(0.0, vf) ** 2
        r_backward = -90.0 * max(0.0, -vf)
        r_vel_track = 0.0
        r_act_w = 0.012
        r_smooth_w = 0.02
        rp_w = 6.0
        level_bonus = 1.2
        gyro_w = 1.0
        bounce_w = 0.35
    else:
        vf = float(v_forward)
        r_fwd = 80.0 * vf
        r_bonus = 0.0
        r_backward = 0.0
        r_vel_track = -40.0 * (vf - vx_target) ** 2
        r_act_w = 0.03
        r_smooth_w = 0.06
        rp_w = 10.0
        level_bonus = 2.5
        gyro_w = 2.2
        bounce_w = 0.55

    tilt = float(roll * roll + pitch * pitch)
    r_lat = -10.0 * abs(v_lateral)
    r_yaw = -3.0 * abs(wz)
    r_z = -35.0 * z_err * z_err
    r_rp = -rp_w * tilt
    r_level = level_bonus * max(
        0.0,
        1.0 - abs(float(roll)) / LEVEL_TARGET_RAD - abs(float(pitch)) / LEVEL_TARGET_RAD,
    )
    r_gyro = -gyro_w * (wx * wx + wy * wy)
    r_bounce = -bounce_w * (az - 9.81) * (az - 9.81)
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
        + r_level
        + r_gyro
        + r_bounce
        + r_act
        + r_smooth
        + r_alive
    )
    return reward, {
        "v_forward": vf,
        "v_lateral": float(v_lateral),
        "reward_fwd": r_fwd + r_bonus + r_backward,
        "reward_mode": mode,
        "imu_roll": float(roll),
        "imu_pitch": float(pitch),
        "reward_level": r_level,
        "reward_gyro": r_gyro,
        "reward_bounce": r_bounce,
    }
