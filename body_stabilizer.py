#!/usr/bin/env python3
"""行走时机身姿态稳定：抑制横漂、偏航乱转与翻滚。"""
from __future__ import annotations

import math

import mujoco
import numpy as np


def _yaw_quat(yaw: float) -> np.ndarray:
    h = 0.5 * yaw
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)], dtype=float)


def stabilize_locomotion_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_z_target: float,
    yaw_hold: float = 0.0,
) -> None:
    """
    保留 +Y 前进与重力响应，抑制侧向漂移和偏航甩动。
    free joint: qpos[0:3] pos, qpos[3:7] quat, qvel[0:3] lin, qvel[3:6] ang
    """
    if model.nq < 7:
        return

    # 线速度：抑制侧向 X，保留 Y 前进；竖直轻度阻尼
    data.qvel[0] *= 0.15
    data.qvel[2] *= 0.55

    # 角速度：强阻尼，避免 IK/摩擦不对称导致乱转
    data.qvel[3] *= 0.12
    data.qvel[4] *= 0.12
    data.qvel[5] *= 0.08

    # 姿态回正到 yaw_hold、roll=pitch=0
    q = data.qpos[3:7].astype(float)
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_err = math.atan2(
        math.sin(yaw - yaw_hold), math.cos(yaw - yaw_hold)
    )
    data.qvel[5] += -2.5 * yaw_err

    target_q = _yaw_quat(yaw_hold)
    data.qpos[3:7] = 0.82 * q + 0.18 * target_q
    n = float(np.linalg.norm(data.qpos[3:7]))
    if n > 1e-9:
        data.qpos[3:7] /= n

    # 高度软约束
    z_err = float(body_z_target) - float(data.qpos[2])
    data.qvel[2] += 1.8 * z_err

    # 侧向位置回拉
    data.qpos[0] *= 0.92
    data.qvel[0] += -1.2 * float(data.qpos[0])

    mujoco.mj_forward(model, data)
