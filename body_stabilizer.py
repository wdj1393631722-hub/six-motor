#!/usr/bin/env python3
"""行走时机身姿态稳定：阻尼侧偏/偏航，保留重力落足与摩擦传力。"""
from __future__ import annotations

import math

import mujoco


def _body_yaw(data: mujoco.MjData) -> float:
    qw, qx, qy, qz = (
        float(data.qpos[3]),
        float(data.qpos[4]),
        float(data.qpos[5]),
        float(data.qpos[6]),
    )
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def stabilize_locomotion_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_z_target: float,
    yaw_hold: float = 0.0,
) -> None:
    """
    轻度速度阻尼 + 弱偏航回正，抑制持续右偏/转圈。
    不直接写 qpos，避免机身被“吊起来”。
    """
    del body_z_target
    if model.nq < 7:
        return

    yaw_err = _body_yaw(data) - float(yaw_hold)
    while yaw_err > math.pi:
        yaw_err -= 2.0 * math.pi
    while yaw_err < -math.pi:
        yaw_err += 2.0 * math.pi
    data.qvel[5] -= 2.0 * yaw_err
    data.qvel[0] *= 0.55
    data.qvel[1] *= 0.88
    data.qvel[2] *= 0.82
    data.qvel[3] *= 0.55
    data.qvel[4] *= 0.55
    data.qvel[5] *= 0.65
