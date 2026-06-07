#!/usr/bin/env python3
"""行走时机身姿态稳定：仅阻尼异常抖动，不强行改写位姿（避免飘浮感）。"""
from __future__ import annotations

import mujoco


def stabilize_locomotion_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_z_target: float,
    yaw_hold: float = 0.0,
) -> None:
    """
    轻度速度阻尼，保留重力落足与摩擦传力。
    不再对 qpos 做高度/姿态回写，避免机身被“吊起来”。
    """
    del body_z_target, yaw_hold
    if model.nq < 7:
        return

    data.qvel[0] *= 0.78
    data.qvel[2] *= 0.82
    data.qvel[3] *= 0.55
    data.qvel[4] *= 0.55
    data.qvel[5] *= 0.88
