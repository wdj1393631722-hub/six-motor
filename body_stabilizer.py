#!/usr/bin/env python3
"""行走时机身姿态稳定：阻尼侧偏/偏航，保留重力落足与摩擦传力。"""
from __future__ import annotations

import math

import mujoco


def _yaw_from_qpos(qpos, adr: int = 0) -> float:
    qw = float(qpos[adr + 3])
    qx = float(qpos[adr + 4])
    qy = float(qpos[adr + 5])
    qz = float(qpos[adr + 6])
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def stabilize_robot_root(
    data: mujoco.MjData,
    root_qposadr: int,
    root_dofadr: int,
    yaw_hold: float = 0.0,
    roll_hold: float = 0.0,
    pitch_hold: float = 0.0,
    level_gain: float = 3.5,
) -> None:
    """多机场景：对单只机器人的 free joint 做侧向/偏航/水平阻尼。"""
    adr = int(root_qposadr)
    dof = int(root_dofadr)
    roll, pitch = _roll_pitch_from_qpos(data.qpos, adr)
    yaw_err = _yaw_from_qpos(data.qpos, adr) - float(yaw_hold)
    while yaw_err > math.pi:
        yaw_err -= 2.0 * math.pi
    while yaw_err < -math.pi:
        yaw_err += 2.0 * math.pi
    roll_err = roll - float(roll_hold)
    pitch_err = pitch - float(pitch_hold)
    data.qvel[dof + 3] -= level_gain * roll_err
    data.qvel[dof + 4] -= level_gain * pitch_err
    data.qvel[dof + 5] -= 2.0 * yaw_err
    data.qvel[dof + 0] *= 0.55
    data.qvel[dof + 1] *= 0.88
    data.qvel[dof + 2] *= 0.82
    data.qvel[dof + 3] *= 0.50
    data.qvel[dof + 4] *= 0.50
    data.qvel[dof + 5] *= 0.65


def _roll_pitch_from_qpos(qpos, adr: int = 0) -> tuple[float, float]:
    qw = float(qpos[adr + 3])
    qx = float(qpos[adr + 4])
    qy = float(qpos[adr + 5])
    qz = float(qpos[adr + 6])
    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
    return roll, pitch


def stabilize_locomotion_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_z_target: float,
    yaw_hold: float = 0.0,
    roll_hold: float = 0.0,
    pitch_hold: float = 0.0,
) -> None:
    """
    轻度速度阻尼 + 弱偏航/水平回正，抑制持续右偏与机身倾斜。
    不直接写 qpos，避免机身被“吊起来”。
    """
    del body_z_target, model
    if data.qpos.shape[0] < 7:
        return
    stabilize_robot_root(
        data,
        0,
        0,
        yaw_hold=yaw_hold,
        roll_hold=roll_hold,
        pitch_hold=pitch_hold,
    )
