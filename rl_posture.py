#!/usr/bin/env python3
"""仿真站立姿态抬高 — 在可站稳的前提下尽量抬高机身。"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

from enable_state import LOCOMOTION_KP, LOCOMOTION_KV, read_joint_pose, set_actuator_gains
from foot_kinematics import (
    FOOT_CONTACT_Z,
    _adjust_leg_foot_height,
    _set_pose,
    _shank_side_down_penalty,
    _tilt_cost,
    foot_pad_bottom_z,
    foot_world,
    load_foot_frames,
    load_stand_pose,
    solve_tibia_for_flat,
)

_MODEL_PATH = None

# 候选抬高方案（仅对称抬高 body_z，m）— 自动选最稳最高的一组
_LIFT_PRESETS = (0.010, 0.014, 0.018, 0.020, 0.022)


def _model_path() -> str:
    import os

    global _MODEL_PATH
    if _MODEL_PATH is None:
        _MODEL_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "generated", "SIX-MOTOR_sim.xml"
        )
    return _MODEL_PATH


def _pose_cost(feet: list[float], body_z: float, base_z: float) -> float:
    """越小越好：足底参差、左右不对称、穿地、悬空惩罚；略奖励更高机身。"""
    spread = float(np.var(feet))
    left = float(np.mean(feet[:3]))
    right = float(np.mean(feet[3:6]))
    pen_lr = 80.0 * (left - right) ** 2
    pen_z = 30.0 * float(np.mean([(f - FOOT_CONTACT_Z) ** 2 for f in feet]))
    pen_low = 5.0 * max(0.0, -min(feet)) ** 2
    pen_high = 2.0 * max(0.0, max(feet) - 0.025) ** 2
    pen_hover = 120.0 * float(
        np.sum([max(0.0, f - 0.010) ** 2 for f in feet])
    )
    bonus = 0.15 * (body_z - base_z)
    return spread + pen_lr + pen_z + pen_low + pen_high + pen_hover - bonus


def _adjust_leg_contact_priority(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg: int,
    pose: Dict[str, float],
    body_z: float,
    frames,
    target_z: float = FOOT_CONTACT_Z,
) -> Dict[str, float]:
    """悬空腿优先贴地：放宽足底倾角惩罚，确保 femur/tibia 能把脚落下来。"""
    out = dict(pose)
    jid_f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_femur_joint")
    jid_t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_tibia_joint")
    fr, tr = model.jnt_range[jid_f], model.jnt_range[jid_t]
    best = (1e9, out[f"leg{leg}_femur_joint"], out[f"leg{leg}_tibia_joint"])

    def leg_err(f: float, t: float) -> float:
        trial = dict(out)
        trial[f"leg{leg}_femur_joint"] = f
        trial[f"leg{leg}_tibia_joint"] = t
        _set_pose(model, data, trial, body_z)
        mujoco.mj_forward(model, data)
        pad_z = foot_pad_bottom_z(model, data, leg)
        _, n = foot_world(model, data, leg, frames)
        return (
            600.0 * (pad_z - target_z) ** 2
            + 8.0 * _tilt_cost(n)
            + 2.0 * _shank_side_down_penalty(model, data, leg, frames)
        )

    for f in np.linspace(fr[0], fr[1], 30):
        for t in np.linspace(tr[0], tr[1], 32):
            e = leg_err(float(f), float(t))
            if e < best[0]:
                best = (e, float(f), float(t))
    _, f1, t1 = best
    for f in np.linspace(max(fr[0], f1 - 0.18), min(fr[1], f1 + 0.18), 16):
        for t in np.linspace(max(tr[0], t1 - 0.25), min(tr[1], t1 + 0.25), 18):
            e = leg_err(float(f), float(t))
            if e < best[0]:
                best = (e, float(f), float(t))
    out[f"leg{leg}_femur_joint"] = best[1]
    out[f"leg{leg}_tibia_joint"] = best[2]
    return out


def _fix_hovering_legs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    body_z: float,
    frames,
    hover_thresh: float = 0.010,
) -> Dict[str, float]:
    out = dict(pose)
    for leg in range(1, 7):
        _set_pose(model, data, out, body_z)
        mujoco.mj_forward(model, data)
        if foot_pad_bottom_z(model, data, leg) <= hover_thresh:
            continue
        out = _adjust_leg_contact_priority(model, data, leg, out, body_z, frames)
    return out


def _trial_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    stand_pose: Dict[str, float],
    body_z: float,
    body_lift_m: float,
    frames,
) -> Tuple[Dict[str, float], float, list[float]]:
    pose = dict(stand_pose)
    bz = float(body_z) + float(body_lift_m)
    for leg in range(1, 7):
        c = pose[f"leg{leg}_coxa_joint"]
        f = pose[f"leg{leg}_femur_joint"]
        pose[f"leg{leg}_tibia_joint"] = solve_tibia_for_flat(
            model, data, leg, c, f, bz, pose, frames
        )
    for leg in range(1, 7):
        pose = _adjust_leg_foot_height(
            model, data, leg, pose, bz, frames, target_z=FOOT_CONTACT_Z
        )
    pose = _fix_hovering_legs(model, data, pose, bz, frames)

    data.qpos[0:3] = 0.0, 0.0, bz
    data.qpos[3:7] = 1.0, 0.0, 0.0, 0.0
    for jn, val in pose.items():
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
        data.qpos[adr] = float(val)
    mujoco.mj_forward(model, data)
    feet = [foot_pad_bottom_z(model, data, leg) for leg in range(1, 7)]
    return pose, bz, feet


def _physics_settle_stand(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    body_z: float,
    *,
    steps: int = 2500,
) -> Tuple[Dict[str, float], float]:
    """重力+摩擦下 PD 平衡，得到与足底支撑一致的关节角与机身高度。"""
    set_actuator_gains(model, LOCOMOTION_KP, LOCOMOTION_KV)
    data.qpos[0:3] = 0.0, 0.0, float(body_z)
    data.qpos[3:7] = 1.0, 0.0, 0.0, 0.0
    for jn, val in pose.items():
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
        data.qpos[adr] = float(val)
    data.qvel[:] = 0.0
    for jn, val in pose.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jn}_act")
        data.ctrl[aid] = float(val)
    for _ in range(int(steps)):
        mujoco.mj_step(model, data)
    return read_joint_pose(model, data), float(data.qpos[2])


def lift_stand_for_rl(
    stand_pose: Dict[str, float],
    body_z: float,
    *,
    model: Optional[mujoco.MjModel] = None,
) -> Tuple[Dict[str, float], float]:
    """在若干预设中选取最稳的抬高站立角（避免两倍抬高导致脚离地趴下）。"""
    if model is None:
        model = mujoco.MjModel.from_xml_path(_model_path())
    data = mujoco.MjData(model)
    frames = load_foot_frames(model, stand_pose, body_z)

    _, _, base_feet = _trial_pose(model, data, stand_pose, body_z, 0.0, frames)
    base_cost = _pose_cost(base_feet, body_z, body_z)
    max_cost = base_cost + 0.006  # 允许略差于标定，换取更高机身

    best_pose = dict(stand_pose)
    best_bz = float(body_z)

    for lift_m in _LIFT_PRESETS:
        pose, bz, feet = _trial_pose(model, data, stand_pose, body_z, lift_m, frames)
        cost = _pose_cost(feet, bz, body_z)
        if cost <= max_cost and bz > best_bz:
            best_pose = pose
            best_bz = bz

    return _physics_settle_stand(model, data, best_pose, best_bz)


def load_sim_stand_pose() -> Tuple[Dict[str, float], float] | None:
    """仿真统一站立姿态（标定 + 安全抬高），步态使能目标 / RL 共用。"""
    loaded = load_stand_pose()
    if loaded is None:
        return None
    return lift_stand_for_rl(*loaded)


def sync_gait_stand(
    gait, stand_pose: Dict[str, float], body_z: float | None = None
) -> None:
    """步态规划器与抬高后的站立角对齐。"""
    gait.stand = dict(stand_pose)
    if body_z is not None and hasattr(gait, "p"):
        gait.p.body_height = float(body_z)
    if getattr(gait, "_joint_crawl", None) is not None:
        gait._joint_crawl.stand = dict(stand_pose)
        gait._joint_crawl._prev_joints = dict(stand_pose)
