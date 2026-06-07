#!/usr/bin/env python3
"""关节限位与实机 18-DOF 映射（仿真同名同单位，1:1 部署）。"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import mujoco

# 步态规划：抬脚偏置在 femur/tibia 间分配（18 关节链内部，非 12→18 映射）
FEMUR_UD_RATIO = 0.65
TIBIA_UD_RATIO = 0.35

LEG_JOINT_SUFFIXES = ("coxa", "femur", "tibia")


def all_joint_names() -> List[str]:
    return [
        f"leg{leg}_{j}_joint"
        for leg in range(1, 7)
        for j in LEG_JOINT_SUFFIXES
    ]


def clamp_joint_targets(
    model: mujoco.MjModel, targets: Dict[str, float]
) -> Dict[str, float]:
    """将关节目标限制在模型 range 内（仿真与实机共用关节名/弧度）。"""
    out: Dict[str, float] = {}
    for jname, val in targets.items():
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        except Exception:
            out[jname] = float(val)
            continue
        lo, hi = model.jnt_range[jid]
        out[jname] = float(max(lo, min(hi, val)))
    return out


def joint_limits_deg(model: mujoco.MjModel) -> Dict[str, Tuple[float, float]]:
    """各关节机械限位（度）。"""
    out: Dict[str, Tuple[float, float]] = {}
    for jname in all_joint_names():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        lo, hi = model.jnt_range[jid]
        out[jname] = (math.degrees(lo), math.degrees(hi))
    return out


def joint_deltas_rad(
    joints: Dict[str, float],
    stand: Dict[str, float],
) -> Dict[str, float]:
    """18 关节相对站立角的增量（rad）。"""
    return {
        jn: float(joints.get(jn, stand[jn]) - stand[jn])
        for jn in stand
    }


def joint_deltas_deg(
    joints: Dict[str, float],
    stand: Dict[str, float],
) -> Dict[str, float]:
    return {jn: math.degrees(d) for jn, d in joint_deltas_rad(joints, stand).items()}


def sim_to_real_joints(
    sim_joints: Dict[str, float],
    stand: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """
    仿真 → 实机：18 关节 1:1，同名同单位（rad）。
    stand 非空时输出相对站立的增量；否则输出绝对角。
    """
    if stand is None:
        return dict(sim_joints)
    return joint_deltas_rad(sim_joints, stand)


def max_abs_delta_deg(
    joints: Dict[str, float],
    stand: Dict[str, float],
) -> Tuple[str, float]:
    deltas = joint_deltas_deg(joints, stand)
    jn = max(deltas, key=lambda k: abs(deltas[k]))
    return jn, abs(deltas[jn])
