#!/usr/bin/env python3
"""支撑足世界坐标锁定：模拟球轴足端可转但不可滑移。"""
from __future__ import annotations

from typing import Dict, Optional

import mujoco
import numpy as np

from foot_kinematics import foot_world


def world_foot_to_base(
    model: mujoco.MjModel, data: mujoco.MjData, base_id: int, p_world: np.ndarray
) -> np.ndarray:
    R = data.xmat[base_id].reshape(3, 3)
    base_pos = data.xpos[base_id]
    return R.T @ (p_world - base_pos)


def apply_stance_world_lock(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik,
    anchors: Dict[int, np.ndarray],
    actuator_map,
) -> None:
    """
    将锚定支撑足的接触点拉回目标世界 XY（允许绕法向旋转）。
    通过 IK 修正关节角并同步 ctrl。
    """
    if not anchors:
        return
    base_id = ik.base_id
    for leg, p_goal in anchors.items():
        p_cur, _ = foot_world(model, data, leg, ik.frames)
        # 仅锁定水平位置，z 由地面接触决定
        p_fix = np.array([p_goal[0], p_goal[1], p_cur[2]])
        target_base = world_foot_to_base(model, data, base_id, p_fix)
        jid_list = ik._joint_ids[leg]
        seed = tuple(float(data.qpos[model.jnt_qposadr[jid]]) for jid in jid_list)
        c, f, t = ik.solve(leg, target_base, seed=seed)
        for jn, val in zip(
            ("coxa", "femur", "tibia"),
            (c, f, t),
        ):
            jname = f"leg{leg}_{jn}_joint"
            adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)]
            data.qpos[adr] = val
            aid = actuator_map.get(jname)
            if aid is not None and aid >= 0:
                data.ctrl[aid] = val
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def build_actuator_map(model: mujoco.MjModel) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for leg in range(1, 7):
        for jn in ("coxa", "femur", "tibia"):
            name = f"leg{leg}_{jn}_joint"
            aname = f"{name}_act"
            try:
                m[name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            except Exception:
                m[name] = -1
    return m
