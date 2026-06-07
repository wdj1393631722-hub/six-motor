#!/usr/bin/env python3
"""支撑足世界坐标锁定：模拟球轴足端可转但不可滑移（依赖地面摩擦）。"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import mujoco
import numpy as np

from foot_kinematics import foot_pad_bottom_z, foot_world

CONTACT_DIST_MAX = 0.004


def world_foot_to_base(
    model: mujoco.MjModel, data: mujoco.MjData, base_id: int, p_world: np.ndarray
) -> np.ndarray:
    R = data.xmat[base_id].reshape(3, 3)
    base_pos = data.xpos[base_id]
    return R.T @ (p_world - base_pos)


def _geom_ids(model: mujoco.MjModel) -> Tuple[Optional[int], Optional[int]]:
    try:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    except Exception:
        floor_id = None
    foot_ids = {}
    for leg in range(1, 7):
        try:
            foot_ids[leg] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"leg{leg}_foot_pad"
            )
        except Exception:
            foot_ids[leg] = None
    return floor_id, foot_ids


def foot_in_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg: int,
    floor_id: Optional[int] = None,
    foot_gid: Optional[int] = None,
) -> bool:
    """足底摩擦垫与地面是否接触。"""
    if floor_id is None or foot_gid is None:
        floor_id2, foot_map = _geom_ids(model)
        floor_id = floor_id if floor_id is not None else floor_id2
        foot_gid = foot_map.get(leg) if foot_gid is None else foot_gid
    if floor_id is None or foot_gid is None:
        return foot_pad_bottom_z(model, data, leg) <= 0.003
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if {g1, g2} == {floor_id, foot_gid} and float(c.dist) <= CONTACT_DIST_MAX:
            return True
    return False


def capture_foot_anchors(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    legs: Iterable[int],
    frames,
    require_contact: bool = True,
) -> Dict[int, np.ndarray]:
    """记录支撑足世界坐标锚点（XY 防滑）。"""
    floor_id, foot_map = _geom_ids(model)
    out: Dict[int, np.ndarray] = {}
    for leg in legs:
        if require_contact and not foot_in_contact(
            model, data, leg, floor_id, foot_map.get(leg)
        ):
            continue
        p, _ = foot_world(model, data, leg, frames)
        out[leg] = p.copy()
    return out


def _leg_qvel_indices(model: mujoco.MjModel, leg: int) -> Tuple[int, ...]:
    idx = []
    for jn in ("coxa", "femur", "tibia"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_{jn}_joint")
        idx.append(int(model.jnt_dofadr[jid]))
    return tuple(idx)


SLIP_CORRECT_ON_M = 0.0025
SLIP_CORRECT_GAIN = 0.12


def blend_stance_ctrl_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik,
    anchors: Dict[int, np.ndarray],
    gait_targets: Dict[str, float],
    slip_on: float = SLIP_CORRECT_ON_M,
    corr_gain: float = SLIP_CORRECT_GAIN,
) -> Dict[str, float]:
    """
    将防滑 IK 修正叠加进步态 ctrl 目标（不直写 qpos），避免 PD 与后处理打架导致腿抖。
    """
    if not anchors:
        return dict(gait_targets)
    out = dict(gait_targets)
    base_id = ik.base_id
    floor_id, foot_map = _geom_ids(model)
    gain = max(0.0, min(float(corr_gain), 0.45))
    for leg, p_goal in anchors.items():
        if not foot_in_contact(model, data, leg, floor_id, foot_map.get(leg)):
            continue
        p_cur, _ = foot_world(model, data, leg, ik.frames)
        slip = math.hypot(p_cur[0] - p_goal[0], p_cur[1] - p_goal[1])
        if slip < slip_on:
            continue
        p_fix = np.array([p_goal[0], p_goal[1], p_cur[2]])
        target_base = world_foot_to_base(model, data, base_id, p_fix)
        jid_list = ik._joint_ids[leg]
        seed = tuple(float(data.qpos[model.jnt_qposadr[jid]]) for jid in jid_list)
        c, f, t = ik.solve(leg, target_base, seed=seed)
        for jn, ik_val in zip(("coxa", "femur", "tibia"), (c, f, t)):
            jname = f"leg{leg}_{jn}_joint"
            gait_val = out.get(
                jname,
                float(
                    data.qpos[
                        model.jnt_qposadr[
                            mujoco.mj_name2id(
                                model, mujoco.mjtObj.mjOBJ_JOINT, jname
                            )
                        ]
                    ]
                ),
            )
            out[jname] = float(gait_val) + gain * (float(ik_val) - float(gait_val))
    return out


def apply_stance_world_lock(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik,
    anchors: Dict[int, np.ndarray],
    actuator_map,
    **_kwargs,
) -> None:
    """槽位步态兼容：仅修正 ctrl，不直写 qpos。"""
    if not anchors:
        return
    gait_targets = {}
    for leg in range(1, 7):
        for jn in ("coxa", "femur", "tibia"):
            jname = f"leg{leg}_{jn}_joint"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            gait_targets[jname] = float(data.qpos[model.jnt_qposadr[jid]])
    merged = blend_stance_ctrl_targets(model, data, ik, anchors, gait_targets)
    for jname, val in merged.items():
        aid = actuator_map.get(jname)
        if aid is not None and aid >= 0:
            data.ctrl[aid] = val


def damp_leg_joint_velocities(
    model: mujoco.MjModel, data: mujoco.MjData, factor: float = 0.52
) -> None:
    """行走后轻度阻尼腿关节角速度，抑制高频抖动。"""
    f = max(0.0, min(float(factor), 1.0))
    for leg in range(1, 7):
        for jn in ("coxa", "femur", "tibia"):
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_{jn}_joint"
            )
            data.qvel[int(model.jnt_dofadr[jid])] *= f


class JointGaitStanceTracker:
    """关节三角步态：按相位捕获/维持支撑足世界锚点。"""

    def __init__(self) -> None:
        self._anchors: Dict[int, np.ndarray] = {}
        self._last_phase_name = ""

    def reset(self) -> None:
        self._anchors.clear()
        self._last_phase_name = ""

    def update(
        self,
        phase_name: str,
        phase_kind: str,
        swing_group: str,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ik,
    ) -> Dict[int, np.ndarray]:
        from joint_tripod_gait import _group_legs, _stance_legs

        if phase_name != self._last_phase_name:
            if phase_kind == "swing":
                captured = capture_foot_anchors(
                    model,
                    data,
                    _stance_legs(swing_group),
                    ik.frames,
                    require_contact=True,
                )
                self._anchors.update(captured)
            elif phase_kind == "place":
                captured = capture_foot_anchors(
                    model,
                    data,
                    _group_legs(swing_group),
                    ik.frames,
                    require_contact=True,
                )
                self._anchors.update(captured)
            elif phase_kind == "push":
                stance = _stance_legs(swing_group)
                captured = capture_foot_anchors(
                    model,
                    data,
                    stance,
                    ik.frames,
                    require_contact=True,
                )
                self._anchors.update(captured)
                for leg in _group_legs(swing_group):
                    self._anchors.pop(leg, None)
            self._last_phase_name = phase_name

        if phase_kind == "push":
            stance = _stance_legs(swing_group)
            return {
                leg: p.copy()
                for leg, p in self._anchors.items()
                if leg in stance
            }
        return {}


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
