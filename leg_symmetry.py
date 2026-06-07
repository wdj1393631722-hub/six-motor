#!/usr/bin/env python3
"""
六足左右对称：以机体 YZ 平面为镜，配对腿足端位置镜像一致。

配对（URDF 安装点 x 坐标相反）:
  leg1 (+x, +y) ↔ leg6 (-x, +y)
  leg3 (+x, -y) ↔ leg4 (-x, -y)
  leg2 (+x,  0) ↔ leg5 (-x,  0)   # leg2 髋轴 X，leg5 髋轴 -Z
"""
from __future__ import annotations

from typing import Dict, Tuple

import mujoco
import numpy as np

# (参考腿, 镜像腿) — 先标定参考腿，再求镜像腿
MIRROR_PAIRS: Tuple[Tuple[int, int], ...] = ((1, 6), (3, 4), (2, 5))
# 新 URDF 下三对均可关节镜像
JOINT_SYMMETRY_PAIRS: Tuple[Tuple[int, int], ...] = MIRROR_PAIRS
REFERENCE_LEGS = (1, 3, 2)

# 髋座在 base_link 下 (m)，用于六边形足端目标
HIP_MOUNT_XY: Dict[int, Tuple[float, float]] = {
    1: (0.1, 0.16209),
    6: (-0.1, 0.16209),
    2: (0.16, 0.002094),
    5: (-0.16, 0.002094),
    3: (0.1, -0.15791),
    4: (-0.1, -0.15791),
}

# 关节角镜像符号 (coxa, femur, tibia)：相对参考腿
MIRROR_JOINT_SIGNS: Dict[Tuple[int, int], Tuple[int, int, int]] = {
    (1, 6): (-1, 1, 1),
    (3, 4): (-1, 1, 1),
    (2, 5): (-1, 1, 1),
}

# 各腿髋座在 base_link 下的方位角 (rad)，用于步态转向分配
LEG_AZIMUTH_DEG = {
    1: 60.0,
    6: 120.0,
    3: -60.0,
    4: -120.0,
    2: 0.0,
    5: 180.0,
}


def mirror_joint_pose(pose: Dict[str, float], ref_leg: int, mir_leg: int) -> None:
    """按镜像符号统一一对腿的关节角幅值（coxa 左右相反）。"""
    sc, sf, st = MIRROR_JOINT_SIGNS[(ref_leg, mir_leg)]
    for j, s_ref, s_mir in zip(
        ("coxa", "femur", "tibia"),
        (1, 1, 1),
        (sc, sf, st),
    ):
        kr = f"leg{ref_leg}_{j}_joint"
        km = f"leg{mir_leg}_{j}_joint"
        mag = 0.5 * (abs(pose[kr]) + abs(pose[km]))
        pose[kr] = mag * s_ref
        pose[km] = mag * s_mir


def apply_strict_mirror_from_refs(pose: Dict[str, float]) -> Dict[str, float]:
    """以 +x 参考腿 (1,3,2) 为准，严格镜像到 (6,4,5)。"""
    out = dict(pose)
    for ref_leg, mir_leg in MIRROR_PAIRS:
        sc, sf, st = MIRROR_JOINT_SIGNS[(ref_leg, mir_leg)]
        out[f"leg{mir_leg}_coxa_joint"] = sc * out[f"leg{ref_leg}_coxa_joint"]
        out[f"leg{mir_leg}_femur_joint"] = sf * out[f"leg{ref_leg}_femur_joint"]
        out[f"leg{mir_leg}_tibia_joint"] = st * out[f"leg{ref_leg}_tibia_joint"]
    return out


def symmetrize_all_pairs(pose: Dict[str, float]) -> Dict[str, float]:
    """三对镜像腿统一关节幅值（左右对称）。"""
    out = dict(pose)
    for ref_leg, mir_leg in MIRROR_PAIRS:
        mirror_joint_pose(out, ref_leg, mir_leg)
    return out


# 参考腿 coxa 期望符号（保证俯视图外展方向）
_REF_COXA_SIGN: Dict[Tuple[int, int], int] = {(1, 6): 1, (3, 4): -1, (2, 5): -1}


def symmetrize_coxa_pairs(pose: Dict[str, float]) -> Dict[str, float]:
    """俯视图对称：coxa_mir = -coxa_ref，幅值取两侧 |coxa| 的较大值。"""
    out = dict(pose)
    for ref_leg, mir_leg in MIRROR_PAIRS:
        cr = out[f"leg{ref_leg}_coxa_joint"]
        cm = out[f"leg{mir_leg}_coxa_joint"]
        mag = max(abs(cr), abs(cm))
        s = _REF_COXA_SIGN[(ref_leg, mir_leg)]
        out[f"leg{ref_leg}_coxa_joint"] = s * mag
        out[f"leg{mir_leg}_coxa_joint"] = -s * mag
    return out


def foot_azimuth_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg: int,
    foot_world_fn,
    frames,
) -> float:
    """足端在 base 系水平方位角 (rad)。"""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    p, _ = foot_world_fn(model, data, leg, frames)
    bp = data.xpos[base_id]
    R = data.xmat[base_id].reshape(3, 3)
    fb = R.T @ (p - bp)
    return float(np.arctan2(fb[1], fb[0]))


def topview_azimuth_symmetry_cost(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_world_fn,
    frames,
) -> float:
    """镜像腿足端方位应满足 az_mir ≈ π - az_ref。"""
    cost = 0.0
    for ref, mir in MIRROR_PAIRS:
        az_r = foot_azimuth_base(model, data, ref, foot_world_fn, frames)
        az_m = foot_azimuth_base(model, data, mir, foot_world_fn, frames)
        da = abs((az_m - (np.pi - az_r) + np.pi) % (2 * np.pi) - np.pi)
        cost += float(da**2)
    return cost


def hip_azimuth_rad(leg: int) -> float:
    x, y = HIP_MOUNT_XY[leg]
    return float(np.arctan2(y, x))


def hex_foot_target_base(leg: int, radius: float, foot_z: float) -> np.ndarray:
    """足端目标：沿髋座方位角、半径 radius、高度 foot_z（base 系）。"""
    a = hip_azimuth_rad(leg)
    return np.array([radius * np.cos(a), radius * np.sin(a), foot_z])


def feet_in_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    body_z: float,
    foot_world_fn,
    frames,
) -> Dict[int, np.ndarray]:
    from foot_kinematics import _set_pose

    _set_pose(model, data, pose, body_z)
    mujoco.mj_forward(model, data)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    bp = data.xpos[base_id]
    R = data.xmat[base_id].reshape(3, 3)
    out: Dict[int, np.ndarray] = {}
    for leg in range(1, 7):
        p, _ = foot_world_fn(model, data, leg, frames)
        out[leg] = R.T @ (p - bp)
    return out


def hexagon_uniformity_cost(feet: Dict[int, np.ndarray]) -> float:
    """六足端在水平面半径一致、相邻腿张角均匀。"""
    radii = [float(np.hypot(feet[l][0], feet[l][1])) for l in range(1, 7)]
    r_mean = float(np.mean(radii))
    cost = float(np.var(radii)) + 0.02 * (r_mean - radii[0]) ** 2
    # 相对 60° 分布：各腿方位与髋座方位一致时 cost 低
    for leg in range(1, 7):
        a_hip = hip_azimuth_rad(leg)
        a_foot = float(np.arctan2(feet[leg][1], feet[leg][0]))
        da = abs((a_foot - a_hip + np.pi) % (2 * np.pi) - np.pi)
        cost += 0.5 * da**2
    return cost


def initial_mirror_guess(
    pose: Dict[str, float], ref_leg: int, mir_leg: int
) -> Tuple[float, float, float]:
    sc, sf, st = MIRROR_JOINT_SIGNS[(ref_leg, mir_leg)]
    return (
        sc * pose[f"leg{ref_leg}_coxa_joint"],
        sf * pose[f"leg{ref_leg}_femur_joint"],
        st * pose[f"leg{ref_leg}_tibia_joint"],
    )


def mirror_point_xy(p: np.ndarray) -> np.ndarray:
    """足端世界坐标关于 YZ 平面镜像 (x 取反)。"""
    return np.array([-p[0], p[1], p[2]])


def mirror_foot_in_base(foot_base: np.ndarray) -> np.ndarray:
    """足端 base 系坐标关于 YZ 平面镜像。"""
    return np.array([-foot_base[0], foot_base[1], foot_base[2]])


def femur_outward_azimuth(
    model: mujoco.MjModel, data: mujoco.MjData, leg: int
) -> float:
    """大腿在 base 系水平面外展方位角 (rad)。"""
    coxa_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"leg{leg}_coxa")
    femur_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"leg{leg}_femur")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    d_world = data.xpos[femur_id] - data.xpos[coxa_id]
    R = data.xmat[base_id].reshape(3, 3)
    d = R.T @ d_world
    return float(np.arctan2(d[1], d[0]))


def radial_angle_cost(model: mujoco.MjModel, data: mujoco.MjData, leg: int) -> float:
    """大腿外展方向与髋座方位角一致时 cost 低。"""
    az = femur_outward_azimuth(model, data, leg)
    tgt = hip_azimuth_rad(leg)
    da = abs((az - tgt + np.pi) % (2 * np.pi) - np.pi)
    return float(da**2)


def symmetry_foot_error(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    body_z: float,
    foot_world_fn,
    frames,
    pairs: Tuple[Tuple[int, int], ...] | None = None,
) -> float:
    from foot_kinematics import _set_pose

    _set_pose(model, data, pose, body_z)
    mujoco.mj_forward(model, data)
    err = 0.0
    for ref, mir in pairs or JOINT_SYMMETRY_PAIRS:
        p_ref, _ = foot_world_fn(model, data, ref, frames)
        p_mir, _ = foot_world_fn(model, data, mir, frames)
        target = mirror_point_xy(p_ref)
        err += float(np.sum((p_mir - target) ** 2))
    return err


def calibrate_mirror_leg(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ref_leg: int,
    mir_leg: int,
    pose: Dict[str, float],
    body_z: float,
    foot_world_fn,
    frames,
    tilt_cost_fn,
) -> Tuple[float, float, float]:
    """在参考腿已标定前提下，搜索镜像腿关节角使足端镜像重合且脚底平行。"""
    from foot_kinematics import _set_pose

    _set_pose(model, data, pose, body_z)
    mujoco.mj_forward(model, data)
    p_target = mirror_point_xy(foot_world_fn(model, data, ref_leg, frames)[0])

    c0, f0, t0 = initial_mirror_guess(pose, ref_leg, mir_leg)
    best = (1e9, c0, f0, t0)

    az_ref = foot_azimuth_base(model, data, ref_leg, foot_world_fn, frames)
    az_tgt_mir = float(np.pi - az_ref)

    def mir_err(c: float, f: float, t: float) -> float:
        trial = dict(pose)
        trial[f"leg{mir_leg}_coxa_joint"] = c
        trial[f"leg{mir_leg}_femur_joint"] = f
        trial[f"leg{mir_leg}_tibia_joint"] = t
        _set_pose(model, data, trial, body_z)
        mujoco.mj_forward(model, data)
        p, n = foot_world_fn(model, data, mir_leg, frames)
        pos_err = float(np.sum((p - p_target) ** 2))
        az = foot_azimuth_base(model, data, mir_leg, foot_world_fn, frames)
        da_m = abs((az - az_tgt_mir + np.pi) % (2 * np.pi) - np.pi)
        da_h = abs(
            (az - hip_azimuth_rad(mir_leg) + np.pi) % (2 * np.pi) - np.pi
        )
        return (
            pos_err
            + 80.0 * tilt_cost_fn(n)
            + 20.0 * (p[2] - p_target[2]) ** 2
            + 35.0 * da_m**2
            + 15.0 * da_h**2
        )

    jid_c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{mir_leg}_coxa_joint")
    jid_f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{mir_leg}_femur_joint")
    jid_t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{mir_leg}_tibia_joint")
    cr, fr, tr = model.jnt_range[jid_c], model.jnt_range[jid_f], model.jnt_range[jid_t]

    n_c, n_f, n_t = (33, 32, 34) if mir_leg in (2, 5) else (25, 28, 30)
    for c in np.linspace(cr[0], cr[1], n_c):
        for f in np.linspace(fr[0], fr[1], n_f):
            for t in np.linspace(tr[0], tr[1], n_t):
                e = mir_err(c, f, t)
                if e < best[0]:
                    best = (e, c, f, t)
    _, c0, f0, t0 = best
    span = 0.28 if mir_leg in (2, 5) else 0.18
    for c in np.linspace(max(cr[0], c0 - span), min(cr[1], c0 + span), 15):
        for f in np.linspace(max(fr[0], f0 - span), min(fr[1], f0 + span), 17):
            for t in np.linspace(max(tr[0], t0 - 0.3), min(tr[1], t0 + 0.3), 19):
                e = mir_err(c, f, t)
                if e < best[0]:
                    best = (e, c, f, t)
    return best[1], best[2], best[3]


def refine_symmetric_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    body_z: float,
    foot_world_fn,
    frames,
    tilt_cost_fn,
    rounds: int = 3,
) -> Dict[str, float]:
    """多轮镜像腿精修。"""
    out = dict(pose)
    for _ in range(rounds):
        for ref_leg, mir_leg in JOINT_SYMMETRY_PAIRS:
            c, f, t = calibrate_mirror_leg(
                model,
                data,
                ref_leg,
                mir_leg,
                out,
                body_z,
                foot_world_fn,
                frames,
                tilt_cost_fn,
            )
            out[f"leg{mir_leg}_coxa_joint"] = c
            out[f"leg{mir_leg}_femur_joint"] = f
            out[f"leg{mir_leg}_tibia_joint"] = t
    return out


def report_symmetry(
    model: mujoco.MjModel,
    pose: Dict[str, float],
    body_z: float,
    foot_world_fn,
    frames,
) -> None:
    from foot_kinematics import _set_pose

    data = mujoco.MjData(model)
    _set_pose(model, data, pose, body_z)
    mujoco.mj_forward(model, data)
    for ref, mir in MIRROR_PAIRS:
        p_ref, _ = foot_world_fn(model, data, ref, frames)
        p_mir, _ = foot_world_fn(model, data, mir, frames)
        tgt = mirror_point_xy(p_ref)
        d = np.linalg.norm(p_mir - tgt)
        print(f"  leg{ref}↔leg{mir}: 镜像误差 {d*1000:.1f} mm")
