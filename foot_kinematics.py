#!/usr/bin/env python3
"""足端姿态：从小腿 STL 估计脚底，标定/求解使脚底平行于地面。"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Tuple

import mujoco
import numpy as np
import trimesh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml")
MESH_DIR = os.path.join(SCRIPT_DIR, "generated", "meshes_decimated")
STAND_POSE_PATH = os.path.join(SCRIPT_DIR, "generated", "stand_pose_flat.json")
PRONE_POSE_PATH = os.path.join(SCRIPT_DIR, "generated", "prone_pose_flat.json")

# 脚底在 tibia 坐标系下的接触点与 outward 法向（指向地心）
FootFrame = Tuple[np.ndarray, np.ndarray]

# 站立时各腿足底接触点目标世界高度（m，z=0 为地面）
FOOT_CONTACT_Z = 0.008
# 机身原点最低高度（m），避免主体下板贴地
MIN_BODY_HEIGHT = 0.065
# 标定目标：主体下沿离地净空（m）
TARGET_BASE_CLEARANCE = 0.045
# 行走时机身参考高度（m），与物理平衡标定一致（勿 kinematic 锁定 qpos）
LOCOMOTION_BODY_HEIGHT = 0.087

def _tilt_cost(n_world: np.ndarray) -> float:
    """脚底法向应指向世界 -Z（朝下），与竖直夹角越小 cost 越小。"""
    n = n_world / (np.linalg.norm(n_world) + 1e-12)
    return float(n[0] ** 2 + n[1] ** 2 + (n[2] + 1.0) ** 2)


def _foot_from_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从小腿 STL 估计足底平面：取小腿远端区域的最大共面面积法向。
    避免把包围盒 -Z 误当成真实足底（会导致侧面/棱边蹭地）。
    """
    m = trimesh.load(mesh_path, force="mesh")
    ext = m.bounds[1] - m.bounds[0]
    axis = int(np.argmax(ext))
    coord = m.vertices[:, axis]
    foot_band = m.vertices[coord <= np.percentile(coord, 15)]

    # 远端三角面按面积统计法向
    face_weights: Dict[Tuple[float, float, float], float] = {}
    face_centroids = []
    for tri, fn, area in zip(m.triangles, m.face_normals, m.area_faces):
        cen = tri.mean(axis=0)
        if cen[axis] > np.percentile(coord, 20):
            continue
        key = tuple(float(np.round(v, 2)) for v in fn)
        face_weights[key] = face_weights.get(key, 0.0) + float(area)
        face_centroids.append(cen)

    if face_weights:
        best_key = max(face_weights, key=face_weights.get)
        foot_n = np.array(best_key, dtype=float)
        foot_n /= np.linalg.norm(foot_n) + 1e-12
    else:
        c = foot_band.mean(axis=0)
        _, _, vh = np.linalg.svd(foot_band - c, full_matrices=False)
        foot_n = vh[2]
        foot_n /= np.linalg.norm(foot_n) + 1e-12

    # 法向指向小腿外侧（足底朝地）
    if foot_n[axis] > 0:
        foot_n = -foot_n

    # 接触点取远端顶点沿足底法向最突出的位置（真实平面最靠地侧）
    proj = foot_band @ foot_n
    foot_pt = foot_band[int(np.argmax(proj))].copy()
    return foot_pt, foot_n


def foot_pad_bottom_z(model: mujoco.MjModel, data: mujoco.MjData, leg: int) -> float:
    """足底碰撞盒最低点世界坐标 z。"""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"leg{leg}_foot_pad")
    pos = data.geom_xpos[gid]
    xmat = data.geom_xmat[gid].reshape(3, 3)
    half = model.geom_size[gid]
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx, sy, sz]) * half
                corners.append((pos + xmat @ local)[2])
    return float(min(corners))


def foot_pad_heights(
    model: mujoco.MjModel, data: mujoco.MjData
) -> Dict[int, float]:
    return {leg: foot_pad_bottom_z(model, data, leg) for leg in range(1, 7)}


def foot_pad_quat(foot_n: np.ndarray) -> str:
    """MuJoCo geom quat：将盒体 +Z 轴对齐到足底法向 foot_n。"""
    n = np.asarray(foot_n, dtype=float)
    n /= np.linalg.norm(n) + 1e-12
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z, n))
    if dot > 0.999:
        return "1 0 0 0"
    if dot < -0.999:
        return "0 1 0 0"
    axis = np.cross(z, n)
    axis /= np.linalg.norm(axis) + 1e-12
    ang = math.acos(np.clip(dot, -1.0, 1.0))
    w = math.cos(ang / 2)
    s = math.sin(ang / 2)
    return f"{w:.6f} {axis[0]*s:.6f} {axis[1]*s:.6f} {axis[2]*s:.6f}"


def _flip_foot_if_points_up(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg: int,
    foot_pt: np.ndarray,
    foot_n: np.ndarray,
    pose: Dict[str, float],
    body_z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """若法向在世界系指向上方，则翻转 link 系法向。"""
    _set_pose(model, data, pose, body_z)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"leg{leg}_tibia")
    R = data.xmat[bid].reshape(3, 3)
    n_w = R @ foot_n
    if n_w[2] > 0:
        foot_n = -foot_n
    return foot_pt, foot_n


def load_foot_frames(
    model: mujoco.MjModel | None = None,
    stand_pose: Dict[str, float] | None = None,
    body_z: float = 0.14,
) -> Dict[int, FootFrame]:
    """每条腿脚底：mesh 底面 + 法向；按站立姿态确保法向朝下。"""
    frames: Dict[int, FootFrame] = {}
    for leg in range(1, 7):
        path = os.path.join(MESH_DIR, f"leg{leg}_tibia.STL")
        frames[leg] = _foot_from_mesh(path)

    if model is not None and stand_pose is not None:
        data = mujoco.MjData(model)
        for leg in range(1, 7):
            pt, n = frames[leg]
            frames[leg] = _flip_foot_if_points_up(
                model, data, leg, pt, n, stand_pose, body_z
            )
    return frames


def _shank_side_down_penalty(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg: int,
    frames: Dict[int, FootFrame],
) -> float:
    """惩罚小腿横躺：足底法向应朝下，若接近水平则小腿侧面贴地。"""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"leg{leg}_tibia")
    R = data.xmat[bid].reshape(3, 3)
    _, foot_n = frames[leg]
    n_world = R @ foot_n
    n_world /= np.linalg.norm(n_world) + 1e-12
    return float(n_world[0] ** 2 + n_world[1] ** 2)


def _calibrate_leg_solo(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg: int,
    frames: Dict[int, FootFrame],
) -> Tuple[float, float, float]:
    """单腿网格搜索，使该腿脚底平行地面。"""
    best = (1e9, 0.0, 0.3, 1.0)
    jid_c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_coxa_joint")
    jid_f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_femur_joint")
    jid_t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_tibia_joint")
    cr = model.jnt_range[jid_c]
    fr = model.jnt_range[jid_f]
    tr = model.jnt_range[jid_t]

    def leg_err(c: float, f: float, t: float) -> float:
        pose = {f"leg{l}_coxa_joint": 0.0 for l in range(1, 7)}
        for l in range(1, 7):
            pose[f"leg{l}_femur_joint"] = 0.03
            pose[f"leg{l}_tibia_joint"] = 0.0
        pose[f"leg{leg}_coxa_joint"] = c
        pose[f"leg{leg}_femur_joint"] = f
        pose[f"leg{leg}_tibia_joint"] = t
        _set_pose(model, data, pose, 0.14)
        mujoco.mj_forward(model, data)
        p, n = foot_world(model, data, leg, frames)
        cost = (
            _tilt_cost(n)
            + 25.0 * (p[2] - 0.008) ** 2
            + 8.0 * _shank_side_down_penalty(model, data, leg, frames)
        )
        for jid, q in zip(
            (jid_c, jid_f, jid_t), (c, f, t)
        ):
            lo, hi = model.jnt_range[jid]
            margin = 0.1
            if q < lo + margin:
                cost += 40.0 * (lo + margin - q) ** 2
            if q > hi - margin:
                cost += 40.0 * (q - hi + margin) ** 2
        return cost

    for c in np.linspace(cr[0], cr[1], 15):
        for f in np.linspace(fr[0], fr[1], 20):
            for t in np.linspace(tr[0], tr[1], 22):
                e = leg_err(c, f, t)
                if e < best[0]:
                    best = (e, c, f, t)
    _, c0, f0, t0 = best
    for c in np.linspace(max(cr[0], c0 - 0.2), min(cr[1], c0 + 0.2), 11):
        for f in np.linspace(max(fr[0], f0 - 0.2), min(fr[1], f0 + 0.2), 13):
            for t in np.linspace(max(tr[0], t0 - 0.25), min(tr[1], t0 + 0.25), 15):
                e = leg_err(c, f, t)
                if e < best[0]:
                    best = (e, c, f, t)
    return best[1], best[2], best[3]


def _calibrate_leg_to_foot_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg: int,
    target_base: np.ndarray,
    frames: Dict[int, FootFrame],
    body_z: float,
    seed: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    """单腿 IK 网格搜索：足端接近 target_base（base 系）且脚底朝下。"""
    from leg_symmetry import feet_in_base

    best = (1e9, seed[0], seed[1], seed[2])
    jid_c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_coxa_joint")
    jid_f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_femur_joint")
    jid_t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_tibia_joint")
    cr, fr, tr = model.jnt_range[jid_c], model.jnt_range[jid_f], model.jnt_range[jid_t]
    target = np.asarray(target_base, dtype=float)

    def leg_err(c: float, f: float, t: float) -> float:
        pose = {f"leg{l}_coxa_joint": 0.0 for l in range(1, 7)}
        for l in range(1, 7):
            pose[f"leg{l}_femur_joint"] = 0.0
            pose[f"leg{l}_tibia_joint"] = 0.0
        pose[f"leg{leg}_coxa_joint"] = c
        pose[f"leg{leg}_femur_joint"] = f
        pose[f"leg{leg}_tibia_joint"] = t
        _set_pose(model, data, pose, body_z)
        mujoco.mj_forward(model, data)
        fb = feet_in_base(model, data, pose, body_z, foot_world, frames)[leg]
        _, n = foot_world(model, data, leg, frames)
        pos_err = float(np.sum((fb - target) ** 2))
        return (
            pos_err
            + 60.0 * _tilt_cost(n)
            + 30.0 * (fb[2] - target[2]) ** 2
            + 8.0 * _shank_side_down_penalty(model, data, leg, frames)
        )

    for c in np.linspace(cr[0], cr[1], 18):
        for f in np.linspace(fr[0], fr[1], 22):
            for t in np.linspace(tr[0], tr[1], 24):
                e = leg_err(c, f, t)
                if e < best[0]:
                    best = (e, c, f, t)
    _, c0, f0, t0 = best
    for c in np.linspace(max(cr[0], c0 - 0.18), min(cr[1], c0 + 0.18), 11):
        for f in np.linspace(max(fr[0], f0 - 0.18), min(fr[1], f0 + 0.18), 13):
            for t in np.linspace(max(tr[0], t0 - 0.22), min(tr[1], t0 + 0.22), 15):
                e = leg_err(c, f, t)
                if e < best[0]:
                    best = (e, c, f, t)
    return best[1], best[2], best[3]


def _set_pose(model, data, pose: Dict[str, float], body_z: float) -> None:
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qpos[2] = body_z
    data.qpos[3] = 1.0
    for jname, val in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        lo, hi = model.jnt_range[jid]
        data.qpos[model.jnt_qposadr[jid]] = float(np.clip(val, lo, hi))


def foot_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg: int,
    frames: Dict[int, FootFrame],
) -> Tuple[np.ndarray, np.ndarray]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"leg{leg}_tibia")
    R = data.xmat[bid].reshape(3, 3)
    foot_pt, foot_n = frames[leg]
    p = data.xpos[bid] + R @ foot_pt
    n = R @ foot_n
    n /= np.linalg.norm(n) + 1e-12
    return p, n


def foot_heights_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    body_z: float,
    frames: Dict[int, FootFrame],
) -> Dict[int, float]:
    _set_pose(model, data, pose, body_z)
    mujoco.mj_forward(model, data)
    return {leg: foot_world(model, data, leg, frames)[0][2] for leg in range(1, 7)}


def foot_height_uniformity_cost(
    heights: Dict[int, float],
    target_z: float = FOOT_CONTACT_Z,
) -> float:
    """六腿足底高度应落在同一接触平面。"""
    return float(sum((heights[leg] - target_z) ** 2 for leg in range(1, 7)))


def flat_cost(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    body_z: float,
    frames: Dict[int, FootFrame],
    legs: range | None = None,
    target_z: float = FOOT_CONTACT_Z,
) -> float:
    _set_pose(model, data, pose, body_z)
    mujoco.mj_forward(model, data)
    total = 0.0
    leg_list = list(legs or range(1, 7))
    for leg in leg_list:
        p, n = foot_world(model, data, leg, frames)
        total += _tilt_cost(n)
        total += 120.0 * (p[2] - target_z) ** 2
        total += 8.0 * _shank_side_down_penalty(model, data, leg, frames)
    if len(leg_list) == 6:
        hs = {leg: foot_world(model, data, leg, frames)[0][2] for leg in leg_list}
        total += 80.0 * foot_height_uniformity_cost(hs, target_z)
    return total


def solve_tibia_for_flat(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg: int,
    coxa: float,
    femur: float,
    body_z: float,
    pose_base: Dict[str, float],
    frames: Dict[int, FootFrame],
    n_try: int = 40,
) -> float:
    """给定 coxa/femur，搜索 tibia 使脚底法向接近竖直向下。"""
    jname = f"leg{leg}_tibia_joint"
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    lo, hi = model.jnt_range[jid]

    def cost_t(t: float) -> float:
        pose = dict(pose_base)
        pose[f"leg{leg}_coxa_joint"] = coxa
        pose[f"leg{leg}_femur_joint"] = femur
        pose[jname] = t
        _set_pose(model, data, pose, body_z)
        mujoco.mj_forward(model, data)
        p, n = foot_world(model, data, leg, frames)
        return (
            80.0 * _tilt_cost(n)
            + 30.0 * (p[2] - FOOT_CONTACT_Z) ** 2
            + 8.0 * _shank_side_down_penalty(model, data, leg, frames)
        )

    ts = np.linspace(lo, hi, n_try)
    best_t, best_c = lo, cost_t(lo)
    for t in ts:
        c = cost_t(t)
        if c < best_c:
            best_c, best_t = c, t
    # 局部细化
    for t in np.linspace(max(lo, best_t - 0.15), min(hi, best_t + 0.15), 15):
        c = cost_t(t)
        if c < best_c:
            best_c, best_t = c, float(t)
    return best_t


def _adjust_leg_foot_height(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg: int,
    pose: Dict[str, float],
    body_z: float,
    frames: Dict[int, FootFrame],
    target_z: float = FOOT_CONTACT_Z,
    allow_coxa: bool = False,
    coarse: bool = False,
) -> Dict[str, float]:
    """搜索关节角使该腿足底落在 target_z，且底面平面贴地。"""
    out = dict(pose)
    jid_c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_coxa_joint")
    jid_f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_femur_joint")
    jid_t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_tibia_joint")
    cr, fr, tr = model.jnt_range[jid_c], model.jnt_range[jid_f], model.jnt_range[jid_t]
    c0 = out[f"leg{leg}_coxa_joint"]
    f0 = out[f"leg{leg}_femur_joint"]
    t0 = out[f"leg{leg}_tibia_joint"]
    best = (1e9, c0, f0, t0)

    def leg_err(c: float, f: float, t: float) -> float:
        trial = dict(out)
        trial[f"leg{leg}_coxa_joint"] = c
        trial[f"leg{leg}_femur_joint"] = f
        trial[f"leg{leg}_tibia_joint"] = t
        _set_pose(model, data, trial, body_z)
        mujoco.mj_forward(model, data)
        pad_z = foot_pad_bottom_z(model, data, leg)
        _, n = foot_world(model, data, leg, frames)
        return (
            220.0 * (pad_z - target_z) ** 2
            + 80.0 * _tilt_cost(n)
            + 6.0 * _shank_side_down_penalty(model, data, leg, frames)
            + 0.5 * (c - c0) ** 2
            + 0.2 * (f - f0) ** 2
            + 0.1 * (t - t0) ** 2
        )

    if coarse:
        n_c, n_f, n_t = (10 if allow_coxa else 1), 14, 16
        n_c2, n_f2, n_t2 = (5 if allow_coxa else 1), 7, 9
        c_span = 0.10
        f_span, t_span = 0.16, 0.20
    else:
        n_c, n_f, n_t = (18 if allow_coxa else 1), 26, 28
        n_c2, n_f2, n_t2 = (9 if allow_coxa else 1), 13, 15
        c_span = 0.15
        f_span, t_span = 0.22, 0.28

    if allow_coxa:
        c_vals = np.linspace(cr[0], cr[1], n_c)
    else:
        c_vals = [c0]
    for c in c_vals:
        for f in np.linspace(fr[0], fr[1], n_f):
            for t in np.linspace(tr[0], tr[1], n_t):
                e = leg_err(float(c), float(f), float(t))
                if e < best[0]:
                    best = (e, float(c), float(f), float(t))
    _, c1, f1, t1 = best
    c_rng = (
        np.linspace(max(cr[0], c1 - c_span), min(cr[1], c1 + c_span), n_c2)
        if allow_coxa
        else [c1]
    )
    for c in c_rng:
        for f in np.linspace(max(fr[0], f1 - f_span), min(fr[1], f1 + f_span), n_f2):
            for t in np.linspace(max(tr[0], t1 - t_span), min(tr[1], t1 + t_span), n_t2):
                e = leg_err(float(c), float(f), float(t))
                if e < best[0]:
                    best = (e, float(c), float(f), float(t))
    out[f"leg{leg}_coxa_joint"] = best[1]
    out[f"leg{leg}_femur_joint"] = best[2]
    out[f"leg{leg}_tibia_joint"] = best[3]
    return out


def _optimize_body_z_for_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    frames: Dict[int, FootFrame],
    target_z: float = FOOT_CONTACT_Z,
    min_body_z: float = MIN_BODY_HEIGHT,
) -> float:
    """粗搜机身高度：在不低于 min_body_z 的前提下平衡足底高度一致性。"""
    max_body_z = min_body_z + 0.012
    best = (1e9, min_body_z)
    for bz in np.linspace(min_body_z, max_body_z, 7):
        trial = dict(pose)
        for leg in range(1, 7):
            c = trial[f"leg{leg}_coxa_joint"]
            f = trial[f"leg{leg}_femur_joint"]
            trial[f"leg{leg}_tibia_joint"] = solve_tibia_for_flat(
                model, data, leg, c, f, float(bz), trial, frames, n_try=24
            )
        hs = foot_heights_world(model, data, trial, float(bz), frames)
        spread = max(hs.values()) - min(hs.values())
        mean_h = float(np.mean(list(hs.values())))
        err = foot_height_uniformity_cost(hs, target_z) + 200.0 * spread**2
        err += 60.0 * max(0.0, mean_h - 0.030) ** 2
        err -= 2.0 * float(bz)
        if err < best[0]:
            best = (err, float(bz))
    return max(best[1], min_body_z)


def calibrate_suspended_stand(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    frames: Dict[int, FootFrame],
    target_pad_z: float = FOOT_CONTACT_Z,
) -> Tuple[Dict[str, float], float]:
    """
    标定可支撑站立：六足足底碰撞体贴地，机身悬空。
    在可达范围内尽量抬高 body_z，使主体下沿接近 TARGET_BASE_CLEARANCE。
    """
    best: Tuple[float, float, Dict[str, float], float] | None = None
    for bz in np.linspace(0.058, 0.078, 7):
        trial = dict(pose)
        for leg in range(1, 7):
            trial = _adjust_leg_foot_height(
                model,
                data,
                leg,
                trial,
                float(bz),
                frames,
                target_pad_z,
                allow_coxa=True,
                coarse=True,
            )
        for leg in range(1, 7):
            trial = _adjust_leg_foot_height(
                model,
                data,
                leg,
                trial,
                float(bz),
                frames,
                target_pad_z,
                allow_coxa=True,
                coarse=False,
            )
        _set_pose(model, data, trial, float(bz))
        mujoco.mj_forward(model, data)
        pads = foot_pad_heights(model, data)
        base_clr = body_bottom_clearance(model, data, float(bz))
        spread = max(pads.values()) - min(pads.values())
        max_pad = max(pads.values())
        if max_pad > 0.018:
            continue
        score = (
            120.0 * spread**2
            + 40.0 * (base_clr - TARGET_BASE_CLEARANCE) ** 2
            + 30.0 * max(0.0, target_pad_z - min(pads.values())) ** 2
            - 3.0 * float(bz)
        )
        if best is None or score < best[0]:
            best = (score, float(bz), trial, base_clr)
    if best is None:
        return equalize_stand_foot_heights(
            model, data, pose, MIN_BODY_HEIGHT, frames, fix_body_z=True
        )
    return best[2], best[1]


def equalize_stand_foot_heights(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    body_z: float,
    frames: Dict[int, FootFrame],
    target_z: float = FOOT_CONTACT_Z,
    rounds: int = 3,
    fix_body_z: bool = False,
    min_body_z: float = MIN_BODY_HEIGHT,
) -> Tuple[Dict[str, float], float]:
    """六腿足底落到同一接触平面；必要时调整 coxa 与机身高度。"""
    out = dict(pose)
    if fix_body_z:
        body_z = max(float(body_z), min_body_z)
    else:
        body_z = _optimize_body_z_for_contact(
            model, data, out, frames, target_z, min_body_z=min_body_z
        )
    for rnd in range(rounds):
        hs = foot_heights_world(model, data, out, body_z, frames)
        target_z = float(np.mean(list(hs.values())))
        for leg in range(1, 7):
            need_coxa = abs(hs[leg] - target_z) > 0.004
            out = _adjust_leg_foot_height(
                model,
                data,
                leg,
                out,
                body_z,
                frames,
                target_z,
                allow_coxa=need_coxa,
                coarse=(rnd == 0),
            )
    return out, body_z


def _refine_topview_symmetric_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    body_z: float,
    frames: Dict[int, FootFrame],
    hex_r: float,
    foot_z: float,
    rounds: int = 4,
) -> Dict[str, float]:
    """多轮：对称 coxa（coxa_mir=-coxa_ref）→ 固定 coxa 优化 femur/tibia。"""
    from leg_symmetry import (
        MIRROR_PAIRS,
        feet_in_base,
        hex_foot_target_base,
        mirror_foot_in_base,
        symmetrize_coxa_pairs,
    )

    out = dict(pose)
    for _ in range(rounds):
        out = symmetrize_coxa_pairs(out)
        for ref_leg, mir_leg in MIRROR_PAIRS:
            c_ref = out[f"leg{ref_leg}_coxa_joint"]
            c_mir = -c_ref
            tgt_ref = hex_foot_target_base(ref_leg, hex_r, foot_z)
            f_ref, t_ref = _calibrate_leg_ft_fixed_coxa(
                model, data, ref_leg, c_ref, tgt_ref, frames, body_z, out
            )
            out[f"leg{ref_leg}_coxa_joint"] = c_ref
            out[f"leg{ref_leg}_femur_joint"] = f_ref
            out[f"leg{ref_leg}_tibia_joint"] = t_ref
            _set_pose(model, data, out, body_z)
            mujoco.mj_forward(model, data)
            tgt_mir = mirror_foot_in_base(
                feet_in_base(model, data, out, body_z, foot_world, frames)[ref_leg]
            )
            f_mir, t_mir = _calibrate_leg_ft_fixed_coxa(
                model, data, mir_leg, c_mir, tgt_mir, frames, body_z, out
            )
            out[f"leg{mir_leg}_coxa_joint"] = c_mir
            out[f"leg{mir_leg}_femur_joint"] = f_mir
            out[f"leg{mir_leg}_tibia_joint"] = t_mir
        for leg in range(1, 7):
            cj = out[f"leg{leg}_coxa_joint"]
            fj = out[f"leg{leg}_femur_joint"]
            out[f"leg{leg}_tibia_joint"] = solve_tibia_for_flat(
                model, data, leg, cj, fj, body_z, out, frames, n_try=36
            )
    return out


def _calibrate_leg_ft_fixed_coxa(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg: int,
    coxa: float,
    target_base: np.ndarray,
    frames: Dict[int, FootFrame],
    body_z: float,
    pose_base: Dict[str, float],
    n_f: int = 26,
    n_t: int = 28,
) -> Tuple[float, float]:
    """固定 coxa，只搜索 femur/tibia 使足端接近 target_base 且脚底朝下。"""
    from leg_symmetry import feet_in_base, radial_angle_cost

    best = (1e9, 0.0, 0.0)
    jid_f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_femur_joint")
    jid_t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{leg}_tibia_joint")
    fr, tr = model.jnt_range[jid_f], model.jnt_range[jid_t]
    target = np.asarray(target_base, dtype=float)

    def leg_err(f: float, t: float) -> float:
        trial = dict(pose_base)
        trial[f"leg{leg}_coxa_joint"] = coxa
        trial[f"leg{leg}_femur_joint"] = f
        trial[f"leg{leg}_tibia_joint"] = t
        _set_pose(model, data, trial, body_z)
        mujoco.mj_forward(model, data)
        fb = feet_in_base(model, data, trial, body_z, foot_world, frames)[leg]
        _, n = foot_world(model, data, leg, frames)
        az = float(np.arctan2(fb[1], fb[0]))
        az_t = float(np.arctan2(target[1], target[0]))
        da = abs((az - az_t + np.pi) % (2 * np.pi) - np.pi)
        return (
            float(np.sum((fb - target) ** 2))
            + 60.0 * _tilt_cost(n)
            + 25.0 * radial_angle_cost(model, data, leg)
            + 40.0 * da**2
            + 8.0 * _shank_side_down_penalty(model, data, leg, frames)
        )

    for f in np.linspace(fr[0], fr[1], n_f):
        for t in np.linspace(tr[0], tr[1], n_t):
            e = leg_err(f, t)
            if e < best[0]:
                best = (e, f, t)
    _, f0, t0 = best
    for f in np.linspace(max(fr[0], f0 - 0.22), min(fr[1], f0 + 0.22), 13):
        for t in np.linspace(max(tr[0], t0 - 0.28), min(tr[1], t0 + 0.28), 15):
            e = leg_err(f, t)
            if e < best[0]:
                best = (e, f, t)
    return best[1], best[2]


def _optimize_mirror_pair_topview(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ref_leg: int,
    mir_leg: int,
    pose: Dict[str, float],
    body_z: float,
    frames: Dict[int, FootFrame],
    hex_r: float,
    foot_z: float,
) -> None:
    """
    一对镜像腿联合标定：强制 coxa_mir = -coxa_ref（俯视图左右对称），
    足端目标为参考腿六边形点及其镜像。
    """
    from leg_symmetry import (
        feet_in_base,
        hex_foot_target_base,
        mirror_foot_in_base,
        radial_angle_cost,
        symmetry_foot_error,
    )

    jid_c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg{ref_leg}_coxa_joint")
    cr = model.jnt_range[jid_c]
    tgt_ref = hex_foot_target_base(ref_leg, hex_r, foot_z)
    best = (1e9, float(pose[f"leg{ref_leg}_coxa_joint"]))

    for c_ref in np.linspace(cr[0], cr[1], 33):
        c_mir = -c_ref
        trial = dict(pose)
        trial[f"leg{ref_leg}_coxa_joint"] = c_ref
        trial[f"leg{mir_leg}_coxa_joint"] = c_mir
        f_ref, t_ref = _calibrate_leg_ft_fixed_coxa(
            model, data, ref_leg, c_ref, tgt_ref, frames, body_z, trial
        )
        trial[f"leg{ref_leg}_femur_joint"] = f_ref
        trial[f"leg{ref_leg}_tibia_joint"] = t_ref
        _set_pose(model, data, trial, body_z)
        mujoco.mj_forward(model, data)
        foot_ref = feet_in_base(model, data, trial, body_z, foot_world, frames)[ref_leg]
        tgt_mir = mirror_foot_in_base(foot_ref)
        f_mir, t_mir = _calibrate_leg_ft_fixed_coxa(
            model, data, mir_leg, c_mir, tgt_mir, frames, body_z, trial
        )
        trial[f"leg{mir_leg}_femur_joint"] = f_mir
        trial[f"leg{mir_leg}_tibia_joint"] = t_mir
        _set_pose(model, data, trial, body_z)
        mujoco.mj_forward(model, data)
        sym = symmetry_foot_error(
            model, data, trial, body_z, foot_world, frames, pairs=((ref_leg, mir_leg),)
        )
        flat = flat_cost(model, data, trial, body_z, frames, legs=(ref_leg, mir_leg))
        rad = radial_angle_cost(model, data, ref_leg) + radial_angle_cost(model, data, mir_leg)
        cost = sym + 0.5 * flat + 40.0 * rad
        if cost < best[0]:
            best = (cost, c_ref)
            pose[f"leg{ref_leg}_coxa_joint"] = c_ref
            pose[f"leg{mir_leg}_coxa_joint"] = c_mir
            pose[f"leg{ref_leg}_femur_joint"] = f_ref
            pose[f"leg{ref_leg}_tibia_joint"] = t_ref
            pose[f"leg{mir_leg}_femur_joint"] = f_mir
            pose[f"leg{mir_leg}_tibia_joint"] = t_mir

    c0 = best[1]
    for c_ref in np.linspace(max(cr[0], c0 - 0.12), min(cr[1], c0 + 0.12), 17):
        c_mir = -c_ref
        trial = dict(pose)
        trial[f"leg{ref_leg}_coxa_joint"] = c_ref
        trial[f"leg{mir_leg}_coxa_joint"] = c_mir
        f_ref, t_ref = _calibrate_leg_ft_fixed_coxa(
            model, data, ref_leg, c_ref, tgt_ref, frames, body_z, trial, n_f=20, n_t=22
        )
        trial[f"leg{ref_leg}_femur_joint"] = f_ref
        trial[f"leg{ref_leg}_tibia_joint"] = t_ref
        _set_pose(model, data, trial, body_z)
        mujoco.mj_forward(model, data)
        tgt_mir = mirror_foot_in_base(
            feet_in_base(model, data, trial, body_z, foot_world, frames)[ref_leg]
        )
        f_mir, t_mir = _calibrate_leg_ft_fixed_coxa(
            model, data, mir_leg, c_mir, tgt_mir, frames, body_z, trial, n_f=20, n_t=22
        )
        trial[f"leg{mir_leg}_femur_joint"] = f_mir
        trial[f"leg{mir_leg}_tibia_joint"] = t_mir
        _set_pose(model, data, trial, body_z)
        mujoco.mj_forward(model, data)
        sym = symmetry_foot_error(
            model, data, trial, body_z, foot_world, frames, pairs=((ref_leg, mir_leg),)
        )
        flat = flat_cost(model, data, trial, body_z, frames, legs=(ref_leg, mir_leg))
        rad = radial_angle_cost(model, data, ref_leg) + radial_angle_cost(model, data, mir_leg)
        cost = sym + 0.5 * flat + 40.0 * rad
        if cost < best[0]:
            best = (cost, c_ref)
            pose[f"leg{ref_leg}_coxa_joint"] = c_ref
            pose[f"leg{mir_leg}_coxa_joint"] = c_mir
            pose[f"leg{ref_leg}_femur_joint"] = f_ref
            pose[f"leg{ref_leg}_tibia_joint"] = t_ref
            pose[f"leg{mir_leg}_femur_joint"] = f_mir
            pose[f"leg{mir_leg}_tibia_joint"] = t_mir


def calibrate_symmetric_hex_stand(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frames: Dict[int, FootFrame],
) -> Tuple[Dict[str, float], float]:
    """左右对称六足站立；末轮固定 coxa_mir=-coxa_ref 并只调左腿 femur/tibia。"""
    from leg_symmetry import (
        MIRROR_PAIRS,
        REFERENCE_LEGS,
        calibrate_mirror_leg,
        feet_in_base,
        hex_foot_target_base,
        mirror_foot_in_base,
        symmetrize_coxa_pairs,
    )

    zero_pose = {
        f"leg{l}_{j}_joint": 0.0
        for l in range(1, 7)
        for j in ("coxa", "femur", "tibia")
    }
    body_z = 0.14
    _set_pose(model, data, zero_pose, body_z)
    mujoco.mj_forward(model, data)
    feet0 = feet_in_base(model, data, zero_pose, body_z, foot_world, frames)
    hex_r = float(np.mean([np.hypot(feet0[l][0], feet0[l][1]) for l in range(1, 7)]))
    foot_z = float(np.mean([feet0[l][2] for l in range(1, 7)]))

    pose: Dict[str, float] = dict(zero_pose)
    for leg in REFERENCE_LEGS:
        tgt = hex_foot_target_base(leg, hex_r, foot_z)
        c, f, t = _calibrate_leg_to_foot_target(
            model, data, leg, tgt, frames, body_z, seed=(0.0, 0.0, 0.0)
        )
        pose[f"leg{leg}_coxa_joint"] = c
        pose[f"leg{leg}_femur_joint"] = f
        pose[f"leg{leg}_tibia_joint"] = t

    for _ in range(4):
        for ref_leg, mir_leg in MIRROR_PAIRS:
            c, f, t = calibrate_mirror_leg(
                model, data, ref_leg, mir_leg, pose, body_z, foot_world, frames, _tilt_cost
            )
            pose[f"leg{mir_leg}_coxa_joint"] = c
            pose[f"leg{mir_leg}_femur_joint"] = f
            pose[f"leg{mir_leg}_tibia_joint"] = t
        for leg in range(1, 7):
            c = pose[f"leg{leg}_coxa_joint"]
            f = pose[f"leg{leg}_femur_joint"]
            pose[f"leg{leg}_tibia_joint"] = solve_tibia_for_flat(
                model, data, leg, c, f, body_z, pose, frames, n_try=35
            )

    body_z = max(body_z, MIN_BODY_HEIGHT)
    best_c = _sym_hex_total(model, data, pose, body_z, frames)
    for bz in np.linspace(MIN_BODY_HEIGHT, MIN_BODY_HEIGHT + 0.012, 5):
        c = _sym_hex_total(model, data, pose, float(bz), frames)
        if c < best_c:
            best_c, body_z = c, float(bz)
            for leg in range(1, 7):
                cj = pose[f"leg{leg}_coxa_joint"]
                fj = pose[f"leg{leg}_femur_joint"]
                pose[f"leg{leg}_tibia_joint"] = solve_tibia_for_flat(
                    model, data, leg, cj, fj, body_z, pose, frames, n_try=28
                )
            for ref_leg, mir_leg in MIRROR_PAIRS:
                c, f, t = calibrate_mirror_leg(
                    model,
                    data,
                    ref_leg,
                    mir_leg,
                    pose,
                    body_z,
                    foot_world,
                    frames,
                    _tilt_cost,
                )
                pose[f"leg{mir_leg}_coxa_joint"] = c
                pose[f"leg{mir_leg}_femur_joint"] = f
                pose[f"leg{mir_leg}_tibia_joint"] = t

    pose = symmetrize_coxa_pairs(pose)
    for leg in range(1, 7):
        cj = pose[f"leg{leg}_coxa_joint"]
        fj = pose[f"leg{leg}_femur_joint"]
        pose[f"leg{leg}_tibia_joint"] = solve_tibia_for_flat(
            model, data, leg, cj, fj, body_z, pose, frames, n_try=35
        )
    pose, body_z = calibrate_suspended_stand(model, data, pose, frames)
    return pose, body_z


def symmetry_foot_error_hex(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Dict[str, float],
    body_z: float,
    frames,
) -> float:
    from leg_symmetry import MIRROR_PAIRS, symmetry_foot_error

    return symmetry_foot_error(
        model, data, pose, body_z, foot_world, frames, pairs=MIRROR_PAIRS
    )


def _sym_hex_total(
    model, data, pose, body_z, frames,
) -> float:
    from leg_symmetry import (
        feet_in_base,
        hexagon_uniformity_cost,
        topview_azimuth_symmetry_cost,
    )

    _set_pose(model, data, pose, body_z)
    mujoco.mj_forward(model, data)
    heights = foot_heights_world(model, data, pose, body_z, frames)
    c = flat_cost(model, data, pose, body_z, frames)
    c += 100.0 * foot_height_uniformity_cost(heights)
    c += 25.0 * hexagon_uniformity_cost(
        feet_in_base(model, data, pose, body_z, foot_world, frames)
    )
    c += 120.0 * symmetry_foot_error_hex(model, data, pose, body_z, frames)
    c += 80.0 * topview_azimuth_symmetry_cost(model, data, foot_world, frames)
    return c


def calibrate_flat_stand(
    model: mujoco.MjModel | None = None,
    iterations: int = 2,
    symmetric: bool = True,
) -> Tuple[Dict[str, float], float]:
    """标定六足站立角：脚底平行地面；symmetric 时强制左右对称与六边形足端分布。"""
    from leg_symmetry import MIRROR_PAIRS, calibrate_mirror_leg, symmetrize_all_pairs

    if model is None:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    pose_seed, bz_seed = load_stand_pose() or ({}, 0.14)
    if not pose_seed:
        pose_seed = {
            f"leg{l}_{j}_joint": 0.0
            for l in range(1, 7)
            for j in ("coxa", "femur", "tibia")
        }
    frames = load_foot_frames(model, pose_seed, bz_seed)

    if symmetric:
        return calibrate_symmetric_hex_stand(model, data, frames)

    pose = {
        f"leg{l}_{j}_joint": 0.0
        for l in range(1, 7)
        for j in ("coxa", "femur", "tibia")
    }
    for leg in range(1, 7):
        c, f, t = _calibrate_leg_solo(model, data, leg, frames)
        pose[f"leg{leg}_coxa_joint"] = c
        pose[f"leg{leg}_femur_joint"] = f
        pose[f"leg{leg}_tibia_joint"] = t
    body_z = 0.14
    for bz in np.linspace(0.10, 0.30, 21):
        if flat_cost(model, data, pose, bz, frames) < flat_cost(
            model, data, pose, body_z, frames
        ):
            body_z = float(bz)
    return pose, body_z


def apply_flat_feet(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets: Dict[str, float],
    body_z: float,
    frames: Dict[int, FootFrame] | None = None,
) -> Dict[str, float]:
    """在步态目标角上，只调整 tibia 使摆动/支撑时脚底仍平行地面。"""
    if frames is None:
        frames = load_foot_frames()
    out = dict(targets)
    for leg in range(1, 7):
        c = out.get(f"leg{leg}_coxa_joint", 0.0)
        f = out.get(f"leg{leg}_femur_joint", 0.0)
        out[f"leg{leg}_tibia_joint"] = solve_tibia_for_flat(
            model, data, leg, c, f, body_z, out, frames, n_try=30
        )
    return out


def _prone_pose_seed(stand_pose: Dict[str, float]) -> Dict[str, float]:
    """从站立角生成趴地搜索初值（偶轴撑起反向）。"""
    from enable_state import make_prone_pose_analytic

    return make_prone_pose_analytic(stand_pose)


def calibrate_prone_pose(
    model: mujoco.MjModel,
    stand_pose: Dict[str, float] | None = None,
    body_z: float | None = None,
) -> Tuple[Dict[str, float], float]:
    """
    标定失能趴地姿态：六足足底碰撞体平面贴地（使能前零力矩状态）。
    """
    data = mujoco.MjData(model)
    if stand_pose is None:
        loaded = load_stand_pose()
        if loaded is not None:
            stand_pose = loaded[0]
        else:
            stand_pose = {
                f"leg{l}_{j}_joint": 0.0
                for l in range(1, 7)
                for j in ("coxa", "femur", "tibia")
            }
    seed = _prone_pose_seed(stand_pose)
    bz = float(MIN_BODY_HEIGHT if body_z is None else body_z)
    frames = load_foot_frames(model, seed, bz)
    out = dict(seed)
    target_z = FOOT_CONTACT_Z
    for rnd in range(5):
        for leg in range(1, 7):
            out = _adjust_leg_foot_height(
                model,
                data,
                leg,
                out,
                bz,
                frames,
                target_z,
                allow_coxa=True,
                coarse=(rnd < 2),
            )
    out, bz = equalize_stand_foot_heights(
        model,
        data,
        out,
        bz,
        frames,
        target_z=target_z,
        rounds=4,
        fix_body_z=True,
        min_body_z=MIN_BODY_HEIGHT,
    )
    return out, bz


def save_prone_pose(pose: Dict[str, float], body_z: float) -> str:
    os.makedirs(os.path.dirname(PRONE_POSE_PATH), exist_ok=True)
    payload = {"body_height": body_z, "joints": pose}
    with open(PRONE_POSE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return PRONE_POSE_PATH


def load_prone_pose() -> Tuple[Dict[str, float], float] | None:
    if not os.path.isfile(PRONE_POSE_PATH):
        return None
    with open(PRONE_POSE_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return d["joints"], float(d["body_height"])


def nominal_prone_pose(
    model: mujoco.MjModel | None = None,
) -> Tuple[Dict[str, float], float]:
    """加载或标定趴地姿态（足底平面贴地）。"""
    loaded = load_prone_pose()
    if loaded is not None:
        return loaded
    if model is None:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    stand_loaded = load_stand_pose()
    stand = stand_loaded[0] if stand_loaded else None
    return calibrate_prone_pose(model, stand_pose=stand)


def save_stand_pose(pose: Dict[str, float], body_z: float) -> str:
    os.makedirs(os.path.dirname(STAND_POSE_PATH), exist_ok=True)
    payload = {"body_height": body_z, "joints": pose}
    with open(STAND_POSE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return STAND_POSE_PATH


def load_stand_pose() -> Tuple[Dict[str, float], float] | None:
    if not os.path.isfile(STAND_POSE_PATH):
        return None
    with open(STAND_POSE_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return d["joints"], float(d["body_height"])


def body_bottom_clearance(
    model: mujoco.MjModel, data: mujoco.MjData, body_z: float
) -> float:
    """主体下板最低点相对地面的净空（m）。"""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    data.qpos[2] = body_z
    mujoco.mj_forward(model, data)
    mesh_path = os.path.join(MESH_DIR, "base_link.STL")
    if os.path.isfile(mesh_path):
        import trimesh

        m = trimesh.load(mesh_path, force="mesh")
        R = data.xmat[bid].reshape(3, 3)
        vw = (R @ m.vertices.T).T + data.xpos[bid]
        return float(vw[:, 2].min())
    return float(data.xpos[bid][2])


def report_flatness(model: mujoco.MjModel, pose: Dict[str, float], body_z: float) -> None:
    data = mujoco.MjData(model)
    frames = load_foot_frames(model, pose, body_z)
    _set_pose(model, data, pose, body_z)
    mujoco.mj_forward(model, data)
    heights = []
    for leg in range(1, 7):
        p, n = foot_world(model, data, leg, frames)
        tilt = np.degrees(np.arccos(np.clip(-n[2], -1.0, 1.0)))
        heights.append(p[2])
        print(f"  leg{leg}: foot_z={p[2]:.4f}m  tilt={tilt:.2f}°")
    print(
        f"  高度范围: {min(heights)*1000:.1f}~{max(heights)*1000:.1f} mm "
        f"(目标 {FOOT_CONTACT_Z*1000:.1f} mm, 极差 {max(heights)-min(heights):.4f} m)"
    )
    pads = foot_pad_heights(model, data)
    print(
        f"  主体下沿净空: {body_bottom_clearance(model, data, body_z)*1000:.1f} mm "
        f"(body_height={body_z*1000:.1f} mm)"
    )
    print(
        f"  足底碰撞体高度: {min(pads.values())*1000:.1f}~{max(pads.values())*1000:.1f} mm"
    )


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    print("标定足底平行站立姿态...")
    pose, bz = calibrate_flat_stand(model)
    path = save_stand_pose(pose, bz)
    print(f"已保存: {path}")
    print(f"body_height={bz:.4f}")
    print("足底倾角:")
    report_flatness(model, pose, bz)
    print("左右对称 (足端镜像误差):")
    from leg_symmetry import (
        MIRROR_PAIRS,
        feet_in_base,
        femur_outward_azimuth,
        hexagon_uniformity_cost,
        report_symmetry,
    )

    report_symmetry(model, pose, bz, foot_world, load_foot_frames(model, pose, bz))
    data = mujoco.MjData(model)
    _set_pose(model, data, pose, bz)
    mujoco.mj_forward(model, data)
    print("俯视图大腿外展角 (deg，应接近髋座方位):")
    for leg in range(1, 7):
        az = np.degrees(femur_outward_azimuth(model, data, leg))
        print(f"  leg{leg}: {az:6.1f}°")
    print("俯视图 coxa 角 (deg，镜像对应相反):")
    for ref, mir in MIRROR_PAIRS:
        cr = np.degrees(pose[f"leg{ref}_coxa_joint"])
        cm = np.degrees(pose[f"leg{mir}_coxa_joint"])
        print(f"  leg{ref}/leg{mir}: {cr:6.1f} / {cm:6.1f}")
    feet = feet_in_base(
        model, mujoco.MjData(model), pose, bz, foot_world, load_foot_frames(model, pose, bz)
    )
    radii = [np.hypot(feet[l][0], feet[l][1]) for l in range(1, 7)]
    print(f"  六边形半径: mean={np.mean(radii):.4f} std={np.std(radii):.4f} m")
    print(f"  六边形均匀度 cost={hexagon_uniformity_cost(feet):.6f}")


if __name__ == "__main__":
    main()
