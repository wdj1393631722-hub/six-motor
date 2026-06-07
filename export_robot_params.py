#!/usr/bin/env python3
"""导出关节/主体参数到 generated/robot_params_for_gait.json（步态设计用）。"""
from __future__ import annotations

import json
import math
import os
import sys

import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml")
OUT_PATH = os.path.join(SCRIPT_DIR, "generated", "robot_params_for_gait.json")

sys.path.insert(0, SCRIPT_DIR)
from enable_state import BODY_LIFT_EVEN_DEG, CRAWL_LIFT_UD_DEG, CRAWL_SWING_FB_DEG
from foot_kinematics import _set_pose, foot_world, load_foot_frames, load_prone_pose, load_stand_pose
from leg_symmetry import HIP_MOUNT_XY, LEG_AZIMUTH_DEG, MIRROR_PAIRS, feet_in_base
from joint_tripod_gait import (
    BODY_FORWARD_AXIS,
    CYCLE_TIME_S,
    LIFT_UD_DEG,
    PHASE_SLOWDOWN,
    PUSH_FWD_DEG,
    STANCE_REAR_DEG,
    SWING_FWD_DEG,
)
from robot_limits import (
    FEMUR_UD_RATIO,
    TIBIA_UD_RATIO,
    all_joint_names,
    joint_limits_deg,
)
from enable_state import LOCOMOTION_KP, LOCOMOTION_KV, STAND_KP, STAND_KV


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    def body_info(name: str) -> dict:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return {
            "name": name,
            "mass_kg": float(model.body_mass[bid]),
            "com_in_body_m": model.body_ipos[bid].tolist(),
            "inertia_diag": model.body_inertia[bid].tolist(),
            "pos_in_parent_m": model.body_pos[bid].tolist(),
            "quat_in_parent_wxyz": model.body_quat[bid].tolist(),
        }

    def joint_info(jname: str) -> dict:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        lo, hi = model.jnt_range[jid]
        dof = model.jnt_dofadr[jid]
        return {
            "name": jname,
            "type": ["free", "ball", "slide", "hinge"][model.jnt_type[jid]],
            "axis": model.jnt_axis[jid].tolist(),
            "range_rad": [float(lo), float(hi)],
            "range_deg": [round(math.degrees(lo), 2), round(math.degrees(hi), 2)],
            "damping": float(model.dof_damping[dof]),
            "armature": float(model.dof_armature[dof]),
        }

    def leg_chain(leg: int) -> dict:
        fem = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"leg{leg}_femur")
        tib = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"leg{leg}_tibia")
        cox = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"leg{leg}_coxa")
        return {
            "coxa_body_offset_m": model.body_pos[cox].tolist(),
            "coxa_to_femur_length_m": round(float(np.linalg.norm(model.body_pos[fem])), 5),
            "femur_to_tibia_length_m": round(float(np.linalg.norm(model.body_pos[tib])), 5),
            "femur_offset_m": model.body_pos[fem].tolist(),
            "tibia_offset_m": model.body_pos[tib].tolist(),
        }

    stand_pose, stand_bz = load_stand_pose() or ({}, 0.087)
    prone_pose, prone_bz = load_prone_pose() or ({}, 0.065)
    frames = load_foot_frames(model, stand_pose, stand_bz)
    _set_pose(model, data, stand_pose, stand_bz)
    mujoco.mj_forward(model, data)
    feet_base = {
        str(k): [round(float(x), 5) for x in v]
        for k, v in feet_in_base(model, data, stand_pose, stand_bz, foot_world, frames).items()
    }

    aid0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "leg1_coxa_joint_act")
    payload = {
        "model": MODEL_PATH,
        "coordinate_frame": {
            "base": "base_link",
            "x": "left_right",
            "y": "forward",
            "z": "up",
            "units": "SI (m, rad, kg)",
        },
        "body": {
            "base_link": body_info("base_link"),
            "total_mass_including_legs_kg": round(
                sum(model.body_mass[i] for i in range(model.nbody)), 4
            ),
            "stand_body_height_m": stand_bz,
            "prone_body_height_m": prone_bz,
            "min_body_height_m": 0.065,
        },
        "legs": {
            "count": 6,
            "dof_per_leg": 3,
            "joint_names": ["coxa", "femur", "tibia"],
            "tripod_A": [1, 3, 5],
            "tripod_B": [2, 4, 6],
            "mirror_pairs": list(MIRROR_PAIRS),
            "hip_mount_xy_m": {str(k): list(v) for k, v in HIP_MOUNT_XY.items()},
            "azimuth_deg": LEG_AZIMUTH_DEG,
            "motor_mapping_18dof": {
                "note": "实机 18 关节与仿真 1:1：同名、同单位 rad",
                "joints_per_leg": [
                    "leg{N}_coxa_joint",
                    "leg{N}_femur_joint",
                    "leg{N}_tibia_joint",
                ],
                "all_joint_names": all_joint_names(),
                "deploy": "ctrl[i] = stand[i] + delta[i]，直接下发各关节目标角",
            },
        },
        "per_leg": {
            str(leg): {
                "chain": leg_chain(leg),
                "joints": {
                    j: joint_info(f"leg{leg}_{j}_joint")
                    for j in ("coxa", "femur", "tibia")
                },
                "stand_foot_in_base_m": feet_base[str(leg)],
            }
            for leg in range(1, 7)
        },
        "actuator": {
            "type": "position",
            "control_law": "tau = kp*(ctrl-qpos) - kv*qvel  (无 Ki)",
            "stand_kp": STAND_KP,
            "stand_kv": STAND_KV,
            "locomotion_kp": LOCOMOTION_KP,
            "locomotion_kv": LOCOMOTION_KV,
            "kp": float(model.actuator_gainprm[aid0, 0]),
            "kv": abs(float(model.actuator_biasprm[aid0, 2])) if model.nu else 12.0,
            "forcerange_Nm": model.actuator_forcerange[aid0].tolist(),
            "count": int(model.nu),
        },
        "joint_limits_deg": joint_limits_deg(model),
        "sim_to_real_mapping": {
            "dof_sim": 18,
            "dof_real": 18,
            "mapping": "1:1 同名关节角（rad）",
            "joint_names": all_joint_names(),
            "gait_lift_split_note": (
                f"步态抬脚偏置在仿真链内分配 femur*{FEMUR_UD_RATIO}+tibia*{TIBIA_UD_RATIO}，"
                "实机直接控制 femur/tibia 两关节"
            ),
            "forward_axis": f"base_link +{BODY_FORWARD_AXIS}",
        },
        "calibrated_poses": {
            "stand": {"body_height_m": stand_bz, "joints_rad": stand_pose},
            "prone": {"body_height_m": prone_bz, "joints_rad": prone_pose},
        },
        "gait_design_reference": {
            "planner": "joint_tripod_gait",
            "cycle_s": CYCLE_TIME_S,
            "sim_slowdown": PHASE_SLOWDOWN,
            "effective_cycle_s": CYCLE_TIME_S,
            "swing_fb_deg": SWING_FWD_DEG,
            "lift_ud_deg": LIFT_UD_DEG,
            "stance_rear_deg": STANCE_REAR_DEG,
            "push_fb_deg": PUSH_FWD_DEG,
            "test40_reference": {
                "swing_fb_deg": CRAWL_SWING_FB_DEG,
                "lift_ud_deg": CRAWL_LIFT_UD_DEG,
                "enable_lift_ud_deg": BODY_LIFT_EVEN_DEG,
            },
            "sim_only_modules": [
                "body_stabilizer",
                "ctrl_smoother",
                "foot_stance_lock blend",
            ],
        },
        "physics": {
            "timestep_s": float(model.opt.timestep),
            "gravity_m_s2": model.opt.gravity.tolist(),
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"已导出: {OUT_PATH}")


if __name__ == "__main__":
    main()
