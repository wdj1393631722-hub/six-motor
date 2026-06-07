#!/usr/bin/env python3
"""仿真环境部署前自检：18 关节限位、增益、1:1 映射、步态目标。"""
from __future__ import annotations

import math
import sys

import mujoco

SCRIPT_DIR = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from enable_state import LOCOMOTION_KP, LOCOMOTION_KV, STAND_KP, STAND_KV
from foot_kinematics import load_prone_pose, load_stand_pose
from gait import create_forward_tripod_gait
from joint_tripod_gait import (
    BODY_FORWARD_AXIS,
    LIFT_UD_DEG,
    PUSH_FWD_DEG,
    STANCE_REAR_DEG,
    SWING_FWD_DEG,
)
from robot_limits import joint_deltas_deg, joint_limits_deg, max_abs_delta_deg

MODEL = __import__("os").path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml")


def main() -> int:
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    issues: list[str] = []
    warns: list[str] = []

    stand_loaded = load_stand_pose()
    prone_loaded = load_prone_pose()
    if not stand_loaded:
        issues.append("缺少 generated/stand_pose_flat.json")
        return 1
    stand, stand_bz = stand_loaded
    prone, _ = prone_loaded or ({}, 0.065)
    limits = joint_limits_deg(model)

    for label, pose in (("stand", stand), ("prone", prone)):
        for jn, ang in pose.items():
            lo, hi = model.jnt_range[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            ]
            if not lo - 1e-6 <= ang <= hi + 1e-6:
                issues.append(
                    f"{label} {jn}={math.degrees(ang):.1f}° 超范围 "
                    f"[{math.degrees(lo):.1f},{math.degrees(hi):.1f}]"
                )

    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "leg1_coxa_joint_act")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "leg1_coxa_joint")
    clo, chi = model.actuator_ctrlrange[aid]
    jlo, jhi = model.jnt_range[jid]
    if abs(clo - jlo) > 0.01 or abs(chi - jhi) > 0.01:
        warns.append(
            f"执行器 ctrlrange [{clo:.3f},{chi:.3f}] 与关节 range 不一致，"
            "建议重新 build_real_mjcf"
        )

    if int(model.nu) != 18:
        issues.append(f"执行器数量 nu={model.nu}，实机应为 18")

    gait = create_forward_tripod_gait(model=model)
    dt = 0.02
    peak_delta = 0.0
    peak_joint = ""
    for _ in range(int(gait.p.cycle_time / dt) + 8):
        targets = gait.step(dt, 0.02, 0, 0, sim_data=data)
        for jn, val in targets.items():
            lo, hi = model.jnt_range[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            ]
            if not lo - 1e-6 <= val <= hi + 1e-6:
                issues.append(
                    f"步态目标 {jn}={math.degrees(val):.1f}° 超关节限位"
                )
        jn, d = max_abs_delta_deg(targets, stand)
        if d > peak_delta:
            peak_delta, peak_joint = d, jn

    gait.reset()
    min_margin: dict[str, float] = {}
    for _ in range(int(gait.p.cycle_time / dt) + 8):
        targets = gait.step(dt, 0.02, 0, 0, sim_data=data)
        for jn, val in targets.items():
            lo_deg, hi_deg = limits[jn]
            abs_deg = math.degrees(val)
            margin = min(abs_deg - lo_deg, hi_deg - abs_deg)
            min_margin[jn] = min(min_margin.get(jn, 1e9), margin)
    for jn, margin in sorted(min_margin.items()):
        lo_deg, hi_deg = limits[jn]
        stand_margin = min(
            math.degrees(stand[jn]) - lo_deg,
            hi_deg - math.degrees(stand[jn]),
        )
        if margin < 3.0 and margin < stand_margin - 0.5:
            warns.append(
                f"{jn} 步态最小余量 {margin:.1f}° "
                f"(range [{lo_deg:.1f},{hi_deg:.1f}]°)"
            )

    print("=== 仿真环境自检（实机 18 关节 1:1）===")
    print(f"模型: {MODEL}")
    print(f"关节/执行器: 18（leg1~6 × coxa/femur/tibia）")
    print(f"坐标系: +{BODY_FORWARD_AXIS} 前进 | 角度单位 rad")
    print(f"PD 增益: 站立 kp={STAND_KP} kv={STAND_KV} | 行走 kp={LOCOMOTION_KP} kv={LOCOMOTION_KV}")
    print(f"步态幅度(°): swing={SWING_FWD_DEG} lift={LIFT_UD_DEG} rear={STANCE_REAR_DEG} push={PUSH_FWD_DEG}")
    print(f"步态峰值增量: {peak_joint} Δ={peak_delta:.1f}°（相对站立）")
    print(f"力矩限幅: ±{model.actuator_forcerange[aid, 1]:.0f} Nm | Ki: 无（PD）")
    print()
    if issues:
        print(f"错误 {len(issues)}:")
        for x in issues[:12]:
            print("  -", x)
        if len(issues) > 12:
            print(f"  ... 另有 {len(issues)-12} 条")
    else:
        print("错误: 0")
    if warns:
        print(f"警告 {len(warns)}:")
        for x in warns[:8]:
            print("  -", x)
        if len(warns) > 8:
            print(f"  ... 另有 {len(warns)-8} 条")
    else:
        print("警告: 0")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
