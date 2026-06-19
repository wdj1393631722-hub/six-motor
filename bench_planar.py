#!/usr/bin/env python3
"""平面三角步态 headless 基准：测前进速度、横向漂移、航向偏差。

用法:
  python3 bench_planar.py                      # 跑当前默认参数
  python3 bench_planar.py cycle_time=0.5 ...   # 覆盖任意 PlanarTripodGait 入参
可附加: dur=8 (仿真秒) max_v=0.06 (前进指令)
"""
from __future__ import annotations

import json
import math
import os
import sys

import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml")
STAND_PATH = os.path.join(SCRIPT_DIR, "generated", "stand_pose_flat.json")
sys.path.insert(0, SCRIPT_DIR)

from planar_tripod_gait import PlanarTripodGait  # noqa: E402

WALK_KP = 200.0
WALK_KV = 15.0
YAW_HOLD_KP = 2.5
YAW_HOLD_MAX = 0.18


def load_stand():
    with open(STAND_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return d["joints"], float(d["body_height"])


def set_gains(model, kp, kv):
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kv


def actuator_id(model, jname):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jname}_act")


def joint_qadr(model, jname):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return model.jnt_qposadr[jid]


def apply_ctrl(model, data, targets):
    for jname, angle in targets.items():
        aid = actuator_id(model, jname)
        if aid >= 0:
            data.ctrl[aid] = float(angle)


def body_yaw(data):
    qw, qx, qy, qz = data.qpos[3:7]
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def reset_standing(model, data, stand_pose, body_z):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, float(body_z)]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for jname, angle in stand_pose.items():
        data.qpos[joint_qadr(model, jname)] = float(angle)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    apply_ctrl(model, data, stand_pose)
    for _ in range(int(0.3 / model.opt.timestep)):
        mujoco.mj_step(model, data)


def main():
    overrides = {}
    dur = 8.0
    max_v = 0.85
    lat_kp = 0.0  # 横向位置反馈增益（0=关闭，复现开环漂移）
    for a in sys.argv[1:]:
        k, _, v = a.partition("=")
        if k == "dur":
            dur = float(v)
        elif k == "max_v":
            max_v = float(v)
        elif k == "lat_kp":
            lat_kp = float(v)
        elif k == "gait_mode":
            overrides[k] = v
        else:
            overrides[k] = float(v)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    stand_pose, body_z = load_stand()

    kwargs = dict(
        gait_mode="linear", cycle_time=0.45, lift_height=0.090,
        height_comp_m=0.055, max_stride=0.28, verbose=False,
    )
    kwargs.update(overrides)
    gait = PlanarTripodGait(model=model, stand_pose=stand_pose,
                            body_height=body_z, **kwargs)
    set_gains(model, WALK_KP, WALK_KV)
    reset_standing(model, data, stand_pose, body_z)

    yaw_ref = body_yaw(data)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    z_start = float(data.qpos[2])

    n = int(dur / model.opt.timestep)
    sim_dt = model.opt.timestep
    max_drift = 0.0
    max_yaw = 0.0
    zsum = 0.0
    fell = False
    LAT_MAX = 0.04
    for i in range(n):
        err = wrap_pi(yaw_ref - body_yaw(data))
        omega_cmd = max(-YAW_HOLD_MAX, min(YAW_HOLD_MAX, YAW_HOLD_KP * err))
        # 横向位置保持：body 系 +X 修正量（航向已锁定，world-X≈body-X）
        x_err = float(data.qpos[0]) - x0
        vx_cmd = max(-LAT_MAX, min(LAT_MAX, -lat_kp * x_err))
        targets = gait.step(sim_dt, vx=vx_cmd, vy=max_v, omega=omega_cmd)
        apply_ctrl(model, data, targets)
        mujoco.mj_step(model, data)
        max_drift = max(max_drift, abs(float(data.qpos[0]) - x0))
        max_yaw = max(max_yaw, abs(wrap_pi(body_yaw(data) - yaw_ref)))
        zsum += float(data.qpos[2])
        if float(data.qpos[2]) < 0.5 * z_start:
            fell = True

    dx = float(data.qpos[0]) - x0
    dy = float(data.qpos[1]) - y0
    speed = dy / dur
    print(f"参数: cycle={gait.cycle_time:.3f} stride={gait.stride_gain:.1f} "
          f"swing={gait.max_coxa_swing:.2f} lift={gait.lift_height*1000:.0f}mm "
          f"comp={gait.height_comp_m*1000:.0f}mm max_v={max_v:.3f}")
    print(f"  前进 dy = {dy*1000:7.1f} mm  速度 = {speed*1000:6.1f} mm/s")
    print(f"  横向 dx = {dx*1000:7.1f} mm  峰值漂移 = {max_drift*1000:5.1f} mm")
    print(f"  航向偏差峰值 = {math.degrees(max_yaw):5.2f}°  平均机身高 = {zsum/n*1000:.1f}mm"
          f"  {'⚠跌倒' if fell else 'OK'}")


if __name__ == "__main__":
    main()
