#!/usr/bin/env python3
"""自动驱动机器人走圆 / S 形，并在地面实时标注机身轨迹。

用法:
  python3 trajectory_demo.py circle   # 走圆
  python3 trajectory_demo.py s        # 走 S 形
  python3 trajectory_demo.py both     # 先圆后 S（重置一次）

轨迹用 viewer.user_scn 叠加红色小球绘制（机身质心在地面的投影）。
"""
from __future__ import annotations

import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from run_planar_tripod import (
    MODEL_PATH, STAND_PATH, WALK_KP, WALK_KV,
    load_stand, set_gains, reset_standing, apply_ctrl,
    body_yaw, wrap_pi,
)
from planar_tripod_gait import PlanarTripodGait

FORWARD = 0.4        # 前进指令（实测约 0.33 m/s）
OMEGA = 1.8          # 转向指令幅度上限（圆形用作恒定转向；S 形用作 omega 限幅）
TRAIL_DT = 0.08      # 轨迹采样间隔 (s)
TRAIL_MAX = 1500     # 轨迹点上限
CIRCLE_T = 14.5      # 走一整圈用时
# S 形：在「空间」上跟踪一条骑在中轴线上的正弦曲线（而非随时间摆头）。
# 这样整条 S 锁死在 yaw0 方向的直线上，步态本身的微小偏航漂移会被横向纠偏拉回，
# 不会再把整条 S 越带越歪。波长/幅值按米计 → S 形状均匀、可预测。
S_WAVELEN = 1.6      # 一个完整左右摆动的空间波长 (m)，越大 S 拉得越长
S_LAT_AMP = 0.35     # 相对中轴线的横向摆幅 (m)，越大 S 甩得越宽
S_RAMP_LEN = 1.2     # 起步缓冲距离 (m)：横向摆幅在前 S_RAMP_LEN 内由 0 平滑升到满幅，
                     # 保证 S 一开始沿中轴线直走，再渐渐摆开（否则 s=0 处正弦斜率最大会立刻甩 ~54°）
S_YAW_KP = 3.0       # 航向闭环比例增益（omega = KP·航向误差，再限幅到 ±OMEGA）
S_CROSS_KP = 1.6     # 横向（cross-track）纠偏增益：把机身拉回正弦曲线上
S_DELTA_MAX = 1.0    # 目标航向相对中轴线的限幅 (rad)，防止纠偏过冲打转


def axis_frame(yaw0: float):
    """中轴线坐标系：û 沿前进方向，n̂ 为其左侧法向（横向）。

    yaw=ψ 时机身世界前进方向为 (-sinψ, cosψ)，故 û=(-sin yaw0, cos yaw0)，
    n̂ 为 û 逆时针转 90°，正的横向位移对应正的航向偏移 δ。
    """
    u = np.array([-np.sin(yaw0), np.cos(yaw0)])
    n = np.array([-np.cos(yaw0), -np.sin(yaw0)])
    return u, n


def schedule(pattern: str, p, yaw: float, p0, yaw0: float):
    """返回 (forward, omega, done)。

    p    当前机身水平位置 (x, y)；
    yaw  当前机身航向；
    p0   该段起始位置，即 S 形中轴线（直线）的起点；
    yaw0 该段起始航向，即中轴线方向。
    """
    if pattern == "circle":
        # 恒定前进 + 恒定转向 → 持续绕圈，永不结束
        return FORWARD, OMEGA, False
    if pattern == "s":
        u, n = axis_frame(yaw0)
        d = np.asarray(p, dtype=float) - np.asarray(p0, dtype=float)
        s = float(d @ u)          # 沿中轴线已前进的距离
        lat = float(d @ n)        # 当前相对中轴线的横向偏移
        k = 2.0 * np.pi / S_WAVELEN
        # 振幅包络：用 smoothstep 在前 S_RAMP_LEN 距离内由 0 平滑升到 1，
        # env(0)=0 且 env'(0)=0 → s=0 处横向偏移与切向斜率都为 0，起步沿中轴线直走。
        r = np.clip(s / S_RAMP_LEN, 0.0, 1.0)
        env = r * r * (3.0 - 2.0 * r)                     # smoothstep 包络
        env_d = 6.0 * r * (1.0 - r) / S_RAMP_LEN          # d(env)/ds
        sin_ks, cos_ks = np.sin(k * s), np.cos(k * s)
        lat_des = S_LAT_AMP * env * sin_ks                # 期望横向（缓冲后的正弦）
        slope = S_LAT_AMP * (env_d * sin_ks + env * k * cos_ks)  # d(lat)/ds → 曲线切向
        # 前馈：沿正弦切向的航向偏移；反馈：把横向误差纠回曲线
        delta = np.arctan(slope) - S_CROSS_KP * (lat - lat_des)
        delta = float(np.clip(delta, -S_DELTA_MAX, S_DELTA_MAX))
        yaw_target = yaw0 + delta
        omega = float(np.clip(S_YAW_KP * wrap_pi(yaw_target - yaw), -OMEGA, OMEGA))
        return FORWARD, omega, False
    return 0.0, 0.0, True


def draw_trail(scn, trail, rgba):
    n = min(len(trail), TRAIL_MAX)
    scn.ngeom = n
    for i in range(n):
        x, y = trail[i]
        mujoco.mjv_initGeom(
            scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.012, 0.0, 0.0]),
            pos=np.array([x, y, 0.006]),
            mat=np.eye(3).flatten(),
            rgba=np.array(rgba, dtype=float),
        )


def run_pattern(model, data, gait, stand_pose, body_z, viewer, pattern, rgba):
    gait.reset()
    reset_standing(model, data, stand_pose, body_z)
    yaw0 = body_yaw(data)   # 锁定中轴线方向（整条 S 围绕它对称延伸）
    p0 = (float(data.qpos[0]), float(data.qpos[1]))   # 锁定中轴线起点
    trail = []
    t0 = time.time()
    last_real = t0
    last_sample = -1.0
    sim_t = 0.0
    while viewer.is_running():
        now = time.time()
        dt = min(max(now - last_real, model.opt.timestep), 0.04)
        last_real = now
        sim_t += dt

        p = (float(data.qpos[0]), float(data.qpos[1]))
        forward, omega, done = schedule(pattern, p, body_yaw(data), p0, yaw0)
        if done:
            break
        targets = gait.step(dt, vx=0.0, vy=forward, omega=omega)
        apply_ctrl(model, data, targets)
        for _ in range(max(1, int(dt / model.opt.timestep))):
            mujoco.mj_step(model, data)

        # 采样机身水平位置（滚动保留最近 TRAIL_MAX 点）
        if sim_t - last_sample >= TRAIL_DT:
            last_sample = sim_t
            trail.append((float(data.qpos[0]), float(data.qpos[1])))
            if len(trail) > TRAIL_MAX:
                trail.pop(0)
        draw_trail(viewer.user_scn, trail, rgba)
        viewer.sync()
    return trail


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "circle"
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    stand_pose, body_z = load_stand()

    print(f"标定步态… 模式={pattern}", flush=True)
    gait = PlanarTripodGait(
        model=model, stand_pose=stand_pose, body_height=body_z,
        gait_mode="linear", cycle_time=0.45, lift_height=0.090,
        height_comp_m=0.055, max_stride=0.28,
    )
    set_gains(model, WALK_KP, WALK_KV)
    reset_standing(model, data, stand_pose, body_z)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.5, 0.0]
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -75      # 接近俯视，方便看轨迹
        viewer.cam.azimuth = 90
        if pattern == "both":
            run_pattern(model, data, gait, stand_pose, body_z, viewer,
                        "circle", [1.0, 0.2, 0.2, 1.0])
            run_pattern(model, data, gait, stand_pose, body_z, viewer,
                        "s", [0.2, 0.4, 1.0, 1.0])
        else:
            rgba = [1.0, 0.2, 0.2, 1.0] if pattern == "circle" else [0.2, 0.4, 1.0, 1.0]
            run_pattern(model, data, gait, stand_pose, body_z, viewer,
                        pattern, rgba)
        print("演示结束，保持窗口。按需关闭。", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
