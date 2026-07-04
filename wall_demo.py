#!/usr/bin/env python3
"""六足机器人**磁吸爬墙**：机身竖直贴在垂直墙面上，靠足底磁铁对抗重力，
用三角步态 + 磁力‑步态联动向上/向下/侧向爬行。

  python3 build_wall_scene.py     # 先生成墙面场景（只需一次）
  python3 wall_demo.py            # 磁吸爬墙

操控（先点一下 MuJoCo 窗口再按键）：
  I 或 ↑   前进（沿墙向上爬）
  K 或 ↓   后退（向下）
  J 或 ←   左转（在墙面内转向）
  L 或 →   右转
  P        停止（所有腿通电吸牢，驻留墙面）
  B        重置到贴墙站立
  1-6      切换第 N 条腿磁力使能   M   全部磁力通/断
鼠标用于旋转/平移视角。直线上爬时自动锁"爬升方向 + 横向"沿竖直走直。

原理：
  1) 机身重新定向贴墙——base +Y(前进)→世界 +Z(沿墙向上)，base +Z(足外法向/抬脚轴)
     →世界 +Y(离墙向外)；重力(世界 -Z)变成沿墙"向下"的面内力。
  2) PlanarTripodGait 是纯机身系相对的，重定向后迈步/摆动核心无需改动即可工作。
  3) 磁力‑步态联动(magnets.follow_gait)：每时刻 3 条支撑腿通电吸墙托住机身，
     3 条摆动腿在抬脚正中段断电释放、落地前重新吸附——既托得住又迈得动。
     单腿吸附仍是 50kg 硬件规格；实测可稳定向上爬 >1.6m。
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import mujoco
import mujoco.viewer

from run_planar_tripod import (
    WALK_KP, WALK_KV,
    load_stand, set_gains, apply_ctrl, joint_qadr,
)
from planar_tripod_gait import PlanarTripodGait
from viewer_controls import VelocityCommand, make_key_handler
from leg_magnets import LegMagnets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WALL_MODEL = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_wall.xml")

# ---- 磁力与步态参数（爬墙专用，已调稳）----
MAGNET_KG = 50.0            # 单腿吸附保持力（硬件规格）
RELEASE = (0.15, 0.80)      # 摆动腿磁力释放窗口（摆动进度 u∈此区间才断电）
LIFT_HEIGHT = 0.05          # 抬脚高度（墙面上调小：足端别离墙太远、落地快、好重吸）
CYCLE_TIME = 0.50           # 步态周期
HEIGHT_COMP = 0.055         # 前馈：支撑足压向墙面（助吸附）
MAX_STRIDE = 0.28

# ---- 贴墙站立初始化 ----
STANDOFF_ADJ = 0.014        # 机身比 body_height 再靠墙 14mm，让足端初始压进墙面吸住
Z_START = 0.6               # 初始挂在墙面的离地高度

MAX_V = 0.6                 # 前进(爬升)指令
MAX_TURN = 1.5              # 转向指令

# ---- 竖直走直闭环（墙面系：向上=世界 +Z，横向=世界 +X）----
WALL_YAW_KP = 2.5           # 爬升方向(航向)保持增益
WALL_YAW_MAX = 0.18
WALL_LAT_KP = 2.5           # 横向位置保持增益（防沿墙侧向漂移）
WALL_LAT_MAX = 0.12

CONTROL_HELP = """
磁吸爬墙操控（先点一下 MuJoCo 窗口再按键）：
  I 或 ↑   向上爬      K 或 ↓   向下
  J 或 ←   左转        L 或 →   右转
  P        停止(吸牢驻留)     B   重置贴墙站立
  1-6      单腿磁力通断        M   全部磁力通/断
鼠标用于旋转/平移视角。直线上爬时自动锁爬升方向+横向走直。
"""

# 机身→世界旋转：base +X→世界 -X, base +Y(前进)→世界 +Z(上), base +Z(抬脚)→世界 +Y(离墙)
_R_WALL = np.array([[-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0]])


def wall_quat() -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, _R_WALL.flatten())
    return q


def ensure_model():
    if not os.path.isfile(WALL_MODEL):
        print("未找到墙面场景，正在生成…", flush=True)
        import build_wall_scene
        build_wall_scene.main()


def reset_on_wall(model, data, stand_pose, body_z, magnets, quat):
    """把机身重定向贴到墙上（feet 压墙），磁力全开沉降吸稳。"""
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, float(body_z) - STANDOFF_ADJ, Z_START]
    data.qpos[3:7] = quat
    for jname, angle in stand_pose.items():
        data.qpos[joint_qadr(model, jname)] = float(angle)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    apply_ctrl(model, data, stand_pose)
    # 磁力全开、沉降，让足端吸附贴稳墙面
    magnets.enable_all()
    for _ in range(int(0.4 / model.opt.timestep)):
        magnets.apply()
        mujoco.mj_step(model, data)


def wall_heading(data) -> float:
    """机身"前进方向"在墙面内相对"正上方(世界+Z)"的偏角，0=竖直向上。"""
    # 机身前进轴(base +Y)在世界系 = free joint 姿态四元数旋转 (0,1,0)
    q = data.qpos[3:7]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    R = R.reshape(3, 3)
    f = R @ np.array([0.0, 1.0, 0.0])   # 世界系前进方向
    # 投影到墙面(世界 X-Z)，与竖直 +Z 的夹角；绕墙法向(+Y)偏转 → 用 (x,z)
    return math.atan2(f[0], f[2])       # 0 表示正对 +Z（竖直向上）


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def main():
    ensure_model()
    model = mujoco.MjModel.from_xml_path(WALL_MODEL)
    data = mujoco.MjData(model)
    stand_pose, body_z = load_stand()

    print("标定步态…", flush=True)
    gait = PlanarTripodGait(
        model=model, stand_pose=stand_pose, body_height=body_z,
        gait_mode="linear", cycle_time=CYCLE_TIME, lift_height=LIFT_HEIGHT,
        height_comp_m=HEIGHT_COMP, max_stride=MAX_STRIDE,
    )
    set_gains(model, WALK_KP, WALK_KV)

    magnets = LegMagnets(model, data, force_kg=MAGNET_KG, start_enabled=True)
    quat = wall_quat()
    reset_on_wall(model, data, stand_pose, body_z, magnets, quat)

    cmd = VelocityCommand()
    head_ref = [None]   # 直线上爬时锁定的爬升方向
    x_ref = [None]      # 直线上爬起点的墙面横向位置(world x)

    def on_reset():
        cmd.vx = cmd.vy = cmd.omega = 0.0
        head_ref[0] = None
        x_ref[0] = None
        gait.reset()
        reset_on_wall(model, data, stand_pose, body_z, magnets, quat)

    key_callback = make_key_handler(
        cmd, max_v=MAX_V, max_turn=MAX_TURN, on_reset=on_reset, magnets=magnets
    )

    print(CONTROL_HELP)
    print(f"足底磁力: 单腿 {MAGNET_KG:.0f}kg | 磁力‑步态联动 release={RELEASE} | {magnets.status_str()}", flush=True)

    def control_tick(dt):
        """一帧：算走直闭环 → 步态 → 磁力联动 → 步进。"""
        forward = cmd.vx           # I/K 映射到前进(爬升)量
        moving_lin = abs(forward) > 1e-6
        turning = abs(cmd.omega) > 1e-6
        omega_cmd = cmd.omega
        lat_cmd = 0.0
        if moving_lin and not turning:
            # 直线上/下爬：锁定爬升方向与横向位置，比例纠偏 → 沿竖直走直
            if head_ref[0] is None:
                head_ref[0] = wall_heading(data)
                x_ref[0] = float(data.qpos[0])
            herr = wrap_pi(head_ref[0] - wall_heading(data))
            omega_cmd = max(-WALL_YAW_MAX, min(WALL_YAW_MAX, WALL_YAW_KP * herr))
            lat_err = float(data.qpos[0]) - x_ref[0]        # 沿墙横向(world x)漂移
            # 注意：机身 base +X → 世界 -X（重定向所致），横向纠偏符号与平面版相反
            lat_cmd = max(-WALL_LAT_MAX, min(WALL_LAT_MAX, WALL_LAT_KP * lat_err))
        elif turning:
            head_ref[0] = wall_heading(data)
            x_ref[0] = float(data.qpos[0])
        else:
            head_ref[0] = None
            x_ref[0] = None

        moving = moving_lin or turning
        targets = gait.step(dt, vx=lat_cmd, vy=forward, omega=omega_cmd)
        apply_ctrl(model, data, targets)
        for _ in range(max(1, int(dt / model.opt.timestep))):
            if moving:
                magnets.follow_gait(gait, release=RELEASE)  # 支撑吸/摆动释放
            else:
                magnets.enable_all()                        # 停止：全部吸牢驻留
            magnets.apply()
            mujoco.mj_step(model, data)

    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, Z_START]
            viewer.cam.distance = 2.4
            viewer.cam.elevation = -8
            viewer.cam.azimuth = 90
            last = time.time()
            while viewer.is_running():
                now = time.time()
                dt = min(max(now - last, model.opt.timestep), 0.04)
                last = now
                control_tick(dt)
                viewer.cam.lookat[:] = [0.0, 0.0, float(data.qpos[2])]  # 跟随上爬
                viewer.sync()
    except Exception as e:
        print(f"无键盘/显示环境（{type(e).__name__}）：自动向上爬 12 秒…", flush=True)
        cmd.vx = MAX_V
        t0 = time.time()
        while time.time() - t0 < 12:
            control_tick(model.opt.timestep)
        print(f"爬升到 z={data.qpos[2]:.2f}（起点 {Z_START}），贴墙={abs(data.qpos[1])<0.2}")


if __name__ == "__main__":
    main()
