#!/usr/bin/env python3
"""手动操控六足在「船底式」坡面上爬行，并在地面标注机身轨迹。

  python3 build_slope_scene.py      # 先生成坡面场景（只需一次，或改完地形后重跑）
  python3 slope_demo.py             # 手动操控爬坡

操控（先点一下 MuJoCo 窗口再按键）：
  I 或 ↑   前进（沿 +y 上坡）
  K 或 ↓   后退（下坡）
  J 或 ←   左转
  L 或 →   右转
  P        停止
  B        重置到坡前站立
鼠标仍用于旋转/平移视角。直线前进/后退时自动锁航向+横向，沿脊线走直。

坡面由 slope_terrain.py 定义（纵向斜面 + 横向凸脊 + 轻微起伏）。
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
    YAW_HOLD_KP, YAW_HOLD_MAX, LAT_HOLD_KP, LAT_HOLD_MAX,
    load_stand, set_gains, reset_standing, apply_ctrl,
    body_yaw, wrap_pi,
)
from planar_tripod_gait import PlanarTripodGait
from viewer_controls import VelocityCommand, make_key_handler
from trajectory_demo import draw_trail
from leg_magnets import LegMagnets
import slope_terrain

# 足底磁力：坡面/船底面上停住吸附、防滑防跌。吸附单腿 50kg，抬脚自动无磁。
# 数字键 1-6 单腿通断、M 键全部通断。默认开机不通电（六腿全通电会钉住步态走不动）；
# 想在坡上停住不下滑时按 M 一键吸附即可。
MAGNET_KG = 50.0
MAGNET_START_ON = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLOPE_MODEL = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_slope.xml")

MAX_V = 0.85         # 前进指令（直线后蹬模式下实测前进约 0.33 m/s）
MAX_TURN = 2.2       # 转向指令
TRAIL_DT = 0.08
TRAIL_MAX = 1500

CONTROL_HELP = """
坡面手动操控（先点一下 MuJoCo 窗口再按键）：
  I 或 ↑   前进（沿 +y 上坡）
  K 或 ↓   后退（下坡）
  J 或 ←   左转
  L 或 →   右转
  P        停止
  B        重置到坡前站立
  1-6      切换第 N 条腿磁力使能（通电吸附/断电）
  M        一键 全部磁力 通电/断电（坡上停住吸附、防下滑）
鼠标用于旋转/平移视角。直线行进时自动锁航向+横向走直。
"""


def ensure_model():
    if not os.path.isfile(SLOPE_MODEL):
        print("未找到坡面场景，正在生成…", flush=True)
        import build_slope_scene
        build_slope_scene.main()


def main():
    ensure_model()
    model = mujoco.MjModel.from_xml_path(SLOPE_MODEL)
    data = mujoco.MjData(model)

    # 用浮点精度填充 heightfield 高程（与 XML 里的 nrow/ncol/size 对应）
    terr = slope_terrain.build()
    model.hfield_data[:] = terr["data"]

    stand_pose, body_z = load_stand()
    print("标定步态…", flush=True)
    gait = PlanarTripodGait(
        model=model, stand_pose=stand_pose, body_height=body_z,
        gait_mode="linear", cycle_time=0.45, lift_height=0.090,
        height_comp_m=0.055, max_stride=0.28,
    )
    set_gains(model, WALK_KP, WALK_KV)
    reset_standing(model, data, stand_pose, body_z)

    magnets = LegMagnets(model, data, force_kg=MAGNET_KG, start_enabled=MAGNET_START_ON)

    cmd = VelocityCommand()
    yaw_ref = [None]      # 直线行进时锁定的目标航向
    pos_ref = [None]      # 直线行进起点的机身水平位置 (world xy)
    trail = []

    def on_reset():
        cmd.vx = cmd.vy = cmd.omega = 0.0
        yaw_ref[0] = None
        pos_ref[0] = None
        trail.clear()
        gait.reset()
        reset_standing(model, data, stand_pose, body_z)

    key_callback = make_key_handler(
        cmd, max_v=MAX_V, max_turn=MAX_TURN, on_reset=on_reset, magnets=magnets
    )

    print(CONTROL_HELP)
    print(f"坡面：倾角≈{terr['angle_deg']}°，坡脚 y={terr['y_foot']}，坡顶 y={terr['y_top']}", flush=True)
    print(f"足底磁力: 单腿 {MAGNET_KG:.0f}kg | 数字键 1-6 单腿通断, M 键全部通断 | {magnets.status_str()}", flush=True)

    last_real = time.time()
    last_sample = -1.0
    sim_t = 0.0

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.0, terr["y_foot"] + 1.5, 0.6]
        viewer.cam.distance = 5.5
        viewer.cam.elevation = -22
        viewer.cam.azimuth = 70

        while viewer.is_running():
            now = time.time()
            dt = min(max(now - last_real, model.opt.timestep), 0.04)
            last_real = now
            sim_t += dt

            # 前进映射到 +Y（cmd.vx 为前进量），与 run_planar_tripod 一致
            forward = cmd.vx
            moving_lin = abs(forward) > 1e-6
            turning = abs(cmd.omega) > 1e-6
            omega_cmd = cmd.omega
            lat_cmd = 0.0
            if moving_lin and not turning:
                # 直线上/下坡：锁定初始航向与位置，分别比例反馈纠偏 → 沿脊线走直
                if yaw_ref[0] is None:
                    yaw_ref[0] = body_yaw(data)
                    pos_ref[0] = data.qpos[0:2].copy()
                err = wrap_pi(yaw_ref[0] - body_yaw(data))
                omega_cmd = max(-YAW_HOLD_MAX, min(YAW_HOLD_MAX, YAW_HOLD_KP * err))
                lat_axis = (math.cos(yaw_ref[0]), math.sin(yaw_ref[0]))
                d = data.qpos[0:2] - pos_ref[0]
                lat_err = float(d[0] * lat_axis[0] + d[1] * lat_axis[1])
                lat_cmd = max(-LAT_HOLD_MAX, min(LAT_HOLD_MAX, -LAT_HOLD_KP * lat_err))
            elif turning:
                yaw_ref[0] = body_yaw(data)
                pos_ref[0] = data.qpos[0:2].copy()
            else:
                yaw_ref[0] = None
                pos_ref[0] = None

            targets = gait.step(dt, vx=lat_cmd, vy=forward, omega=omega_cmd)
            apply_ctrl(model, data, targets)
            for _ in range(max(1, int(dt / model.opt.timestep))):
                magnets.apply()          # 施加足底吸附外力（沿坡面法向压紧）
                mujoco.mj_step(model, data)

            if sim_t - last_sample >= TRAIL_DT:
                last_sample = sim_t
                trail.append((float(data.qpos[0]), float(data.qpos[1])))
                if len(trail) > TRAIL_MAX:
                    trail.pop(0)
            draw_trail(viewer.user_scn, trail, [1.0, 0.55, 0.1, 1.0])
            viewer.sync()


if __name__ == "__main__":
    main()
