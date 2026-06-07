#!/usr/bin/env python3
"""
六足前进 — 三角步态（Tripod Gait）演示。

按键（不用 W/A/S/D）:
  I 或 ↑   前进
  K 或 ↓   后退
  P        停止
  B        重置站立
"""
from __future__ import annotations

import os
import sys
import time

import mujoco
import mujoco.viewer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml")

sys.path.insert(0, SCRIPT_DIR)
from gait import create_forward_tripod_gait
from run_locomotion import apply_ctrl, joint_qposadr
from tripod_planner import TRIPOD_A, TRIPOD_B
from viewer_controls import CONTROL_HELP, VelocityCommand, make_key_handler


def reset_stand(model, data, gait) -> None:
    """物理站立：足底贴地，不锁定机身。"""
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = gait.p.body_height
    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0
    for jname, angle in gait.stand.items():
        data.qpos[joint_qposadr(model, jname)] = angle
    data.qvel[:] = 0.0
    apply_ctrl(model, data, gait.stand)
    mujoco.mj_forward(model, data)
    for _ in range(800):
        mujoco.mj_step(model, data)
        apply_ctrl(model, data, gait.stand)


def main() -> None:
    if not os.path.isfile(MODEL_PATH):
        print("未找到模型，正在生成...")
        import build_real_mjcf

        build_real_mjcf.main()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    max_v = 0.06
    gait = create_forward_tripod_gait(model=model, speed_mps=max_v)
    reset_stand(model, data, gait)

    cmd = VelocityCommand(vx=0.0)

    def on_reset():
        gait.reset()
        reset_stand(model, data, gait)

    key_callback = make_key_handler(cmd, max_v=max_v, max_turn=0.0, on_reset=on_reset)

    print("模型:", MODEL_PATH)
    print("三角步态 | A:", TRIPOD_A, "B:", TRIPOD_B)
    print(CONTROL_HELP)
    print("提示: 点击窗口后按 I 前进；启动后自动演示 3 秒。")
    auto_until = time.time() + 3.0

    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.12]
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -25
            viewer.cam.azimuth = 135
            last = time.time()
            while viewer.is_running():
                now = time.time()
                dt = max(now - last, model.opt.timestep)
                last = now
                if now < auto_until and cmd.vx == 0.0:
                    cmd.vx = max_v
                targets = gait.step(dt, vx=cmd.vx, vy=0.0, omega=0.0)
                apply_ctrl(model, data, targets)
                steps = max(1, int(dt / model.opt.timestep))
                for _ in range(steps):
                    mujoco.mj_step(model, data)
                viewer.sync()
    except TypeError:
        print("无键盘回调，自动前进 12 秒…")
        cmd.vx = max_v
        t0 = time.time()
        while time.time() - t0 < 12.0:
            dt = model.opt.timestep
            apply_ctrl(model, data, gait.step(dt, vx=cmd.vx))
            mujoco.mj_step(model, data)
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
