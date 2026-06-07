#!/usr/bin/env python3
"""
SIX-MOTOR 六足 MuJoCo 仿真 — 槽位三角步态（中/前/后 + 机身推进）

按键（viewer 窗口激活时）:
  E        使能（4s 撑起 → 0.6s 软站立）
  D        失能（5s 缓降 → 无力矩）
  I/↑ K/↓ J/← L/→  前进/后退/左转/右转（须先使能）
  P        停止
  B        重置为失能趴地
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
from enable_state import EnableController, EnablePhase, make_prone_pose_analytic
from gait import create_forward_tripod_gait
from tripod_planner import TRIPOD_A, TRIPOD_B
from foot_stance_lock import apply_stance_world_lock, build_actuator_map
from viewer_controls import CONTROL_HELP, VelocityCommand, make_key_handler

try:
    from foot_kinematics import MIN_BODY_HEIGHT, load_prone_pose, nominal_prone_pose
except ImportError:
    MIN_BODY_HEIGHT = 0.065
    load_prone_pose = None
    nominal_prone_pose = None

def apply_kinematic_pose(model, data, pose: dict, body_z: float) -> None:
    """无力矩时保持标定趴地/站立姿态（仅更新位姿，不施力）。"""
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = float(body_z)
    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0
    for jname, angle in pose.items():
        adr = joint_qposadr(model, jname)
        data.qpos[adr] = angle
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def joint_qposadr(model, jname: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return model.jnt_qposadr[jid]


def joint_actuator_id(model, jname: str) -> int:
    aname = f"{jname}_act"
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
    except Exception:
        return -1


def apply_ctrl(model, data, targets: dict):
    for jname, angle in targets.items():
        aid = joint_actuator_id(model, jname)
        if aid >= 0:
            data.ctrl[aid] = angle


def apply_joint_qpos(model, data, targets: dict, body_x: float | None = None):
    """运动学相：直接写关节角，避免足端滑移。"""
    if body_x is not None:
        data.qpos[0] = float(body_x)
        data.qpos[1] = 0.0
    for jname, angle in targets.items():
        adr = joint_qposadr(model, jname)
        data.qpos[adr] = angle
    apply_ctrl(model, data, targets)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _prone_pose_and_height(model, gait) -> tuple[dict, float]:
    if load_prone_pose is not None:
        loaded = load_prone_pose()
        if loaded is not None:
            return loaded[0], loaded[1]
    if nominal_prone_pose is not None:
        return nominal_prone_pose(model)
    prone = make_prone_pose_analytic(gait.stand)
    return prone, MIN_BODY_HEIGHT


def reset_disabled(model, data, gait, enable_ctrl: EnableController) -> None:
    """失能趴地：无力矩，六足足底平面贴地。"""
    mujoco.mj_resetData(model, data)
    prone, body_z = _prone_pose_and_height(model, gait)
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = body_z
    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0
    for jname, angle in prone.items():
        adr = joint_qposadr(model, jname)
        data.qpos[adr] = angle
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    enable_ctrl.load_stand(gait.stand)
    enable_ctrl.init_disabled(model, data, prone=prone)
    enable_ctrl.prone_body_z = body_z
    apply_kinematic_pose(model, data, prone, body_z)


def main():
    if not os.path.isfile(MODEL_PATH):
        print("未找到模型，正在生成...")
        import build_real_mjcf

        build_real_mjcf.main()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    max_v = 0.02
    gait = create_forward_tripod_gait(model=model, speed_mps=max_v)
    gait.use_slot_gait = True
    enable_ctrl = EnableController()
    reset_disabled(model, data, gait, enable_ctrl)
    act_map = build_actuator_map(model)

    cmd = VelocityCommand()
    max_turn = 0.4

    def on_reset():
        gait.reset()
        reset_disabled(model, data, gait, enable_ctrl)

    def on_enable():
        enable_ctrl.enable(model, data)

    def on_disable():
        enable_ctrl.disable(model, data)

    key_callback = make_key_handler(
        cmd,
        max_v=max_v,
        max_turn=max_turn,
        on_reset=on_reset,
        on_enable=on_enable,
        on_disable=on_disable,
    )

    print("加载:", MODEL_PATH)
    print("关节数:", model.njnt, "执行器:", model.nu)
    print("三角步态分组: A", TRIPOD_A, "| B", TRIPOD_B)
    print(
        "  周期",
        round(gait.p.cycle_time, 1),
        "s | 步长",
        gait.p.stride_length,
        "m | 抬脚",
        gait.p.step_height,
        "m | 槽位步态",
    )
    print(CONTROL_HELP)
    print("提示: 请先点击 MuJoCo 窗口；按 E 使能后再用 I/K/J/L 行走。")

    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.10]
            viewer.cam.distance = 2.2
            viewer.cam.elevation = -28
            viewer.cam.azimuth = 140
            last = time.time()
            last_phase = ""
            while viewer.is_running():
                now = time.time()
                dt = now - last
                last = now
                if dt <= 0:
                    dt = model.opt.timestep

                enable_targets = enable_ctrl.step(model, data, dt)

                if enable_ctrl.allows_gait and (
                    abs(cmd.vx) > 1e-6
                    or abs(cmd.vy) > 1e-6
                    or abs(cmd.omega) > 1e-6
                ):
                    targets = gait.step(dt, cmd.vx, cmd.vy, cmd.omega, sim_data=data)
                elif enable_targets is not None:
                    targets = enable_targets
                else:
                    targets = None

                if targets is not None:
                    apply_ctrl(model, data, targets)

                if enable_ctrl.phase == EnablePhase.DISABLED:
                    body_z = getattr(enable_ctrl, "prone_body_z", MIN_BODY_HEIGHT)
                    apply_kinematic_pose(
                        model, data, enable_ctrl.prone_pose, body_z
                    )
                else:
                    walking = enable_ctrl.allows_gait and (
                        abs(cmd.vx) > 1e-6
                        or abs(cmd.vy) > 1e-6
                        or abs(cmd.omega) > 1e-6
                    )
                    if walking and gait.use_slot_gait and targets is not None:
                        apply_joint_qpos(
                            model, data, targets, body_x=gait.last_body_x
                        )
                    else:
                        if walking:
                            data.qpos[0] = gait.last_body_x
                        for _ in range(max(1, int(dt / model.opt.timestep))):
                            mujoco.mj_step(model, data)
                        if (
                            walking
                            and gait.last_stance_lock
                            and gait._ik is not None
                        ):
                            apply_stance_world_lock(
                                model,
                                data,
                                gait._ik,
                                gait.last_stance_lock,
                                act_map,
                            )

                if gait.last_phase and gait.last_phase != last_phase:
                    last_phase = gait.last_phase
                    print(
                        f"步态相: {last_phase} | body_x={gait.last_body_x:.4f} m"
                    )

                viewer.sync()
    except TypeError:
        print("无键盘控制：自动使能并前进 15 秒…")
        enable_ctrl.enable(model, data)
        cmd.vx = max_v
        t0 = time.time()
        while time.time() - t0 < 15:
            dt = model.opt.timestep
            enable_targets = enable_ctrl.step(model, data, dt)
            if enable_ctrl.allows_gait:
                targets = gait.step(dt, cmd.vx, cmd.vy, cmd.omega, sim_data=data)
            elif enable_targets is not None:
                targets = enable_targets
            else:
                targets = None
            if targets is not None:
                apply_ctrl(model, data, targets)
            if enable_ctrl.allows_gait and gait.use_slot_gait and targets:
                apply_joint_qpos(model, data, targets, body_x=gait.last_body_x)
            else:
                if enable_ctrl.allows_gait:
                    data.qpos[0] = gait.last_body_x
                mujoco.mj_step(model, data)
                if enable_ctrl.allows_gait and gait.last_stance_lock and gait._ik:
                    apply_stance_world_lock(
                        model, data, gait._ik, gait.last_stance_lock, act_map
                    )
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
