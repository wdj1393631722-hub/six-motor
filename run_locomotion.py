#!/usr/bin/env python3
"""
SIX-MOTOR 六足 MuJoCo 仿真 — 关节三角步态（抬腿/前摆/蹬进）

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
ENABLE_CTRL_DT = 0.005

sys.path.insert(0, SCRIPT_DIR)
from ctrl_smoother import CtrlSmoother
from enable_state import (
    EnableController,
    EnablePhase,
    LOCOMOTION_KP,
    LOCOMOTION_KV,
    STAND_KP,
    STAND_KV,
    make_prone_pose_analytic,
    set_actuator_gains,
)
from gait import create_forward_tripod_gait
from tripod_planner import TRIPOD_A, TRIPOD_B
from body_stabilizer import stabilize_locomotion_body
from foot_stance_lock import (
    apply_stance_world_lock,
    blend_stance_ctrl_targets,
    build_actuator_map,
    damp_leg_joint_velocities,
)
from robot_limits import clamp_joint_targets
from viewer_controls import CONTROL_HELP, VelocityCommand, make_key_handler

try:
    from foot_kinematics import (
        ENABLE_BODY_LIFT_M,
        MIN_BODY_HEIGHT,
        PRONE_BODY_HEIGHT,
        body_bottom_clearance,
        load_prone_pose,
        load_stand_pose,
        nominal_prone_pose,
        prone_foot_max_hover,
        resolve_prone_body_height,
        solve_stand_at_height,
    )
except ImportError:
    MIN_BODY_HEIGHT = 0.050
    PRONE_BODY_HEIGHT = 0.058
    ENABLE_BODY_LIFT_M = 0.022
    body_bottom_clearance = None
    prone_foot_max_hover = None
    resolve_prone_body_height = None
    solve_stand_at_height = None
    load_prone_pose = None
    load_stand_pose = None
    nominal_prone_pose = None

def apply_kinematic_pose(model, data, pose: dict, body_z: float) -> None:
    """无力矩时保持标定趴地/站立姿态（仅更新位姿，不施力）。"""
    from foot_kinematics import _set_pose

    _set_pose(model, data, clamp_joint_targets(model, pose), float(body_z))
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
    for jname, angle in clamp_joint_targets(model, targets).items():
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
    stand_pose = None
    if load_stand_pose is not None:
        loaded_stand = load_stand_pose()
        if loaded_stand is not None:
            stand_pose = loaded_stand[0]
    if load_prone_pose is not None:
        loaded = load_prone_pose()
        if loaded is not None:
            prone, body_z = loaded
            if resolve_prone_body_height is not None:
                data = mujoco.MjData(model)
                fixed_bz = resolve_prone_body_height(model, data, prone, body_z)
                if fixed_bz > body_z + 1e-6:
                    print(
                        f"[趴地] 主体穿地，body_z {body_z*1000:.1f}→{fixed_bz*1000:.1f} mm",
                        flush=True,
                    )
                    body_z = fixed_bz
                if body_bottom_clearance is not None:
                    clr = body_bottom_clearance(model, data, body_z)
                    hover = (
                        prone_foot_max_hover(model, prone, body_z)
                        if prone_foot_max_hover is not None
                        else 0.0
                    )
                    print(
                        f"[趴地] body_z={body_z*1000:.1f} mm  "
                        f"主体下沿={clr*1000:.1f} mm  足底悬空={hover*1000:.1f} mm",
                        flush=True,
                    )
                    if hover > 0.012:
                        print(
                            "[趴地] 足底偏高，可运行: bash run.sh prone",
                            flush=True,
                        )
            return prone, body_z
    if nominal_prone_pose is not None:
        return nominal_prone_pose(model)
    prone = make_prone_pose_analytic(gait.stand)
    return prone, PRONE_BODY_HEIGHT


def _apply_file_stand_pose(gait) -> None:
    """步态基准角：直接读 stand_pose_flat.json（启动不重算，避免卡住）。"""
    if load_stand_pose is None:
        return
    loaded = load_stand_pose()
    if loaded is None:
        return
    stand_pose, body_z = loaded
    try:
        from rl_posture import sync_gait_stand

        sync_gait_stand(gait, stand_pose, body_z)
    except ImportError:
        gait.stand = dict(stand_pose)
        gait.p.body_height = float(body_z)


def reset_disabled(model, data, gait, enable_ctrl: EnableController) -> None:
    """失能趴地：无力矩，六足足底平面贴地。"""
    _apply_file_stand_pose(gait)
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
    enable_ctrl.load_stand(
        gait.stand, body_z=gait.p.body_height, prone_body_z=body_z
    )
    stand_bz = float(body_z) + float(ENABLE_BODY_LIFT_M)
    if solve_stand_at_height is not None:
        stand_pose, stand_bz = solve_stand_at_height(
            model, prone, gait.stand, stand_bz
        )
    else:
        stand_pose = dict(prone)
    enable_ctrl.stand_pose = dict(stand_pose)
    enable_ctrl.stand_body_z = float(stand_bz)
    enable_ctrl.init_disabled(model, data, prone=prone)
    enable_ctrl.prone_body_z = body_z
    gait.stand = dict(stand_pose)
    gait.p.body_height = float(stand_bz)
    if getattr(gait, "_model", None) is not None:
        gait._init_kinematics()
    else:
        gait.reset()
    apply_kinematic_pose(model, data, prone, body_z)


def main():
    if not os.path.isfile(MODEL_PATH):
        print("未找到模型，正在生成...")
        import build_real_mjcf

        build_real_mjcf.main()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    max_v = 0.06
    gait = create_forward_tripod_gait(model=model, speed_mps=max_v)
    gait.use_joint_gait = True
    gait.use_slot_gait = False
    print("正在加载模型与姿态…", flush=True)
    _apply_file_stand_pose(gait)
    enable_ctrl = EnableController()
    reset_disabled(model, data, gait, enable_ctrl)
    if load_stand_pose is not None:
        loaded = load_stand_pose()
        if loaded is not None:
            import math

            _, sbz = loaded
            print(
                f"站立姿态: generated/stand_pose_flat.json  "
                f"名义高度={sbz*1000:.0f} mm  "
                f"物理站立={enable_ctrl.stand_body_z*1000:.0f} mm"
            )
            for leg in range(1, 7):
                c = gait.stand[f"leg{leg}_coxa_joint"]
                print(f"  leg{leg} coxa {math.degrees(c):+.1f}°")
    act_map = build_actuator_map(model)
    ctrl_smoother = CtrlSmoother(tau=0.14)
    locomotion_gains = False

    cmd = VelocityCommand()
    max_turn = 0.4

    def on_reset():
        nonlocal locomotion_gains
        gait.reset()
        ctrl_smoother.reset(gait.stand)
        locomotion_gains = False
        reset_disabled(model, data, gait, enable_ctrl)

    def on_enable():
        enable_ctrl.enable(model, data)

    def on_disable():
        enable_ctrl.disable(model, data)

    KEY_Z, KEY_X, KEY_S = 90, 88, 83
    base_key_callback = make_key_handler(
        cmd,
        max_v=max_v,
        max_turn=max_turn,
        on_reset=on_reset,
        on_enable=on_enable,
        on_disable=on_disable,
    )

    def key_callback(keycode: int) -> None:
        standing_idle = (
            enable_ctrl.phase == EnablePhase.ENABLED
            and abs(cmd.vx) < 1e-6
            and abs(cmd.vy) < 1e-6
            and abs(cmd.omega) < 1e-6
        )
        if standing_idle and keycode in (KEY_Z, KEY_X):
            from foot_kinematics import (
                WALK_BODY_HEIGHT,
                adjust_stand_body_height,
                recalibrate_stand_contact_pose,
            )

            dz = 0.001 if keycode == KEY_Z else -0.001
            new_pose, new_bz = adjust_stand_body_height(
                model,
                data,
                enable_ctrl.stand_pose,
                enable_ctrl.stand_body_z,
                dz,
                min_body_z=WALK_BODY_HEIGHT,
            )
            new_pose, new_bz = recalibrate_stand_contact_pose(
                model, data, new_pose, new_bz, keep_coxa=True
            )
            enable_ctrl.stand_pose = dict(new_pose)
            enable_ctrl.stand_body_z = max(
                new_bz, enable_ctrl._physics_body_z_floor(enable_ctrl.prone_body_z)
            )
            try:
                from rl_posture import sync_gait_stand

                sync_gait_stand(gait, new_pose, enable_ctrl.stand_body_z)
            except ImportError:
                gait.stand = dict(new_pose)
                gait.p.body_height = enable_ctrl.stand_body_z
            ctrl_smoother.reset(gait.stand)
            apply_ctrl(model, data, new_pose)
            print(
                f"[机身] 关节撑起 body_z={new_bz * 1000:.1f} mm "
                f"（femur/tibia 已重算，足底贴地）"
            )
            return
        if standing_idle and keycode == KEY_S:
            try:
                from foot_kinematics import save_stand_pose

                path = save_stand_pose(
                    dict(enable_ctrl.stand_pose), enable_ctrl.stand_body_z
                )
                print(f"[保存] 站立关节角+高度 → {path}")
            except ImportError:
                print("[保存] 缺少 foot_kinematics.save_stand_pose")
            return
        base_key_callback(keycode)

    print("加载:", MODEL_PATH)
    print("关节数:", model.njnt, "执行器:", model.nu)
    print("三角步态分组: A", TRIPOD_A, "| B", TRIPOD_B)
    mode = "物理+摩擦" if getattr(gait, "use_physics_gait", True) else "运动学"
    print(
        "  周期",
        round(gait.p.cycle_time, 1),
        "s | 步长",
        gait.p.stride_length,
        "m | 抬脚",
        gait.p.step_height,
        "m | 关节步态 |",
        mode,
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
                dt = min(dt, 0.04)

                if enable_ctrl.phase in (
                    EnablePhase.LIFT_RAMP,
                    EnablePhase.SOFT_STAND,
                    EnablePhase.RAMP_DOWN,
                ):
                    enable_targets = None
                    elapsed = 0.0
                    while elapsed + 1e-9 < dt:
                        sub_dt = min(ENABLE_CTRL_DT, dt - elapsed)
                        enable_targets = enable_ctrl.step(model, data, sub_dt)
                        elapsed += sub_dt
                else:
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

                walking = enable_ctrl.allows_gait and (
                    abs(cmd.vx) > 1e-6
                    or abs(cmd.vy) > 1e-6
                    or abs(cmd.omega) > 1e-6
                )
                if walking and gait.use_joint_gait and gait.use_physics_gait:
                    if not locomotion_gains:
                        set_actuator_gains(model, LOCOMOTION_KP, LOCOMOTION_KV)
                        locomotion_gains = True
                elif locomotion_gains and enable_ctrl.phase == EnablePhase.ENABLED:
                    set_actuator_gains(model, STAND_KP, STAND_KV)
                    locomotion_gains = False

                if targets is not None:
                    ctrl_targets = targets
                    if (
                        walking
                        and gait.use_joint_gait
                        and gait.use_physics_gait
                        and gait.last_stance_lock
                        and gait._ik is not None
                    ):
                        ctrl_targets = blend_stance_ctrl_targets(
                            model,
                            data,
                            gait._ik,
                            gait.last_stance_lock,
                            targets,
                        )
                    if walking and gait.use_joint_gait and gait.use_physics_gait:
                        ctrl_targets = ctrl_smoother.filter(ctrl_targets, dt)
                    apply_ctrl(model, data, ctrl_targets)

                standing_idle = (
                    enable_ctrl.phase == EnablePhase.ENABLED
                    and not walking
                )
                if enable_ctrl.phase == EnablePhase.DISABLED:
                    body_z = getattr(enable_ctrl, "prone_body_z", MIN_BODY_HEIGHT)
                    apply_kinematic_pose(
                        model, data, enable_ctrl.prone_pose, body_z
                    )
                elif enable_ctrl.phase in (
                    EnablePhase.LIFT_RAMP,
                    EnablePhase.SOFT_STAND,
                    EnablePhase.RAMP_DOWN,
                ) or standing_idle:
                    pose_z = (
                        enable_ctrl.ramp_body_z
                        if enable_ctrl.phase
                        in (
                            EnablePhase.LIFT_RAMP,
                            EnablePhase.SOFT_STAND,
                            EnablePhase.RAMP_DOWN,
                        )
                        else enable_ctrl.stand_body_z
                    )
                    pose = (
                        targets
                        if targets is not None
                        else enable_ctrl.stand_pose
                    )
                    apply_kinematic_pose(model, data, pose, pose_z)
                    if targets is not None:
                        apply_ctrl(model, data, targets)
                else:
                    if (
                        walking
                        and gait.use_joint_gait
                        and targets is not None
                        and gait.last_kinematic_only
                    ):
                        data.qpos[1] = getattr(gait, "last_body_y", 0.0)
                        apply_joint_qpos(model, data, targets)
                    elif (
                        walking
                        and gait.use_slot_gait
                        and targets is not None
                    ):
                        apply_joint_qpos(
                            model, data, targets, body_x=gait.last_body_x
                        )
                    else:
                        if walking and not gait.use_joint_gait:
                            data.qpos[0] = gait.last_body_x
                        elif walking and enable_ctrl.phase == EnablePhase.ENABLED:
                            data.qpos[2] = enable_ctrl.stand_body_z
                            data.qvel[2] = 0.0
                        for _ in range(max(1, int(dt / model.opt.timestep))):
                            mujoco.mj_step(model, data)
                        if walking and gait.use_joint_gait and gait.use_physics_gait:
                            stabilize_locomotion_body(
                                model,
                                data,
                                body_z_target=gait.p.body_height,
                            )
                            damp_leg_joint_velocities(model, data)
                        elif (
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
                    if gait.use_joint_gait:
                        print(f"步态相: {last_phase}")
                    else:
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
            if (
                enable_ctrl.allows_gait
                and gait.use_joint_gait
                and targets
                and gait.last_kinematic_only
            ):
                data.qpos[1] = getattr(gait, "last_body_y", 0.0)
                apply_joint_qpos(model, data, targets)
            elif enable_ctrl.allows_gait and gait.use_slot_gait and targets:
                apply_joint_qpos(model, data, targets, body_x=gait.last_body_x)
            else:
                if enable_ctrl.allows_gait and not gait.use_joint_gait:
                    data.qpos[0] = gait.last_body_x
                mujoco.mj_step(model, data)
                if enable_ctrl.allows_gait and gait.use_joint_gait and gait.use_physics_gait:
                    stabilize_locomotion_body(
                        model, data, body_z_target=gait.p.body_height
                    )
                if enable_ctrl.allows_gait and gait.last_stance_lock and gait._ik:
                    apply_stance_world_lock(
                        model,
                        data,
                        gait._ik,
                        gait.last_stance_lock,
                        act_map,
                        blend=0.35,
                    )
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
