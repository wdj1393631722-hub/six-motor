#!/usr/bin/env python3
"""交互式调节初始姿态 — 实时改关节角，打印/保存后发给我即可。"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import mujoco
import mujoco.viewer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml")

KEY_1, KEY_2, KEY_3 = 49, 50, 51
KEY_4, KEY_5, KEY_6 = 52, 53, 54
KEY_C, KEY_F, KEY_T = 67, 70, 84
KEY_U, KEY_J = 85, 74
KEY_H, KEY_P, KEY_S, KEY_R = 72, 80, 83, 82
KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET = 91, 93
KEY_COMMA, KEY_PERIOD = 44, 46
KEY_PAGE_UP, KEY_PAGE_DOWN = 266, 267
KEY_Z, KEY_X = 90, 88
KEY_EQUAL, KEY_MINUS = 61, 45

JOINTS = ("coxa", "femur", "tibia")

HELP = """
姿态手动调节（先点击 MuJoCo 窗口）:
  1~6        选择腿号
  [ / ]      上一条 / 下一条腿
  C / F / T  选 coxa / femur / tibia
  U / J      当前关节 +1° / -1°
  , / .      当前关节 +0.2° / -0.2°
  Z / X      关节撑起/降低 ±1mm（改 femur/tibia，足底贴地）
  = / -      机身高度 +1mm / -1mm（备用）
  PgUp/PgDn  机身高度 ±1mm（有独立键的键盘）
  P          打印当前 JSON（复制发给我）
  S          保存到 generated/*_pose_flat.json
  R          恢复启动时加载的姿态
  H          显示本帮助
"""


def _joint_names() -> list[str]:
    return [f"leg{l}_{j}_joint" for l in range(1, 7) for j in JOINTS]


def _load_pose(which: str) -> tuple[dict[str, float], float, str]:
    from foot_kinematics import load_prone_pose, load_stand_pose

    if which == "prone":
        loaded = load_prone_pose()
        path = os.path.join(SCRIPT_DIR, "generated", "prone_pose_flat.json")
    else:
        loaded = load_stand_pose()
        path = os.path.join(SCRIPT_DIR, "generated", "stand_pose_flat.json")
    if loaded is None:
        raise FileNotFoundError(f"未找到 {which} 姿态文件")
    return dict(loaded[0]), float(loaded[1]), path


def _apply(model, data, pose: dict[str, float], body_z: float) -> None:
    data.qpos[0:3] = 0.0, 0.0, float(body_z)
    data.qpos[3:7] = 1.0, 0.0, 0.0, 0.0
    for jn, val in pose.items():
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
        data.qpos[adr] = float(val)
    data.qvel[:] = 0.0
    for jn, val in pose.items():
        try:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jn}_act")
            data.ctrl[aid] = float(val)
        except Exception:
            pass
    mujoco.mj_forward(model, data)


def _clamp_joint(model, jn: str, val: float) -> float:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
    lo, hi = model.jnt_range[jid]
    return float(max(lo, min(hi, val)))


def _payload(pose: dict[str, float], body_z: float) -> dict:
    return {"body_height": round(body_z, 6), "joints": {k: float(v) for k, v in pose.items()}}


def _print_pose(pose: dict[str, float], body_z: float, leg: int, joint: str) -> None:
    jn = f"leg{leg}_{joint}_joint"
    print(
        f"\n[当前] leg={leg} joint={joint} "
        f"值={math.degrees(pose[jn]):.2f}° ({pose[jn]:.6f} rad) body_z={body_z*1000:.1f}mm"
    )
    print(json.dumps(_payload(pose, body_z), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="交互式调节站立/趴地姿态")
    parser.add_argument(
        "--pose",
        choices=("stand", "prone"),
        default="stand",
        help="调节站立或趴地姿态（默认 stand）",
    )
    parser.add_argument(
        "--step-deg",
        type=float,
        default=1.0,
        help="U/J 每次调节角度（度）",
    )
    args = parser.parse_args()

    if not os.path.isfile(MODEL_PATH):
        print("未找到模型，请先: bash run.sh build")
        sys.exit(1)

    pose, body_z, save_path = _load_pose(args.pose)
    backup_pose = dict(pose)
    backup_bz = body_z

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    _apply(model, data, pose, body_z)

    leg = 1
    joint_idx = 0
    step_rad = math.radians(args.step_deg)
    fine_rad = math.radians(0.2)

    state = {"leg": leg, "joint_idx": joint_idx, "pose": pose, "body_z": body_z}

    def _bump(delta_rad: float) -> None:
        leg_i = state["leg"]
        ji = state["joint_idx"]
        jn = f"leg{leg_i}_{JOINTS[ji]}_joint"
        p = state["pose"]
        p[jn] = _clamp_joint(model, jn, p[jn] + delta_rad)
        _apply(model, data, p, state["body_z"])
        print(
            f"leg{leg_i} {JOINTS[ji]} → {math.degrees(p[jn]):.2f}° "
            f"({p[jn]:.6f} rad)"
        )

    def key_callback(keycode: int) -> None:
        nonlocal leg, joint_idx
        if keycode in (KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6):
            state["leg"] = keycode - KEY_1 + 1
            leg = state["leg"]
            print(f"选中 leg{leg}")
        elif keycode == KEY_LEFT_BRACKET:
            state["leg"] = (state["leg"] - 2) % 6 + 1
            leg = state["leg"]
            print(f"选中 leg{leg}")
        elif keycode == KEY_RIGHT_BRACKET:
            state["leg"] = state["leg"] % 6 + 1
            leg = state["leg"]
            print(f"选中 leg{leg}")
        elif keycode == KEY_C:
            state["joint_idx"] = 0
            joint_idx = 0
            print("关节: coxa")
        elif keycode == KEY_F:
            state["joint_idx"] = 1
            joint_idx = 1
            print("关节: femur")
        elif keycode == KEY_T:
            state["joint_idx"] = 2
            joint_idx = 2
            print("关节: tibia")
        elif keycode == KEY_U:
            _bump(step_rad)
        elif keycode == KEY_J:
            _bump(-step_rad)
        elif keycode == KEY_COMMA:
            _bump(-fine_rad)
        elif keycode == KEY_PERIOD:
            _bump(fine_rad)
        elif keycode in (KEY_PAGE_UP, KEY_Z, KEY_EQUAL):
            from foot_kinematics import adjust_stand_body_height

            pose, bz = adjust_stand_body_height(
                model, data, state["pose"], state["body_z"], 0.001
            )
            state["pose"] = pose
            state["body_z"] = bz
            _apply(model, data, state["pose"], state["body_z"])
            print(f"关节撑起 body_z = {state['body_z']*1000:.1f} mm")
        elif keycode in (KEY_PAGE_DOWN, KEY_X, KEY_MINUS):
            from foot_kinematics import adjust_stand_body_height

            pose, bz = adjust_stand_body_height(
                model, data, state["pose"], state["body_z"], -0.001
            )
            state["pose"] = pose
            state["body_z"] = bz
            _apply(model, data, state["pose"], state["body_z"])
            print(f"关节撑起 body_z = {state['body_z']*1000:.1f} mm")
        elif keycode == KEY_P:
            _print_pose(state["pose"], state["body_z"], state["leg"], JOINTS[state["joint_idx"]])
        elif keycode == KEY_S:
            payload = _payload(state["pose"], state["body_z"])
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"已保存: {save_path}")
        elif keycode == KEY_R:
            state["pose"] = dict(backup_pose)
            state["body_z"] = backup_bz
            _apply(model, data, state["pose"], state["body_z"])
            print("已恢复初始加载姿态")
        elif keycode == KEY_H:
            print(HELP)

    print(f"加载: {args.pose} 姿态 ({save_path})")
    print(f"body_z = {body_z*1000:.1f} mm | 当前 leg{leg} {JOINTS[joint_idx]}")
    print(HELP)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.10]
        viewer.cam.distance = 2.4
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 140
        while viewer.is_running():
            viewer.sync()


if __name__ == "__main__":
    main()
