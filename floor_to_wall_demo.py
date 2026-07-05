#!/usr/bin/env python3
"""六足机器人**平地→垂直墙 连贯乘墙**：先在平地正常行走，走到墙角按 T 键，
机身沿墙角轴翻转 90° 乘上竖直墙面，再切入现有磁吸爬墙步态向上爬。

  python3 build_wall_scene.py     # 先生成含 floor+wall 的场景（只需一次）
  python3 floor_to_wall_demo.py   # 平地→乘墙→爬墙

操控（先点一下 MuJoCo 窗口再按键）：
  I 或 ↑   前进（平地朝墙 / 墙面向上爬）
  K 或 ↓   后退      J 或 ←  左转      L 或 →  右转
  T        在墙前触发**乘墙过渡**（机身翻立贴墙）
  P        停止      B   重置到平地站立
  1-6 单腿磁力通断    M   全部磁力通/断

三阶段状态机：
  FLOOR  平地行走（磁力关，自然站立），朝墙前进。
  MOUNT  按 T 触发：目标姿态沿墙角轴(世界 +X)从水平球面插值到竖直(wall_quat)，
         引导控制器把 free joint 角速度推向目标；步态持续前进+磁力联动，前腿触墙
         吸住把机身撑立。竖直后 → WALL。
  WALL   完全复用现有爬墙逻辑（走直闭环 + 磁力‑步态联动）向上爬。

引导控制：不直接写 qpos（保物理），每子步用 mju_subQuat 求当前→目标的姿态误差矢量，
按比例设为 free joint 角速度伺服（目标随时间平滑推进，跟踪误差很小），线速度轻阻尼防甩。
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
from wall_demo import (
    RELEASE, LIFT_HEIGHT, CYCLE_TIME, HEIGHT_COMP, MAX_STRIDE,
    MAGNET_KG, STANDOFF_ADJ, MAX_V, MAX_TURN,
    WALL_YAW_KP, WALL_YAW_MAX, WALL_LAT_KP, WALL_LAT_MAX,
    wall_quat, wall_heading, wrap_pi,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAMP_MODEL = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_wall_ramp.xml")


def ensure_ramp_model():
    if not os.path.isfile(RAMP_MODEL):
        print("未找到斜面过渡场景，正在生成…", flush=True)
        import build_wall_ramp_scene
        build_wall_ramp_scene.main()


# ---- 平地初始化 ----
Y_START = 1.10           # 机身初始离墙(y)距离：斜面延伸到 y≈0.36，前脚前伸~0.35，
                        #   起点要足够靠后让六足初始都落在平地上、再走向斜面。

# ---- 乘墙过渡参数（沿斜面"贴面引导"上爬）----
# 尖直角处机身无支撑、纯腿/引导都难（悬臂翻落）；改在 45° 斜面上过渡：机身沿一条
# **贴着斜面的圆弧路径(二次 Bezier 绕墙角)**被引导俯仰上爬，腿用 follow_gait 在斜面/
# 墙面上迈步磁吸——全程有支撑面在身下，观感是"贴斜坡爬上墙"而非腾空飞。
# 引导=对 free joint 做位姿速度伺服(不写 qpos，接触/磁吸仍生效)。
MOUNT_TIME = 3.5         # 沿斜面从平地站姿爬到墙面站姿的时长 (s)
ORI_KP = 12.0            # 姿态伺服增益（角速度 = KP·姿态误差矢量）
ORI_WMAX = 3.0           # 角速度限幅 (rad/s)
POS_KP = 6.0             # 位置伺服增益（线速度 = KP·位置误差）
POS_VMAX = 0.6           # 线速度限幅 (m/s)
MOUNT_V = 0.20           # 乘墙期间步态前进量：让腿在斜面/墙面上自然迈步(观感)
Z_WALL = 0.75            # 乘墙终点离地高度：够高让"六足全部"落在斜面顶(0.36)以上的
                        #   竖直墙面——否则下侧脚仍吸在 45°斜面上、法向把机身往外拉致下滑。
BEZIER_C = np.array([0.0, 0.30, 0.40])  # 贴斜面的 Bezier 控制点(斜面上方，引导贴坡拐上墙)
UPRIGHT_DONE = 0.95      # base +Y 在世界 +Z 的投影 > 此值判定竖直，切 WALL

# 机身→世界（平地朝墙）旋转：base +X→-X, base +Y(前进)→世界 -Y(朝墙),
# base +Z(足底法向)→世界 +Z(压地)。绕世界 +X 转 -90° 即得 wall_quat()。
_R_FLOOR = np.array([[-1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0]])

CONTROL_HELP = """
平地→乘墙→爬墙操控（先点一下 MuJoCo 窗口再按键）：
  I/↑ 前进(朝墙/向上)   K/↓ 后退   J/← 左转   L/→ 右转
  T   墙前触发乘墙       P 停止     B 重置平地站立
  1-6 单腿磁力          M 全部磁力通/断
状态：FLOOR 平地行走 → (按T) MOUNT 翻立贴墙 → WALL 磁吸爬墙。
"""

FLOOR, MOUNT, WALL = "FLOOR", "MOUNT", "WALL"


def floor_quat() -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, _R_FLOOR.flatten())
    return q


def slerp_quat(qa: np.ndarray, qb: np.ndarray, s: float) -> np.ndarray:
    """测地插值 qa→qb：q = qa ⊕ (subQuat(qb,qa)·s)。"""
    dv = np.zeros(3)
    mujoco.mju_subQuat(dv, np.asarray(qb, float), np.asarray(qa, float))
    q = np.array(qa, dtype=float)
    mujoco.mju_quatIntegrate(q, dv * float(s), 1.0)
    mujoco.mju_normalize4(q)
    return q


def uprightness(data) -> float:
    """base +Y(前进/爬升轴) 在世界 +Z 的投影：0=水平站立, 1=完全竖直贴墙。"""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    return float(R.reshape(3, 3)[:, 1][2])   # (R@[0,1,0])[2]


def _clamp_norm(v, vmax):
    n = float(np.linalg.norm(v))
    return v * (vmax / n) if n > vmax else v


def drive_root_pose(data, q_target, pos_target,
                    ori_kp=ORI_KP, w_max=ORI_WMAX,
                    pos_kp=POS_KP, v_max=POS_VMAX) -> None:
    """引导控制：把 free joint 的角速度/线速度伺服到目标姿态与目标位置。

    目标位姿随时间平滑推进（球面插值 + 直线插值），跟踪误差小、速度有界。不写 qpos，
    接触与磁吸力仍参与积分——即"引导"而非"瞬移"。以此把机身沿墙角平滑翻立到墙面站姿。
    """
    qc = np.array(data.qpos[3:7], dtype=float)
    mujoco.mju_normalize4(qc)
    err = np.zeros(3)
    mujoco.mju_subQuat(err, np.asarray(q_target, float), qc)  # 机体系旋转矢量误差
    data.qvel[3:6] = _clamp_norm(ori_kp * err, w_max)         # 角速度伺服(机体系)
    perr = np.asarray(pos_target, float) - np.array(data.qpos[0:3], float)
    data.qvel[0:3] = _clamp_norm(pos_kp * perr, v_max)        # 线速度伺服(世界系)


def reset_on_floor(model, data, stand_pose, body_z, magnets):
    """平地站立：机身水平朝墙，磁力关，沉降贴地。"""
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, Y_START, float(body_z)]
    data.qpos[3:7] = floor_quat()
    for jname, angle in stand_pose.items():
        data.qpos[joint_qadr(model, jname)] = float(angle)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    apply_ctrl(model, data, stand_pose)
    magnets.disable_all()
    for _ in range(int(0.3 / model.opt.timestep)):
        mujoco.mj_step(model, data)


def main():
    ensure_ramp_model()
    model = mujoco.MjModel.from_xml_path(RAMP_MODEL)
    data = mujoco.MjData(model)
    stand_pose, body_z = load_stand()

    print("标定步态…", flush=True)
    gait = PlanarTripodGait(
        model=model, stand_pose=stand_pose, body_height=body_z,
        gait_mode="linear", cycle_time=CYCLE_TIME, lift_height=LIFT_HEIGHT,
        height_comp_m=HEIGHT_COMP, max_stride=MAX_STRIDE,
    )
    set_gains(model, WALK_KP, WALK_KV)
    magnets = LegMagnets(model, data, force_kg=MAGNET_KG, start_enabled=False)
    reset_on_floor(model, data, stand_pose, body_z, magnets)

    cmd = VelocityCommand()
    state = [FLOOR]
    t_mount0 = [0.0]
    p_mount0 = [np.zeros(3)]   # 乘墙起点机身位置
    q_floor = floor_quat()
    q_wall = wall_quat()
    head_ref = [None]
    x_ref = [None]
    clock = [0.0]   # 仿真累计时间（不依赖 wall clock，便于自测）

    def on_reset():
        cmd.vx = cmd.vy = cmd.omega = 0.0
        state[0] = FLOOR
        head_ref[0] = None
        x_ref[0] = None
        gait.reset()
        reset_on_floor(model, data, stand_pose, body_z, magnets)

    def on_mount():
        if state[0] == FLOOR:
            state[0] = MOUNT
            t_mount0[0] = clock[0]
            p_mount0[0] = np.array(data.qpos[0:3], dtype=float)
            magnets.enable_all()
            gait.reset()
            print(f"[过渡] 开始乘墙 @ y={data.qpos[1]:.3f} z={data.qpos[2]:.3f}", flush=True)

    key_callback = make_key_handler(
        cmd, max_v=MAX_V, max_turn=MAX_TURN,
        on_reset=on_reset, on_mount=on_mount, magnets=magnets,
    )

    def control_tick(dt):
        clock[0] += dt
        forward = cmd.vx
        n_sub = max(1, int(dt / model.opt.timestep))

        if state[0] == FLOOR:
            # 平地：磁力关，自然站立行走（朝墙 -Y）
            targets = gait.step(dt, vx=0.0, vy=forward, omega=cmd.omega)
            apply_ctrl(model, data, targets)
            for _ in range(n_sub):
                magnets.disable_all()
                mujoco.mj_step(model, data)

        elif state[0] == MOUNT:
            # 沿斜面贴面引导：机身沿"贴斜面的 Bezier 圆弧"从平地站姿爬到墙面站姿，
            # 腿用 follow_gait 在斜面/墙面上迈步磁吸——身下全程有支撑面，观感贴坡爬升。
            s = min(1.0, (clock[0] - t_mount0[0]) / max(MOUNT_TIME, 1e-6))
            sm = s * s * (3.0 - 2.0 * s)                 # smoothstep 缓入缓出
            q_tgt = slerp_quat(q_floor, q_wall, sm)
            # 二次 Bezier: 起点 → 贴斜面控制点 → 墙面站姿，路径贴着斜面拐上墙
            p_wall = np.array([0.0, float(body_z) - STANDOFF_ADJ, Z_WALL])
            p_tgt = ((1 - sm) ** 2 * p_mount0[0]
                     + 2 * (1 - sm) * sm * BEZIER_C
                     + sm ** 2 * p_wall)
            targets = gait.step(dt, vx=0.0, vy=MOUNT_V, omega=0.0)  # 腿在面上迈步
            apply_ctrl(model, data, targets)
            for _ in range(n_sub):
                if gait.active:
                    magnets.follow_gait(gait, release=RELEASE)     # 支撑吸/摆动释放
                else:
                    magnets.enable_all()
                magnets.apply()
                drive_root_pose(data, q_tgt, p_tgt)
                mujoco.mj_step(model, data)
            if s >= 1.0:
                # 交接：保持墙面站姿并沉降吸附，等多数腿抓牢再松伺服交给爬墙步态
                p_wall = np.array([0.0, float(body_z) - STANDOFF_ADJ, Z_WALL])
                for _ in range(int(1.2 / model.opt.timestep)):
                    magnets.enable_all()
                    magnets.apply()
                    drive_root_pose(data, q_wall, p_wall)
                    mujoco.mj_step(model, data)
                data.qvel[:] = 0.0
                state[0] = WALL
                head_ref[0] = None
                x_ref[0] = None
                print(f"[过渡] 乘墙完成 → 爬墙 (贴墙度={uprightness(data):.3f} "
                      f"y={data.qpos[1]:.3f} z={data.qpos[2]:.2f} "
                      f"吸附={len(magnets.attached_legs())}/6)", flush=True)

        else:  # WALL —— 复用现有爬墙走直闭环
            moving_lin = abs(forward) > 1e-6
            turning = abs(cmd.omega) > 1e-6
            omega_cmd = cmd.omega
            lat_cmd = 0.0
            if moving_lin and not turning:
                if head_ref[0] is None:
                    head_ref[0] = wall_heading(data)
                    x_ref[0] = float(data.qpos[0])
                herr = wrap_pi(head_ref[0] - wall_heading(data))
                omega_cmd = max(-WALL_YAW_MAX, min(WALL_YAW_MAX, WALL_YAW_KP * herr))
                lat_err = float(data.qpos[0]) - x_ref[0]
                lat_cmd = max(-WALL_LAT_MAX, min(WALL_LAT_MAX, WALL_LAT_KP * lat_err))
            elif turning:
                head_ref[0] = wall_heading(data)
                x_ref[0] = float(data.qpos[0])
            else:
                head_ref[0] = None
                x_ref[0] = None
            targets = gait.step(dt, vx=lat_cmd, vy=forward, omega=omega_cmd)
            apply_ctrl(model, data, targets)
            for _ in range(n_sub):
                if gait.active:
                    magnets.follow_gait(gait, release=RELEASE)
                else:
                    magnets.enable_all()
                magnets.apply()
                mujoco.mj_step(model, data)

    print(CONTROL_HELP)
    print(f"足底磁力 单腿 {MAGNET_KG:.0f}kg | {magnets.status_str()}", flush=True)

    headless = bool(os.environ.get("F2W_HEADLESS")) or not os.environ.get("DISPLAY")
    if not headless:
        try:
            with mujoco.viewer.launch_passive(
                model, data, key_callback=key_callback
            ) as viewer:
                viewer.cam.distance = 2.4
                viewer.cam.elevation = -12
                viewer.cam.azimuth = 110
                last = time.time()
                while viewer.is_running():
                    now = time.time()
                    dt = min(max(now - last, model.opt.timestep), 0.04)
                    last = now
                    control_tick(dt)
                    viewer.cam.lookat[:] = data.qpos[0:3]
                    viewer.sync()
            return
        except Exception as e:
            print(f"无显示环境（{type(e).__name__}）：转自动演示…", flush=True)

    if True:
        # 无显示/自测：自动跑完整套 平地前进→到墙角乘墙→爬墙
        print("自动演示 平地→乘墙→爬墙…", flush=True)
        ts = model.opt.timestep
        z0 = float(data.qpos[2])
        cmd.vx = MAX_V
        # 1) 平地前进直到抵达斜面脚下
        for _ in range(int(6.0 / ts)):
            control_tick(ts)
            if data.qpos[1] < 0.80:
                break
        print(f"  到斜面脚下 y={data.qpos[1]:.3f} z={data.qpos[2]:.3f}，触发乘墙", flush=True)
        # 2) 触发乘墙并跑完过渡
        on_mount()
        for _ in range(int((MOUNT_TIME + 1.5) / ts)):
            control_tick(ts)
            if state[0] == WALL:
                break
        # 3a) 竖直后先静止保持 1.5s，检验能否贴稳
        cmd.vx = 0.0
        for _ in range(int(1.5 / ts)):
            control_tick(ts)
        print(f"  保持1.5s后：贴墙度={uprightness(data):.3f} y={data.qpos[1]:.3f} "
              f"z={data.qpos[2]:.2f} 吸附={len(magnets.attached_legs())}/6", flush=True)
        # 3b) 再上爬 3s，中途采样
        cmd.vx = MAX_V
        for i in range(int(3.0 / ts)):
            control_tick(ts)
            if i % int(1.0 / ts) == 0:
                print(f"  爬升 t={i*ts:.1f}s 贴墙度={uprightness(data):.3f} "
                      f"z={data.qpos[2]:.2f} 吸附={len(magnets.attached_legs())}/6", flush=True)
        up = uprightness(data)
        print(f"结果：状态={state[0]} 贴墙度={up:.3f} z={data.qpos[2]:.2f}(起点{z0:.2f}) "
              f"|y|={abs(data.qpos[1]):.3f}", flush=True)
        ok = state[0] == WALL and up > UPRIGHT_DONE and data.qpos[2] > z0 + 0.05
        print("自测：通过 ✅" if ok else "自测：未达标 ⚠（需调参 ORI_KP/MOUNT_TIME/Z_WALL）")


if __name__ == "__main__":
    main()
