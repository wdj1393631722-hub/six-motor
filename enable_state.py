#!/usr/bin/env python3
"""
六足使能/失能状态机 — 对齐 TEST-4.0 Cloud-TAC App.c 行为。

实机 12-DOF（每腿 fb+ud）映射到仿真 18-DOF（coxa+femur+tibia）：
  奇数电机 fb → coxa  水平摆幅 ±30°
  偶数电机 ud → femur/tibia  竖直抬腿 24° / 撑起 30°

状态：
  DISABLED   — 无力矩，腿趴地
  LIFT_RAMP  — 使能后 4s smootherstep 撑起（偶轴 ud）
  SOFT_STAND — 0.6s 软站立
  ENABLED    — 允许三角步态
  RAMP_DOWN  — 失能 5s 缓降后无力矩
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple

import mujoco
import numpy as np

# --- TEST-4.0 App.c / RobStride04.c 对齐参数（度） ---
CRAWL_SWING_FB_DEG = 34.0  # R04_CRAWL_SWING_FB_DEG
CRAWL_LIFT_UD_DEG = 24.0   # R04_CRAWL_LIFT_UD_DEG
CRAWL_PUSH_HALF_FB_DEG = 17.0
BODY_LIFT_EVEN_DEG = 30.0
EVEN_ENABLE_RAMP_S = 4.0
POST_ENABLE_SOFT_S = 0.6
EVEN_RAMP_DOWN_S = 5.0

# 腿号 1~6：左上前、左中、左下、右下、右中、右上前（与 TRIPOD_A/B 一致）
# 偶轴撑起方向：腿 1/2/3 femur 减小，腿 4/5/6 femur 增大（对应 m2/m4/m6 ud 下压）
_LEG_UD_LIFT_SIGN = {1: -1.0, 2: -1.0, 3: -1.0, 4: 1.0, 5: 1.0, 6: 1.0}

LEG_JOINTS = ("coxa", "femur", "tibia")


class EnablePhase(Enum):
    DISABLED = auto()
    LIFT_RAMP = auto()
    SOFT_STAND = auto()
    ENABLED = auto()
    RAMP_DOWN = auto()


def _deg2rad(d: float) -> float:
    return math.radians(d)


def smootherstep(u: float) -> float:
    """TEST-4.0 R04_RunEvenRampEnable 同款插值。"""
    u = max(0.0, min(1.0, u))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def _joint_names() -> List[str]:
    return [f"leg{l}_{j}_joint" for l in range(1, 7) for j in LEG_JOINTS]


def read_joint_pose(model: mujoco.MjModel, data: mujoco.MjData) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for jn in _joint_names():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        out[jn] = float(data.qpos[model.jnt_qposadr[jid]])
    return out


def _leg_femur_tibia_split(ud_delta_rad: float) -> Tuple[float, float]:
    """竖直轴偏置分配到 femur / tibia（2:1）。"""
    return ud_delta_rad * 0.65, ud_delta_rad * 0.35


def _apply_ud_delta(
    pose: Dict[str, float],
    leg: int,
    ud_delta_rad: float,
) -> None:
    df, dt = _leg_femur_tibia_split(ud_delta_rad)
    pose[f"leg{leg}_femur_joint"] += df
    pose[f"leg{leg}_tibia_joint"] += dt


@dataclass
class EnableController:
    """使能状态机 + 关节目标生成。"""

    phase: EnablePhase = EnablePhase.DISABLED
    stand_pose: Dict[str, float] = field(default_factory=dict)
    zero_pose: Dict[str, float] = field(default_factory=dict)
    prone_pose: Dict[str, float] = field(default_factory=dict)
    lift_start: Dict[int, float] = field(default_factory=dict)
    lift_target: Dict[int, float] = field(default_factory=dict)
    ramp_down_start: Dict[str, float] = field(default_factory=dict)
    ramp_t: float = 0.0
    soft_remain_s: float = 0.0
    prone_body_z: float = 0.065
    _kp_backup: float = 280.0

    def load_stand(self, stand: Dict[str, float]) -> None:
        self.stand_pose = dict(stand)

    def init_disabled(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        prone: Dict[str, float] | None = None,
    ) -> None:
        """上电：无力矩，记录趴地零位。"""
        self.phase = EnablePhase.DISABLED
        self.ramp_t = 0.0
        self.soft_remain_s = 0.0
        if prone is not None:
            self.prone_pose = dict(prone)
        else:
            self.prone_pose = read_joint_pose(model, data)
        self.prone_body_z = float(data.qpos[2])
        self.zero_pose = dict(self.prone_pose)
        set_actuator_torque_enabled(model, False, self._kp_backup)

    def enable(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Sbus[4] 上升沿：从当前趴地角 4s 插值到标定站立角（物理支撑）。"""
        if self.phase in (EnablePhase.ENABLED, EnablePhase.LIFT_RAMP, EnablePhase.SOFT_STAND):
            return
        if not self.stand_pose:
            self.stand_pose = read_joint_pose(model, data)
        self.zero_pose = read_joint_pose(model, data)
        self.prone_pose = dict(self.zero_pose)
        self.lift_start.clear()
        self.lift_target.clear()
        self.ramp_t = 0.0
        self.soft_remain_s = 0.0
        self.phase = EnablePhase.LIFT_RAMP
        set_actuator_torque_enabled(model, True, self._kp_backup)
        print("[使能] 开始撑起 (4s smootherstep)…")

    def disable(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """失能：5s 缓降回趴地角。"""
        if self.phase == EnablePhase.DISABLED:
            return
        self.ramp_down_start = read_joint_pose(model, data)
        self.ramp_t = 0.0
        self.phase = EnablePhase.RAMP_DOWN
        print("[失能] 开始缓降 (5s)…")

    @property
    def is_enabled(self) -> bool:
        return self.phase == EnablePhase.ENABLED

    @property
    def allows_gait(self) -> bool:
        return self.phase == EnablePhase.ENABLED

    def step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dt: float,
    ) -> Dict[str, float] | None:
        """
        推进状态机，返回本帧关节目标；DISABLED 返回 None（无力矩）。
        """
        if self.phase == EnablePhase.DISABLED:
            set_actuator_torque_enabled(model, False, self._kp_backup)
            return None

        if self.phase == EnablePhase.LIFT_RAMP:
            self.ramp_t += dt
            u = smootherstep(self.ramp_t / EVEN_ENABLE_RAMP_S)
            targets: Dict[str, float] = {}
            for jn in _joint_names():
                s = self.zero_pose.get(jn, self.stand_pose[jn])
                e = self.stand_pose[jn]
                targets[jn] = s + (e - s) * u
            if self.ramp_t >= EVEN_ENABLE_RAMP_S:
                self.phase = EnablePhase.SOFT_STAND
                self.soft_remain_s = POST_ENABLE_SOFT_S
                print("[使能] 软站立 0.6s…")
            return targets

        if self.phase == EnablePhase.SOFT_STAND:
            self.soft_remain_s -= dt
            if self.soft_remain_s <= 0.0:
                self.phase = EnablePhase.ENABLED
                print("[使能] 完成，三角步态已解锁（I/K/J/L）")
            return dict(self.stand_pose)

        if self.phase == EnablePhase.RAMP_DOWN:
            self.ramp_t += dt
            alpha = min(1.0, self.ramp_t / EVEN_RAMP_DOWN_S)
            targets: Dict[str, float] = {}
            for jn in _joint_names():
                s = self.ramp_down_start.get(jn, 0.0)
                e = self.prone_pose.get(jn, s)
                if jn.endswith("_coxa_joint"):
                    targets[jn] = s
                else:
                    targets[jn] = s + (e - s) * alpha
            if alpha >= 1.0:
                self.phase = EnablePhase.DISABLED
                set_actuator_torque_enabled(model, False, self._kp_backup)
                print("[失能] 完成，已进入无力矩状态")
                return None
            return targets

        return dict(self.stand_pose)


STAND_KP = 280.0
STAND_KV = 12.0
LOCOMOTION_KP = 130.0
LOCOMOTION_KV = 30.0


def set_actuator_gains(
    model: mujoco.MjModel,
    kp: float,
    kv: float,
) -> None:
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kv


def set_actuator_torque_enabled(
    model: mujoco.MjModel,
    enabled: bool,
    kp_nominal: float = STAND_KP,
    kv_nominal: float = STAND_KV,
) -> None:
    """仿真 MIT 使能/失能：失能时 kp=0 无力矩。"""
    if enabled:
        set_actuator_gains(model, kp_nominal, kv_nominal)
    else:
        set_actuator_gains(model, 0.0, 0.0)


def make_prone_pose_analytic(stand_pose: Dict[str, float]) -> Dict[str, float]:
    """从站立角生成趴地角初值（偶轴撑起反向，未保证足底贴地）。"""
    prone = dict(stand_pose)
    lift_rad = _deg2rad(BODY_LIFT_EVEN_DEG)
    for leg in range(1, 7):
        sign = _LEG_UD_LIFT_SIGN[leg]
        df, dt = _leg_femur_tibia_split(-sign * lift_rad)
        prone[f"leg{leg}_femur_joint"] += df
        prone[f"leg{leg}_tibia_joint"] += dt
    return prone


def make_prone_pose(stand_pose: Dict[str, float]) -> Dict[str, float]:
    """趴地关节角：优先使用标定结果（六足平面贴地）。"""
    try:
        from foot_kinematics import load_prone_pose

        loaded = load_prone_pose()
        if loaded is not None:
            return loaded[0]
    except ImportError:
        pass
    return make_prone_pose_analytic(stand_pose)


def crawl_gait_scale_from_test40() -> Tuple[float, float]:
    """将 TEST-4.0 摆幅（度）换算为仿真步长/抬脚高度（米）。"""
    # 经验换算：30° 水平摆 ≈ 0.11m 步长，24° 抬腿 ≈ 0.10m（18-DOF 链更长）
    stride = 0.11 * (CRAWL_SWING_FB_DEG / 30.0)
    lift = 0.10 * (CRAWL_LIFT_UD_DEG / 24.0)
    return stride, lift
