#!/usr/bin/env python3
"""
三角步态（Tripod Gait）运动学规划。

两组腿交替支撑/摆动：
  Tripod A: leg1, leg3, leg5
  Tripod B: leg2, leg4, leg6

摆动相：足端在机体坐标系内沿行进方向划摆线（抬脚 + 前后摆）。
支撑相：足端相对机体向后移动（等效于机体前进）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

TRIPOD_A = (1, 3, 5)
TRIPOD_B = (2, 4, 6)


@dataclass
class TripodGaitConfig:
    """三角步态参数。swing_fraction=0.5 表示周期一半摆动、一半支撑。"""

    cycle_time: float = 0.6
    step_height: float = 0.035
    stride_length: float = 0.055
    swing_fraction: float = 0.5
    cmd_deadband: float = 1e-3
    use_radial_stride: bool = True  # 各腿沿髋座方位做前后摆

    @property
    def stance_duty(self) -> float:
        """兼容旧字段名。"""
        return self.swing_fraction


def forward_tripod_config(
    speed_mps: float = 0.08,
) -> TripodGaitConfig:
    """前进三角步态推荐参数（对齐 TEST-4.0：水平 ±30°、竖直 24°）。"""
    try:
        from enable_state import crawl_gait_scale_from_test40

        base_stride, base_lift = crawl_gait_scale_from_test40()
    except ImportError:
        base_stride, base_lift = 0.11, 0.10
    scale = min(max(speed_mps / 0.08, 0.5), 1.3)
    # TEST-4.0 10 步爬行一周期 400 ms（见 test40_crawl.py）
    return TripodGaitConfig(
        cycle_time=0.40,
        step_height=base_lift,
        stride_length=base_stride * scale,
        swing_fraction=0.5,
        use_radial_stride=True,
    )


def swing_lift_profile(u: float, step_h: float) -> float:
    """
    摆动相抬脚高度 u∈[0,1]。
    梯形：起落快、中间保持全高，比正弦更容易看出抬腿。
    """
    if u < 0.12:
        return step_h * (u / 0.12)
    if u < 0.68:
        return step_h
    if u < 1.0:
        return step_h * ((1.0 - u) / 0.32)
    return 0.0


def _unit_velocity(vx: float, vy: float) -> Tuple[float, float, float]:
    speed = math.hypot(vx, vy)
    if speed < 1e-6:
        return 1.0, 0.0, 0.0
    return speed, vx / speed, vy / speed


def _leg_heading_rad(leg: int) -> float:
    try:
        from leg_symmetry import LEG_AZIMUTH_DEG

        return math.radians(LEG_AZIMUTH_DEG.get(leg, 0.0))
    except ImportError:
        return {1: math.radians(60), 2: 0.0, 3: math.radians(-60),
                4: math.radians(-120), 5: math.pi, 6: math.radians(120)}.get(leg, 0.0)


def _stride_vector(
    stride: float,
    dir_xy: Tuple[float, float],
    leg: int,
    radial: bool,
) -> Tuple[float, float]:
    """将步长投影到 base 系 (dx, dy)。"""
    dx, dy = dir_xy
    if not radial:
        return stride * dx, stride * dy
    az = _leg_heading_rad(leg)
    lx, ly = math.cos(az), math.sin(az)
    along = stride * (dx * lx + dy * ly)
    return along * lx, along * ly


def swing_foot_offset(
    u: float,
    stride: float,
    step_h: float,
    dir_xy: Tuple[float, float],
    leg: int = 1,
    radial: bool = True,
) -> np.ndarray:
    """
    摆动相足端偏移 (base 系)，u∈[0,1]。
    落足点从后向前，正弦抬脚。
    """
    sx, sy = _stride_vector(stride, dir_xy, leg, radial)
    along = u - 0.5
    lift = swing_lift_profile(u, step_h)
    return np.array([along * sx * 2.0, along * sy * 2.0, lift])


def stance_foot_offset(
    u: float,
    stride: float,
    dir_xy: Tuple[float, float],
    leg: int = 1,
    radial: bool = True,
) -> np.ndarray:
    """
    支撑相足端相对机体位移 (base 系)，u∈[0,1]。
    足端从前往后扫，推动机体沿指令方向前进。
    """
    sx, sy = _stride_vector(stride, dir_xy, leg, radial)
    along = 0.5 - u
    return np.array([along * sx * 2.0, along * sy * 2.0, 0.0])


def turn_foot_offset(omega: float, leg: int, scale: float = 0.02) -> np.ndarray:
    """绕机体 Z 转向时，按腿方位角给切向偏移。"""
    az = _leg_heading_rad(leg)
    # 左转 (omega>0)：外侧腿足端略向前/内侧略向后
    tang = np.array([-math.sin(az), math.cos(az), 0.0])
    return tang * (omega * scale)


__all__ = [
    "TRIPOD_A",
    "TRIPOD_B",
    "TripodGaitConfig",
    "TripodFootPlanner",
    "forward_tripod_config",
    "swing_foot_offset",
    "stance_foot_offset",
]


class TripodFootPlanner:
    """输出每条腿在 base_link 系下的足端目标位置。"""

    def __init__(self, nominal_feet_base: Dict[int, np.ndarray], cfg: TripodGaitConfig | None = None):
        self.nominal = {k: np.asarray(v, dtype=float).copy() for k, v in nominal_feet_base.items()}
        self.cfg = cfg or TripodGaitConfig()
        self.t = 0.0

    def reset(self) -> None:
        self.t = 0.0

    def step(
        self,
        dt: float,
        vx: float = 0.0,
        vy: float = 0.0,
        omega: float = 0.0,
        yaw: float = 0.0,
    ) -> Dict[int, np.ndarray]:
        """
        返回 {leg_id: foot_target_base}。
        yaw: 机体绕 Z 偏航 (rad)，用于随转弯微调足端（可选）。
        """
        speed, ux, uy = _unit_velocity(vx, vy)
        moving = speed > self.cfg.cmd_deadband or abs(omega) > self.cfg.cmd_deadband
        if not moving:
            return {leg: self.nominal[leg].copy() for leg in range(1, 7)}

        self.t += dt
        phase = (self.t % self.cfg.cycle_time) / self.cfg.cycle_time
        sf = self.cfg.swing_fraction
        swing_a = phase < sf

        stride = self.cfg.stride_length * min(speed / 0.08, 1.0)
        if stride < 1e-4 and abs(omega) > self.cfg.cmd_deadband:
            stride = self.cfg.stride_length * 0.35

        dir_xy = (ux, uy)
        radial = self.cfg.use_radial_stride
        out: Dict[int, np.ndarray] = {}

        for leg in range(1, 7):
            in_a = leg in TRIPOD_A
            swinging = swing_a if in_a else not swing_a

            if swinging:
                # A 在 [0, sf) 摆动；B 在 [sf, 1) 摆动，须分别归一化到 u∈[0,1]
                if in_a:
                    local_u = phase / max(sf, 1e-6)
                else:
                    local_u = (phase - sf) / max(sf, 1e-6)
                local_u = max(0.0, min(1.0, local_u))
                delta = swing_foot_offset(
                    local_u,
                    stride,
                    self.cfg.step_height,
                    dir_xy,
                    leg=leg,
                    radial=radial,
                )
            else:
                local_u = (phase - sf) / max(1e-6, 1.0 - sf)
                local_u = max(0.0, min(1.0, local_u))
                delta = stance_foot_offset(
                    local_u, stride, dir_xy, leg=leg, radial=radial
                )

            delta += turn_foot_offset(omega, leg)
            # 轻微偏航补偿：前进方向旋转足端目标
            if abs(yaw) > 1e-4:
                c, s = math.cos(yaw), math.sin(yaw)
                rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                delta = rot @ delta

            out[leg] = self.nominal[leg] + delta

        return out
