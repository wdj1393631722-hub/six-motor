#!/usr/bin/env python3
"""
关节空间三角步态 — 1/3/5 与 2/4/6 交替，优化为连续平滑前进。

腿号: 1 左前 | 2 左中 | 3 左后 | 4 右后 | 5 右中 | 6 右前
前进轴: base_link +Y（1/6 在前方）

半周期三相位（摆动组）:
  swing  — 抬腿 + coxa 前摆（单相连续）
  place  — 落地
  push   — 六腿同步前蹬，摆动组回中、支撑组到后位
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from enable_state import _leg_femur_tibia_split
from tripod_planner import TRIPOD_A, TRIPOD_B

# --- 幅度（度）---
LIFT_UD_DEG = 28.0
SWING_FWD_DEG = 38.0
STANCE_REAR_DEG = 32.0
PUSH_FWD_DEG = 22.0

# 单半周期时长 (s)：swing / place / push（缩短 → 步频更快）
PHASE_SWING_S = 0.40
PHASE_PLACE_S = 0.12
PHASE_PUSH_S = 0.30
PHASE_SLOWDOWN = 1.2
HALF_CYCLE_BODY_Y = 0.058

# 关节指令一阶平滑时间常数 (s)，越小越跟手、越大越柔
JOINT_SMOOTH_TAU = 0.09

BODY_FORWARD_AXIS = "Y"

_FWD_SIGN: Dict[int, float] = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: -1.0,
    5: -1.0,
    6: -1.0,
}

_LIFT_UD_SIGN: Dict[int, float] = {leg: -1.0 for leg in range(1, 7)}

LegOffset = Tuple[float, float]
TripodGroup = str  # "A" or "B"


def _o(fb: float, ud: float) -> LegOffset:
    return (fb, ud)


def _z() -> LegOffset:
    return (0.0, 0.0)


def _fwd(leg: int, deg: float) -> float:
    return _FWD_SIGN[leg] * deg


def _rear(leg: int, deg: float) -> float:
    return -_fwd(leg, deg)


def _lift_ud(leg: int, deg: float = LIFT_UD_DEG) -> float:
    return _LIFT_UD_SIGN[leg] * deg


def _ease(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def _lift_profile(u: float, height: float) -> float:
    """摆动相抬脚：起落平滑，中间最高。"""
    u = max(0.0, min(1.0, u))
    return height * math.sin(math.pi * _ease(u))


def _group_legs(group: TripodGroup) -> Tuple[int, ...]:
    return TRIPOD_A if group == "A" else TRIPOD_B


def _stance_legs(group: TripodGroup) -> Tuple[int, ...]:
    return _group_legs("B" if group == "A" else "A")


@dataclass
class MacroPhase:
    name: str
    duration_s: float
    kind: str  # swing | place | push
    group: TripodGroup


def build_macro_phases() -> List[MacroPhase]:
    s = PHASE_SLOWDOWN
    return [
        MacroPhase("A_swing", PHASE_SWING_S * s, "swing", "A"),
        MacroPhase("A_place", PHASE_PLACE_S * s, "place", "A"),
        MacroPhase("A_push", PHASE_PUSH_S * s, "push", "A"),
        MacroPhase("B_swing", PHASE_SWING_S * s, "swing", "B"),
        MacroPhase("B_place", PHASE_PLACE_S * s, "place", "B"),
        MacroPhase("B_push", PHASE_PUSH_S * s, "push", "B"),
    ]


def _stance_prep_fraction(phase_kind: str, u: float) -> float:
    """对方组 swing+place 内，支撑腿 0→后位 的归一化进度。"""
    total = PHASE_SWING_S + PHASE_PLACE_S
    if phase_kind == "swing":
        t = u * PHASE_SWING_S
    elif phase_kind == "place":
        t = PHASE_SWING_S + u * PHASE_PLACE_S
    else:
        return 1.0
    return _ease(t / total)


def _stance_hold_during_swing_place(
    group: TripodGroup,
    phase_kind: str,
    u: float,
    first_cycle: bool,
) -> Dict[int, LegOffset]:
    """对方组摆动/落地时，本组支撑腿由中位平滑后撤到位。"""
    out: Dict[int, LegOffset] = {}
    prep = _stance_prep_fraction(phase_kind, u)
    for leg in _stance_legs(group):
        if first_cycle and group == "A":
            out[leg] = _z()
        else:
            out[leg] = _o(_rear(leg, STANCE_REAR_DEG * prep), 0.0)
    return out


def _offsets_swing(
    group: TripodGroup, u: float, first_cycle: bool = False
) -> Dict[int, LegOffset]:
    swing = _group_legs(group)
    out = {leg: _z() for leg in range(1, 7)}
    su = _ease(u)
    rear_start = (
        0.0 if (first_cycle and group == "A") else STANCE_REAR_DEG
    )
    for leg in swing:
        fb = _rear(leg, rear_start * (1.0 - su)) + _fwd(leg, SWING_FWD_DEG * su)
        ud = _lift_profile(u, LIFT_UD_DEG)
        out[leg] = _o(fb, ud)
    out.update(_stance_hold_during_swing_place(group, "swing", u, first_cycle))
    return out


def _offsets_place(
    group: TripodGroup, u: float, first_cycle: bool = False
) -> Dict[int, LegOffset]:
    swing = _group_legs(group)
    out = {leg: _z() for leg in range(1, 7)}
    su = _ease(u)
    for leg in swing:
        fb = _fwd(leg, SWING_FWD_DEG)
        ud = _lift_profile(1.0 - su, LIFT_UD_DEG)
        out[leg] = _o(fb, ud)
    out.update(_stance_hold_during_swing_place(group, "place", u, first_cycle))
    return out


def _offsets_push(
    group: TripodGroup, u: float, first_cycle: bool = False
) -> Dict[int, LegOffset]:
    """六腿前蹬；摆动组由前位回中，支撑组保持/到达后位。"""
    swing = _group_legs(group)
    su = _ease(u)
    push_wave = PUSH_FWD_DEG * math.sin(math.pi * su)
    out: Dict[int, LegOffset] = {}
    for leg in range(1, 7):
        if leg in swing:
            fb = _fwd(leg, SWING_FWD_DEG * (1.0 - su))
        elif first_cycle and group == "A":
            fb = _rear(leg, STANCE_REAR_DEG * su)
        else:
            fb = _rear(leg, STANCE_REAR_DEG)
        fb += _fwd(leg, push_wave)
        out[leg] = _o(fb, 0.0)
    return out


def offsets_for_phase(
    phase: MacroPhase, u: float, first_cycle: bool = False
) -> Dict[int, LegOffset]:
    if phase.kind == "swing":
        return _offsets_swing(phase.group, u, first_cycle)
    if phase.kind == "place":
        return _offsets_place(phase.group, u, first_cycle)
    return _offsets_push(phase.group, u, first_cycle)


def offsets_to_joints(
    stand: Dict[str, float],
    offsets: Dict[int, LegOffset],
) -> Dict[str, float]:
    out = dict(stand)
    for leg in range(1, 7):
        fb, ud = offsets.get(leg, (0.0, 0.0))
        out[f"leg{leg}_coxa_joint"] += math.radians(fb)
        if abs(ud) > 1e-6:
            ud_rad = math.radians(_LIFT_UD_SIGN[leg] * abs(ud))
            df, dt = _leg_femur_tibia_split(ud_rad)
            out[f"leg{leg}_femur_joint"] += df
            out[f"leg{leg}_tibia_joint"] += dt
    return out


class JointTripodCrawlPlanner:
    """平滑关节三角步态。"""

    def __init__(
        self,
        stand_pose: Dict[str, float],
        phase_slowdown: float = PHASE_SLOWDOWN,
        smooth_tau: float = JOINT_SMOOTH_TAU,
    ):
        self.stand = dict(stand_pose)
        self.phase_slowdown = phase_slowdown
        self.smooth_tau = smooth_tau
        self.t = 0.0
        self._phases = build_macro_phases()
        self._durs = [p.duration_s for p in self._phases]
        self.cycle_time = sum(self._durs)
        self._prev_joints: Dict[str, float] = dict(stand_pose)
        self._last_dt = 0.02
        self.body_y = 0.0
        self.last_body_y = 0.0

    def reset(self) -> None:
        self.t = 0.0
        self._prev_joints = dict(self.stand)
        self.body_y = 0.0
        self.last_body_y = 0.0

    def _phase_index(self, phase_t: float) -> Tuple[int, float]:
        acc = 0.0
        for i, dur in enumerate(self._durs):
            if phase_t < acc + dur:
                return i, (phase_t - acc) / max(dur, 1e-9)
            acc += dur
        return len(self._durs) - 1, 1.0

    def _body_y_in_cycle(self, phase_t: float, direction: float) -> float:
        sign = 1.0 if direction >= 0 else -1.0
        acc = 0.0
        y = 0.0
        for phase, dur in zip(self._phases, self._durs):
            if phase.kind != "push":
                acc += dur
                continue
            if phase_t < acc:
                break
            if phase_t >= acc + dur:
                y += HALF_CYCLE_BODY_Y
                acc += dur
                continue
            u = (phase_t - acc) / max(dur, 1e-9)
            y += HALF_CYCLE_BODY_Y * _ease(u)
            break
        else:
            if phase_t >= acc:
                pass
        return sign * y

    def _raw_joints(
        self, phase_t: float, direction: float = 1.0, first_cycle: bool = False
    ) -> Dict[str, float]:
        idx, u = self._phase_index(phase_t)
        phase = self._phases[idx]
        offs = offsets_for_phase(
            phase, max(0.0, min(1.0, u)), first_cycle=first_cycle
        )
        sign = 1.0 if direction >= 0 else -1.0
        if sign < 0:
            offs = {
                leg: (fb * sign, ud) for leg, (fb, ud) in offs.items()
            }
        return offsets_to_joints(self.stand, offs)

    def _smooth_joints(
        self, target: Dict[str, float], dt: float
    ) -> Dict[str, float]:
        if dt <= 0:
            return dict(target)
        alpha = 1.0 - math.exp(-dt / max(self.smooth_tau, 1e-4))
        out = dict(self._prev_joints)
        for jn, val in target.items():
            prev = out.get(jn, val)
            out[jn] = prev + alpha * (val - prev)
        self._prev_joints = out
        return out

    def step(
        self,
        dt: float,
        speed_scale: float = 1.0,
        direction: float = 1.0,
        physics: bool = True,
    ) -> Dict[str, float]:
        rate = max(0.45, min(speed_scale, 2.2))
        self._last_dt = dt
        self.t += dt * rate
        phase_t = self.t % self.cycle_time
        cycles = int(self.t // self.cycle_time)
        raw = self._raw_joints(phase_t, direction, first_cycle=(cycles == 0))
        if physics:
            self.body_y = 0.0
            self.last_body_y = 0.0
        else:
            target_y = (
                cycles * 2.0 * HALF_CYCLE_BODY_Y * (1.0 if direction >= 0 else -1.0)
                + self._body_y_in_cycle(phase_t, direction)
            )
            alpha = 1.0 - math.exp(-dt / max(self.smooth_tau, 1e-4))
            self.body_y += alpha * (target_y - self.body_y)
            self.last_body_y = self.body_y
        return self._smooth_joints(raw, dt)

    def current_phase(self) -> MacroPhase:
        idx, _ = self._phase_index(self.t % self.cycle_time)
        return self._phases[idx]

    def current_phase_name(self) -> str:
        return self.current_phase().name


# 兼容旧接口
FORWARD_JOINT_STEPS = build_macro_phases()
CYCLE_TIME_S = sum(p.duration_s for p in FORWARD_JOINT_STEPS)


def build_forward_joint_steps():
    return build_macro_phases()


__all__ = [
    "JointTripodCrawlPlanner",
    "build_forward_joint_steps",
    "build_macro_phases",
    "FORWARD_JOINT_STEPS",
    "BODY_FORWARD_AXIS",
    "CYCLE_TIME_S",
    "TRIPOD_A",
    "TRIPOD_B",
    "LIFT_UD_DEG",
    "SWING_FWD_DEG",
]
