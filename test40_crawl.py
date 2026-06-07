#!/usr/bin/env python3
"""
TEST-4.0 前进爬行步态表 — 对齐 App.c R04_ForwardCrawlSteps。

实机 12-DOF（fb 奇轴 / ud 偶轴）→ 仿真 18-DOF（coxa / femur+tibia）。
10 步周期 400 ms：A 组(1/3/5) ①–③ → B 组(2/4/6) ④–⑥。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from enable_state import _leg_femur_tibia_split

# App.c 宏
CRAWL_LIFT_UD_DEG = 24.0
CRAWL_LIFT_STANCE_FB_DEG = 28.0
CRAWL_SWING_FB_DEG = 30.0  # 步态表实际 ±30°（宏 34° 未引用）
CRAWL_PUSH_HALF_FB_DEG = 15.0

LegOffset = Tuple[float, float]  # (fb_deg, ud_deg) 相对站立零位


def _o(fb: float, ud: float) -> LegOffset:
    return (fb, ud)


def _full(
    l1: LegOffset,
    l2: LegOffset,
    l3: LegOffset,
    l4: LegOffset,
    l5: LegOffset,
    l6: LegOffset,
) -> Dict[int, LegOffset]:
    return {1: l1, 2: l2, 3: l3, 4: l4, 5: l5, 6: l6}


@dataclass
class CrawlStepDef:
    """单步末端偏置（度）；步内线性插值。"""

    name: str
    duration_ms: int
    ends: Dict[int, LegOffset] = field(default_factory=dict)


# 来源：App.c R04_ForwardCrawlSteps（步 ①②a 原文确认，②b–⑥b 按相位+镜像推断）
FORWARD_CRAWL_STEPS: List[CrawlStepDef] = [
    CrawlStepDef(
        "1_A_lift",
        26,
        _full(
            _o(-CRAWL_SWING_FB_DEG, CRAWL_LIFT_UD_DEG),
            _o(0, 0),
            _o(-CRAWL_SWING_FB_DEG, CRAWL_LIFT_UD_DEG),
            _o(0, 0),
            _o(CRAWL_SWING_FB_DEG, -CRAWL_LIFT_UD_DEG),
            _o(0, 0),
        ),
    ),
    CrawlStepDef(
        "2a_A_swing",
        70,
        _full(
            _o(CRAWL_SWING_FB_DEG, CRAWL_LIFT_UD_DEG),
            _o(0, 0),
            _o(CRAWL_SWING_FB_DEG, CRAWL_LIFT_UD_DEG),
            _o(0, 0),
            _o(-CRAWL_SWING_FB_DEG, -CRAWL_LIFT_UD_DEG),
            _o(0, 0),
        ),
    ),
    CrawlStepDef(
        "2b_A_land",
        66,
        _full(
            _o(CRAWL_SWING_FB_DEG, 0),
            _o(0, 0),
            _o(CRAWL_SWING_FB_DEG, 0),
            _o(0, 0),
            _o(-CRAWL_SWING_FB_DEG, 0),
            _o(0, 0),
        ),
    ),
    CrawlStepDef(
        "3a_A_push_mid",
        19,
        _full(
            _o(CRAWL_PUSH_HALF_FB_DEG, 0),
            _o(0, 0),
            _o(CRAWL_PUSH_HALF_FB_DEG, 0),
            _o(0, 0),
            _o(-CRAWL_PUSH_HALF_FB_DEG, 0),
            _o(0, 0),
        ),
    ),
    CrawlStepDef(
        "3b_A_push_end",
        19,
        _full(
            _o(0, 0),
            _o(-CRAWL_SWING_FB_DEG, 0),
            _o(0, 0),
            _o(-CRAWL_SWING_FB_DEG, 0),
            _o(0, 0),
            _o(-CRAWL_SWING_FB_DEG, 0),
        ),
    ),
    CrawlStepDef(
        "4_B_lift",
        26,
        _full(
            _o(CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(-CRAWL_SWING_FB_DEG, CRAWL_LIFT_UD_DEG),
            _o(CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(-CRAWL_SWING_FB_DEG, CRAWL_LIFT_UD_DEG),
            _o(-CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(CRAWL_SWING_FB_DEG, -CRAWL_LIFT_UD_DEG),
        ),
    ),
    CrawlStepDef(
        "5a_B_swing",
        70,
        _full(
            _o(CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(CRAWL_SWING_FB_DEG, CRAWL_LIFT_UD_DEG),
            _o(CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(CRAWL_SWING_FB_DEG, CRAWL_LIFT_UD_DEG),
            _o(-CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(-CRAWL_SWING_FB_DEG, -CRAWL_LIFT_UD_DEG),
        ),
    ),
    CrawlStepDef(
        "5b_B_land",
        66,
        _full(
            _o(CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(CRAWL_SWING_FB_DEG, 0),
            _o(CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(CRAWL_SWING_FB_DEG, 0),
            _o(-CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(-CRAWL_SWING_FB_DEG, 0),
        ),
    ),
    CrawlStepDef(
        "6a_B_push_mid",
        19,
        _full(
            _o(CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(CRAWL_PUSH_HALF_FB_DEG, 0),
            _o(CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(CRAWL_PUSH_HALF_FB_DEG, 0),
            _o(-CRAWL_LIFT_STANCE_FB_DEG, 0),
            _o(-CRAWL_PUSH_HALF_FB_DEG, 0),
        ),
    ),
    CrawlStepDef(
        "6b_B_push_end",
        19,
        _full(
            _o(-CRAWL_SWING_FB_DEG, 0),
            _o(0, 0),
            _o(-CRAWL_SWING_FB_DEG, 0),
            _o(0, 0),
            _o(CRAWL_SWING_FB_DEG, 0),
            _o(0, 0),
        ),
    ),
]

CYCLE_TIME_S = sum(s.duration_ms for s in FORWARD_CRAWL_STEPS) / 1000.0  # 0.4 s（实机表值）

# 仿真相位减速：实机 400 ms/周期在 MuJoCo 中过快，放大有效周期
CRAWL_PHASE_SLOWDOWN = 8.0


def _merge_ends(step: CrawlStepDef, carry: Dict[int, LegOffset]) -> Dict[int, LegOffset]:
    out = dict(carry)
    out.update(step.ends)
    return out


def _build_step_chain() -> Tuple[List[Dict[int, LegOffset]], List[float]]:
    """每步起点→终点与时长（秒）。"""
    zero = {leg: (0.0, 0.0) for leg in range(1, 7)}
    starts: List[Dict[int, LegOffset]] = []
    durs: List[float] = []
    carry = dict(zero)
    for i, step in enumerate(FORWARD_CRAWL_STEPS):
        starts.append(dict(carry))
        carry = _merge_ends(step, carry)
        durs.append(step.duration_ms / 1000.0)
    return starts, durs


def offsets_to_joints(
    stand: Dict[str, float],
    offsets: Dict[int, LegOffset],
) -> Dict[str, float]:
    """12-DOF 偏置 → 18 关节角。"""
    out = dict(stand)
    for leg in range(1, 7):
        fb, ud = offsets.get(leg, (0.0, 0.0))
        out[f"leg{leg}_coxa_joint"] += math.radians(fb)
        ud_rad = math.radians(ud)
        df, dt = _leg_femur_tibia_split(ud_rad)
        out[f"leg{leg}_femur_joint"] += df
        out[f"leg{leg}_tibia_joint"] += dt
    return out


def _lerp_offset(a: LegOffset, b: LegOffset, u: float) -> LegOffset:
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


class Test40CrawlPlanner:
    """TEST-4.0 10 步离散爬行规划器（关节空间）。"""

    def __init__(self, stand_pose: Dict[str, float]):
        self.stand = dict(stand_pose)
        self.t = 0.0
        self._starts, self._durs = _build_step_chain()
        self._ends = [
            _merge_ends(FORWARD_CRAWL_STEPS[i], self._starts[i])
            for i in range(len(FORWARD_CRAWL_STEPS))
        ]
        self.cycle_time = CYCLE_TIME_S
        self.phase_slowdown = CRAWL_PHASE_SLOWDOWN

    def reset(self) -> None:
        self.t = 0.0

    def step_index_at(self, phase_t: float) -> Tuple[int, float]:
        acc = 0.0
        for i, dur in enumerate(self._durs):
            if phase_t < acc + dur:
                u = (phase_t - acc) / max(dur, 1e-9)
                return i, max(0.0, min(1.0, u))
            acc += dur
        return len(self._durs) - 1, 1.0

    def offsets_at(self, phase_t: float, direction: float = 1.0) -> Dict[int, LegOffset]:
        """phase_t ∈ [0, cycle_time)；direction=-1 为后退（水平反相）。"""
        idx, u = self.step_index_at(phase_t)
        start = self._starts[idx]
        end = self._ends[idx]
        out: Dict[int, LegOffset] = {}
        sign = 1.0 if direction >= 0 else -1.0
        for leg in range(1, 7):
            lo = _lerp_offset(start[leg], end[leg], u)
            out[leg] = (lo[0] * sign, lo[1])
        return out

    def step(
        self,
        dt: float,
        speed_scale: float = 1.0,
        direction: float = 1.0,
    ) -> Dict[str, float]:
        """
        speed_scale: 相对标称 0.08 m/s 的相位倍率。
        direction: +1 前进，-1 后退。
        """
        rate = max(0.15, min(speed_scale, 1.0)) / max(self.phase_slowdown, 1.0)
        self.t += dt * rate
        phase_t = self.t % self.cycle_time
        offs = self.offsets_at(phase_t, direction)
        return offsets_to_joints(self.stand, offs)

    def current_phase_name(self) -> str:
        phase_t = self.t % self.cycle_time
        idx, _ = self.step_index_at(phase_t)
        return FORWARD_CRAWL_STEPS[idx].name
