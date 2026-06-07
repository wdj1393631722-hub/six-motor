#!/usr/bin/env python3
"""
三角步态 — 足端槽位（中/前/后）+ 机身推进。

单周期（以 A 组 1/3/5 摆动为例）:
  1. 摆动腿抬起 → 前移到前方槽位 → 落地
  2. 六足定锚，机身运动学前进（支撑足不滑移）
  3. 摆动腿回中间槽位，原支撑腿移到后方槽位
  4. B 组 (2/4/6) 镜像重复
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from tripod_planner import TRIPOD_A, TRIPOD_B


class PhaseKind(Enum):
    SWING_LIFT = auto()
    SWING_FORWARD = auto()
    SWING_PLACE = auto()
    BODY_ADVANCE = auto()
    REPOSITION = auto()


@dataclass
class PhaseStep:
    kind: PhaseKind
    duration_s: float
    swing_group: str  # "A" or "B"


@dataclass
class SlotGaitConfig:
    step_length: float = 0.04
    step_height: float = 0.03
    phase_slowdown: float = 2.0
    # 各相时长 (s)
    lift_s: float = 0.7
    swing_fwd_s: float = 0.9
    place_s: float = 0.6
    body_advance_s: float = 1.0
    reposition_s: float = 0.9


@dataclass
class GaitStepOutput:
    """单步输出：关节目标 + 机身 x + 需锁定世界坐标的支撑足。"""

    joints: Dict[str, float]
    body_x: float
    stance_world_lock: Dict[int, np.ndarray] = field(default_factory=dict)
    phase_name: str = ""
    kinematic_only: bool = False


def _stride_delta(distance: float) -> np.ndarray:
    """沿机体前进方向 (+X) 的步进偏移。"""
    return np.array([distance, 0.0, 0.0])


def _lerp3(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    return a + (b - a) * max(0.0, min(1.0, u))


def _smooth(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


class TripodSlotGait:
    """
    足端槽位三角步态规划器。

    摆动相：另一组支撑足世界坐标锚定不动。
    机身推进相：六足世界锚定，仅机身前进。
    """

    def __init__(self, ik, stand_pose: Dict[str, float], cfg: SlotGaitConfig | None = None):
        self.ik = ik
        self.stand_pose = dict(stand_pose)
        self.cfg = cfg or SlotGaitConfig()
        self.body_x = 0.0
        self._phase_idx = 0
        self._phase_t = 0.0
        self._anchors_world: Dict[int, np.ndarray] = {}
        self._body_adv_start_x = 0.0
        self._body_adv_armed = False
        self._swing_stance_armed = False
        self._half = 0  # 0=A swing, 1=B swing
        self._last_joints = {
            leg: self.ik.stand[leg].joints for leg in range(1, 7)
        }
        self._steps = self._build_steps()
        self._cycle_time = sum(s.duration_s for s in self._steps)

    def _build_steps(self) -> List[PhaseStep]:
        c = self.cfg
        s = c.phase_slowdown
        return [
            PhaseStep(PhaseKind.SWING_LIFT, c.lift_s * s, "A"),
            PhaseStep(PhaseKind.SWING_FORWARD, c.swing_fwd_s * s, "A"),
            PhaseStep(PhaseKind.SWING_PLACE, c.place_s * s, "A"),
            PhaseStep(PhaseKind.BODY_ADVANCE, c.body_advance_s * s, "A"),
            PhaseStep(PhaseKind.REPOSITION, c.reposition_s * s, "A"),
            PhaseStep(PhaseKind.SWING_LIFT, c.lift_s * s, "B"),
            PhaseStep(PhaseKind.SWING_FORWARD, c.swing_fwd_s * s, "B"),
            PhaseStep(PhaseKind.SWING_PLACE, c.place_s * s, "B"),
            PhaseStep(PhaseKind.BODY_ADVANCE, c.body_advance_s * s, "B"),
            PhaseStep(PhaseKind.REPOSITION, c.reposition_s * s, "B"),
        ]

    def reset(self) -> None:
        self.body_x = 0.0
        self._phase_idx = 0
        self._phase_t = 0.0
        self._anchors_world.clear()
        self._body_adv_start_x = 0.0
        self._body_adv_armed = False
        self._swing_stance_armed = False
        self._half = 0
        self._last_joints = {
            leg: self.ik.stand[leg].joints for leg in range(1, 7)
        }

    def _group_legs(self, group: str) -> Tuple[int, ...]:
        return TRIPOD_A if group == "A" else TRIPOD_B

    def _stance_legs(self, group: str) -> Tuple[int, ...]:
        return self._group_legs("B" if group == "A" else "A")

    def _slot_center(self, leg: int) -> np.ndarray:
        return self.ik.stand[leg].foot_base.copy()

    def _slot_forward(self, leg: int) -> np.ndarray:
        return self._slot_center(leg) + _stride_delta(self.cfg.step_length * 0.5)

    def _slot_rear(self, leg: int) -> np.ndarray:
        return self._slot_center(leg) - _stride_delta(self.cfg.step_length * 0.5)

    def _capture_anchors_world(
        self, legs: Tuple[int, ...], sim_data=None
    ) -> None:
        """记录支撑足当前世界坐标（摆动相起锚）。"""
        import mujoco
        from foot_kinematics import foot_world

        model = self.ik.model
        data = sim_data if sim_data is not None else self.ik.data
        if sim_data is None:
            data.qpos[0] = self.body_x
            mujoco.mj_forward(model, data)
        for leg in legs:
            p, _ = foot_world(model, data, leg, self.ik.frames)
            self._anchors_world[leg] = p.copy()

    def _foot_target_base(
        self, leg: int, slot: np.ndarray, lift: float = 0.0
    ) -> np.ndarray:
        t = slot.copy()
        t[2] += lift
        return t

    def _phase_foot_targets(
        self, step: PhaseStep, u: float, sim_data=None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], float, bool]:
        """
        返回 (swing_targets, stance_world_lock, body_x)。
        stance_world_lock 非空时表示该腿世界锚定。
        """
        swing = self._group_legs(step.swing_group)
        stance = self._stance_legs(step.swing_group)
        swing_t: Dict[int, np.ndarray] = {}
        lock: Dict[int, np.ndarray] = {}
        body_x = self.body_x
        h = self.cfg.step_height
        su = _smooth(u)
        # 全程运动学：足端世界锚定 + 机身 x 推进，避免物理步进把机身推回原地
        kinematic_only = True

        if step.kind == PhaseKind.SWING_LIFT:
            if not self._swing_stance_armed:
                self._capture_anchors_world(stance, sim_data)
                self._swing_stance_armed = True
            for leg in swing:
                c = self._slot_center(leg)
                swing_t[leg] = self._foot_target_base(leg, c, h * su)
            for leg in stance:
                if leg in self._anchors_world:
                    lock[leg] = self._anchors_world[leg].copy()

        elif step.kind == PhaseKind.SWING_FORWARD:
            for leg in swing:
                c = self._slot_center(leg)
                f = self._slot_forward(leg)
                swing_t[leg] = self._foot_target_base(leg, _lerp3(c, f, su), h)
            for leg in stance:
                if leg in self._anchors_world:
                    lock[leg] = self._anchors_world[leg].copy()

        elif step.kind == PhaseKind.SWING_PLACE:
            for leg in swing:
                f = self._slot_forward(leg)
                swing_t[leg] = self._foot_target_base(leg, f, h * (1.0 - su))
            for leg in stance:
                if leg in self._anchors_world:
                    lock[leg] = self._anchors_world[leg].copy()

        elif step.kind == PhaseKind.BODY_ADVANCE:
            if not self._body_adv_armed or len(self._anchors_world) < 6:
                self._body_adv_start_x = self.body_x
                self._capture_anchors_world(tuple(range(1, 7)), sim_data)
                self._body_adv_armed = True
            body_x = self._body_adv_start_x + self.cfg.step_length * su
            for leg in range(1, 7):
                lock[leg] = self._anchors_world[leg].copy()
                swing_t[leg] = self._world_to_base_at(body_x, lock[leg])

        elif step.kind == PhaseKind.REPOSITION:
            swing_grp = step.swing_group
            for leg in self._group_legs(swing_grp):
                f = self._slot_forward(leg)
                c = self._slot_center(leg)
                swing_t[leg] = self._foot_target_base(leg, _lerp3(f, c, su))
            for leg in self._stance_legs(swing_grp):
                c = self._slot_center(leg)
                r = self._slot_rear(leg)
                swing_t[leg] = self._foot_target_base(leg, _lerp3(c, r, su))

        return swing_t, lock, body_x, kinematic_only

    def _world_to_base_at(self, body_x: float, p_world: np.ndarray) -> np.ndarray:
        """固定机体位姿 (body_x, 0, z) 下世界点 → base 系。"""
        import mujoco

        model = self.ik.model
        data = self.ik.data
        data.qpos[0] = body_x
        data.qpos[1] = 0.0
        data.qpos[2] = self.ik.body_height
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        mujoco.mj_forward(model, data)
        bid = self.ik.base_id
        R = data.xmat[bid].reshape(3, 3)
        base_pos = data.xpos[bid]
        return R.T @ (p_world - base_pos)

    def step(
        self, dt: float, speed_scale: float = 1.0, sim_data=None
    ) -> GaitStepOutput:
        if not self._steps:
            return GaitStepOutput(joints=dict(self.stand_pose), body_x=self.body_x)

        rate = max(0.2, min(speed_scale, 1.0))
        self._phase_t += dt * rate
        step = self._steps[self._phase_idx]

        while self._phase_t >= step.duration_s:
            self._phase_t -= step.duration_s
            if step.kind == PhaseKind.BODY_ADVANCE:
                self.body_x = self._body_adv_start_x + self.cfg.step_length
            self._phase_idx = (self._phase_idx + 1) % len(self._steps)
            step = self._steps[self._phase_idx]
            if step.kind == PhaseKind.SWING_LIFT:
                self._anchors_world.clear()
                self._swing_stance_armed = False
            elif step.kind == PhaseKind.BODY_ADVANCE:
                self._body_adv_armed = False

        u = self._phase_t / max(step.duration_s, 1e-9)
        swing_t, lock, body_x, kinematic_only = self._phase_foot_targets(
            step, u, sim_data
        )

        # 合并目标：摆动相只动 swing 腿；支撑相由 lock 或 swing_t 决定
        all_foot: Dict[int, np.ndarray] = {}
        for leg in range(1, 7):
            if leg in swing_t:
                all_foot[leg] = swing_t[leg]
            elif leg in lock:
                all_foot[leg] = self._world_to_base_at(body_x, lock[leg])
            else:
                all_foot[leg] = self._slot_center(leg)

        joints: Dict[str, float] = {}
        for leg in range(1, 7):
            seed = self._last_joints[leg]
            c, f, t = self.ik.solve(leg, all_foot[leg], seed=seed)
            self._last_joints[leg] = (c, f, t)
            joints[f"leg{leg}_coxa_joint"] = c
            joints[f"leg{leg}_femur_joint"] = f
            joints[f"leg{leg}_tibia_joint"] = t

        name = f"{step.swing_group}_{step.kind.name}"
        return GaitStepOutput(
            joints=joints,
            body_x=body_x,
            stance_world_lock=lock,
            phase_name=name,
            kinematic_only=kinematic_only,
        )
