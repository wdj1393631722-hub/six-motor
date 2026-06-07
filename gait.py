#!/usr/bin/env python3
"""
三角步态（Tripod Gait）运动学规划 + 平面速度指令。

规划流程：
  1. TripodFootPlanner — 支撑/摆动相足端轨迹（机体坐标系）
  2. HexapodIK — 足端目标 → 关节角
  3. apply_flat_feet — 可选，微调 tibia 使足底平行地面
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from foot_kinematics import apply_flat_feet, load_stand_pose
except ImportError:
    apply_flat_feet = None
    load_stand_pose = None

from leg_ik import HexapodIK
from joint_tripod_gait import JointTripodCrawlPlanner
from robot_limits import clamp_joint_targets
from tripod_planner import (
    TRIPOD_A,
    TRIPOD_B,
    TripodFootPlanner,
    TripodGaitConfig,
    forward_tripod_config,
)

LEG_JOINTS = ["coxa", "femur", "tibia"]


@dataclass
class GaitParams:
    cycle_time: float = 0.6
    step_height: float = 0.035
    stride_length: float = 0.055
    body_height: float = 0.14
    turn_rate_scale: float = 0.35
    cmd_deadband: float = 1e-3
    swing_fraction: float = 0.5
    use_radial_stride: bool = True


CRAWL_NOMINAL_SPEED_MPS = 0.04
SLOT_NOMINAL_SPEED_MPS = 0.04


def forward_gait_params(speed_mps: float = CRAWL_NOMINAL_SPEED_MPS) -> GaitParams:
    """前进三角步态默认参数（与 forward_tripod_config 对齐）。"""
    cfg = forward_tripod_config(speed_mps)
    p = GaitParams(
        cycle_time=cfg.cycle_time,
        step_height=cfg.step_height,
        stride_length=cfg.stride_length,
        swing_fraction=cfg.swing_fraction,
        use_radial_stride=cfg.use_radial_stride,
    )
    p.body_height = default_body_height()
    return p


def leg_joint_names(leg_id: int) -> List[str]:
    return [f"leg{leg_id}_{j}_joint" for j in LEG_JOINTS]


def nominal_stand_pose() -> Dict[str, float]:
    if load_stand_pose is not None:
        loaded = load_stand_pose()
        if loaded is not None:
            return loaded[0]
    try:
        from rl_posture import load_sim_stand_pose

        lifted = load_sim_stand_pose()
        if lifted is not None:
            return lifted[0]
    except ImportError:
        pass
    pose = {}
    defaults = {
        1: (0.275, 0.014, 0.0),
        2: (-0.46, 0.8, 2.2),
        3: (0.084, 0.030, 0.0),
        4: (-0.084, 0.030, 0.0),
        5: (0.006, 0.056, 0.0),
        6: (-0.275, 0.014, 0.0),
    }
    for leg, (c, f, t) in defaults.items():
        pose[f"leg{leg}_coxa_joint"] = c
        pose[f"leg{leg}_femur_joint"] = f
        pose[f"leg{leg}_tibia_joint"] = t
    return pose


def default_body_height() -> float:
    if load_stand_pose is not None:
        loaded = load_stand_pose()
        if loaded is not None:
            return loaded[1]
    try:
        from rl_posture import load_sim_stand_pose

        lifted = load_sim_stand_pose()
        if lifted is not None:
            return lifted[1]
    except ImportError:
        pass
    try:
        from foot_kinematics import LOCOMOTION_BODY_HEIGHT

        return LOCOMOTION_BODY_HEIGHT
    except ImportError:
        pass
    return 0.14


class TripodGait:
    """
    三角步态运动学控制器。

    Tripod A (摆动/支撑交替): leg1, leg3, leg5
    Tripod B: leg2, leg4, leg6
    """

    def __init__(self, params: GaitParams | None = None, model=None):
        self.p = params or GaitParams()
        self.p.body_height = default_body_height()
        self.stand = nominal_stand_pose()
        self.t = 0.0
        self.yaw = 0.0
        self._model = model
        self._ik: Optional[HexapodIK] = None
        self._planner: Optional[TripodFootPlanner] = None
        self._flat_frames = None
        self._flat_data = None
        self._last_joints: Dict[int, tuple] = {}
        self._slot = None
        self._joint_crawl: Optional[JointTripodCrawlPlanner] = None
        self.use_slot_gait = False
        self.use_joint_gait = True
        self.use_physics_gait = True
        self._joint_stance = None
        self.last_phase = ""
        self.last_body_x = 0.0
        self.last_stance_lock: Dict[int, object] = {}
        self.last_kinematic_only = False
        self.last_body_y = 0.0

        if model is not None:
            self._init_kinematics()

    def _init_kinematics(self) -> None:
        import mujoco
        from foot_kinematics import load_foot_frames

        self._flat_data = mujoco.MjData(self._model)
        self._flat_frames = load_foot_frames(
            self._model, self.stand, self.p.body_height
        )
        # 手动标定 stand_pose_flat.json 时保持 JSON 角度，不再改 tibia
        has_file_stand = (
            load_stand_pose is not None and load_stand_pose() is not None
        )
        if apply_flat_feet is not None and not has_file_stand:
            self.stand = apply_flat_feet(
                self._model,
                self._flat_data,
                self.stand,
                self.p.body_height,
                self._flat_frames,
            )
        self._ik = HexapodIK(self._model, self.stand, self.p.body_height)
        nominal = {leg: sf.foot_base for leg, sf in self._ik.stand.items()}
        cfg = TripodGaitConfig(
            cycle_time=self.p.cycle_time,
            step_height=self.p.step_height,
            stride_length=self.p.stride_length,
            swing_fraction=self.p.swing_fraction,
            use_radial_stride=self.p.use_radial_stride,
            cmd_deadband=self.p.cmd_deadband,
        )
        self._planner = TripodFootPlanner(nominal, cfg)
        self._last_joints = {leg: self._ik.stand[leg].joints for leg in range(1, 7)}
        if self.use_joint_gait:
            self._joint_crawl = JointTripodCrawlPlanner(self.stand)
            self.p.cycle_time = self._joint_crawl.cycle_time
            if self.use_physics_gait:
                from foot_stance_lock import JointGaitStanceTracker

                self._joint_stance = JointGaitStanceTracker()
        elif self.use_slot_gait:
            from tripod_slot_gait import SlotGaitConfig, TripodSlotGait

            cfg = SlotGaitConfig(step_length=self.p.stride_length, step_height=self.p.step_height)
            self._slot = TripodSlotGait(self._ik, self.stand, cfg)
            self.p.cycle_time = self._slot._cycle_time

    def reset(self) -> None:
        self.t = 0.0
        self.yaw = 0.0
        if self._planner is not None:
            self._planner.reset()
        self.last_stance_lock = {}
        self.last_body_x = 0.0
        self.last_kinematic_only = False
        self.last_body_y = 0.0
        if self._joint_crawl is not None:
            self._joint_crawl.reset()
            self._joint_crawl.stand = dict(self.stand)
        if self._joint_stance is not None:
            self._joint_stance.reset()
        elif self._slot is not None:
            self._slot.reset()
        elif self.use_joint_gait:
            self._joint_crawl = JointTripodCrawlPlanner(self.stand)
            self.p.cycle_time = self._joint_crawl.cycle_time
            if self.use_physics_gait:
                from foot_stance_lock import JointGaitStanceTracker

                self._joint_stance = JointGaitStanceTracker()
        elif self._ik is not None and self.use_slot_gait:
            from tripod_slot_gait import SlotGaitConfig, TripodSlotGait

            cfg = SlotGaitConfig(step_length=self.p.stride_length, step_height=self.p.step_height)
            self._slot = TripodSlotGait(self._ik, self.stand, cfg)
            self.p.cycle_time = self._slot._cycle_time
        if self._ik is not None:
            self._last_joints = {leg: self._ik.stand[leg].joints for leg in range(1, 7)}

    def step(
        self,
        dt: float,
        vx: float = 0.0,
        vy: float = 0.0,
        omega: float = 0.0,
        sim_data=None,
    ) -> Dict[str, float]:
        """
        vx: 前进 (m/s)，沿 base_link +Y（1/6 前方）
        vy: 侧向 (m/s)
        omega: 绕 Z 角速度 (rad/s)，正=左转
        """
        speed = math.hypot(vx, vy)
        moving = speed > self.p.cmd_deadband or abs(omega) > self.p.cmd_deadband

        if not moving:
            self.last_stance_lock = {}
            self.last_kinematic_only = False
            self.last_body_y = 0.0
            if self._model is not None:
                return clamp_joint_targets(self._model, dict(self.stand))
            return dict(self.stand)

        # 关节空间三角步态：抬腿→coxa 前摆→六腿蹬进（大振幅）
        if (
            self.use_joint_gait
            and self._joint_crawl is not None
            and abs(omega) <= self.p.cmd_deadband
            and abs(vy) <= self.p.cmd_deadband
            and abs(vx) > self.p.cmd_deadband
        ):
            scale = min(speed / CRAWL_NOMINAL_SPEED_MPS, 2.2)
            direction = 1.0 if vx >= 0 else -1.0
            self.last_body_x = 0.0
            physics = self.use_physics_gait
            self.last_kinematic_only = not physics
            joints = self._joint_crawl.step(
                dt, speed_scale=scale, direction=direction, physics=physics
            )
            phase = self._joint_crawl.current_phase()
            self.last_phase = phase.name
            if physics:
                self.last_body_y = 0.0
                if (
                    sim_data is not None
                    and self._ik is not None
                    and self._joint_stance is not None
                ):
                    self.last_stance_lock = self._joint_stance.update(
                        phase.name,
                        phase.kind,
                        phase.group,
                        self._model,
                        sim_data,
                        self._ik,
                    )
                else:
                    self.last_stance_lock = {}
            else:
                self.last_body_y = getattr(self._joint_crawl, "last_body_y", 0.0)
                self.last_stance_lock = {}
            if self._model is not None:
                return clamp_joint_targets(self._model, joints)
            return joints

        # 槽位三角步态（备用）
        if (
            self.use_slot_gait
            and self._slot is not None
            and abs(omega) <= self.p.cmd_deadband
            and vx > self.p.cmd_deadband
        ):
            scale = min(speed / CRAWL_NOMINAL_SPEED_MPS, 1.0)
            out = self._slot.step(dt, speed_scale=scale, sim_data=sim_data)
            self.last_phase = out.phase_name
            self.last_body_x = out.body_x
            self.last_stance_lock = dict(out.stance_world_lock)
            self.last_kinematic_only = out.kinematic_only
            if out.phase_name.endswith("BODY_ADVANCE"):
                joints = out.joints
            else:
                joints = self._enforce_flat_feet(out.joints)
            if self._model is not None:
                return clamp_joint_targets(self._model, joints)
            return joints

        if self._planner is None or self._ik is None:
            return self._step_fallback(dt, vx, vy, omega)

        self.t += dt
        self.yaw += omega * dt

        omega_scaled = omega * self.p.turn_rate_scale
        feet = self._planner.step(dt, vx, vy, omega_scaled, self.yaw)

        targets: Dict[str, float] = {}
        for leg in range(1, 7):
            seed = self._last_joints.get(leg, self._ik.stand[leg].joints)
            c, f, t = self._ik.solve(leg, feet[leg], seed=seed, yaw=self.yaw)
            self._last_joints[leg] = (c, f, t)
            targets[f"leg{leg}_coxa_joint"] = c
            targets[f"leg{leg}_femur_joint"] = f
            targets[f"leg{leg}_tibia_joint"] = t

        targets = self._enforce_flat_feet(targets)
        if self._model is not None:
            return clamp_joint_targets(self._model, targets)
        return targets

    def _step_fallback(
        self,
        dt: float,
        vx: float,
        vy: float,
        omega: float,
    ) -> Dict[str, float]:
        """无 MuJoCo 模型时的简化关节调制（兼容旧逻辑）。"""
        self.t += dt
        phase = (self.t % self.p.cycle_time) / self.p.cycle_time
        swing_a = phase < self.p.swing_fraction
        stride = self.p.stride_length * min(math.hypot(vx, vy) / 0.1, 1.0)
        direction = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-4 else 0.0
        targets = dict(self.stand)
        for leg in range(1, 7):
            is_a = leg in TRIPOD_A
            swinging = swing_a if is_a else not swing_a
            sf = self.p.swing_fraction
            local_phase = (phase / sf) if swinging else max(
                0.0, (phase - sf) / max(1e-6, 1.0 - sf)
            )
            if swinging:
                from tripod_planner import swing_lift_profile

                lift = swing_lift_profile(local_phase, self.p.step_height)
            else:
                lift = 0.0
            sweep = stride * (local_phase - 0.5) * 2.0 if swinging else -stride * 0.25
            try:
                from leg_symmetry import LEG_AZIMUTH_DEG

                leg_angle = math.radians(LEG_AZIMUTH_DEG.get(leg, 0.0))
            except ImportError:
                leg_angle = 0.0
            turn_offset = omega * self.p.turn_rate_scale * math.sin(leg_angle)
            targets[f"leg{leg}_coxa_joint"] = self.stand[f"leg{leg}_coxa_joint"] + turn_offset
            targets[f"leg{leg}_femur_joint"] = (
                self.stand[f"leg{leg}_femur_joint"] - sweep * 2.0 - lift * 5.0
            )
            targets[f"leg{leg}_tibia_joint"] = (
                self.stand[f"leg{leg}_tibia_joint"] + sweep * 1.5 + lift * 7.0
            )
            targets[f"leg{leg}_femur_joint"] += math.cos(direction) * stride * 0.5
            targets[f"leg{leg}_tibia_joint"] += math.sin(direction) * stride * 0.3
        return targets

    def _enforce_flat_feet(self, targets: Dict[str, float]) -> Dict[str, float]:
        if apply_flat_feet is None or self._model is None:
            return targets
        import mujoco

        if self._flat_frames is None:
            from foot_kinematics import load_foot_frames

            self._flat_frames = load_foot_frames(
                self._model, self.stand, self.p.body_height
            )
        if self._flat_data is None:
            self._flat_data = mujoco.MjData(self._model)
        return apply_flat_feet(
            self._model,
            self._flat_data,
            targets,
            self.p.body_height,
            self._flat_frames,
        )


# 导出步态分组供文档/测试
def create_forward_tripod_gait(
    model=None, speed_mps: float = CRAWL_NOMINAL_SPEED_MPS
) -> TripodGait:
    """创建用于前进行走的三角步态控制器。"""
    return TripodGait(params=forward_gait_params(speed_mps), model=model)


__all__ = [
    "TripodGait",
    "GaitParams",
    "TRIPOD_A",
    "TRIPOD_B",
    "create_forward_tripod_gait",
    "forward_gait_params",
    "nominal_stand_pose",
    "leg_joint_names",
]
