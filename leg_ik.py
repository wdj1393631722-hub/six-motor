#!/usr/bin/env python3
"""单腿足端逆运动学：在机体坐标系下求关节角（基于 MuJoCo 正解 + 数值优化）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import mujoco
import numpy as np

try:
    from foot_kinematics import _tilt_cost, foot_world, load_foot_frames
except ImportError:
    _tilt_cost = None
    foot_world = None
    load_foot_frames = None


@dataclass
class LegStandFrame:
    """站立标定下的足端与默认关节角。"""
    foot_base: np.ndarray  # 足端在 base_link 系 (3,)
    joints: Tuple[float, float, float]  # coxa, femur, tibia


class HexapodIK:
    """
    六足 IK：每条腿独立 3 自由度数值解。
    规划在 base_link 坐标系（机体水平、原点=机身中心投影）。
    """

    def __init__(self, model: mujoco.MjModel, stand_pose: Dict[str, float], body_height: float):
        self.model = model
        self.data = mujoco.MjData(model)
        self.body_height = body_height
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.frames = (
            load_foot_frames(model, stand_pose, body_height)
            if load_foot_frames
            else {}
        )
        self.stand: Dict[int, LegStandFrame] = {}
        self._joint_ids = {}
        for leg in range(1, 7):
            names = [f"leg{leg}_coxa_joint", f"leg{leg}_femur_joint", f"leg{leg}_tibia_joint"]
            self._joint_ids[leg] = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names
            ]
        self._cache_stand(stand_pose)

    def _set_body(self, yaw: float = 0.0) -> None:
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        self.data.qpos[2] = self.body_height
        # qw, qx, qy, qz — 绕 Z 偏航
        self.data.qpos[3] = float(np.cos(yaw / 2))
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = float(np.sin(yaw / 2))

    def _foot_in_base(self, leg: int) -> np.ndarray:
        p_w, _ = foot_world(self.model, self.data, leg, self.frames)
        base_pos = self.data.xpos[self.base_id]
        R = self.data.xmat[self.base_id].reshape(3, 3)
        return R.T @ (p_w - base_pos)

    def _cache_stand(self, stand_pose: Dict[str, float]) -> None:
        self._set_body(0.0)
        for leg in range(1, 7):
            for j, jn in enumerate(["coxa", "femur", "tibia"]):
                jid = self._joint_ids[leg][j]
                self.data.qpos[self.model.jnt_qposadr[jid]] = stand_pose[
                    f"leg{leg}_{jn}_joint"
                ]
        mujoco.mj_forward(self.model, self.data)
        for leg in range(1, 7):
            q = tuple(
                float(self.data.qpos[self.model.jnt_qposadr[jid]])
                for jid in self._joint_ids[leg]
            )
            self.stand[leg] = LegStandFrame(self._foot_in_base(leg).copy(), q)

    def foot_base(self, leg: int, joints: Tuple[float, float, float], yaw: float = 0.0) -> np.ndarray:
        self._set_body(yaw)
        for jid, v in zip(self._joint_ids[leg], joints):
            adr = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self.data.qpos[adr] = float(np.clip(v, lo, hi))
        mujoco.mj_forward(self.model, self.data)
        return self._foot_in_base(leg)

    def solve(
        self,
        leg: int,
        target_base: np.ndarray,
        seed: Tuple[float, float, float] | None = None,
        yaw: float = 0.0,
    ) -> Tuple[float, float, float]:
        """足端目标 (base 系) → 关节角，带限位惩罚。"""
        if seed is None:
            seed = self.stand[leg].joints
        target = np.asarray(target_base, dtype=float)
        best_q = np.array(seed, dtype=float)
        best_cost = self._ik_cost(leg, best_q, target, yaw)

        # 种子较差时再小范围网格搜索
        if best_cost > 2e-3:
            for dc in (-0.22, 0.0, 0.22):
                for df in (-0.28, 0.0, 0.28):
                    for dt in (-0.45, 0.0, 0.45):
                        q = np.array(seed) + [dc, df, dt]
                        c = self._ik_cost(leg, q, target, yaw)
                        if c < best_cost:
                            best_cost, best_q = c, q.copy()

        step = np.array([0.08, 0.08, 0.10])
        for _ in range(10):
            improved = False
            for i in range(3):
                for sgn in (-1.0, 1.0):
                    trial = best_q.copy()
                    trial[i] += sgn * step[i]
                    c = self._ik_cost(leg, trial, target, yaw)
                    if c < best_cost:
                        best_cost, best_q, improved = c, trial, True
            if not improved:
                step *= 0.5
                if np.max(step) < 0.012:
                    break

        lo_hi = [self.model.jnt_range[jid] for jid in self._joint_ids[leg]]
        return tuple(
            float(np.clip(best_q[i], lo_hi[i][0], lo_hi[i][1])) for i in range(3)
        )

    def _ik_cost(
        self,
        leg: int,
        q: np.ndarray,
        target_base: np.ndarray,
        yaw: float,
    ) -> float:
        p = self.foot_base(leg, tuple(q), yaw)
        dp = p - target_base
        # 高度误差权重大一些，否则抬脚目标达不到
        err = float(dp[0] ** 2 + dp[1] ** 2 + 12.0 * dp[2] ** 2)
        for i, jid in enumerate(self._joint_ids[leg]):
            lo, hi = self.model.jnt_range[jid]
            if q[i] < lo:
                err += 80.0 * (lo - q[i]) ** 2
            elif q[i] > hi:
                err += 80.0 * (q[i] - hi) ** 2
        # 行走时减弱“拉回站立”的惩罚，否则腿几乎不动
        s = self.stand[leg].joints
        foot_move = float(np.linalg.norm(target_base - self.stand[leg].foot_base))
        stand_w = 0.008 if foot_move > 0.012 else 0.10
        err += stand_w * sum((q[i] - s[i]) ** 2 for i in range(3))
        if foot_world is not None and self.frames and _tilt_cost is not None:
            self._set_body(yaw)
            for jid, v in zip(self._joint_ids[leg], q):
                adr = self.model.jnt_qposadr[jid]
                lo, hi = self.model.jnt_range[jid]
                self.data.qpos[adr] = float(np.clip(v, lo, hi))
            mujoco.mj_forward(self.model, self.data)
            _, n = foot_world(self.model, self.data, leg, self.frames)
            # 支撑/摆动时足底平面尽量平行地面
            err += 120.0 * _tilt_cost(n)
        return err
