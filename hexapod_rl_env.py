#!/usr/bin/env python3
"""
六足平地行走 Gymnasium 环境 — 规则三角步态 + RL 残差关节修正。

动作: 18 维残差 Δq（叠加在 joint_tripod_gait 目标上）
观测: 机身姿态/速度 + 关节状态 + 步态相位
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from enable_state import LOCOMOTION_KP, LOCOMOTION_KV, set_actuator_gains
from body_stabilizer import stabilize_locomotion_body
from ctrl_smoother import CtrlSmoother
from foot_kinematics import load_stand_pose, resolve_post_enable_stand
from foot_stance_lock import blend_stance_ctrl_targets, damp_leg_joint_velocities
from gait import create_forward_tripod_gait
from rl_posture import sync_gait_stand
from rl_reward import MAX_SPEED_VX_MPS, TRACK_VX_MPS, compute_walk_reward
from robot_limits import all_joint_names, clamp_joint_targets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml")

WALK_SPEED_MPS = 0.06  # 跟踪模式目标速度 m/s（沿 base_link +Y）
VX_CMD = WALK_SPEED_MPS


def _quat_rp(q: np.ndarray) -> Tuple[float, float]:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def _actuator_id(model: mujoco.MjModel, jname: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jname}_act")


class HexapodFlatWalkEnv(gym.Env):
    """平地前进：步态基准 + 可学习残差。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        control_dt: float = 0.02,
        max_episode_steps: int = 500,
        residual_scale: float = 0.06,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        reward_mode: str = "track",
        vx_cmd: Optional[float] = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.control_dt = control_dt
        self.max_episode_steps = max_episode_steps
        self.residual_scale = residual_scale
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        if vx_cmd is not None:
            self.vx_cmd = float(vx_cmd)
        elif reward_mode == "max_speed":
            self.vx_cmd = MAX_SPEED_VX_MPS
        else:
            self.vx_cmd = VX_CMD

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"未找到模型: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._n_substeps = max(1, int(round(control_dt / self.model.opt.timestep)))

        loaded = load_stand_pose()
        if loaded is None:
            raise RuntimeError("缺少 generated/stand_pose_flat.json，请先标定站立姿态")
        coxa_pose, _ = loaded
        stand_pose, body_z = resolve_post_enable_stand(self.model, coxa_pose)
        self._nominal_stand_pose, self._nominal_body_z = stand_pose, body_z
        self.stand_pose, self.body_z = stand_pose, body_z

        self.joint_names = all_joint_names()
        self._j_qposadr = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                ]
                for jn in self.joint_names
            ],
            dtype=int,
        )
        self._j_dofadr = np.array(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                ]
                for jn in self.joint_names
            ],
            dtype=int,
        )
        self._act_ids = np.array(
            [_actuator_id(self.model, jn) for jn in self.joint_names], dtype=int
        )

        self.gait = create_forward_tripod_gait(model=self.model, speed_mps=self.vx_cmd)
        self.gait.use_joint_gait = True
        self.gait.use_physics_gait = True
        sync_gait_stand(self.gait, self.stand_pose, self.body_z)

        n_j = len(self.joint_names)
        # obs: height_err, roll, pitch, v_body(3), wz, q_rel(18), qd(18), phase_sin, phase_cos
        obs_dim = 1 + 2 + 3 + 1 + n_j + n_j + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_j,), dtype=np.float32
        )

        self._step_count = 0
        self._last_action = np.zeros(n_j, dtype=np.float32)
        self._ctrl_smoother = CtrlSmoother(tau=0.14)
        self._viewer = None
        self._np_random: Optional[np.random.Generator] = None

        if seed is not None:
            self.reset(seed=seed)

    def _set_stand_pose(self, noise_rad: float = 0.0) -> None:
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.body_z
        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0
        for jn, val in self.stand_pose.items():
            adr = self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            ]
            v = float(val)
            if noise_rad > 0 and self._np_random is not None:
                v += float(self._np_random.uniform(-noise_rad, noise_rad))
            self.data.qpos[adr] = v
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _gait_phase_u(self) -> float:
        planner = self.gait._joint_crawl
        if planner is None:
            return 0.0
        phase_t = planner.t % planner.cycle_time
        acc = 0.0
        for dur in planner._durs:
            if phase_t < acc + dur:
                return (phase_t - acc) / max(dur, 1e-9)
            acc += dur
        return 0.0

    def _get_obs(self) -> np.ndarray:
        roll, pitch = _quat_rp(self.data.qpos[3:7])
        base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        R = self.data.xmat[base_id].reshape(3, 3)
        v_body = R.T @ self.data.qvel[0:3]
        wz = float(self.data.qvel[5])

        q = self.data.qpos[self._j_qposadr].astype(np.float64)
        q_stand = np.array(
            [self.stand_pose[jn] for jn in self.joint_names], dtype=np.float64
        )
        q_rel = q - q_stand
        qd = self.data.qvel[self._j_dofadr].astype(np.float64)

        u = self._gait_phase_u()
        parts = [
            np.array([self.data.qpos[2] - self.body_z], dtype=np.float32),
            np.array([roll, pitch], dtype=np.float32),
            v_body.astype(np.float32),
            np.array([wz], dtype=np.float32),
            q_rel.astype(np.float32),
            (qd * 0.05).astype(np.float32),
            np.array([math.sin(2 * math.pi * u), math.cos(2 * math.pi * u)], dtype=np.float32),
        ]
        return np.concatenate(parts)

    def _is_fallen(self) -> bool:
        z = float(self.data.qpos[2])
        roll, pitch = _quat_rp(self.data.qpos[3:7])
        if z < self.body_z - 0.045:
            return True
        if abs(roll) > 0.55 or abs(pitch) > 0.55:
            return True
        return False

    def _compute_reward(
        self, action: np.ndarray, q_ref: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        R = self.data.xmat[base_id].reshape(3, 3)
        v_body = R.T @ self.data.qvel[0:3]
        v_forward = float(v_body[1])
        v_lateral = float(v_body[0])
        wz = float(self.data.qvel[5])
        z_err = float(self.data.qpos[2] - self.body_z)

        roll, pitch = _quat_rp(self.data.qpos[3:7])
        reward, rinfo = compute_walk_reward(
            v_forward,
            v_lateral,
            wz,
            z_err,
            roll,
            pitch,
            action,
            self._last_action,
            mode=self.reward_mode,
            vx_target=TRACK_VX_MPS,
        )
        info = {
            **rinfo,
            "v_lateral": v_lateral,
            "body_z": float(self.data.qpos[2]),
        }
        return reward, info

    def _apply_stand_hold(self) -> None:
        """PD 持站立角。"""
        for jn, val in self.stand_pose.items():
            self.data.ctrl[_actuator_id(self.model, jn)] = float(val)
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)
        damp_leg_joint_velocities(self.model, self.data)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        self.gait.reset()
        set_actuator_gains(self.model, LOCOMOTION_KP, LOCOMOTION_KV)

        noise = 0.008
        if options and "noise_rad" in options:
            noise = float(options["noise_rad"])
        self.stand_pose = dict(self._nominal_stand_pose)
        self.body_z = float(self._nominal_body_z)
        sync_gait_stand(self.gait, self.stand_pose, self.body_z)
        self._set_stand_pose(noise_rad=noise)
        self._ctrl_smoother.reset(self.stand_pose)
        self._apply_stand_hold()

        self._step_count = 0
        self._last_action[:] = 0.0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        q_ref = self.gait.step(
            self.control_dt, self.vx_cmd, 0.0, 0.0, sim_data=self.data
        )
        targets = dict(q_ref)
        for i, jn in enumerate(self.joint_names):
            targets[jn] = float(targets.get(jn, self.stand_pose[jn])) + (
                self.residual_scale * float(action[i])
            )
        targets = clamp_joint_targets(self.model, targets)

        ctrl_targets = targets
        if (
            self.gait.use_joint_gait
            and self.gait.use_physics_gait
            and self.gait.last_stance_lock
            and self.gait._ik is not None
        ):
            ctrl_targets = blend_stance_ctrl_targets(
                self.model,
                self.data,
                self.gait._ik,
                self.gait.last_stance_lock,
                targets,
            )
        if self.gait.use_joint_gait and self.gait.use_physics_gait:
            ctrl_targets = self._ctrl_smoother.filter(ctrl_targets, self.control_dt)

        for jn, val in ctrl_targets.items():
            self.data.ctrl[_actuator_id(self.model, jn)] = val

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        if self.gait.use_joint_gait and self.gait.use_physics_gait:
            stabilize_locomotion_body(
                self.model,
                self.data,
                body_z_target=self.gait.p.body_height,
            )
            damp_leg_joint_velocities(self.model, self.data)

        reward, rinfo = self._compute_reward(action, q_ref)
        self._step_count += 1
        fallen = self._is_fallen()
        terminated = fallen
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = {**rinfo, "fallen": fallen, "phase": self.gait.last_phase}

        if self.render_mode == "human":
            self.render()

        self._last_action = action.copy()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def make_hexapod_env(**kwargs) -> HexapodFlatWalkEnv:
    return HexapodFlatWalkEnv(**kwargs)
