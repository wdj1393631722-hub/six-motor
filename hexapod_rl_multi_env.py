#!/usr/bin/env python3
"""
同屏多机器人平地 RL 环境 — 一个 MuJoCo 场景内 N 只六足并行训练/可视化。

适用于「十几个机器人在屏幕上同时强化学习」演示：单窗口、共享 GPU 策略。
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from build_multi_mjcf import build_multi_mjcf, random_spawn_positions
from body_stabilizer import stabilize_robot_root
from ctrl_smoother import CtrlSmoother
from enable_state import LOCOMOTION_KP, LOCOMOTION_KV, set_actuator_gains
from foot_kinematics import load_stand_pose, resolve_post_enable_stand
from gait import create_forward_tripod_gait
from hexapod_rl_env import VX_CMD, _quat_rp
from imu_sensor import ImuBinding, imu_obs_vector
from rl_posture import RL_WARMUP_STAND_STEPS, settle_multi_robots, sync_gait_stand
from rl_reward import MAX_SPEED_VX_MPS, TRACK_VX_MPS, compute_walk_reward
from robot_limits import all_joint_names, clamp_joint_targets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_spawn_layout(n_robots: int) -> tuple[float, float, float]:
    """按机器人数返回 (spawn_span, min_spawn_dist, camera_distance_hint)。"""
    if n_robots <= 8:
        return 7.0, 1.25, 10.0
    if n_robots <= 16:
        return 12.0, 1.5, 16.0
    if n_robots <= 24:
        return 16.0, 1.7, 22.0
    # 30 机：更大散布，减少互撞
    return 20.0, 1.9, 26.0


def _prefixed_joint_names(prefix: str) -> List[str]:
    return [f"{prefix}{jn}" for jn in all_joint_names()]


def _actuator_id(model: mujoco.MjModel, jname: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jname}_act")


class _RobotSlot:
    def __init__(
        self,
        index: int,
        prefix: str,
        model: mujoco.MjModel,
        stand_pose: Dict[str, float],
        body_z: float,
        offset_xy: Tuple[float, float],
        reward_mode: str = "track",
        vx_cmd: float = VX_CMD,
    ):
        # stand_pose 为未加前缀的 18 关节角
        self.index = index
        self.prefix = prefix
        self.offset_xy = offset_xy
        self.body_z = body_z
        self.joint_names = _prefixed_joint_names(prefix)
        self.stand_pose = {f"{prefix}{k}": v for k, v in stand_pose.items()}
        self.root_joint = f"{prefix}root"
        self.base_body = f"{prefix}base_link"

        self._root_qposadr = int(
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.root_joint)
            ]
        )
        self._root_dofadr = int(
            model.jnt_dofadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.root_joint)
            ]
        )
        self._j_qposadr = np.array(
            [
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                ]
                for jn in self.joint_names
            ],
            dtype=int,
        )
        self._j_dofadr = np.array(
            [
                model.jnt_dofadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                ]
                for jn in self.joint_names
            ],
            dtype=int,
        )
        self._act_ids = np.array(
            [_actuator_id(model, jn) for jn in self.joint_names], dtype=int
        )
        self._base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.base_body
        )
        self._imu = ImuBinding(
            model,
            prefix=prefix,
            root_qposadr=self._root_qposadr,
            root_dofadr=self._root_dofadr,
            base_body_id=self._base_id,
        )

        # 多机场景勿传整场景 model（会重复做 12 次 IK/足底标定，极慢）
        self.gait = create_forward_tripod_gait(model=None, speed_mps=vx_cmd)
        self.gait.use_joint_gait = True
        self.gait.use_physics_gait = True
        sync_gait_stand(self.gait, stand_pose, body_z)
        self._clamp_model = model

        n_j = len(self.joint_names)
        self._last_action = np.zeros(n_j, dtype=np.float32)
        self._step_count = 0
        self.vx_scale = 1.0
        self.vx_cmd = float(vx_cmd)
        self.reward_mode = reward_mode
        self.spawn_xy = offset_xy
        self.spawn_yaw = 0.0
        self.home_xy = offset_xy
        self.home_yaw = 0.0
        self._inactive = False
        self._inactive_until = 0
        self._ctrl_smoother = CtrlSmoother(tau=0.14)

    def apply_stand_ctrl(self, data: mujoco.MjData) -> None:
        for jn, val in self.stand_pose.items():
            data.ctrl[_actuator_id(self._clamp_model, jn)] = float(val)

    def reset_pose(
        self,
        data: mujoco.MjData,
        rng: np.random.Generator,
        noise_rad: float,
        xy: Optional[Tuple[float, float]] = None,
        yaw: Optional[float] = None,
        phase_offset: Optional[float] = None,
        vx_scale: Optional[float] = None,
    ) -> None:
        if xy is not None:
            self.spawn_xy = xy
            self.home_xy = xy
        if yaw is not None:
            self.spawn_yaw = yaw
            self.home_yaw = yaw
        if vx_scale is not None:
            self.vx_scale = float(vx_scale)

        adr = self._root_qposadr
        z = self.body_z + float(rng.uniform(-0.004, 0.004))
        data.qpos[adr : adr + 3] = (self.spawn_xy[0], self.spawn_xy[1], z)
        half = self.spawn_yaw * 0.5
        data.qpos[adr + 3 : adr + 7] = (
            math.cos(half),
            0.0,
            0.0,
            math.sin(half),
        )
        for jn, val in self.stand_pose.items():
            jadr = data.model.jnt_qposadr[
                mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            ]
            v = float(val)
            if noise_rad > 0:
                v += float(rng.uniform(-noise_rad, noise_rad))
            data.qpos[jadr] = v
        data.qvel[self._root_dofadr : self._root_dofadr + 6] = 0.0
        data.qvel[self._j_dofadr] = 0.0
        self.gait.reset()
        planner = self.gait._joint_crawl
        if planner is not None:
            # 从步态周期起点开始，避免 reset 后立刻处于抬腿相
            if phase_offset is None:
                phase_offset = 0.0
            planner.t = float(phase_offset)
            planner._prev_joints = dict(self.gait.stand)
        self._ctrl_smoother.reset(
            {jn[len(self.prefix) :]: v for jn, v in self.stand_pose.items()}
        )
        self._last_action[:] = 0.0
        self._step_count = 0
        self._inactive = False
        self._inactive_until = 0

    def hold_stand(self, data: mujoco.MjData) -> None:
        """摔倒等待重生时保持站立角，避免抽搐。"""
        self.apply_stand_ctrl(data)

    def damp_legs(self, data: mujoco.MjData) -> None:
        for dof in self._j_dofadr:
            data.qvel[int(dof)] *= 0.52

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

    def get_obs(self, data: mujoco.MjData) -> np.ndarray:
        adr = self._root_qposadr
        q_root = data.qpos[adr : adr + 7]
        imu = self._imu.read(data)
        R = data.xmat[self._base_id].reshape(3, 3)
        v_body = R.T @ data.qvel[self._root_dofadr : self._root_dofadr + 3]

        q = data.qpos[self._j_qposadr].astype(np.float64)
        q_stand = np.array(
            [self.stand_pose[jn] for jn in self.joint_names], dtype=np.float64
        )
        q_rel = q - q_stand
        qd = data.qvel[self._j_dofadr].astype(np.float64)
        u = self._gait_phase_u()

        parts = [
            np.array([float(q_root[2]) - self.body_z], dtype=np.float32),
            imu_obs_vector(imu),
            v_body.astype(np.float32),
            q_rel.astype(np.float32),
            (qd * 0.05).astype(np.float32),
            np.array(
                [math.sin(2 * math.pi * u), math.cos(2 * math.pi * u)],
                dtype=np.float32,
            ),
        ]
        return np.concatenate(parts)

    def is_fallen(self, data: mujoco.MjData) -> bool:
        adr = self._root_qposadr
        z = float(data.qpos[adr + 2])
        roll, pitch = _quat_rp(data.qpos[adr + 3 : adr + 7])
        if z < self.body_z - 0.045:
            return True
        if abs(roll) > 0.55 or abs(pitch) > 0.55:
            return True
        return False

    def compute_reward(self, data: mujoco.MjData, action: np.ndarray) -> Tuple[float, Dict]:
        imu = self._imu.read(data)
        R = data.xmat[self._base_id].reshape(3, 3)
        v_body = R.T @ data.qvel[self._root_dofadr : self._root_dofadr + 3]
        v_forward = float(v_body[1])
        v_lateral = float(v_body[0])
        wz = float(imu.gyro[2])
        z_err = float(data.qpos[self._root_qposadr + 2] - self.body_z)

        reward, rinfo = compute_walk_reward(
            v_forward,
            v_lateral,
            wz,
            z_err,
            imu.roll,
            imu.pitch,
            action,
            self._last_action,
            mode=self.reward_mode,
            vx_target=TRACK_VX_MPS,
            gyro=imu.gyro,
            acc=imu.acc,
        )
        return reward, {
            **rinfo,
            "v_lateral": v_lateral,
            "body_z": float(data.qpos[self._root_qposadr + 2]),
            "fallen": self.is_fallen(data),
            "phase": self.gait.last_phase,
        }

    def apply_action(
        self,
        data: mujoco.MjData,
        action: np.ndarray,
        residual_scale: float,
        control_dt: float,
    ) -> None:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        vx = self.vx_cmd * self.vx_scale
        q_ref = self.gait.step(control_dt, vx, 0.0, 0.0, sim_data=None)
        targets: Dict[str, float] = {}
        for i, jn in enumerate(self.joint_names):
            base_name = jn[len(self.prefix) :]
            val = float(q_ref.get(base_name, self.stand_pose[jn])) + (
                residual_scale * float(action[i])
            )
            targets[jn] = val
        targets = clamp_joint_targets(self._clamp_model, targets)
        ph = self.gait.last_phase or ""
        self._ctrl_smoother.tau = (
            0.006 if ("swing" in ph or "place" in ph) else 0.038
        )
        smoothed = self._ctrl_smoother.filter(
            {jn[len(self.prefix) :]: targets[jn] for jn in self.joint_names},
            control_dt,
        )
        for jn in self.joint_names:
            base_name = jn[len(self.prefix) :]
            data.ctrl[_actuator_id(self._clamp_model, jn)] = float(
                smoothed.get(base_name, targets[jn])
            )
        self._last_action = action.copy()


class HexapodMultiArena:
    """N 只机器人共用一个 MuJoCo 场景。"""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        n_robots: int = 30,
        spacing: float = 1.15,
        control_dt: float = 0.02,
        max_episode_steps: int = 500,
        residual_scale: float = 0.08,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        random_layout: bool = True,
        spawn_span: float | None = None,
        min_spawn_dist: float | None = None,
        respawn_mode: str = "soft",
        respawn_delay_steps: int = 60,
        render_stride: int = 2,
        reward_mode: str = "track",
        vx_cmd: Optional[float] = None,
    ):
        self.n_robots = n_robots
        self.control_dt = control_dt
        self.max_episode_steps = max_episode_steps
        self.residual_scale = residual_scale
        self.render_mode = render_mode
        self.random_layout = random_layout
        default_span, default_dist, cam_hint = _default_spawn_layout(n_robots)
        self.spawn_span = float(spawn_span if spawn_span is not None else default_span)
        self.min_spawn_dist = float(
            min_spawn_dist if min_spawn_dist is not None else default_dist
        )
        self._camera_distance_hint = cam_hint
        self.respawn_mode = respawn_mode
        self.respawn_delay_steps = max(0, int(respawn_delay_steps))
        self.render_stride = max(1, int(render_stride))
        self._render_counter = 0
        self.reward_mode = reward_mode
        if vx_cmd is not None:
            self.vx_cmd = float(vx_cmd)
        elif reward_mode == "max_speed":
            self.vx_cmd = MAX_SPEED_VX_MPS
        else:
            self.vx_cmd = VX_CMD

        print(f"[arena] 生成/加载 {n_robots} 机器人场景…", flush=True)
        self.model_path = build_multi_mjcf(n_robots, spacing, layout="scatter")
        print(f"[arena] 编译 MuJoCo 模型: {self.model_path}", flush=True)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        print(f"[arena] 模型就绪 nq={self.model.nq} nu={self.model.nu}", flush=True)
        self.data = mujoco.MjData(self.model)
        self._n_substeps = max(1, int(round(control_dt / self.model.opt.timestep)))

        raw = load_stand_pose()
        if raw is None:
            raise RuntimeError("缺少 generated/stand_pose_flat.json")
        coxa_pose, _ = raw
        # 站立角标定只需单机模型；多机 MJCF 编译极慢且关节名带前缀
        single_model_path = os.path.join(
            SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml"
        )
        single_model = mujoco.MjModel.from_xml_path(single_model_path)
        stand_pose, body_z = resolve_post_enable_stand(single_model, coxa_pose)
        print(f"[arena] 使能完成站立 body_z={body_z:.3f} m", flush=True)
        self._settle_steps = 160 if n_robots >= 24 else 200

        from build_multi_mjcf import _grid_xy

        positions = _grid_xy(n_robots, spacing)
        self.robots: List[_RobotSlot] = []
        for i in range(n_robots):
            prefix = f"r{i}_"
            self.robots.append(
                _RobotSlot(
                    i,
                    prefix,
                    self.model,
                    stand_pose,
                    body_z,
                    positions[i],
                    reward_mode=self.reward_mode,
                    vx_cmd=self.vx_cmd,
                )
            )

        n_j = len(all_joint_names())
        obs_dim = 1 + 8 + 3 + n_j + n_j + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_j,), dtype=np.float32
        )
        self._viewer = None
        self._np_random: Optional[np.random.Generator] = None
        self._global_step = 0

        if seed is not None:
            self.reset(seed=seed)

    def _occupied_xy(self, skip_index: Optional[int] = None) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        for bot in self.robots:
            if skip_index is not None and bot.index == skip_index:
                continue
            adr = bot._root_qposadr
            pts.append((float(self.data.qpos[adr]), float(self.data.qpos[adr + 1])))
        return pts

    def _sample_xy(self, occupied: List[Tuple[float, float]]) -> Tuple[float, float]:
        assert self._np_random is not None
        for _ in range(600):
            x = float(self._np_random.uniform(-self.spawn_span, self.spawn_span))
            y = float(self._np_random.uniform(-self.spawn_span, self.spawn_span))
            ok = True
            for px, py in occupied:
                if (x - px) ** 2 + (y - py) ** 2 < self.min_spawn_dist**2:
                    ok = False
                    break
            if ok:
                return x, y
        return random_spawn_positions(1, self._np_random, self.spawn_span, self.min_spawn_dist)[0]

    def _respawn_bot(
        self,
        bot: _RobotSlot,
        noise_rad: float = 0.012,
        *,
        randomize: bool = False,
    ) -> None:
        assert self._np_random is not None
        if randomize and self.random_layout and self.respawn_mode == "random":
            occupied = self._occupied_xy(skip_index=bot.index)
            xy = self._sample_xy(occupied)
            yaw = float(self._np_random.uniform(-math.pi, math.pi))
        else:
            xy = bot.home_xy
            yaw = bot.home_yaw
        vx_scale = float(self._np_random.uniform(0.92, 1.08))
        bot.reset_pose(
            self.data,
            self._np_random,
            noise_rad,
            xy=xy,
            yaw=yaw,
            phase_offset=0.0,
            vx_scale=vx_scale,
        )
        bot.apply_stand_ctrl(self.data)
        for _ in range(80):
            mujoco.mj_step(self.model, self.data)
        adr = bot._root_qposadr
        bot.body_z = float(self.data.qpos[adr + 2])
        for jn in bot.joint_names:
            jadr = self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            ]
            bot.stand_pose[jn] = float(self.data.qpos[jadr])
        base_name = {jn[len(bot.prefix) :]: v for jn, v in bot.stand_pose.items()}
        sync_gait_stand(bot.gait, base_name, bot.body_z)
        bot._ctrl_smoother.reset(base_name)

    def _mark_inactive(self, bot: _RobotSlot) -> None:
        bot._inactive = True
        bot._inactive_until = self._global_step + self.respawn_delay_steps

    def _process_inactive(self) -> list[int]:
        """处理摔倒等待与重生，返回本步刚重生的机器人索引。"""
        revived: list[int] = []
        for bot in self.robots:
            if not bot._inactive:
                continue
            bot.hold_stand(self.data)
            if self._global_step >= bot._inactive_until:
                self._respawn_bot(bot, noise_rad=0.008, randomize=False)
                revived.append(bot.index)
        return revived

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        set_actuator_gains(self.model, LOCOMOTION_KP, LOCOMOTION_KV)
        noise = 0.015
        if options and "noise_rad" in options:
            noise = float(options["noise_rad"])

        self.data.qvel[:] = 0.0
        if self.random_layout:
            spawns = random_spawn_positions(
                self.n_robots,
                self._np_random,
                self.spawn_span,
                self.min_spawn_dist,
            )
        else:
            spawns = [bot.offset_xy for bot in self.robots]

        for bot, xy in zip(self.robots, spawns):
            yaw = float(self._np_random.uniform(-math.pi, math.pi))
            vx_scale = float(self._np_random.uniform(0.86, 1.14))
            bot.reset_pose(
                self.data,
                self._np_random,
                noise,
                xy=xy,
                yaw=yaw,
                phase_offset=0.0,
                vx_scale=vx_scale,
            )

        settle_multi_robots(
            self.model, self.data, self.robots, steps=self._settle_steps
        )
        for bot in self.robots:
            adr = bot._root_qposadr
            bot.body_z = float(self.data.qpos[adr + 2])
            for jn in bot.joint_names:
                jadr = self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                ]
                bot.stand_pose[jn] = float(self.data.qpos[jadr])
            base_name = {jn[len(bot.prefix) :]: v for jn, v in bot.stand_pose.items()}
            sync_gait_stand(bot.gait, base_name, bot.body_z)
            bot._ctrl_smoother.reset(base_name)
            bot._warmup_remaining = RL_WARMUP_STAND_STEPS
            bot._warmup_active = False
        mujoco.mj_forward(self.model, self.data)
        self._global_step = 0
        obs = self._get_obs_batch()
        return obs, {}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.n_robots, self.action_space.shape[0]):
            raise ValueError(f"动作形状应为 ({self.n_robots}, 18)，收到 {actions.shape}")

        for bot, act in zip(self.robots, actions):
            if bot._inactive:
                bot.hold_stand(self.data)
                continue
            bot.apply_action(self.data, act, self.residual_scale, self.control_dt)

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        for bot in self.robots:
            if bot._inactive:
                continue
            stabilize_robot_root(
                self.data,
                bot._root_qposadr,
                bot._root_dofadr,
                yaw_hold=bot.spawn_yaw,
                roll_hold=0.0,
                pitch_hold=0.0,
            )
            bot.damp_legs(self.data)

        rewards = np.zeros(self.n_robots, dtype=np.float32)
        terminated = np.zeros(self.n_robots, dtype=bool)
        infos: list = []
        for bot, act in zip(self.robots, actions):
            if not bot._inactive:
                bot._step_count += 1
            r, info = bot.compute_reward(self.data, act)
            rewards[bot.index] = r
            fallen = bool(info["fallen"])
            trunc = (not bot._inactive) and bot._step_count >= self.max_episode_steps
            term = (not bot._inactive) and fallen
            terminated[bot.index] = term or trunc
            infos.append({**info, "terminated": term, "truncated": trunc})

        self._global_step += 1

        # 摔倒/超时：先原地趴一会再同位置重生，避免画面不停闪跳
        need_reset = [i for i, inf in enumerate(infos) if inf["terminated"] or inf["truncated"]]
        if need_reset and self.respawn_mode in ("soft", "delayed"):
            for i in need_reset:
                self._mark_inactive(self.robots[i])
        elif need_reset:
            for i in need_reset:
                self._respawn_bot(self.robots[i], randomize=True)

        revived = self._process_inactive()
        if need_reset or revived:
            mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs_batch()

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminated, np.array([inf["truncated"] for inf in infos]), infos

    def _get_obs_batch(self) -> np.ndarray:
        return np.stack([bot.get_obs(self.data) for bot in self.robots], axis=0)

    def render(self):
        if self.render_mode is None:
            return None
        if self._viewer is None:
            import mujoco.viewer

            print("[arena] 正在打开 MuJoCo 3D 窗口…", flush=True)
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("[arena] 窗口已打开（鼠标旋转视角，关闭窗口结束）", flush=True)
            # 拉远相机，俯瞰整个机器人阵列
            self._viewer.cam.lookat[:] = (0.0, 0.0, 0.08)
            self._viewer.cam.distance = max(
                self._camera_distance_hint, self.spawn_span * 1.55 + 2.5
            )
            self._viewer.cam.elevation = -22
            self._viewer.cam.azimuth = 140
            self._last_render_time = time.time()
        # 时间基限帧：最多 30 FPS，防止 viewer.sync() 阻塞物理循环导致卡顿
        now = time.time()
        min_interval = 1.0 / 30.0
        if now - self._last_render_time >= min_interval:
            self._viewer.sync()
            self._last_render_time = now
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


try:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv as _VecEnv
except ImportError:  # pragma: no cover
    _VecEnv = object  # type: ignore


class HexapodMultiVecEnv(_VecEnv):
    """Stable-Baselines3 兼容的 VecEnv：num_envs = n_robots。"""

    def __init__(self, arena: HexapodMultiArena):
        if _VecEnv is object:
            raise ImportError("需要 stable-baselines3: pip install -r requirements-rl.txt")
        super().__init__(
            arena.n_robots,
            arena.observation_space,
            arena.action_space,
        )
        self.arena = arena
        self.metadata = arena.metadata
        self.render_mode = arena.render_mode
        self._actions: Optional[np.ndarray] = None

    def reset(self):
        obs, _ = self.arena.reset()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self.arena.step(self._actions)
        dones = np.logical_or(terminated, truncated)
        return obs, rewards, dones, infos

    def close(self) -> None:
        self.arena.close()

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * self.num_envs

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self.arena, attr_name)] * self.num_envs

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self.arena, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        return [getattr(self.arena, method_name)(*args, **kwargs)] * self.num_envs

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

    def getattr_depth_check(self, name: str, already_found: bool) -> str:
        return name

    def getattr(self, name: str):
        if hasattr(self.arena, name):
            return getattr(self.arena, name)
        raise AttributeError(name)
