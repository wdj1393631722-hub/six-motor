#!/usr/bin/env python3
"""
同屏多机器人可视化强化学习 — GPU 训练 + MuJoCo 3D 阵列窗口。

用法:
  python train_rl_arena.py                    # 12 机器人同屏训练
  python train_rl_arena.py --n-robots 16      # 16 机器人
  python train_rl_arena.py --eval             # 仅可视化已训练策略
"""
from __future__ import annotations

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DEFAULT_SAVE = os.path.join(SCRIPT_DIR, "rl_models", "ppo_arena")


def _find_arena_model(save_dir: str, explicit: str | None) -> str | None:
    if explicit and os.path.isfile(explicit):
        return explicit
    candidates = [
        os.path.join(save_dir, n + ".zip")
        for n in ("ppo_arena_final", "best_model")
    ]
    flat = os.path.join(SCRIPT_DIR, "rl_models", "ppo_flat_walk")
    candidates += [
        os.path.join(flat, n + ".zip")
        for n in ("ppo_flat_walk_final", "best_model")
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("警告: 未检测到 CUDA，回退 CPU")
        return "cpu"
    return requested


class _ArenaWindowCallback:
    """窗口关闭时停止训练。"""

    def __init__(self, arena):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Cb(BaseCallback):
            def __init__(self, arena_):
                super().__init__()
                self._arena = arena_

            def _on_step(self) -> bool:
                v = self._arena._viewer
                if v is not None and not v.is_running():
                    return False
                return True

        self._cb_factory = lambda: _Cb(arena)

    def make(self):
        return self._cb_factory()


def train(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    from hexapod_rl_multi_env import HexapodMultiArena, HexapodMultiVecEnv

    device = _resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "tb")
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    arena = HexapodMultiArena(
        n_robots=args.n_robots,
        spacing=args.spacing,
        max_episode_steps=args.max_steps,
        residual_scale=args.residual_scale,
        render_mode="human",
        respawn_mode="soft",
        respawn_delay_steps=args.respawn_delay,
        render_stride=args.render_stride,
        reward_mode=args.reward_mode,
        vx_cmd=args.vx_cmd,
    )
    env = HexapodMultiVecEnv(arena)

    import torch

    print("══ 多机器人可视化 RL ══")
    print(f"  奖励模式: {args.reward_mode}" + (
        "（越快奖励越高）" if args.reward_mode == "max_speed" else ""
    ))
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"  GPU 策略网络: {torch.cuda.get_device_name(0)}")
    print(f"  同屏机器人数: {args.n_robots}")
    print(f"  场景文件: {arena.model_path}")
    print("  MuJoCo 窗口会实时显示所有机器人行走（同位置软重生，减少闪烁）")
    print("  关闭 MuJoCo 窗口可结束训练")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps_rollout,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        seed=args.seed,
        device=device,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(20_000 // max(args.n_robots, 1), 1),
                save_path=ckpt_dir,
                name_prefix="ppo_arena",
            ),
            _ArenaWindowCallback(arena).make(),
        ]
    )

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("用户中断训练。")
    finally:
        final = os.path.join(args.save_dir, "ppo_arena_final")
        model.save(final)
        print(f"模型已保存: {final}.zip")
        env.close()


def evaluate(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO

    from hexapod_rl_multi_env import HexapodMultiArena, HexapodMultiVecEnv

    device = _resolve_device(args.device)
    model_path = _find_arena_model(args.save_dir, args.model)
    if not model_path:
        raise FileNotFoundError("未找到模型，请先训练: bash run.sh rl-arena")

    model = PPO.load(model_path, device=device)
    arena = HexapodMultiArena(
        n_robots=args.n_robots,
        spacing=args.spacing,
        max_episode_steps=args.max_steps,
        residual_scale=args.residual_scale,
        render_mode="human",
        respawn_mode="soft",
        respawn_delay_steps=args.respawn_delay,
        render_stride=args.render_stride,
        reward_mode=args.reward_mode,
        vx_cmd=args.vx_cmd,
    )
    env = HexapodMultiVecEnv(arena)
    obs = env.reset()

    print(f"加载: {model_path}")
    print(f"同屏 {args.n_robots} 只机器人 — 关闭窗口结束")

    stochastic = args.stochastic
    print(f"策略模式: {'随机采样' if stochastic else '确定性'}（各机器人观测不同 → 动作不同）")

    try:
        while arena._viewer is None or arena._viewer.is_running():
            action, _ = model.predict(obs, deterministic=not stochastic)
            obs, _, _, _, _ = env.step(action)
            time.sleep(arena.control_dt)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def smoke(args: argparse.Namespace) -> None:
    import numpy as np

    from hexapod_rl_multi_env import HexapodMultiArena

    model = None
    model_path = _find_arena_model(args.save_dir, args.model)
    if model_path:
        from stable_baselines3 import PPO

        device = _resolve_device(args.device)
        model = PPO.load(model_path, device=device)
        print(f"已加载 RL 策略: {model_path}", flush=True)
    else:
        print("未找到已训练模型，使用随机残差演示（可先训练 rl-arena）", flush=True)

    print(
        f"演示: {args.n_robots} 只机器人随机分布 + 随机朝向/步态相位",
        flush=True,
    )
    arena = HexapodMultiArena(
        n_robots=args.n_robots,
        spacing=args.spacing,
        render_mode="human",
        max_episode_steps=200,
        random_layout=True,
        respawn_mode="soft",
        respawn_delay_steps=args.respawn_delay,
        render_stride=args.render_stride,
        reward_mode=args.reward_mode,
        vx_cmd=args.vx_cmd,
    )
    obs, _ = arena.reset()
    print(f"场景: {arena.model_path} | 奖励: {args.reward_mode}")
    print(f"观测 batch: {obs.shape} — 关闭窗口结束")
    rng = np.random.default_rng(args.seed)

    step_i = 0
    try:
        while arena._viewer is None or arena._viewer.is_running():
            if model is not None:
                act, _ = model.predict(obs, deterministic=not args.stochastic)
            else:
                # 无模型时：每只机器人不同随机残差，便于看出差异
                act = rng.uniform(-0.35, 0.35, (args.n_robots, 18)).astype(np.float32)
            obs, rew, _, _, infos = arena.step(act)
            step_i += 1
            if step_i == 1 or step_i % 100 == 0:
                print(
                    f"  step {step_i} | 平均奖励 {float(rew.mean()):.2f} | "
                    f"v_fwd≈{float(np.mean([inf['v_forward'] for inf in infos])):.4f}",
                    flush=True,
                )
            time.sleep(arena.control_dt)
    except KeyboardInterrupt:
        pass
    finally:
        arena.close()
    print("smoke OK")


def main():
    p = argparse.ArgumentParser(description="同屏多机器人可视化 RL")
    p.add_argument("--n-robots", type=int, default=12, help="同屏机器人数量")
    p.add_argument("--spacing", type=float, default=1.05, help="机器人间距(m)")
    p.add_argument("--steps", type=int, default=500_000, help="总环境步数")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--residual-scale", type=float, default=0.08)
    p.add_argument(
        "--reward-mode",
        type=str,
        default="max_speed",
        choices=["track", "max_speed"],
        help="track=跟踪固定速度; max_speed=尽量跑快",
    )
    p.add_argument(
        "--vx-cmd",
        type=float,
        default=None,
        help="步态基准前进速度 m/s（默认 max_speed=0.14, track=0.06）",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps-rollout", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=DEFAULT_SAVE)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--eval", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="策略随机采样（各机器人动作更有差异，推荐演示/评估时开启）",
    )
    p.add_argument("--model", type=str, default=None)
    p.add_argument(
        "--respawn-delay",
        type=int,
        default=60,
        help="摔倒后等待多少步再原地重生（减少闪烁，默认 60≈1.2s）",
    )
    p.add_argument(
        "--render-stride",
        type=int,
        default=2,
        help="每 N 步刷新一次 3D 画面（默认 2，越大越省资源）",
    )
    args = p.parse_args()

    if args.smoke:
        smoke(args)
    elif args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
