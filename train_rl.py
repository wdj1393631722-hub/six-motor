#!/usr/bin/env python3
"""
平地行走强化学习训练 — PPO + 步态残差策略。

用法:
  python train_rl.py                    # 默认 50 万步
  python train_rl.py --steps 2000000    # 更长训练
  python train_rl.py --eval             # 加载最新模型可视化
"""
from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DEFAULT_SAVE = os.path.join(SCRIPT_DIR, "rl_models", "ppo_flat_walk")


def _resolve_device(requested: str) -> str:
    """auto → 有 CUDA 则用 GPU；否则 CPU。"""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("警告: 未检测到 CUDA，回退到 CPU")
        return "cpu"
    return requested


def _print_hw_plan(device: str, n_envs: int) -> None:
    import torch

    print("── 硬件分工 ──")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"  GPU (PPO 神经网络): {torch.cuda.get_device_name(0)}")
    else:
        print("  GPU: 未使用（策略网络在 CPU）")
    print(f"  CPU (MuJoCo 物理仿真): {n_envs} 个并行环境")
    print(
        "  说明: 本项目的仿真在 CPU 上跑；4090 负责策略网络的矩阵运算。"
        "提速主要靠增加 --n-envs（建议 16~24）。"
    )


def train(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from hexapod_rl_env import HexapodFlatWalkEnv

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "tb")
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    def _make():
        return Monitor(
            HexapodFlatWalkEnv(
                control_dt=0.02,
                max_episode_steps=args.max_steps,
                residual_scale=args.residual_scale,
                reward_mode=args.reward_mode,
                vx_cmd=args.vx_cmd,
            )
        )

    if args.n_envs > 1:
        env = make_vec_env(
            _make,
            n_envs=args.n_envs,
            vec_env_cls=SubprocVecEnv,
            seed=args.seed,
        )
        eval_env = DummyVecEnv([_make])
    else:
        env = DummyVecEnv([_make])
        eval_env = DummyVecEnv([_make])

    device = _resolve_device(args.device)

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
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        seed=args.seed,
        device=device,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(10_000 // max(args.n_envs, 1), 1),
            save_path=ckpt_dir,
            name_prefix="ppo_hex",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=args.save_dir,
            log_path=os.path.join(args.save_dir, "eval"),
            eval_freq=max(20_000 // max(args.n_envs, 1), 1),
            n_eval_episodes=3,
            deterministic=True,
        ),
    ]

    print(f"训练开始: {args.total_steps} 步 | 并行环境 {args.n_envs} | device={device}")
    _print_hw_plan(device, args.n_envs)
    print(f"模型保存: {args.save_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")

    model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=True)
    final_path = os.path.join(args.save_dir, "ppo_flat_walk_final")
    model.save(final_path)
    print(f"训练完成，已保存: {final_path}.zip")


def _run_viewer_loop(env, action_fn, label: str) -> None:
    """MuJoCo 3D 窗口循环，关闭窗口即退出。"""
    import time

    print(f"正在打开 MuJoCo 3D 窗口（{label}）…")
    print("操作：鼠标拖拽旋转视角，滚轮缩放，关闭窗口结束。")

    obs, _ = env.reset()
    if env.render_mode == "human":
        env.render()
    total_r = 0.0
    episode = 1
    while True:
        action = action_fn(obs, env)
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        if term or trunc:
            print(
                f"[回合 {episode}] return={total_r:.1f} "
                f"v_fwd={info.get('v_forward', 0):.4f} fallen={info.get('fallen')}"
            )
            obs, _ = env.reset()
            total_r = 0.0
            episode += 1
        if env._viewer is not None and not env._viewer.is_running():
            break
        time.sleep(env.control_dt)
    env.close()
    print("窗口已关闭。")


def watch_demo(args: argparse.Namespace) -> None:
    """仅规则步态 + 零残差，无需训练模型即可看机器人走动。"""
    import numpy as np

    from hexapod_rl_env import HexapodFlatWalkEnv

    env = HexapodFlatWalkEnv(
        render_mode="human",
        max_episode_steps=args.max_steps,
        residual_scale=args.residual_scale,
    )

    def zero_action(_obs, env_):
        return np.zeros(env_.action_space.shape, dtype=np.float32)

    _run_viewer_loop(env, zero_action, "规则三角步态演示")


def evaluate(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO

    from hexapod_rl_env import HexapodFlatWalkEnv

    model_path = args.model
    if model_path is None:
        for name in ("best_model", "ppo_flat_walk_final"):
            p = os.path.join(args.save_dir, name + ".zip")
            if os.path.isfile(p):
                model_path = p
                break
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"未找到模型，请先训练，或用演示模式: bash run.sh rl-view"
        )

    device = _resolve_device(args.device)
    print(f"加载模型: {model_path} (device={device})")
    model = PPO.load(model_path, device=device)
    env = HexapodFlatWalkEnv(
        render_mode="human",
        max_episode_steps=args.max_steps,
        residual_scale=args.residual_scale,
    )

    def policy_action(obs, _env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    _run_viewer_loop(env, policy_action, f"PPO 策略 · {os.path.basename(model_path)}")


def smoke_test(view: bool = False) -> None:
    """零策略 + 步态残差=0，确认环境可跑。"""
    import numpy as np

    from hexapod_rl_env import HexapodFlatWalkEnv

    if view:
        watch_demo(argparse.Namespace(max_steps=500, residual_scale=0.06))
        return

    env = HexapodFlatWalkEnv(max_episode_steps=100)
    obs, _ = env.reset()
    total = 0.0
    for _ in range(100):
        obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        total += r
        if term or trunc:
            break
    print(
        f"smoke OK: return={total:.2f} v_fwd={info.get('v_forward', 0):.4f} "
        f"fallen={info.get('fallen')}"
    )
    env.close()


def main():
    parser = argparse.ArgumentParser(description="六足平地行走 PPO 训练")
    parser.add_argument("--steps", type=int, default=500_000, help="总训练步数")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=16,
        help="并行 MuJoCo 环境数（4090 机器建议 16~24，吃满 CPU）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PPO 策略网络设备（auto=有 GPU 则用 CUDA）",
    )
    parser.add_argument("--max-steps", type=int, default=500, help="每回合最大步数")
    parser.add_argument("--residual-scale", type=float, default=0.08, help="残差幅度(rad)")
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="max_speed",
        choices=["track", "max_speed"],
        help="track=跟踪速度; max_speed=尽量跑快",
    )
    parser.add_argument("--vx-cmd", type=float, default=None, help="步态基准速度 m/s")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps-rollout", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE)
    parser.add_argument("--eval", action="store_true", help="加载 PPO 模型，MuJoCo 3D 窗口评估")
    parser.add_argument(
        "--view",
        action="store_true",
        help="打开 MuJoCo 3D 窗口（无模型=规则步态演示；配合 --eval=加载策略）",
    )
    parser.add_argument("--model", type=str, default=None, help="评估用模型路径 .zip")
    parser.add_argument("--smoke", action="store_true", help="仅测试环境（加 --view 可看 3D）")
    args = parser.parse_args()
    args.total_steps = args.steps

    if args.smoke:
        smoke_test(view=args.view)
    elif args.view and not args.eval:
        watch_demo(args)
    elif args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
