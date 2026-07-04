#!/usr/bin/env python3
"""Viewer 键盘控制（避免与 MuJoCo 默认 W/S/A/D 等冲突）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

# GLFW 键码（与 mujoco.viewer 一致）
KEY_I = 73
KEY_J = 74
KEY_K = 75
KEY_L = 76
KEY_P = 80
KEY_B = 66
KEY_E = 69
KEY_D = 68
KEY_M = 77
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262
# 数字键 1-6：切换第 N 条腿磁力使能（GLFW 数字键码 = 48 + 数字）
KEY_DIGIT = {49: 1, 50: 2, 51: 3, 52: 4, 53: 5, 54: 6}

CONTROL_HELP = """
六足步态按键（关节三角步态：1/3/5 与 2/4/6 交替抬腿蹬进）:
  I 或 ↑   前进（coxa 前摆 + 六腿蹬进）
  K 或 ↓   后退
  J 或 ←   左转
  L 或 →   右转
  P        停止
  Z / X    站立时 关节撑起/降低 ±1mm（改 femur/tibia，足底仍贴地）
  S        站立时 保存关节角+高度到 stand_pose_flat.json
  B        重置为站立
  1-6      切换第 N 条腿磁力使能（通电吸附/断电）
  M        一键 全部磁力 通电/断电
鼠标仍用于旋转/平移视角（MuJoCo 默认）。
"""


@dataclass
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


def make_key_handler(
    cmd: VelocityCommand,
    *,
    max_v: float = 0.02,
    max_turn: float = 0.4,
    on_reset: Optional[Callable[[], None]] = None,
    on_enable: Optional[Callable[[], None]] = None,
    on_disable: Optional[Callable[[], None]] = None,
    magnets=None,
) -> Callable[[int], None]:
    """返回 mujoco.viewer.launch_passive 的 key_callback。

    magnets: 可选 LegMagnets 实例；提供后数字键 1-6 切换单腿磁力，M 键切换全部。
    """

    def key_callback(keycode: int) -> None:
        if keycode in (KEY_I, KEY_UP):
            cmd.vx = max_v
            cmd.omega = 0.0
            print(f"[步态] 前进 vx={max_v:.2f} m/s")
        elif keycode in (KEY_K, KEY_DOWN):
            cmd.vx = -max_v
            cmd.omega = 0.0
            print(f"[步态] 后退 vx={-max_v:.2f} m/s")
        elif keycode in (KEY_J, KEY_LEFT):
            cmd.vx = 0.0
            cmd.omega = max_turn
            print(f"[步态] 左转 omega={max_turn:.2f}")
        elif keycode in (KEY_L, KEY_RIGHT):
            cmd.vx = 0.0
            cmd.omega = -max_turn
            print(f"[步态] 右转 omega={-max_turn:.2f}")
        elif keycode == KEY_P:
            cmd.vx = cmd.vy = cmd.omega = 0.0
            print("[步态] 停止")
        elif keycode == KEY_B:
            cmd.vx = cmd.vy = cmd.omega = 0.0
            print("[步态] 重置为站立")
            if on_reset is not None:
                on_reset()
        elif magnets is not None and keycode in KEY_DIGIT:
            leg = KEY_DIGIT[keycode]
            on = magnets.toggle(leg)
            print(f"[磁力] 腿{leg} {'通电吸附' if on else '断电'} | {magnets.status_str()}")
        elif magnets is not None and keycode == KEY_M:
            on = magnets.toggle_all()
            print(f"[磁力] 全部 {'通电吸附' if on else '断电'} | {magnets.status_str()}")

    return key_callback
