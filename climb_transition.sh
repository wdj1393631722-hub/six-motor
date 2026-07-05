#!/usr/bin/env bash
# 一键启动【平地→斜面→垂直墙 连贯乘墙】：开窗 + 自动把 MuJoCo 窗口置顶到最前。
# 流程：平地按 I 前进到斜面脚下 → 按 T 触发乘墙(沿45°斜面贴坡爬上墙) → 竖直后 I/K/J/L 爬墙。
# 首次运行自动生成斜面过渡场景 SIX-MOTOR_wall_ramp.xml。
set -e
cd "$(cd "$(dirname "$0")" && pwd)"
export DISPLAY="${DISPLAY:-:0}"
PY="../.venv/bin/python3"

# 后台监视：MuJoCo 窗口一出现就激活、移到显眼位置、置顶
(
  for _ in $(seq 1 40); do
    WID=$(xdotool search --class MuJoCo 2>/dev/null | head -1)
    if [ -n "$WID" ]; then
      wmctrl -ia "$WID" 2>/dev/null || true
      xdotool windowactivate "$WID" 2>/dev/null || true
      xdotool windowmove "$WID" 300 150 2>/dev/null || true
      xdotool windowraise "$WID" 2>/dev/null || true
      break
    fi
    sleep 0.5
  done
) &

# 前台跑（当前终端即控制终端，viewer 线程才能正常开窗）
exec "$PY" -u floor_to_wall_demo.py
