#!/usr/bin/env bash
# 一键启动【平地→斜面→垂直墙 连贯乘墙】：开窗 + 自动把 MuJoCo 窗口置顶到最前。
# 流程：平地按 I 前进到斜面脚下 → 按 T 触发乘墙(沿45°斜面贴坡爬上墙) → 竖直后 I/K/J/L 爬墙。
# 首次运行自动生成斜面过渡场景 SIX-MOTOR_wall_ramp.xml。
set -e
cd "$(cd "$(dirname "$0")" && pwd)"
export DISPLAY="${DISPLAY:-:0}"
PY="../.venv/bin/python3"

echo "启动中：先做步态标定(几秒)，随后会自动弹出 MuJoCo 窗口并置于最前……"

# 后台监视：MuJoCo 窗口一出现就激活/移到显眼位置/置顶；持续 ~30s 反复置顶，
# 防止被终端盖住或窗口管理器抢焦点（按 class 或 name 双重查找更稳）。
(
  moved=""
  for _ in $(seq 1 60); do
    WID=$(xdotool search --class MuJoCo 2>/dev/null | head -1)
    [ -z "$WID" ] && WID=$(xdotool search --name "MuJoCo : " 2>/dev/null | head -1)
    if [ -n "$WID" ]; then
      wmctrl -ia "$WID" 2>/dev/null || true
      xdotool windowactivate "$WID" 2>/dev/null || true
      if [ -z "$moved" ]; then
        xdotool windowmove "$WID" 300 150 2>/dev/null || true
        xdotool windowsize "$WID" 1100 800 2>/dev/null || true
        moved=1
        echo "MuJoCo 窗口已打开（如被终端挡住，看任务栏或 Alt+Tab 切过去）。"
      fi
      xdotool windowraise "$WID" 2>/dev/null || true
    fi
    sleep 0.5
  done
) &

# 前台跑（当前终端即控制终端，viewer 线程才能正常开窗）
exec "$PY" -u floor_to_wall_demo.py
