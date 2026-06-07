#!/usr/bin/env bash
# 六足仿真启动（自动用 venv 里的 python3）
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${ROOT}/../.venv"
PY="${VENV}/bin/python3"

if [[ ! -x "$PY" ]]; then
  echo "未找到虚拟环境，请先执行:"
  echo "  cd ~/hexapod_sim && bash setup.sh"
  exit 1
fi

cd "$ROOT"
case "${1:-locomotion}" in
  locomotion|loco|full)
    echo "启动 locomotion…"
    exec "$PY" -u run_locomotion.py
    ;;
  forward|fwd|tripod)
    exec "$PY" run_forward_tripod.py
    ;;
  calibrate|stand)
    exec "$PY" foot_kinematics.py
    ;;
  export|params)
    exec "$PY" export_robot_params.py
    ;;
  validate|check)
    exec "$PY" validate_robot_env.py "$@"
    ;;
  prone)
    exec "$PY" -c "
import mujoco
from foot_kinematics import load_stand_pose, make_prone_from_stand_pose, save_prone_pose, report_flatness
m = mujoco.MjModel.from_xml_path('generated/SIX-MOTOR_sim.xml')
stand, _ = load_stand_pose()
pose, bz = make_prone_from_stand_pose(m, stand, body_z=0.058)
print('saved', save_prone_pose(pose, bz))
report_flatness(m, pose, bz)
"
    ;;
  build)
    "$PY" prepare_model.py
    "$PY" build_real_mjcf.py
    ;;
  rl|train)
    shift
    exec "$PY" train_rl.py "$@"
    ;;
  rl-eval|eval-rl)
    shift
    exec "$PY" train_rl.py --eval "$@"
    ;;
  rl-view|view-rl|rl-demo)
    shift
    exec "$PY" train_rl.py --view "$@"
    ;;
  rl-arena|arena)
    shift
    exec "$PY" -u train_rl_arena.py "$@"
    ;;
  tune|tune-pose)
    shift
    exec "$PY" tune_pose.py "$@"
    ;;
  *)
    echo "用法: $0 [locomotion|forward|calibrate|prone|export|validate|build|tune|rl|rl-view|rl-eval|rl-arena]"
    echo "  locomotion  前进/后退/转弯（默认）"
    echo "  forward     仅前进三角步态"
    echo "  calibrate   标定站立姿态"
    echo "  prone       标定失能趴地姿态（足底贴地）"
    echo "  export      导出关节/主体参数（步态设计）"
    echo "  validate    部署前环境自检（自动用 venv）"
    echo "  build       生成 MuJoCo 模型"
    echo "  tune        手动调节姿态（例: bash run.sh tune --pose stand）"
    echo "  rl          平地行走 PPO 训练（例: bash run.sh rl --steps 500000 --device auto）"
    echo "  rl-view     MuJoCo 3D 窗口看规则步态（无需训练模型）"
    echo "  rl-eval     MuJoCo 3D 窗口看 PPO 策略行走"
    echo "  rl-arena    十几只机器人同屏可视化 RL + GPU（例: bash run.sh rl-arena --n-robots 12）"
    exit 1
    ;;
esac
