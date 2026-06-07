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
    exec "$PY" run_locomotion.py
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
  prone)
    exec "$PY" -c "
import mujoco
from foot_kinematics import calibrate_prone_pose, load_stand_pose, save_prone_pose, report_flatness
m = mujoco.MjModel.from_xml_path('generated/SIX-MOTOR_sim.xml')
stand = load_stand_pose()[0]
pose, bz = calibrate_prone_pose(m, stand_pose=stand)
print('saved', save_prone_pose(pose, bz))
report_flatness(m, pose, bz)
"
    ;;
  build)
    "$PY" prepare_model.py
    "$PY" build_real_mjcf.py
    ;;
  *)
    echo "用法: $0 [locomotion|forward|calibrate|prone|export|build]"
    echo "  locomotion  前进/后退/转弯（默认）"
    echo "  forward     仅前进三角步态"
    echo "  calibrate   标定站立姿态"
    echo "  prone       标定失能趴地姿态（足底贴地）"
    echo "  export      导出关节/主体参数（步态设计）"
    echo "  build       生成 MuJoCo 模型"
    exit 1
    ;;
esac
