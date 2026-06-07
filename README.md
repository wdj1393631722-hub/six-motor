# SIX-MOTOR 六足 MuJoCo 仿真

模型路径（默认）：`/media/dash/OS/six-motor-URDF/SIX-MOTOR-NEW`（SW 重导出 URDF + STL）

旧模型仍可用：`export SIX_MOTOR_PATH=~/桌面/SIX-MOTOR python prepare_model.py`

## 1. 安装环境（Ubuntu，一次性）

```bash
cd ~/hexapod_sim
bash setup.sh
source .venv/bin/activate
```

## 2. 生成仿真模型

```bash
cd ~/hexapod_sim/six_motor
python3 prepare_model.py
# 或指定目录: python3 prepare_model.py /media/dash/OS/six-motor-URDF/SIX-MOTOR-NEW
python3 build_real_mjcf.py
```

说明：使用 SW 导出的 **真实 STL**（`base_link` = 六边上板 + 6×RS03，各腿 coxa/femur/tibia），关节树与 `SIX-MOTOR.urdf` 一致。面数在 MuJoCo 上限内做高精度简化（机身约 20 万面、髋部约 8 万面、小腿保持原模）。

## 3. 运行仿真（前进 / 后退 / 转向）

```bash
source ~/hexapod_sim/.venv/bin/activate
python3 run_locomotion.py
# 或不用 activate：
# ~/hexapod_sim/.venv/bin/python3 run_locomotion.py
# 或：
# bash run.sh
# bash run.sh forward
```

说明：Debian/Ubuntu 上通常没有 `python` 命令，请用 **`python3`** 或上面的 **`run.sh`**。

| 按键 | 功能 |
|------|------|
| I 或 ↑ | 前进 |
| K 或 ↓ | 后退 |
| J 或 ← | 左转 |
| L 或 → | 右转 |
| P | 停止 |
| B | 重置站立 |

（不用 W/A/S/D，避免与 MuJoCo 默认快捷键冲突，例如 W 会切换线框显示。）

仿真场景含 **15×15 m 棋盘格平面地貌**（摩擦系数适合六足行走），带阴影与默认俯视视角。

足底平行标定（使小腿底部贴地、不斜戳）：

```bash
python3 foot_kinematics.py   # 生成 generated/stand_pose_flat.json（含左右对称）
```

对称配对（相对机体 YZ 平面）: leg1↔leg6、leg3↔leg4、leg2↔leg5。  
SW 导出 URDF 的关节 0 位并不左右对称，仿真以 `stand_pose_flat.json` 为对称站立零位。

**leg2 / leg5 髋轴统一：** 若 SW 仍导出为 X 轴 + pitch90°，`prepare_model.py` 会自动改为 Z 轴（与 leg5 同约定）。SW 重导出后若已是 Z 轴则跳过。详见 [docs/SW_统一leg2_leg5髋关节.md](docs/SW_统一leg2_leg5髋关节.md)。

## 4. 三角步态运动学（Tripod Gait）

| 组 | 腿 | 相位 |
|----|-----|------|
| Tripod A | leg1, leg3, leg5 | 与 B 交替摆动/支撑 |
| Tripod B | leg2, leg4, leg6 | 各占半周期 |

规划链路（`tripod_planner.py` → `leg_ik.py` → `gait.py`）：

1. **足端轨迹**（机体 `base_link` 系）：摆动相抬脚 + 前后摆；支撑相足端相对机体后移。
2. **逆运动学**：MuJoCo 正解 + 数值 IK，将足端目标转为 coxa/femur/tibia。
3. **速度指令**：`vx`/`vy` 定步向，`omega` 按腿方位角加切向偏移。

```bash
python3 run_locomotion.py   # I/K 前进后退，J/L 转向
```

可调参数见 `gait.GaitParams`：`cycle_time`、`stride_length`、`step_height`。

## 5. 调试顺序

1. `build_real_mjcf.py` 能打印 `nq=25 nu=18 nmesh=19`
2. `foot_kinematics.py` 生成 `stand_pose_flat.json`
3. `run_locomotion.py` 能站立不塌陷
4. 调 `GaitParams`：`stride_length`、`step_height`、`cycle_time`
5. 确认 `TRIPOD_A/B` 与真机安装一致

## 6. 文件说明

| 文件 | 作用 |
|------|------|
| `prepare_model.py` | 修复 URDF 路径与关节限位 |
| `build_real_mjcf.py` | 生成真实外形 `generated/SIX-MOTOR_sim.xml` |
| `decimate_meshes.py` | 简化 STL 面数（MuJoCo 限制） |
| `tripod_planner.py` | 三角步态足端轨迹（运动学规划） |
| `leg_ik.py` | 足端 → 关节角逆解 |
| `gait.py` | 步态控制器（整合规划 + IK） |
| `run_locomotion.py` | 主仿真程序 |

## 7. 与真机 1:1 的说明

| 项目 | 当前状态 |
|------|----------|
| 关节位置 / 轴向 | 与 URDF 一致（MuJoCo 原生导入） |
| 机身外形 | `base_link.STL`（含上板 + 髋电机） |
| 腿外形 | 各 leg `*_coxa/femur/tibia.STL` |
| 面数 | 为载入 MuJoCo 做了简化，非 SW 原始 53 万面 |

若要在 SW 里进一步贴近真机：单独导出「六边上板」并入 `base_link`，或导出低面数但保留孔细节的 STL；改 `decimate_meshes.py` 里 `MAX_FACES` 可再调高（单 mesh 建议 ≤199000）。

运行 `python calibrate_stand.py` 可在换 mesh 后重新标定站立高度。

## 7. 下一步（运动控制）

- 用真实连杆长度标定 `nominal_stand_pose()`（或 IK）
- 加入足端位置 IK 替代关节角开环
- 接入 IMU/关节反馈做闭环
