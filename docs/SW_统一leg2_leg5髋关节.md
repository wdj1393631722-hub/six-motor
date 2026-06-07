# SolidWorks：统一 leg2 / leg5 髋关节

在重新导出 URDF 前，仿真流水线会用 `prepare_model.py` 自动修补 leg2，使与 leg5 约定一致。你在 SW 里改完后，源 URDF 已是 Z 轴时修补会自动跳过。

## 目标（与 leg5 一致）

| 项目 | leg5（模板） | leg2（应改为） |
|------|-------------|----------------|
| 装配 | 单腿装配-5 | 单腿装配-4 |
| 髋座 | (-0.16, 0, 0) | (+0.16, 0, 0) |
| coxa 轴 | 0 0 **-1** | 0 0 **+1**（右侧镜像） |
| joint rpy | 0 0 0 | **0 0 0**（去掉 pitch=90°） |

## SW 操作步骤

1. 打开总装配 `SIX-MOTOR`，对比 **单腿装配-4** 与 **单腿装配-5** 在机身上的安装是否左右对称。
2. 进入 **单腿装配-4**，找到 URDF 导出用的 **髋旋转副**（髋连接版 / coxa）。
3. 将关节轴改为绕 **竖直 Z**（与 leg5 同类型；右侧腿用 +Z，左侧 leg5 为 -Z）。
4. 删除仅为导出产生的 **90° pitch** 配合偏移。
5. 站立零位下，大腿走向应与 leg5 镜像对称。
6. 重新导出到 `~/桌面/SIX-MOTOR/urdf` 与 `meshes/`。
7. 打开 `SIX-MOTOR.urdf`，确认 `leg2_coxa_joint` 为 `axis="0 0 1"`（或 `0 0 -1`）且 `rpy="0 0 0"`。

## 导出后验证

```bash
cd ~/hexapod_sim/six_motor
python prepare_model.py ~/桌面/SIX-MOTOR
python build_real_mjcf.py
python foot_kinematics.py
```

通过标准：leg2↔leg5 镜像误差明显小于 110 mm，且无需 leg2 脚底 +X 特例。
