#!/usr/bin/env python3
"""船底式坡面地形（MuJoCo heightfield）的「唯一真源」。

build_slope_scene.py 用这里的尺寸写 XML 里的 <hfield>，
slope_demo.py 用这里的高程数据填 model.hfield_data —— 两者共用同一份参数，
保证 XML 声明的 nrow/ncol/size 与运行时写入的高程永远一致、不会错位。

地形 = 纵向 ~ANGLE_DEG° 斜面（沿 +y 上坡）
       + 横向「凸脊」截面（中间高两边低，类比倒扣船底的外侧）
       + 轻微平滑起伏（类比船板焊缝/微变形）。

起点处用平滑包络把斜度与凸脊都从 0 渐入，所以坡面底边与平地齐平、无台阶。
"""
from __future__ import annotations

import numpy as np

# ---- 可调参数（改这里即可重塑坡面）------------------------------------------
# 默认值已调到「现成平地步态能直接爬上去」的范围：纯斜面本步态约能爬到 15°，
# 而沿凸脊正中线走是最陡的一条路，所以默认倾角取 12°、凸脊 0.14m、渐入拉长。
# 想要更陡（你最初想要的 ~20°）或更高的脊：把 ANGLE_DEG/CURV 调大即可改地形，
# 但当前步态会在坡脚打滑停住——那需要给步态加「随坡调姿」（俯仰补偿/重心前移），
# 是下一步的控制工作，不是地形问题。
ANGLE_DEG = 12.0     # 纵向倾角（°）：机器人沿 +y 往上爬的坡度（本步态可爬上限≈15°纯斜面）
RUNUP = 0.5          # 坡前平地助跑距离 (m)：机器人在平地起步再上坡
RX = 2.0             # 坡面半宽 (m) → 横向总宽 2*RX
RY = 2.0             # 坡面半长 (m) → 上坡方向总长 2*RY
NX = 121             # 横向网格点数（x），越大越平滑
NY = 221             # 纵向网格点数（y）
CURV = 0.14          # 凸脊高度 (m)：中轴比两边高多少（船底外凸程度）
NOISE_AMP = 0.025    # 起伏幅值 (m)：轻微不平整的最大高度
N_HARM = 4           # 起伏用的正弦谐波数（越多越细碎）
V_KNEE = 0.06        # 坡脚「拐点」平滑长度（占总长比例）：软化平地→斜面的折角
V_ONSET = 0.45       # 凸脊/起伏的渐入长度（占总长比例）：拉长→坡脚凸脊隆起更缓、不挡步
Z_BASE = 0.5         # heightfield 底座厚度 (m)
SEED = 7             # 起伏随机种子（固定→可复现）


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _elevation():
    """返回 (E, z_top)：E 为 (NY, NX) 物理高程 (m)，z_top 为归一化用的最大高程。"""
    slope_h = np.tan(np.deg2rad(ANGLE_DEG))
    vlen = 2.0 * RY                     # 坡面纵向物理长度 (m)
    v = np.linspace(0.0, 1.0, NY)       # 纵向归一化坐标 0→1（坡脚→坡顶）
    u = np.linspace(-1.0, 1.0, NX)      # 横向归一化坐标 -1→1（左→右）

    # 纵向斜面：对斜率做坡脚软化后累积积分 → 折角处平滑过渡
    dv = 1.0 / (NY - 1)
    slope_profile = slope_h * _smoothstep(v / V_KNEE)
    incline = np.cumsum(slope_profile) * vlen * dv          # (NY,)

    # 横向凸脊：中间高两边低（船底外凸）
    ridge = CURV * (1.0 - u ** 2)                           # (NX,)

    # 渐入包络：坡脚处把凸脊与起伏从 0 拉起，避免起步台阶
    onset = _smoothstep(v / V_ONSET)                        # (NY,)

    # 轻微起伏：几条低频正弦叠加，平滑且可复现
    rng = np.random.default_rng(SEED)
    Xg, Yg = np.meshgrid(u, v)                              # (NY, NX)
    noise = np.zeros((NY, NX))
    for _ in range(N_HARM):
        fx, fy = rng.uniform(0.4, 1.4), rng.uniform(0.4, 1.6)
        px, py = rng.uniform(0.0, 2 * np.pi, size=2)
        amp = rng.uniform(0.5, 1.0)
        noise += amp * np.sin(np.pi * fx * Xg + px) * np.sin(2 * np.pi * fy * Yg + py)
    noise *= NOISE_AMP / (np.max(np.abs(noise)) + 1e-9)

    E = incline[:, None] + onset[:, None] * (ridge[None, :] + noise)
    E = np.clip(E, 0.0, None)                               # 去掉坡脚处的极小负坑
    z_top = float(E.max()) * 1.02
    return E, z_top


def build():
    """返回 heightfield 描述 dict，供 XML 写入与运行时填充共用。

    keys:
      nrow, ncol         网格点数（写入 <hfield>）
      size               (RX, RY, z_top, Z_BASE)，写入 <hfield size=...>
      pos                (0, RUNUP+RY, 0)，写入坡面 geom 的 pos（坡脚在 y=RUNUP）
      data               归一化高程 float32，flatten 后写入 model.hfield_data
      y_foot, y_top      坡脚 / 坡顶的世界 y 坐标，供 demo 判断终点
      z_top, angle_deg   便于打印/调试
    """
    E, z_top = _elevation()
    data = (E / z_top).clip(0.0, 1.0).astype(np.float32).ravel()  # 行主序，行=y
    y_foot = RUNUP
    y_top = RUNUP + 2.0 * RY
    return {
        "nrow": NY,
        "ncol": NX,
        "size": (RX, RY, z_top, Z_BASE),
        "pos": (0.0, RUNUP + RY, 0.0),
        "data": data,
        "y_foot": y_foot,
        "y_top": y_top,
        "z_top": z_top,
        "angle_deg": ANGLE_DEG,
    }


if __name__ == "__main__":
    h = build()
    print(f"船底坡面: 倾角={h['angle_deg']}°  长={2*RY}m 宽={2*RX}m")
    print(f"  网格 nrow×ncol = {h['nrow']}×{h['ncol']}")
    print(f"  size = {tuple(round(x,3) for x in h['size'])}  (RX RY z_top z_base)")
    print(f"  坡脚 y={h['y_foot']}  坡顶 y={h['y_top']}  顶部高度≈{h['z_top']:.2f}m")
