#!/usr/bin/env python3
"""用真实运动学把六足摆成正六边形（等半径、等角60°、贴地平整）。

模型六腿安装朝向相同、髋座非等半径，故站立角层面凑不出正六边形。这里直接
对真实模型做足端 IK：足端目标 = 共同半径 R、方位角锁 60° 等分、脚底贴地。
同时扫描机身高度 body_z，找到能让正六边形落地的高度。
"""
from __future__ import annotations
import json
import math
import numpy as np
import mujoco

MODEL = "generated/SIX-MOTOR_sim.xml"
OUT = "/tmp/hex_pose.json"
# 方位角锁到 60° 等分（与髋座方位最近、左右镜像对称）
IDEAL_AZ = {1: 60.0, 2: 0.0, 3: -60.0, 4: -120.0, 5: 180.0, 6: 120.0}
FOOT_Z = 0.002  # 足底盒底目标高度（贴地）

m = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(m)


def jid(j):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)


def qadr(j):
    return m.jnt_qposadr[jid(j)]


def rng(leg, j):
    return tuple(m.jnt_range[jid(f"leg{leg}_{j}_joint")])


ZERO = {f"leg{l}_{j}_joint": 0.0 for l in range(1, 7) for j in ("coxa", "femur", "tibia")}


def set_leg(bz, leg, c, f, t):
    data.qpos[:] = 0
    data.qpos[2] = bz
    data.qpos[3] = 1
    for jn, a in ZERO.items():
        data.qpos[qadr(jn)] = a
    data.qpos[qadr(f"leg{leg}_coxa_joint")] = c
    data.qpos[qadr(f"leg{leg}_femur_joint")] = f
    data.qpos[qadr(f"leg{leg}_tibia_joint")] = t
    mujoco.mj_forward(m, data)


def pad(leg):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"leg{leg}_foot_pad")
    pos = data.geom_xpos[g]
    R = data.geom_xmat[g].reshape(3, 3)
    half = m.geom_size[g]
    zb = min((pos + R @ (np.array([sx, sy, sz]) * half))[2]
             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1))
    n = R @ np.array([0.0, 0.0, 1.0])
    return np.array([pos[0], pos[1], zb]), n[0] ** 2 + n[1] ** 2


def solve_leg(leg, R, bz, fine=False):
    cr, fr, tr = rng(leg, "coxa"), rng(leg, "femur"), rng(leg, "tibia")
    az = math.radians(IDEAL_AZ[leg])
    tx, ty = R * math.cos(az), R * math.sin(az)

    def cost(c, f, t):
        set_leg(bz, leg, c, f, t)
        p, tilt = pad(leg)
        exy = (p[0] - tx) ** 2 + (p[1] - ty) ** 2
        ez = (p[2] - FOOT_Z) ** 2
        lim = 0.0
        for (lo, hi), q in ((cr, c), (fr, f), (tr, t)):
            if q < lo + 0.06:
                lim += (lo + 0.06 - q) ** 2
            if q > hi - 0.06:
                lim += (q - hi + 0.06) ** 2
        return 1000.0 * exy + 1000.0 * ez + 300.0 * tilt + 20.0 * lim

    nc, nf, nt = (17, 22, 24) if fine else (11, 14, 16)
    best = (1e18, 0.0, 0.0, 0.3)
    for c in np.linspace(cr[0], cr[1], nc):
        for f in np.linspace(fr[0], fr[1], nf):
            for t in np.linspace(tr[0], tr[1], nt):
                e = cost(c, f, t)
                if e < best[0]:
                    best = (e, c, f, t)
    for _ in range(2 if fine else 1):
        _, c1, f1, t1 = best
        for c in np.linspace(max(cr[0], c1 - 0.12), min(cr[1], c1 + 0.12), 9):
            for f in np.linspace(max(fr[0], f1 - 0.12), min(fr[1], f1 + 0.12), 11):
                for t in np.linspace(max(tr[0], t1 - 0.14), min(tr[1], t1 + 0.14), 13):
                    e = cost(c, f, t)
                    if e < best[0]:
                        best = (e, c, f, t)
    return best  # (cost,c,f,t)


def build(R, bz, fine=False):
    pose = dict(ZERO)
    tot = 0.0
    for leg in range(1, 7):
        e, c, f, t = solve_leg(leg, R, bz, fine)
        tot += e
        pose[f"leg{leg}_coxa_joint"] = float(c)
        pose[f"leg{leg}_femur_joint"] = float(f)
        pose[f"leg{leg}_tibia_joint"] = float(t)
    return pose, tot


def main():
    # 阶段1：粗扫 (R, body_z) 找正六边形能落地的组合
    best = None
    for bz in np.linspace(0.055, 0.085, 7):
        for R in np.linspace(0.34, 0.46, 7):
            _, tot = build(R, bz, fine=False)
            if best is None or tot < best[0]:
                best = (tot, R, bz)
        print("PY  bz=%.3f 扫描完, 当前最优 R=%.3f bz=%.3f cost=%.3f"
              % (bz, best[1], best[2], best[0]))
    _, R, bz = best
    print("PY 选定 R=%.3f body_z=%.3f" % (R, bz))

    # 阶段2：在最优点高精度求解
    pose, tot = build(R, bz, fine=True)

    # 校验
    data.qpos[:] = 0
    data.qpos[2] = bz
    data.qpos[3] = 1
    for jn, a in pose.items():
        data.qpos[qadr(jn)] = a
    mujoco.mj_forward(m, data)
    print("PY === 正六边形结果 (R=%.3f body_z=%.1fmm) ===" % (R, bz * 1000))
    radii, zs, azs = [], [], []
    for leg in range(1, 7):
        p, tilt = pad(leg)
        r = math.hypot(p[0], p[1])
        az = math.degrees(math.atan2(p[1], p[0]))
        radii.append(r); zs.append(p[2]); azs.append(az)
        print("PY leg%d c=%+.3f f=%+.3f t=%+.3f | foot(%.3f,%.3f) r=%.3f z=%5.1fmm tilt=%4.2f° az=%+6.1f"
              % (leg, pose[f"leg{leg}_coxa_joint"], pose[f"leg{leg}_femur_joint"],
                 pose[f"leg{leg}_tibia_joint"], p[0], p[1], r, p[2] * 1000,
                 math.degrees(math.acos(min(1, math.sqrt(max(0, 1 - tilt))))), az))
    print("PY 半径 mean=%.3f std=%.4f | 足高极差=%.2fmm" % (
        np.mean(radii), np.std(radii), 1000 * (max(zs) - min(zs))))
    order = sorted(range(6), key=lambda i: azs[i])
    sp = []
    for k in range(6):
        a0 = azs[order[k]]; a1 = azs[order[(k + 1) % 6]]
        d = (a1 - a0) % 360
        sp.append(d)
    print("PY 相邻方位间隔: " + " ".join("%.1f" % s for s in sp) + " (理想全=60)")

    json.dump({"body_height": bz, "joints": pose},
              open(OUT, "w"), indent=2, ensure_ascii=False)
    print("PY 候选已写 %s" % OUT)


if __name__ == "__main__":
    main()
