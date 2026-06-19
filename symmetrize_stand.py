#!/usr/bin/env python3
"""任务空间对称化站立姿态（不依赖 trimesh，只用 foot_pad geom）。

思路：腿的安装非关节镜像，故无法靠关节镜像得到对称。改为把六个足端摆成
左右镜像六边形（每对镜像腿足端 x 取反、y/z 相同），每条腿独立解
coxa/femur/tibia 命中各自目标且脚掌平贴地面。body_height 锁死不变。

镜像对: (1,6) (3,4) (2,5)，参考腿用左侧 (1,3,2)。
"""
from __future__ import annotations
import json
import math
import sys
import numpy as np
import mujoco

MODEL = "generated/SIX-MOTOR_sim.xml"
STAND = "generated/stand_pose_flat.json"
PAIRS = ((1, 6), (3, 4), (2, 5))
HIP = {
    1: (0.1, 0.16209), 6: (-0.1, 0.16209),
    2: (0.16, 0.002094), 5: (-0.16, 0.002094),
    3: (0.1, -0.15791), 4: (-0.1, -0.15791),
}
FOOT_Z_TARGET = 0.002  # 足底盒底目标高度（贴地）


def main():
    m = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(m)
    d = json.load(open(STAND))
    pose0 = dict(d["joints"])
    bz = float(d["body_height"])

    def jid(j):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)

    def qadr(j):
        return m.jnt_qposadr[jid(j)]

    def rng(leg, j):
        return tuple(m.jnt_range[jid(f"leg{leg}_{j}_joint")])

    def set_leg(pose, leg, c, f, t):
        data.qpos[:] = 0
        data.qpos[2] = bz
        data.qpos[3] = 1
        for jn, a in pose.items():
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
        zb = min(
            (pos + R @ (np.array([sx, sy, sz]) * half))[2]
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        )
        n = R @ np.array([0.0, 0.0, 1.0])
        tilt = n[0] ** 2 + n[1] ** 2  # 水平分量平方=倾斜代价
        return np.array([pos[0], pos[1], zb]), tilt

    # 理想正六边形方位角（deg）：髋座布局本就≈这些角，左右镜像对称
    IDEAL_AZ = {1: 60.0, 2: 0.0, 3: -60.0, 4: -120.0, 5: 180.0, 6: 120.0}

    def solve_leg(leg, R, c0):
        cr, fr, tr = rng(leg, "coxa"), rng(leg, "femur"), rng(leg, "tibia")
        az_t = math.radians(IDEAL_AZ[leg])

        def cost(c, f, t):
            set_leg(pose0, leg, c, f, t)
            p, tilt = pad(leg)
            # 只锁方位角(看起来规整), 半径放开; 脚贴地+平整
            az = math.atan2(p[1], p[0])
            daz = abs((az - az_t + math.pi) % (2 * math.pi) - math.pi)
            ez = (p[2] - FOOT_Z_TARGET) ** 2
            lim = 0.0
            for (lo, hi), q in ((cr, c), (fr, f), (tr, t)):
                if q < lo + 0.08:
                    lim += (lo + 0.08 - q) ** 2
                if q > hi - 0.08:
                    lim += (q - hi + 0.08) ** 2
            return (
                1500.0 * daz ** 2 + 800.0 * ez + 300.0 * tilt
                + 20.0 * lim + 0.05 * (c - c0) ** 2
            )

        cf = pose0[f"leg{leg}_coxa_joint"]
        ff = pose0[f"leg{leg}_femur_joint"]
        tf = pose0[f"leg{leg}_tibia_joint"]
        best = (cost(cf, ff, tf), cf, ff, tf)
        for c in np.linspace(cr[0], cr[1], 17):
            for f in np.linspace(fr[0], fr[1], 20):
                for t in np.linspace(tr[0], tr[1], 22):
                    e = cost(c, f, t)
                    if e < best[0]:
                        best = (e, c, f, t)
        for _ in range(2):
            _, c1, f1, t1 = best
            for c in np.linspace(max(cr[0], c1 - 0.12), min(cr[1], c1 + 0.12), 11):
                for f in np.linspace(max(fr[0], f1 - 0.12), min(fr[1], f1 + 0.12), 13):
                    for t in np.linspace(max(tr[0], t1 - 0.14), min(tr[1], t1 + 0.14), 15):
                        e = cost(c, f, t)
                        if e < best[0]:
                            best = (e, c, f, t)
        return best[1], best[2], best[3]

    out = dict(pose0)
    for leg in range(1, 7):
        c0 = pose0[f"leg{leg}_coxa_joint"]
        c, f, t = solve_leg(leg, 0.0, c0)
        out[f"leg{leg}_coxa_joint"] = float(c)
        out[f"leg{leg}_femur_joint"] = float(f)
        out[f"leg{leg}_tibia_joint"] = float(t)

    # 校验
    data.qpos[:] = 0
    data.qpos[2] = bz
    data.qpos[3] = 1
    for jn, a in out.items():
        data.qpos[qadr(jn)] = a
    mujoco.mj_forward(m, data)
    print("PY === 对称化结果 (body_z=%.1fmm) ===" % (bz * 1000))
    zs = []
    feet = {}
    for leg in range(1, 7):
        p, tilt = pad(leg)
        feet[leg] = p
        zs.append(p[2])
        hx, hy = HIP[leg]
        az = math.degrees(math.atan2(p[1] - hy, p[0] - hx))
        print(
            "PY leg%d c=%+.3f f=%+.3f t=%+.3f | foot(%.3f,%.3f) z=%5.1fmm tilt=%4.2f° az=%+6.1f"
            % (leg, out[f"leg{leg}_coxa_joint"], out[f"leg{leg}_femur_joint"],
               out[f"leg{leg}_tibia_joint"], p[0], p[1], p[2] * 1000,
               math.degrees(math.acos(min(1, math.sqrt(max(0, 1 - tilt))))), az)
        )
    print("PY 足高极差=%.2fmm 范围[%.1f,%.1f]" % (
        1000 * (max(zs) - min(zs)), 1000 * min(zs), 1000 * max(zs)))
    print("PY 左右镜像误差 (足端 x和应≈0, y差应≈0):")
    for ref, mir in PAIRS:
        dx = feet[ref][0] + feet[mir][0]
        dy = feet[ref][1] - feet[mir][1]
        dz = feet[ref][2] - feet[mir][2]
        print("PY  leg%d/leg%d  x和=%+.4f y差=%+.4f z差=%+.4f m" % (ref, mir, dx, dy, dz))

    json.dump({"body_height": bz, "joints": out},
              open("/tmp/sym_pose.json", "w"), indent=2, ensure_ascii=False)
    print("PY 候选已写 /tmp/sym_pose.json")


if __name__ == "__main__":
    main()
