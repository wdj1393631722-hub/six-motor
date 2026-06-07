#!/usr/bin/env python3
"""生成实心六角上板 STL（无中心孔、无 RS03 电机）。"""
import os

import trimesh
import numpy as np

OUT = os.path.join(
    os.path.dirname(__file__), "generated", "meshes_decimated", "base_plate_only.STL"
)

# 与原始六边上板外廓接近（米）
HEX_RADIUS = 0.225   # 外接圆半径 ~0.45m 对边距
HEIGHT = 0.058
Z_TOP = 0.0          # 顶面在 base_link 坐标 z=0


def main():
    # 正六棱柱（实心，顶面平整无孔）
    plate = trimesh.creation.cylinder(
        radius=HEX_RADIUS, height=HEIGHT, sections=6, transform=None
    )
    plate.apply_translation([0, 0, Z_TOP - HEIGHT / 2])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plate.export(OUT)
    print(f"已生成: {OUT} ({len(plate.faces)} 面)")


if __name__ == "__main__":
    main()
