#!/usr/bin/env python3
"""在机器人模型里注入【竖直墙 + 墙角 45° 斜面过渡】，生成
generated/SIX-MOTOR_wall_ramp.xml（平地→斜面→竖直墙 连贯乘墙用）。

与 build_wall_scene.py 同构，但在 floor 之后额外插入：
  1) 竖直墙平面 wall：y=0，法向 +Y，机器人贴在 y>0 侧（同 wall 场景）。
  2) 斜面过渡 chamfer：一块 45° 斜面(用薄 box 表示)，把地面(z=0)与墙面(y=0)
     之间的尖直角"削"成斜坡——机器人可沿它从平地平滑爬上墙，避免尖角处
     中途悬臂托不住而翻落（实测尖直角纯腿/引导都难，斜面过渡是真实爬壁做法）。

斜面 box：绕世界 +X 转 -45°，上表面法向=(0, .707, .707)（朝 +Y+Z 即机器人一侧），
其顶面构成从 (y=CHAMFER, z=0) 到 (y=0, z=CHAMFER) 的斜坡；材料填在墙角三角内。
"""
from __future__ import annotations

import math
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml")
DST = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_wall_ramp.xml")

WALL_Y = 0.0
WALL_FRICTION = "3.5 0.12 0.0001"
CHAMFER = 0.36          # 斜面在 floor/wall 上的跨度 (m)：从 (y=CHAMFER,z=0) 到 (y=0,z=CHAMFER)
CHAMFER_THICK = 0.10    # 斜面 box 半厚（填入墙角材料侧）


def main():
    with open(SRC, encoding="utf-8") as f:
        xml = f.read()

    # 竖直墙（法向 +Y）
    wall = (
        f'\n    <geom name="wall" type="plane" size="8 8 0.05" '
        f'pos="0 {WALL_Y:.4f} 0" euler="-1.5708 0 0" material="mat_ground" '
        f'friction="{WALL_FRICTION}" solref="0.004 1.2" solimp="0.92 0.98 0.001" condim="3"/>'
    )
    # 45° 斜面过渡（薄 box，绕 +X 转 -45°，顶面即斜坡）
    n = 0.7071067811865476           # cos45=sin45
    cy = CHAMFER / 2.0 - CHAMFER_THICK * n
    cz = CHAMFER / 2.0 - CHAMFER_THICK * n
    hy = CHAMFER / math.sqrt(2.0)    # 斜面宽度半长
    chamfer = (
        f'\n    <geom name="chamfer" type="box" '
        f'size="8 {hy:.4f} {CHAMFER_THICK:.4f}" '
        f'pos="0 {cy:.4f} {cz:.4f}" euler="-0.785398 0 0" material="mat_ground" '
        f'friction="{WALL_FRICTION}" solref="0.004 1.2" solimp="0.92 0.98 0.001" condim="3"/>'
    )
    new_xml, k = re.subn(
        r'(<geom name="floor"[\s\S]*?/>)', r'\1' + wall + chamfer, xml, count=1
    )
    if k != 1:
        raise RuntimeError("未找到 floor geom，无法注入墙面/斜面")

    with open(DST, "w", encoding="utf-8") as f:
        f.write(new_xml)
    print(f"已生成斜面过渡场景: {DST}")
    print(f"  竖直墙 y={WALL_Y:.2f} 法向+Y | 45°斜面 跨度{CHAMFER:.2f}m")


if __name__ == "__main__":
    main()
