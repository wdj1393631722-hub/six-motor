#!/usr/bin/env python3
"""在机器人模型里注入一面竖直墙，生成 generated/SIX-MOTOR_wall.xml（磁吸爬墙用）。

做法与 build_slope_scene.py 同构：读取原始 SIX-MOTOR_sim.xml 文本，在地面 floor
geom 之后插入一块**竖直平面**当墙，其余（机身、网格、灯光、actuator）原样保留。

墙是一块无限大平面（plane 的 size 只影响视觉、对碰撞无限大）：
  - pos="0 0 0"，euler="-1.5708 0 0" 把平面法向从世界 +Z 转到 **+Y**，
    即墙在 y=0 竖直面上、法向指向 +Y，机器人贴在 +Y 侧（feet 压向 -Y 触墙）。
  - 摩擦取足垫同款 3.5（抗滑），与磁吸配合把机器人牢牢固定在墙上。
保留 floor（z=0 地面）作参照/兜底：机器人万一脱墙会掉到地面而非穿出世界。
无需 hfield，也无需运行时填充数据——比坡面场景更简单。
"""
from __future__ import annotations

import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml")
DST = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_wall.xml")

# 墙面 y 位置（机器人贴在 y>0 一侧）与摩擦
WALL_Y = 0.0
WALL_FRICTION = "3.5 0.12 0.0001"


def main():
    with open(SRC, encoding="utf-8") as f:
        xml = f.read()

    # 在 floor geom 之后插入竖直墙平面（法向 +Y，高摩擦，condim=3）
    wall_geom = (
        f'\n    <geom name="wall" type="plane" size="8 8 0.05" '
        f'pos="0 {WALL_Y:.4f} 0" euler="-1.5708 0 0" material="mat_ground" '
        f'friction="{WALL_FRICTION}" solref="0.004 1.2" solimp="0.92 0.98 0.001" condim="3"/>'
    )
    new_xml, n = re.subn(
        r'(<geom name="floor"[\s\S]*?/>)', r'\1' + wall_geom, xml, count=1
    )
    if n != 1:
        raise RuntimeError("未找到 floor geom，无法注入墙面")
    xml = new_xml

    with open(DST, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"已生成壁面场景: {DST}")
    print(f"  竖直墙 y={WALL_Y:.2f}，法向 +Y，摩擦 {WALL_FRICTION}")


if __name__ == "__main__":
    main()
