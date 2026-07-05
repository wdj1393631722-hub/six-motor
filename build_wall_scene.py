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
# 墙体尺寸（用有限厚度 box，非无限 plane）
WALL_HALF_X = 8.0       # 墙宽半长 (x)
WALL_HALF_Z = 4.0       # 墙高半长 (z)，立在地面上 → z∈[0, 2*HALF_Z]
WALL_THICK = 0.05       # 墙半厚 (y)；+Y 面位于 y=WALL_Y 即攀爬贴合面


def main():
    with open(SRC, encoding="utf-8") as f:
        xml = f.read()

    # 在 floor geom 之后插入竖直墙。**用有限厚度 box 代替无限 plane**：
    #   - plane 是单面渲染(背面透明像"空气墙")且碰撞无限大；
    #   - box 六面都渲染(两面可见)、碰撞为有限范围(可见外不再有隐形墙)。
    # box 的 +Y 面置于 y=WALL_Y(=0)，即机器人贴合/吸附的攀爬面；法向仍 +Y，物理不变。
    wall_geom = (
        f'\n    <geom name="wall" type="box" '
        f'size="{WALL_HALF_X:.2f} {WALL_THICK:.4f} {WALL_HALF_Z:.2f}" '
        f'pos="0 {WALL_Y - WALL_THICK:.4f} {WALL_HALF_Z:.4f}" material="mat_ground" '
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
    print(f"  竖直墙 box y面={WALL_Y:.2f} 宽{2*WALL_HALF_X:.0f}m 高{2*WALL_HALF_Z:.0f}m "
          f"厚{2*WALL_THICK:.2f}m 法向+Y（有限体积，无空气墙）")


if __name__ == "__main__":
    main()
