#!/usr/bin/env python3
"""把船底式坡面 heightfield 注入机器人模型，生成 generated/SIX-MOTOR_slope.xml。

做法：读取原始 SIX-MOTOR_sim.xml 文本，在 <asset> 里加一条 <hfield> 声明，
在地面 floor geom 之后加一条 hfield geom。其余（机身、网格、灯光）原样保留。
高程数据不写进 XML（避免 PNG 量化台阶），由 slope_demo.py 运行时按浮点精度填充；
此处只需保证 nrow/ncol/size/pos 与 slope_terrain 一致。
"""
from __future__ import annotations

import os
import re

import slope_terrain

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml")
DST = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_slope.xml")


def main():
    h = slope_terrain.build()
    rx, ry, z_top, z_base = h["size"]
    px, py, pz = h["pos"]

    with open(SRC, encoding="utf-8") as f:
        xml = f.read()

    # 1) 在 </asset> 前插入 hfield 资源声明
    hfield_asset = (
        f'    <hfield name="ship_slope" nrow="{h["nrow"]}" ncol="{h["ncol"]}" '
        f'size="{rx:.4f} {ry:.4f} {z_top:.4f} {z_base:.4f}"/>\n  </asset>'
    )
    if "</asset>" not in xml:
        raise RuntimeError("源模型缺少 </asset>，无法注入 hfield")
    xml = xml.replace("</asset>", hfield_asset, 1)

    # 2) 在 floor geom 之后插入 hfield geom（摩擦/接触参数对齐地面，condim=3）
    hfield_geom = (
        f'\n    <geom name="ship_slope" type="hfield" hfield="ship_slope" '
        f'pos="{px:.4f} {py:.4f} {pz:.4f}" material="mat_ground" '
        f'friction="2.0 0.05 0.0001" solref="0.004 1.2" solimp="0.92 0.98 0.001" condim="3"/>'
    )
    new_xml, n = re.subn(r'(<geom name="floor"[\s\S]*?/>)', r'\1' + hfield_geom, xml, count=1)
    if n != 1:
        raise RuntimeError("未找到 floor geom，无法注入坡面")
    xml = new_xml

    with open(DST, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"已生成坡面场景: {DST}")
    print(f"  倾角 {h['angle_deg']}°，坡脚 y={h['y_foot']}，坡顶 y={h['y_top']}，顶高≈{z_top:.2f}m")


if __name__ == "__main__":
    main()
