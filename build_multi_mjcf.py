#!/usr/bin/env python3
"""从单机 SIX-MOTOR_sim.xml 生成同屏多机器人 MJCF（网格排列）。"""
from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "generated", "SIX-MOTOR_sim.xml")


def _grid_xy(n: int, spacing: float) -> list[tuple[float, float]]:
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    out: list[tuple[float, float]] = []
    for i in range(n):
        c = i % cols
        r = i // cols
        x = (c - (cols - 1) / 2.0) * spacing
        y = (r - (rows - 1) / 2.0) * spacing
        out.append((x, y))
    return out


def _scatter_xy(
    n: int,
    span: float,
    min_dist: float,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """随机散布，保证最小间距。"""
    import random

    rng = random.Random(seed)
    pts: list[tuple[float, float]] = []
    for i in range(n):
        placed = False
        for _ in range(500):
            x = rng.uniform(-span, span)
            y = rng.uniform(-span, span)
            if all((x - px) ** 2 + (y - py) ** 2 >= min_dist**2 for px, py in pts):
                pts.append((x, y))
                placed = True
                break
        if not placed:
            pts.append(_grid_xy(n, min_dist * 1.1)[i])
    return pts


def random_spawn_positions(
    n: int,
    rng: np.random.Generator,
    span: float = 3.5,
    min_dist: float = 0.85,
) -> list[tuple[float, float]]:
    """运行时随机出生点（每次 reset 可不同）。"""
    pts: list[tuple[float, float]] = []
    for i in range(n):
        for _ in range(800):
            x = float(rng.uniform(-span, span))
            y = float(rng.uniform(-span, span))
            if all((x - px) ** 2 + (y - py) ** 2 >= min_dist**2 for px, py in pts):
                pts.append((x, y))
                break
        else:
            pts.append(_grid_xy(n, min_dist * 1.15)[i])
    return pts


def _prefix_tree(elem: ET.Element, prefix: str) -> None:
    """为 body/joint/geom/site/actuator 名称加前缀；mesh 资源名保持不变。"""
    if elem.tag == "site" and "name" in elem.attrib:
        val = elem.attrib["name"]
        if not val.startswith(prefix):
            elem.attrib["name"] = f"{prefix}{val}"
    else:
        for key in ("name", "joint"):
            if key in elem.attrib:
                val = elem.attrib[key]
                if val.startswith(prefix):
                    continue
                if not val.startswith("r") or "_leg" not in val:
                    if key == "joint" or elem.tag in (
                        "body",
                        "joint",
                        "geom",
                        "actuator",
                    ):
                        elem.attrib[key] = f"{prefix}{val}"
    if elem.tag == "geom" and "foot_pad" in elem.attrib.get("name", ""):
        elem.attrib["contype"] = "2"
        elem.attrib["conaffinity"] = "1"
    for child in list(elem):
        _prefix_tree(child, prefix)


def build_multi_mjcf(
    n_robots: int = 12,
    spacing: float = 1.05,
    src_path: str = SRC,
    out_path: str | None = None,
    force: bool = False,
    layout: str = "scatter",
) -> str:
    if out_path is None:
        tag = "scatter" if layout == "scatter" else "grid"
        out_path = os.path.join(
            SCRIPT_DIR, "generated", f"SIX-MOTOR_multi_{n_robots}_{tag}.xml"
        )
    if not force and os.path.isfile(out_path):
        return out_path
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"缺少单机模型: {src_path}")

    tree = ET.parse(src_path)
    root = tree.getroot()
    world = root.find("worldbody")
    actuator_sec = root.find("actuator")
    sensor_sec = root.find("sensor")
    if world is None or actuator_sec is None:
        raise RuntimeError("XML 结构异常：缺少 worldbody / actuator")

    base_tpl: ET.Element | None = None
    for child in list(world):
        if child.tag == "body" and child.attrib.get("name") == "base_link":
            base_tpl = child
            break
    if base_tpl is None:
        raise RuntimeError("未找到 base_link body")

    world.remove(base_tpl)

    sensor_tpl: list[ET.Element] = []
    if sensor_sec is not None:
        sensor_tpl = [ET.fromstring(ET.tostring(s)) for s in list(sensor_sec)]
        sensor_sec.clear()
    else:
        sensor_sec = ET.SubElement(root, "sensor")

    half = max(22.0, 8.0 + 0.75 * math.sqrt(max(n_robots, 1)))
    floor = world.find(".//geom[@name='floor']")
    if floor is not None:
        floor.attrib["size"] = f"{half:.0f} {half:.0f} 0.05"
        floor.attrib["contype"] = "1"
        floor.attrib["conaffinity"] = "1"

    act_tpl = [ET.fromstring(ET.tostring(a)) for a in list(actuator_sec)]
    actuator_sec.clear()

    if layout == "scatter":
        scatter_span = max(6.0, spacing * math.sqrt(max(n_robots, 1)) * 1.2)
        positions = _scatter_xy(
            n_robots, span=scatter_span, min_dist=max(spacing, 1.1)
        )
    else:
        positions = _grid_xy(n_robots, spacing)
    for i, (gx, gy) in enumerate(positions):
        prefix = f"r{i}_"
        body = ET.fromstring(ET.tostring(base_tpl))
        body.attrib["pos"] = f"{gx:.4f} {gy:.4f} 0"
        _prefix_tree(body, prefix)
        for j in body.iter("joint"):
            if j.attrib.get("name", "").endswith("root") or j.attrib.get("type") == "free":
                j.attrib["name"] = f"{prefix}root"
        world.append(body)

        for act in act_tpl:
            a = ET.fromstring(ET.tostring(act))
            jname = a.attrib.get("joint", "")
            aname = a.attrib.get("name", "")
            a.attrib["joint"] = f"{prefix}{jname}"
            a.attrib["name"] = f"{prefix}{aname}"
            actuator_sec.append(a)

        for sens in sensor_tpl:
            s = ET.fromstring(ET.tostring(sens))
            s.attrib["name"] = f"{prefix}{s.attrib['name']}"
            if "site" in s.attrib:
                s.attrib["site"] = f"{prefix}{s.attrib['site']}"
            sensor_sec.append(s)

    root.attrib["model"] = f"SIX-MOTOR_MULTI_{n_robots}"
    visual = root.find("visual")
    if visual is not None:
        map_elem = visual.find("map")
        if map_elem is not None:
            map_elem.attrib["zfar"] = str(max(45, int(half * 1.8)))
            map_elem.attrib["fogend"] = str(max(40, int(half * 1.5)))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tree.write(out_path, encoding="unicode", xml_declaration=False)

    # 修正 meshdir 绝对路径（ElementTree 可能转义）
    with open(out_path, "r", encoding="utf-8") as f:
        xml = f.read()
    xml = re.sub(
        r'<compiler([^>]*?)meshdir="[^"]*"',
        f'<compiler\\1meshdir="{os.path.join(SCRIPT_DIR, "generated")}/"',
        xml,
        count=1,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)

    return out_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="生成多机器人 MuJoCo 场景")
    p.add_argument("-n", "--n-robots", type=int, default=12)
    p.add_argument("--spacing", type=float, default=1.05)
    p.add_argument("-o", "--out", type=str, default=None)
    args = p.parse_args()
    out = build_multi_mjcf(args.n_robots, args.spacing, out_path=args.out)
    print(f"已生成: {out} ({args.n_robots} 只机器人)")


if __name__ == "__main__":
    main()
