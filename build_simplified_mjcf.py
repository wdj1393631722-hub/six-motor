#!/usr/bin/env python3
"""从 SIX-MOTOR URDF 生成简化 MJCF（胶囊几何，避免超大 STL）。"""
import os
import xml.etree.ElementTree as ET

OUT = os.path.join(os.path.dirname(__file__), "generated", "SIX-MOTOR_sim.xml")

# 每条腿三段胶囊尺寸（米，按你的机构可再调）
LINK_SIZE = {
    "coxa": (0.04, 0.06),
    "femur": (0.025, 0.14),
    "tibia": (0.02, 0.16),
}
BASE_SIZE = (0.28, 0.18, 0.06)  # 机身 box half-size


def parse_urdf(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    for j in root.findall("joint"):
        if j.get("type") != "revolute":
            continue
        origin = j.find("origin")
        axis = j.find("axis")
        limit = j.find("limit")
        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
        rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
        ax = [float(x) for x in axis.get("xyz", "0 0 1").split()]
        lo = float(limit.get("lower", "-0.8")) if limit is not None else -0.8
        hi = float(limit.get("upper", "0.8")) if limit is not None else 0.8
        joints.append({
            "name": j.get("name"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": xyz,
            "rpy": rpy,
            "axis": ax,
            "lo": lo,
            "hi": hi,
        })
    return joints


def rpy_to_quat(rpy):
    # 简化：只处理 rpy 很小或单轴；完整可 scipy
    roll, pitch, yaw = rpy
    cr, sr = __import__("math").cos(roll / 2), __import__("math").sin(roll / 2)
    cp, sp = __import__("math").cos(pitch / 2), __import__("math").sin(pitch / 2)
    cy, sy = __import__("math").cos(yaw / 2), __import__("math").sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return f"{w} {x} {y} {z}"


def link_geom_xml(link_name: str) -> str:
    if link_name == "base_link":
        sx, sy, sz = BASE_SIZE
        return f'<geom name="{link_name}_g" type="box" size="{sx} {sy} {sz}" mass="6.6" rgba="0.3 0.5 0.8 1"/>'
    part = "coxa" if "coxa" in link_name else "femur" if "femur" in link_name else "tibia"
    r, half = LINK_SIZE.get(part, (0.02, 0.08))
    mass = 1.5 if part == "coxa" else 1.0 if part == "femur" else 0.35
    return (
        f'<geom name="{link_name}_g" type="capsule" fromto="0 0 0 0 0 -{half}" '
        f'size="{r}" mass="{mass}" rgba="0.8 0.35 0.2 1"/>'
    )


def build_body_tree(joints, link_name, indent=2):
    """递归构建 body 树（从 base 的子关节开始）。"""
    sp = " " * indent
    lines = []
    child_joints = [j for j in joints if j["parent"] == link_name]
    for j in child_joints:
        cn = j["child"]
        xyz = " ".join(f"{v:.6f}" for v in j["xyz"])
        quat = rpy_to_quat(j["rpy"])
        ax = " ".join(f"{v:.6f}" for v in j["axis"])
        jn = j["name"]
        lines.append(f'{sp}<body name="{cn}" pos="{xyz}" quat="{quat}">')
        lines.append(
            f'{sp}  <joint name="{jn}" type="hinge" axis="{ax}" '
            f'range="{j["lo"]} {j["hi"]}" damping="1.0"/>'
        )
        lines.append(f"{sp}  {link_geom_xml(cn)}")
        lines.extend(build_body_tree(joints, cn, indent + 2))
        lines.append(f"{sp}</body>")
    return lines


def main():
    urdf = os.path.join(os.path.dirname(__file__), "generated", "SIX-MOTOR_fixed.urdf")
    if not os.path.isfile(urdf):
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from prepare_model import fix_urdf, DEFAULT_SRC
        urdf = fix_urdf(os.environ.get("SIX_MOTOR_PATH", DEFAULT_SRC),
                        os.path.join(os.path.dirname(__file__), "generated"))

    joints = parse_urdf(urdf)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    body_lines = build_body_tree(joints, "base_link", indent=4)

    actuators = []
    for j in joints:
        actuators.append(
            f'    <position name="{j["name"]}_act" joint="{j["name"]}" kp="250" '
            f'ctrlrange="{j["lo"]} {j["hi"]}"/>'
        )

    xml = f"""<mujoco model="SIX-MOTOR">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <default>
    <geom friction="1.2 0.02 0.001" condim="3"/>
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="4 4 0.05" rgba="0.35 0.38 0.42 1"/>
    <body name="base_link" pos="0 0 0.38">
      <freejoint name="root"/>
      {link_geom_xml("base_link")}
{chr(10).join(body_lines)}
    </body>
  </worldbody>
  <actuator>
{chr(10).join(actuators)}
  </actuator>
  <keyframe>
    <key name="stand" qpos="0 0 0.38 1 0 0 0"/>
  </keyframe>
</mujoco>
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"已写入: {OUT}")

    import mujoco
    m = mujoco.MjModel.from_xml_path(OUT)
    print(f"验证加载: nq={m.nq}, nv={m.nv}, nu={m.nu}")


if __name__ == "__main__":
    main()
