#!/usr/bin/env python3
"""
从 SIX-MOTOR URDF 生成 MuJoCo 模型（MuJoCo 原生 URDF 导入，保留真实外形）。
"""
from __future__ import annotations

import os
import re
import sys

import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED = os.path.join(SCRIPT_DIR, "generated")
URDF_IN = os.path.join(GENERATED, "SIX-MOTOR_fixed.urdf")
OUT = os.path.join(GENERATED, "SIX-MOTOR_sim.xml")
DEFAULT_SRC = "/media/dash/OS/six-motor-URDF/SIX-MOTOR-NEW"

# MuJoCo geom friction: (滑动, 扭转, 滚动)
FLOOR_FRICTION = (2.0, 0.05, 0.0001)
FOOT_FRICTION = (3.5, 0.12, 0.0001)


def _friction_attr(friction: tuple[float, float, float]) -> str:
    return f'"{friction[0]} {friction[1]} {friction[2]}"'


def ensure_assets(src_dir: str) -> None:
    import decimate_meshes
    import prepare_model

    os.makedirs(GENERATED, exist_ok=True)
    prepare_model.fix_urdf(src_dir, GENERATED)
    mesh_src = os.path.join(GENERATED, "meshes")
    decimate_meshes.main(mesh_src)


def build_with_mjspec(urdf_path: str, out_path: str) -> mujoco.MjModel:
    """用 MuJoCo 原生 URDF 解析，避免手写 body 树误差。"""
    cwd = os.path.dirname(os.path.abspath(urdf_path))
    os.chdir(cwd)

    spec = mujoco.MjSpec.from_file(os.path.basename(urdf_path))
    spec.modelname = "SIX-MOTOR"
    spec.compiler.meshdir = cwd
    spec.compiler.degree = False  # 弧度
    spec.compiler.autolimits = True

    # 平面地貌（纹理与光照在 _patch_terrain 中写入 XML）
    floor = spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[15, 15, 0.05],
        rgba=[0.4, 0.43, 0.46, 1],
        friction=list(FLOOR_FRICTION),
    )
    _ = floor

    # 机身：自由关节（位置由 run_locomotion 重置时的 qpos 决定，不改 URDF 几何）
    base = spec.body("base_link")
    base.add_freejoint(name="root")

    model = spec.compile()
    spec.to_file(out_path)
    _patch_sim_xml(out_path)
    return mujoco.MjModel.from_xml_path(out_path)


def _patch_terrain(xml: str) -> str:
    """添加可见的平面地貌（棋盘格地面 + 光照）。"""
    import re

    if "tex_ground" not in xml:
        terrain_assets = """
    <texture name="tex_ground" type="2d" builtin="checker" mark="edge" markrgb="0.12 0.14 0.18"
             rgb1="0.38 0.41 0.44" rgb2="0.48 0.51 0.54" width="512" height="512"/>
    <material name="mat_ground" texture="tex_ground" texrepeat="14 14" reflectance="0.08"/>
"""
        xml = xml.replace("<asset>", "<asset>" + terrain_assets, 1)

    xml = re.sub(r'\s*<texture name="tex_sky"[^/]*/>\s*', "\n", xml)

    if "<visual>" not in xml:
        xml = xml.replace(
            "<worldbody>",
            """  <visual>
    <quality shadowsize="2048"/>
    <map fogstart="4" fogend="18" zfar="25"/>
  </visual>
  <worldbody>""",
            1,
        )

    floor_geom = """    <geom name="floor" type="plane" size="15 15 0.05" pos="0 0 0" material="mat_ground"
          friction={_friction_attr(FLOOR_FRICTION)} solref="0.004 1.2" solimp="0.92 0.98 0.001" condim="3"/>"""

    if 'light name="sun"' not in xml:
        lights = """    <light name="sun" pos="0 0 4" dir="0 0 -1" diffuse="0.95 0.95 0.92" castshadow="true"/>
    <light name="fill" pos="2 2 2.5" dir="-0.6 -0.6 -1" diffuse="0.4 0.42 0.45"/>
"""
        xml = re.sub(
            r'(\s*<worldbody>\s*)',
            r"\1" + lights,
            xml,
            count=1,
        )

    xml = re.sub(
        r'\s*<geom name="floor"[^>]*/>\s*',
        "\n" + floor_geom + "\n",
        xml,
        count=1,
    )
    # 清理重复光照（旧版重复 patch 时产生）
    while xml.count('light name="sun"') > 1:
        xml = re.sub(
            r'    <light name="sun"[^/]*/>\s*    <light name="fill"[^/]*/>\s*',
            "",
            xml,
            count=1,
        )
    return xml


def _patch_sim_xml(path: str) -> None:
    """补仿真选项、关节阻尼、地貌，并添加 position 伺服执行器。"""
    import re

    with open(path, "r", encoding="utf-8") as f:
        xml = f.read()

    xml = _patch_terrain(xml)

    if "<option" not in xml:
        xml = xml.replace(
            "<compiler",
            '<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>\n  <default>\n'
            '    <joint damping="3" armature="0.04" frictionloss="0.2"/>\n'
            "  </default>\n  <compiler",
            1,
        )

    def _position_actuator(m: re.Match) -> str:
        jname = m.group(2)
        lo_hi = m.group(3)
        return (
            f'    <position name="{m.group(1)}" joint="{jname}" kp="280" kv="12" '
            f'ctrlrange="{lo_hi}" forcelimited="true" forcerange="-100 100"/>'
        )

    xml = re.sub(
        r'    <general name="([^"]+)" joint="([^"]+)" ctrlrange="([^"]+)" gainprm="250"/>',
        _position_actuator,
        xml,
    )

    if "<actuator>" not in xml:
        joints = re.findall(
            r'<joint name="([^"]+)"[^>]*range="([^"]+)"',
            xml,
        )
        lines = [
            '  <actuator>',
            *[
                f'    <position name="{j}_act" joint="{j}" kp="280" kv="12" '
                f'ctrlrange="{lo} {hi}" forcelimited="true" forcerange="-100 100"/>'
                for j, lo_hi in joints
                if "_joint" in j
                for lo, hi in [lo_hi.split()]
            ],
            "  </actuator>",
        ]
        xml = xml.replace("</mujoco>", "\n".join(lines) + "\n</mujoco>")

    xml = _patch_foot_contact(xml)
    xml = _patch_imu(xml)

    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


def _patch_imu(xml: str) -> str:
    from imu_sensor import patch_imu_xml

    return patch_imu_xml(xml)


def _patch_foot_contact(xml: str) -> str:
    """小腿 mesh 仅用于显示；足底用小盒碰撞体，避免侧面蹭地。"""
    import re

    try:
        from foot_kinematics import foot_pad_quat, load_foot_frames, load_stand_pose
    except ImportError:
        return xml

    sim_path = os.path.join(os.path.dirname(__file__), "generated", "SIX-MOTOR_sim.xml")
    loaded = load_stand_pose()
    if loaded and os.path.isfile(sim_path):
        frames = load_foot_frames(mujoco.MjModel.from_xml_path(sim_path), loaded[0], loaded[1])
    else:
        frames = load_foot_frames()
    for leg in range(1, 7):
        pt, n = frames[leg]
        n = np.asarray(n, dtype=float)
        n /= np.linalg.norm(n) + 1e-12
        pad_half = 0.005
        # 盒心略高于足底接触点，使薄边底面与 mesh 底平面对齐
        center = pt - n * pad_half
        pos = " ".join(f"{v:.6f}" for v in center)
        quat = foot_pad_quat(n)
        size = "0.028 0.022 0.005"
        foot_geom = (
            f'            <geom name="leg{leg}_foot_pad" type="box" pos="{pos}" '
            f'quat="{quat}" size="{size}" rgba="0.75 0.78 0.82 0.35" '
            f'friction={_friction_attr(FOOT_FRICTION)} solref="0.004 1.2" condim="3"/>\n'
        )
        pat = (
            rf'(<body name="leg{leg}_tibia"[^>]*>\s*'
            rf'(?:.*?\n)*?\s*'
            rf'<geom type="mesh"[^>]*/>)'
        )

        def _repl(m: re.Match) -> str:
            g = m.group(1)
            if 'contype="' not in g:
                g = g.replace("<geom ", '<geom contype="0" conaffinity="0" ', 1)
            return g + "\n" + foot_geom

        xml, n = re.subn(pat, _repl, xml, count=1, flags=re.DOTALL)
        if n == 0:
            continue
    return xml


def main():
    src = os.environ.get("SIX_MOTOR_PATH", DEFAULT_SRC)
    ensure_assets(src)

    if not os.path.isfile(URDF_IN):
        print(f"缺少 URDF: {URDF_IN}")
        sys.exit(1)

    print("正在用 MuJoCo 原生 URDF 编译模型...")
    model = build_with_mjspec(URDF_IN, OUT)
    print(f"已生成: {OUT}")
    print(f"加载成功: nq={model.nq}, nv={model.nv}, nu={model.nu}, nmesh={model.nmesh}")
    print()
    print("说明: 使用 SW 导出的 base_link.STL（六边上板+髋电机）及腿 STL，高精度简化后加载。")


if __name__ == "__main__":
    main()
