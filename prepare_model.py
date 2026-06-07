#!/usr/bin/env python3
"""修复 SIX-MOTOR URDF 并生成 MuJoCo 场景 XML。"""
from __future__ import annotations

import os
import re
import sys

# 默认模型路径（可通过环境变量 SIX_MOTOR_PATH 覆盖）
DEFAULT_SRC = "/media/dash/OS/six-motor-URDF/SIX-MOTOR-NEW"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "generated")

# SW 旧版 leg2：coxa 绕 X + origin pitch=90°；与 leg5（绕 -Z）不一致
LEG2_COXA_JOINT_OLD = re.compile(
    r"<joint\s+name=\"leg2_coxa_joint\"[^>]*>.*?</joint>",
    re.DOTALL,
)
LEG5_FEMUR_ORIGIN = re.compile(
    r'<joint\s+name="leg5_femur_joint"[^>]*>\s*<origin\s+xyz="([^"]*)"\s+rpy="[^"]*"\s*/>',
    re.DOTALL,
)
LEG5_TIBIA_ORIGIN = re.compile(
    r'<joint\s+name="leg5_tibia_joint"[^>]*>\s*<origin\s+xyz="([^"]*)"\s+rpy="[^"]*"\s*/>',
    re.DOTALL,
)
LEG5_FEMUR_AXIS = re.compile(
    r'(<joint\s+name="leg5_femur_joint"[^>]*>.*?<axis\s+xyz=")([^"]+)(")',
    re.DOTALL,
)
LEG2_FEMUR_ORIGIN = re.compile(
    r"(<joint\s+name=\"leg2_femur_joint\"[^>]*>\s*<origin\s+xyz=\"[^\"]*\"\s+rpy=\")([^\"]+)(\")",
    re.DOTALL,
)
LEG2_TIBIA_ORIGIN = re.compile(
    r'(<joint\s+name="leg2_tibia_joint"[^>]*>\s*<origin\s+xyz=")[^"]*("\s+rpy=")[^"]*(")',
    re.DOTALL,
)
LEG2_FEMUR_AXIS = re.compile(
    r'(<joint\s+name="leg2_femur_joint"[^>]*>.*?<axis\s+xyz=")[^"]*(")',
    re.DOTALL,
)
LEG2_TIBIA_AXIS = re.compile(
    r'(<joint\s+name="leg2_tibia_joint"[^>]*>.*?<axis\s+xyz=")[^"]*(")',
    re.DOTALL,
)
LEG2_TIBIA_MESH_RPY = re.compile(
    r'(<link\s+name="leg2_tibia">.*?<(?:visual|collision)>)\s*<origin\s+xyz="([^"]*)"\s+rpy="[^"]*"\s*/>',
    re.DOTALL,
)


def _mirror_x_xyz(xyz_str: str) -> str:
    parts = [float(x) for x in xyz_str.split()]
    parts[0] = -parts[0]
    return " ".join(str(x) for x in parts)


def _mirror_x_axis(axis_str: str) -> str:
    parts = [float(x) for x in axis_str.split()]
    parts[0] = -parts[0]
    return " ".join(str(x) for x in parts)


def mirror_leg2_chain_from_leg5(text: str) -> tuple[str, bool]:
    """leg2 大腿/小腿关节与 leg5 关于 YZ 平面对称（去掉 femur pitch90）。"""
    m_f5 = LEG5_FEMUR_ORIGIN.search(text)
    m_t5 = LEG5_TIBIA_ORIGIN.search(text)
    m_ax5 = LEG5_FEMUR_AXIS.search(text)
    if not (m_f5 and m_t5 and m_ax5):
        return text, False

    femur_xyz = _mirror_x_xyz(m_f5.group(1))
    tibia_xyz = _mirror_x_xyz(m_t5.group(1))
    axis_xyz = _mirror_x_axis(m_ax5.group(2))

    def _origin_repl(xyz: str):
        def _fn(m: re.Match) -> str:
            return f"{m.group(1)}{xyz}{m.group(2)}0 0 0{m.group(3)}"

        return _fn

    text = LEG2_FEMUR_ORIGIN.sub(_origin_repl(femur_xyz), text, count=1)
    text = LEG2_TIBIA_ORIGIN.sub(_origin_repl(tibia_xyz), text, count=1)
    text = LEG2_FEMUR_AXIS.sub(lambda m: f"{m.group(1)}{axis_xyz}{m.group(2)}", text, count=1)
    text = LEG2_TIBIA_AXIS.sub(lambda m: f"{m.group(1)}{axis_xyz}{m.group(2)}", text, count=1)

    text, n = LEG2_TIBIA_MESH_RPY.subn(
        rf'\1\n      <origin xyz="\2" rpy="0 0 0" />',
        text,
    )
    return text, True


def unify_leg2_hip_with_leg5(text: str) -> tuple[str, bool]:
    """
    将 leg2 髋关节改为与 leg5 相同的 Z 轴约定（绕竖直轴转髋）。

    旧: coxa joint rpy=pitch90 + axis X
    新: coxa joint rpy=0 + axis Z+（右侧镜像 leg5 的 Z-）
    大腿/小腿关节与 leg5 镜像对称（不再使用 femur pitch90）。
    若源 URDF 已由 SW 正确导出则跳过。
    """
    m = LEG2_COXA_JOINT_OLD.search(text)
    if not m or 'xyz="1 0 0"' not in m.group(0):
        return text, False

    new_coxa_joint = """<joint
    name="leg2_coxa_joint"
    type="revolute">
    <origin
      xyz="0.16 0.00209403969745811 0"
      rpy="0 0 0" />
    <parent
      link="base_link" />
    <child
      link="leg2_coxa" />
    <axis
      xyz="0 0 1" />
    <limit lower="-0.8" upper="0.8"
      effort="100"
      velocity="10" />
  </joint>"""

    text = LEG2_COXA_JOINT_OLD.sub(new_coxa_joint, text, count=1)

    # SW 导出的 leg2 mesh 仍按旧 coxa pitch 坐标系建模，需在 femur 保留 pitch90
    def _femur_rpy(match: re.Match) -> str:
        return f"{match.group(1)}0 1.5707963267949 0{match.group(3)}"

    text = LEG2_FEMUR_ORIGIN.sub(_femur_rpy, text, count=1)
    return text, True


def resolve_src_paths(src_dir: str) -> tuple[str, str]:
    """解析 URDF 与 meshes 目录（兼容 SIX-MOTOR / SIX-MOTOR-NEW 包名）。"""
    urdf_dir = os.path.join(src_dir, "urdf")
    if not os.path.isdir(urdf_dir):
        raise FileNotFoundError(f"未找到 urdf 目录: {urdf_dir}")
    urdf_files = [f for f in os.listdir(urdf_dir) if f.endswith(".urdf")]
    if not urdf_files:
        raise FileNotFoundError(f"urdf 目录内无 .urdf 文件: {urdf_dir}")
    urdf_files.sort()
    src_urdf = os.path.join(urdf_dir, urdf_files[0])
    mesh_dir = os.path.join(src_dir, "meshes")
    if not os.path.isdir(mesh_dir):
        raise FileNotFoundError(f"未找到 meshes 目录: {mesh_dir}")
    return src_urdf, mesh_dir


# (参考腿, 镜像腿) — 参考腿在 +x 侧
LEG_MIRROR_PAIRS = ((1, 6), (3, 4), (2, 5))
# 中腿 5 保留 SW 原始关节（单腿装配-5 与 leg2 几何约定不同）；角腿 4/6 仅翻转 X
SKIP_MIRROR_LEGS = frozenset({5})
JOINT_ORIGIN_RE = re.compile(
    r'(<joint\s+name="(leg\d+_(?:femur|tibia)_joint)"[^>]*>\s*<origin\s+xyz=")([^"]+)("\s+rpy=")([^"]*)(")',
    re.DOTALL,
)
JOINT_AXIS_RE = re.compile(
    r'(<joint\s+name="(leg\d+_(?:femur|tibia)_joint)"[^>]*>.*?<axis\s+xyz=")([^"]+)(")',
    re.DOTALL,
)


def _mirror_xyz(xyz_str: str, flip_y: bool = False) -> str:
    """关于机体 YZ 平面镜像关节 origin。"""
    x, y, z = [float(v) for v in xyz_str.split()]
    return f"{-x} {-y if flip_y else y} {z}"


def _mirror_axis(axis_str: str, flip_y: bool = False) -> str:
    ax, ay, az = [float(v) for v in axis_str.split()]
    return f"{-ax} {-ay if flip_y else ay} {az}"


def mirror_left_leg_kinematics(text: str) -> tuple[str, bool]:
    """
    SW 导出时左右腿 femur/tibia 的 joint origin、axis 常未镜像。
    将 leg6/4/5 的 femur、tibia 关节与 leg1/3/2 关于 YZ 平面对称（仅翻转 X）。
    """
    origins: dict[str, str] = {}
    for m in JOINT_ORIGIN_RE.finditer(text):
        origins[m.group(2)] = m.group(3)

    axes: dict[str, str] = {}
    for m in JOINT_AXIS_RE.finditer(text):
        axes[m.group(2)] = m.group(3)

    patched = False
    for ref_leg, mir_leg in LEG_MIRROR_PAIRS:
        for jtype in ("femur", "tibia"):
            ref_name = f"leg{ref_leg}_{jtype}_joint"
            mir_name = f"leg{mir_leg}_{jtype}_joint"
            if ref_name not in origins or mir_name not in origins:
                continue
            if mir_leg in SKIP_MIRROR_LEGS:
                continue
            flip_y = False
            new_xyz = _mirror_xyz(origins[ref_name], flip_y=flip_y)
            text = text.replace(
                f'<joint\n    name="{mir_name}"',
                f'<joint\n    name="{mir_name}"',
                1,
            )
            # 替换该 joint 块内的 origin xyz
            pat = re.compile(
                rf'(<joint\s+name="{mir_name}"[^>]*>\s*<origin\s+xyz=")[^"]*(")',
                re.DOTALL,
            )

            def _orig_repl(m: re.Match, xyz: str = new_xyz) -> str:
                return f"{m.group(1)}{xyz}{m.group(2)}"

            text, n = pat.subn(_orig_repl, text, count=1)
            if n:
                patched = True
            if ref_name in axes and mir_name in axes:
                new_ax = _mirror_axis(axes[ref_name], flip_y=flip_y)
                pat_ax = re.compile(
                    rf'(<joint\s+name="{mir_name}"[^>]*>.*?<axis\s+xyz=")[^"]*(")',
                    re.DOTALL,
                )

                def _ax_repl(m: re.Match, ax: str = new_ax) -> str:
                    return f"{m.group(1)}{ax}{m.group(2)}"

                text, n2 = pat_ax.subn(_ax_repl, text, count=1)
                if n2:
                    patched = True
    return text, patched


def sync_meshes(mesh_dir: str, out_dir: str) -> str:
    """复制源 STL 到 generated/meshes，供 decimate_meshes 读取。"""
    import shutil

    dst = os.path.join(out_dir, "meshes")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(mesh_dir):
        if f.upper().endswith(".STL"):
            shutil.copy2(os.path.join(mesh_dir, f), os.path.join(dst, f))
    return dst


def fix_urdf(src_dir: str, out_dir: str) -> str:
    src_urdf, mesh_dir = resolve_src_paths(src_dir)
    sync_meshes(mesh_dir, out_dir)

    with open(src_urdf, "r", encoding="utf-8") as f:
        text = f.read()

    decimated_dir = "meshes_decimated/"
    for pkg in ("SIX-MOTOR-NEW", "SIX-MOTOR"):
        text = text.replace(f"package://{pkg}/meshes/", decimated_dir)
    text = text.replace(f"{mesh_dir}/", decimated_dir)
    text = text.replace(f"{src_dir}/meshes/", decimated_dir)

    # SW 导出关节限位常为 0，改为合理范围（弧度在 MJCF 里设置；URDF 用 degree 需确认）
    # MuJoCo 读 URDF 时 limit 单位为弧度
    limit_map = {
        "coxa_joint": (-0.8, 0.8),
        "femur_joint": (-1.4, 0.8),
        "tibia_joint": (0.0, 2.2),
    }

    def fix_limits(match: re.Match) -> str:
        block = match.group(0)
        jname = ""
        m = re.search(r'name="([^"]+)"', block)
        if m:
            jname = m.group(1)
        for key, (lo, hi) in limit_map.items():
            if key in jname:
                block = re.sub(
                    r'<limit\s+lower="[^"]*"\s+upper="[^"]*"',
                    f'<limit lower="{lo}" upper="{hi}"',
                    block,
                    count=1,
                )
                block = re.sub(r'effort="0"', 'effort="100"', block)
                block = re.sub(r'velocity="0"', 'velocity="10"', block)
                break
        return block

    text = re.sub(r"<joint[^>]*>.*?</joint>", fix_limits, text, flags=re.DOTALL)

    text, patched = unify_leg2_hip_with_leg5(text)
    if patched:
        print("已修补 leg2 髋关节（旧 URDF）: coxa 改为 Z 轴")
    else:
        print("leg2/leg5 髋关节已是 SW 新模型约定（Z 轴、rpy=0），跳过修补")

    # 勿改写左腿 femur/tibia 关节：SW 导出的 STL 与关节 origin 配套，强行镜像会导致腿链“炸开”
    print("左腿 femur/tibia 保留 SW 原始关节（与 mesh 坐标系一致）")

    os.makedirs(out_dir, exist_ok=True)
    out_urdf = os.path.join(out_dir, "SIX-MOTOR_fixed.urdf")
    with open(out_urdf, "w", encoding="utf-8") as f:
        f.write(text)
    return out_urdf


def write_mjcf_scene(urdf_path: str, out_dir: str) -> str:
    """生成带地面、执行器、自由关节的 MuJoCo 场景。"""
    urdf_abs = os.path.abspath(urdf_path)
    scene_path = os.path.join(out_dir, "six_motor_scene.xml")

    # 使用 mujoco <include> 编译 URDF 需在运行时；这里写组合场景
    content = f"""<mujoco model="SIX-MOTOR_scene">
  <compiler angle="radian" meshdir="{os.path.dirname(os.path.dirname(urdf_abs))}/meshes" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <default>
    <joint damping="1.0" armature="0.02"/>
    <geom friction="1.2 0.02 0.001" condim="3" solref="0.02 1" solimp="0.9 0.95 0.001"/>
    <motor ctrllimited="true" ctrlrange="-3.14 3.14"/>
  </default>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" width="512" height="512"
             rgb1="0.15 0.25 0.35" rgb2="0.1 0.15 0.2"/>
    <material name="groundplane" texture="groundplane" texrepeat="6 6" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="groundplane"/>
    <body name="robot_mount" pos="0 0 0.35">
      <freejoint name="root"/>
      <!-- 机器人由 compile 脚本在运行时从 URDF 合并；见 run_locomotion.py -->
    </body>
  </worldbody>
</mujoco>
"""
    with open(scene_path, "w", encoding="utf-8") as f:
        f.write(content)
    return scene_path


def compile_urdf_to_mjcf(urdf_path: str, out_dir: str) -> str:
    """用 MuJoCo 将 URDF 编译为 MJCF（含 mesh）。"""
    try:
        import mujoco
    except ImportError:
        raise SystemExit("请先: pip install mujoco")

    urdf_abs = os.path.abspath(urdf_path)
    # 切换到 URDF 所在目录，便于解析 mesh 相对路径
    model = mujoco.MjModel.from_xml_path(urdf_abs)

    # 添加自由关节：在 base 上需要 freejoint。URDF 固定基座，用 wrapper 加载
    mjcf_out = os.path.join(out_dir, "SIX-MOTOR.mjcf")

    # 导出为 XML（MuJoCo 3.2+ save_last_xml 或手动构建）
    # 使用 include 方式：保存编译结果
    mujoco.mj_saveLastXML(mjcf_out.encode(), model)
    return mjcf_out


def build_combined_model(urdf_path: str, out_dir: str) -> str:
    """构建带 freejoint + position 执行器的完整 MJCF。"""
    import mujoco

    urdf_abs = os.path.abspath(urdf_path)
    mesh_dir = os.path.join(os.path.dirname(os.path.dirname(urdf_abs)), "..", "meshes")
    mesh_dir = os.path.normpath(os.path.join(os.path.dirname(urdf_path), "..", "meshes"))
    if not os.path.isdir(mesh_dir):
        mesh_dir = os.path.join(os.path.dirname(os.path.dirname(urdf_abs)), "meshes")

    # 读取 fixed urdf 文本，嵌入 mujoco 根
    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_body = f.read()

    combined = os.path.join(out_dir, "SIX-MOTOR_sim.xml")

    # 直接用 MuJoCo 加载 URDF 再保存
    model = mujoco.MjModel.from_xml_path(urdf_abs)
    # 给根 link 加 freejoint 需要修改 XML；用 keyframe 初始高度
    xml_path = combined
    mujoco.mj_saveLastXML(xml_path.encode(), model)

    # 后处理：插入 freejoint 和 actuator（若保存的 xml 无执行器）
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()

    if "freejoint" not in xml and "base_link" in xml:
        xml = xml.replace(
            '<body name="base_link"',
            '<body name="base_link" pos="0 0 0.35"',
            1,
        )
        # 在 base_link body 开头加 freejoint
        xml = xml.replace(
            "<body name=\"base_link\"",
            '<freejoint/><body name="base_link"',
            1,
        )

    if "<actuator>" not in xml:
        actuators = []
        for i in range(model.nu):
            aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if aname:
                actuators.append(f'    <position name="{aname}_act" joint="{aname}" kp="80"/>')
        if not actuators:
            for i in range(model.njnt):
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if jname and jname != "root":
                    actuators.append(
                        f'    <position name="{jname}_act" joint="{jname}" kp="80"/>'
                    )
        if actuators:
            xml = xml.replace(
                "</mujoco>",
                "  <actuator>\n" + "\n".join(actuators) + "\n  </actuator>\n</mujoco>",
            )

    if "<worldbody>" in xml and "floor" not in xml:
        floor = """
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="4 4 0.05" rgba="0.3 0.35 0.4 1" friction="1.2 0.02 0.001"/>"""
        xml = xml.replace("<worldbody>", "<worldbody>" + floor, 1)

    with open(combined, "w", encoding="utf-8") as f:
        f.write(xml)

    return combined


def main():
    src = os.environ.get("SIX_MOTOR_PATH", DEFAULT_SRC)
    if len(sys.argv) > 1:
        src = sys.argv[1]

    print(f"源目录: {src}")
    os.makedirs(OUT_DIR, exist_ok=True)

    fixed_urdf = fix_urdf(src, OUT_DIR)
    print(f"已生成: {fixed_urdf}")

    print("提示: 请运行 python build_real_mjcf.py 生成带真实外形的 MuJoCo 模型")


if __name__ == "__main__":
    main()
