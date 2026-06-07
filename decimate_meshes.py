#!/usr/bin/env python3
"""将 SW 导出的 STL 简化到 MuJoCo 可加载的面数（保留外形）。"""
import os
import sys

try:
    import trimesh
except ImportError:
    print("安装: pip install trimesh")
    sys.exit(1)

# 尽量贴近 SW 外形；单 mesh 面数需低于 MuJoCo 约 20 万上限
MAX_FACES = {
    "base_link": 199000,  # 六边上板 + 6×RS03（原 53 万面）
    "coxa": 80000,        # 髋部细节（原 38 万面）
    "femur": 35000,       # 大腿（原 9.8 万面）
    "tibia": 5722,        # 小腿原模已足够，不简化
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "generated", "meshes")
DST = os.path.join(SCRIPT_DIR, "generated", "meshes_decimated")


def target_faces(name: str) -> int:
    if "base" in name:
        return MAX_FACES["base_link"]
    if "coxa" in name:
        return MAX_FACES["coxa"]
    if "femur" in name:
        return MAX_FACES["femur"]
    return MAX_FACES["tibia"]


def decimate_one(src_path: str, dst_path: str) -> None:
    m = trimesh.load(src_path, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump(concatenate=True)
    n = len(m.faces)
    tgt = target_faces(os.path.basename(src_path))
    if n <= tgt:
        m.export(dst_path)
        print(f"  保留 {os.path.basename(src_path)}: {n} 面")
        return
    face_count = max(4, min(tgt, n - 1))
    simplified = m.simplify_quadric_decimation(face_count=face_count)
    simplified.export(dst_path)
    print(f"  简化 {os.path.basename(src_path)}: {n} -> {len(simplified.faces)} 面")


def main(src_dir: str | None = None):
    if src_dir is None:
        src_dir = sys.argv[1] if len(sys.argv) > 1 else SRC
    if not os.path.isdir(src_dir):
        # 从原始 SIX-MOTOR 转换
        from stl import mesh as stlmesh
        os.makedirs(SRC, exist_ok=True)
        raw = os.path.expanduser("~/桌面/SIX-MOTOR/meshes")
        for f in os.listdir(raw):
            if f.upper().endswith(".STL"):
                m = stlmesh.Mesh.from_file(os.path.join(raw, f))
                m.save(os.path.join(SRC, f))
        src_dir = SRC

    os.makedirs(DST, exist_ok=True)
    print(f"输出: {DST}")
    for f in sorted(os.listdir(src_dir)):
        if not f.upper().endswith(".STL"):
            continue
        decimate_one(os.path.join(src_dir, f), os.path.join(DST, f))
    print("完成")


if __name__ == "__main__":
    main()
