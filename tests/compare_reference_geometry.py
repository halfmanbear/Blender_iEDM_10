"""Compare matching mesh names against a bundled Blender reference scene.

Run in background Blender with -- EDM_PATH.
Reports symmetric nearest-vertex distances in world space at argument zero.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import bpy
from mathutils.kdtree import KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import iEDM_10
from iEDM_10 import reader


def geometry():
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = {}
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        evaluated = ob.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            result[ob.name] = [evaluated.matrix_world @ v.co for v in mesh.vertices]
        finally:
            evaluated.to_mesh_clear()
    return result


def distance(left, right):
    tree = KDTree(len(right))
    for i, point in enumerate(right):
        tree.insert(point, i)
    tree.balance()
    return max(tree.find(point)[2] for point in left)


def main():
    args = sys.argv[sys.argv.index("--") + 1 :]
    path = Path(args[0]).resolve()
    iEDM_10.register()
    with contextlib.redirect_stdout(io.StringIO()):
        bpy.ops.wm.open_mainfile(filepath=str(path.with_suffix(".blend")))
        bpy.context.scene.frame_set(100)
        reference = geometry()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        reader.read_file(
            str(path),
            options={"mesh_origin_mode": "RAW", "preserve_scene_boxes": False},
        )
        imported = geometry()
    matched = reference.keys() & imported.keys()
    assert matched, "No matching reference meshes"
    failures = []
    for name in sorted(matched):
        left, right = reference[name], imported[name]
        if left and right:
            error = max(distance(left, right), distance(right, left))
            print("REFERENCE " + json.dumps({"object": name, "error": error}))
            if error > 1e-5:
                failures.append(name)
    assert not failures, "Reference geometry mismatch: " + ", ".join(failures)


if __name__ == "__main__":
    main()
