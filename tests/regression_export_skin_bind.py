"""Round-trip skins whose bind is not the arg-0 pose.

Run with Blender --background --factory-startup --python-exit-code 1
--python tests/regression_export_skin_bind.py -- [model.edm ...].

Defaults to tests/assets/Bones.edm, whose skin bind differs from its arg-0
pose. Imports each model, exports with edm.export, then moves one argument
at a time and deforms every skin the way DCS does (packed bone index + 1,
missing weight to palette[0]). Every source vertex must have a round-trip
vertex within 1 mm.
"""

import struct
import sys
import tempfile
from pathlib import Path

import addon_utils
import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.kdtree import KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import iEDM_10
from iEDM_10 import reader
from iEDM_10.blender_importer.anim_actions import _scale_orientation_quaternion
from iEDM_10.edm_format.types import EDMFile

VALUES = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
TOLERANCE = 1e-3


def _interp(keys, t):
    keys = sorted(keys, key=lambda k: k.frame)
    if t <= keys[0].frame:
        return keys[0].value
    for a, b in zip(keys, keys[1:], strict=False):
        if t <= b.frame:
            f = (t - a.frame) / (b.frame - a.frame)
            if len(a.value) == 4:
                return Quaternion(a.value).slerp(Quaternion(b.value), f)
            return Vector(a.value).lerp(Vector(b.value), f)
    return keys[-1].value


def _local(node, arg, value):
    base = getattr(node, "base", None)
    if base is None or not hasattr(base, "quat_1"):
        return Matrix(getattr(node, "matrix", Matrix.Identity(4)))
    at = {arg: value}
    pos = Vector()
    for a, keys in node.posData or []:
        if keys:
            pos += Vector(_interp(keys, at.get(a, 0.0)))
    rot = Quaternion()
    for a, keys in node.rotData or []:
        if keys:
            rot = rot @ Quaternion(_interp(keys, at.get(a, 0.0)))
    scale = Matrix.Identity(4)
    for a, (orient_keys, scale_keys) in node.scaleData or []:
        if not scale_keys:
            continue
        t = at.get(a, 0.0)
        orient = Quaternion()
        if orient_keys:  # orientation steps to the nearest key
            nearest = min(orient_keys, key=lambda k: abs(k.frame - t))
            orient = _scale_orientation_quaternion(nearest.value)
        r = orient.to_matrix().to_4x4()
        diagonal = Matrix.Diagonal(Vector(_interp(scale_keys, t)).to_4d())
        scale = scale @ r @ diagonal @ r.inverted()
    return (
        Matrix(base.matrix)
        @ Matrix.Translation(Vector(base.position) + pos)
        @ base.quat_1.to_matrix().to_4x4()
        @ rot.to_matrix().to_4x4()
        @ scale
        @ Matrix.Diagonal(Vector(base.scale).to_4d())
    )


def _world(node, arg, value):
    matrix = Matrix.Identity(4)
    while node is not None and not isinstance(node, (int, list)):
        matrix = _local(node, arg, value) @ matrix
        node = getattr(node, "parent", None)
    return matrix


def _skin_matrix(bone, arg, value):
    inverse = getattr(bone, "inv_base_bone_matrix", None)
    if inverse is None:
        inverse = bone.bone_matrix
    return _world(bone, arg, value) @ Matrix(inverse)


def _vertices(skin):
    offsets, cursor = {}, 0
    for channel, count in enumerate(skin.material.vertex_format.data):
        if count:
            offsets[channel] = (cursor, cursor + int(count))
            cursor += int(count)
    used = (
        sorted(set(skin.indexData)) if skin.indexData else range(len(skin.vertexData))
    )
    for i in used:
        row = skin.vertexData[i]
        packed = struct.unpack("<I", struct.pack("<f", float(row[3])))[0]
        start, end = offsets[21]
        weights = list(row[start:end]) + [0.0] * (4 - (end - start))
        yield Vector(row[0:3]), [(packed >> (8 * k)) & 0xFF for k in range(4)], weights


def deformed(path, arg, value):
    points = []
    for skin in EDMFile(str(path)).renderNodes:
        if type(skin).__name__ != "SkinNode":
            continue
        palette = [_skin_matrix(b, arg, value) for b in skin.bones]
        for co, indices, weights in _vertices(skin):
            out = max(0.0, 1.0 - sum(weights)) * (palette[0] @ co)
            for index, weight in zip(indices, weights, strict=True):
                if weight:
                    out += weight * (palette[index + 1] @ co)
            points.append(out)
    return points


def skin_args(path):
    args = set()
    for skin in EDMFile(str(path)).renderNodes:
        if type(skin).__name__ != "SkinNode":
            continue
        for bone in skin.bones:
            node = bone
            while node is not None and not isinstance(node, (int, list)):
                args |= {a for a, k in (getattr(node, "rotData", None) or []) if k}
                args |= {a for a, k in (getattr(node, "posData", None) or []) if k}
                node = getattr(node, "parent", None)
    return sorted(args)


def worst_distance(source, output, arg, value):
    rt = deformed(output, arg, value)
    tree = KDTree(len(rt))
    for i, co in enumerate(rt):
        tree.insert(co, i)
    tree.balance()
    return max(tree.find(co)[2] for co in deformed(source, arg, value))


def round_trip(source, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("io_scene_edm", default_set=True)
    iEDM_10.register()
    reader.read_file(str(source), options={"mesh_origin_mode": "RAW"})
    assert bpy.ops.edm.export(filepath=str(output)) == {"FINISHED"}
    iEDM_10.unregister()


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    paths = [Path(p).resolve() for p in argv] or [ROOT / "tests/assets/Bones.edm"]
    with tempfile.TemporaryDirectory(prefix="iedm_skin_bind_") as directory:
        for path in paths:
            output = Path(directory) / (path.stem + ".edm")
            round_trip(path, output)
            args = skin_args(path)
            assert args, (path.name, "has no animated skin bones")
            worst = max(
                worst_distance(path, output, arg, value)
                for arg in args
                for value in VALUES
            )
            assert worst < TOLERANCE, (path.name, worst)
            print("PASS", path.name, "skin args", len(args), "worst", worst, "m")
    print("Blender", bpy.app.version_string)


if __name__ == "__main__":
    main()
