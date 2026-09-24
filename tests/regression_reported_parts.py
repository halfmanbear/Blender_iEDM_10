"""Source-matrix oracle for reported aircraft geometry and animated skinning.

Blender --background --factory-startup --python \
    tests/regression_reported_parts.py -- paths...

Target object names per model are read from the git-ignored
``local/reported_parts_targets.json``; models without an entry check nothing.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import iEDM_10
from iEDM_10.blender_importer import session

# Per-model target object names live outside the repository, keyed by the
# lower-case file stem: {"<model stem>": ["<object name>", ...]}.
TARGETS_FILE = ROOT / "local" / "reported_parts_targets.json"
TARGETS = (
    json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    if TARGETS_FILE.exists()
    else {}
)
BASIS = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def sample(keys, t):
    if t <= keys[0].frame:
        return keys[0].value.copy()
    if t >= keys[-1].frame:
        return keys[-1].value.copy()
    for a, b in zip(keys, keys[1:], strict=False):
        if a.frame <= t <= b.frame:
            u = (t - a.frame) / (b.frame - a.frame)
            va, vb = a.value, b.value
            if isinstance(va, Quaternion):
                if va.dot(vb) < 0:
                    vb = -vb
                return Quaternion(
                    tuple(x * (1 - u) + y * u for x, y in zip(va, vb, strict=False))
                ).normalized()
            return va.lerp(vb, u)
    raise AssertionError(t)


def oriented(scale, quat):
    r = quat.normalized().to_matrix().to_4x4()
    return r @ Matrix.Diagonal((*scale, 1)) @ r.inverted()


def source_local(node, args):
    if not hasattr(node, "base"):
        return Matrix(getattr(node, "matrix", Matrix.Identity(4)))
    b = node.base
    pos = b.position.copy()
    rotation = b.quat_1.copy()
    scale = oriented(b.scale, b.quat_2)
    for arg, keys in node.posData:
        if keys:
            pos += sample(keys, args(arg))
    for arg, keys in node.rotData:
        if keys:
            rotation = rotation @ sample(keys, args(arg))
    for arg, (qs, vs) in node.scaleData:
        q = sample(qs, args(arg)) if qs else (0, 0, 0, 1)
        v = sample(vs, args(arg)) if vs else (1, 1, 1)
        scale = scale @ oriented(v, Quaternion((q[3], q[0], q[1], q[2])))
    return b.matrix @ Matrix.Translation(pos) @ rotation.to_matrix().to_4x4() @ scale


def source_world(node, args, cache):
    if node is None:
        return Matrix.Identity(4)
    if node not in cache:
        cache[node] = source_world(node.parent, args, cache) @ source_local(node, args)
    return cache[node]


def check(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    captured = []
    original = session.build_graph

    def capture(edm):
        graph = original(edm)
        captured.append(graph)
        return graph

    session.build_graph = capture
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            session.read_file(
                str(path),
                options={"mesh_origin_mode": "RAW", "preserve_scene_boxes": False},
            )
    finally:
        session.build_graph = original
    graph = captured[0]
    ctx = session._import_ctx.bone_import_ctx or {}
    bone_source = {
        name: node.transform for node, name in ctx.get("bone_name_by_node", {}).items()
    }
    names = TARGETS.get(path.stem.lower(), [])
    nodes = {
        node.blender.name: node
        for node in graph.nodes
        if node.render is not None and node.blender is not None
    }
    for obj in bpy.data.objects:
        if obj.animation_data:
            for curve in obj.animation_data.drivers:
                if curve.data_path in {"hide_viewport", "hide_render"}:
                    curve.mute = True
        obj.hide_viewport = False
        obj.hide_set(False)
    errors = []
    for frame in (100, 125, 150, 175, 200):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        def args(arg, frame=frame):
            return frame / 100 - 1

        cache = {}
        dg = bpy.context.evaluated_depsgraph_get()
        for name in names:
            obj = bpy.data.objects[name]
            node = nodes[name]
            rn = node.render
            ev = obj.evaluated_get(dg)
            skin = type(rn).__name__ == "SkinNode"
            indices = (
                list(range(len(rn.vertexData))) if skin else sorted(set(rn.indexData))
            )
            worst = 0.0
            for vi, source_index in enumerate(indices):
                if vi % max(1, len(indices) // 100):
                    continue
                raw = Vector(rn.vertexData[source_index][:3])
                if skin:
                    expected = Vector()
                    for g in obj.data.vertices[vi].groups:
                        source = bone_source[obj.vertex_groups[g.group].name]
                        inverse = Matrix(
                            getattr(
                                source,
                                "bone_matrix",
                                getattr(
                                    source, "inv_base_bone_matrix", Matrix.Identity(4)
                                ),
                            )
                        )
                        expected += g.weight * (
                            BASIS @ source_world(source, args, cache) @ inverse @ raw
                        )
                else:
                    expected = BASIS @ source_world(rn.parent, args, cache) @ raw
                actual = ev.matrix_world @ ev.data.vertices[vi].co
                worst = max(worst, (expected - actual).length)
            print(
                "PART", path.name, frame, name, "max_error", round(worst, 7), flush=True
            )
            if worst > 0.002:
                errors.append((frame, name, worst))
    print("RESULT", path.name, "errors", errors, flush=True)
    if "--save" in sys.argv:
        bpy.context.scene.frame_set(100)
        bpy.ops.wm.save_as_mainfile(
            filepath=str(
                ROOT
                / "local"
                / "diagnostics"
                / ("source_fixed_" + path.stem + ".blend")
            )
        )
    assert not errors, errors


iEDM_10.register()
for arg in sys.argv[sys.argv.index("--") + 1 :]:
    if arg != "--save":
        check(Path(arg).resolve())
