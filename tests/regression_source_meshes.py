"""Source-matrix oracle for every static RenderNode mesh in each file.

Blender --background --factory-startup --python \
    tests/regression_source_meshes.py -- paths...
"""

import contextlib
import io
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import iEDM_10
from iEDM_10.blender_importer import session

BASIS = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
FRAMES = (100, 150, 200)
TOLERANCE = 0.002
SAMPLES_PER_MESH = 8


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


def import_graph(path):
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
    # Visibility must not stop hidden meshes from being evaluated.
    for obj in bpy.data.objects:
        if obj.animation_data:
            for curve in obj.animation_data.drivers:
                if curve.data_path in {"hide_viewport", "hide_render"}:
                    curve.mute = True
        obj.hide_viewport = False
        obj.hide_set(False)
    return captured[0]


def check(path):
    graph = import_graph(path)
    targets = []
    skipped = 0
    for node in graph.nodes:
        render, obj = node.render, node.blender
        if render is None or obj is None or obj.type != "MESH":
            continue
        if type(render).__name__ == "SkinNode" or not hasattr(render, "indexData"):
            continue
        indices = sorted(set(render.indexData))
        if len(indices) != len(obj.data.vertices):
            skipped += 1
            continue
        targets.append((render, obj, indices))

    worst = {}
    for frame in FRAMES:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        def args(arg, frame=frame):
            return frame / 100 - 1

        cache = {}
        for render, obj, indices in targets:
            step = max(1, len(indices) // SAMPLES_PER_MESH)
            parent_world = BASIS @ source_world(render.parent, args, cache)
            error = 0.0
            for vi in range(0, len(indices), step):
                raw = Vector(render.vertexData[indices[vi]][:3])
                actual = obj.matrix_world @ obj.data.vertices[vi].co
                error = max(error, (parent_world @ raw - actual).length)
            if error > TOLERANCE and error > worst.get(obj.name, (0.0,))[0]:
                worst[obj.name] = (error, frame)

    errors = sorted(
        ((round(e, 4), f, name) for name, (e, f) in worst.items()), reverse=True
    )
    print(
        "RESULT",
        path.name,
        "meshes",
        len(targets),
        "skipped",
        skipped,
        "errors",
        errors,
        flush=True,
    )
    return errors


iEDM_10.register()
failures = {}
for arg in sys.argv[sys.argv.index("--") + 1 :]:
    path = Path(arg).resolve()
    errors = check(path)
    if errors:
        failures[path.name] = errors
assert not failures, failures
