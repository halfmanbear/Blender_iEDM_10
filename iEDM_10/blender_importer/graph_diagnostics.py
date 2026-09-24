"""Configure transform diagnostics and describe graph-node transforms."""

import math as math

import bpy as bpy
from mathutils import Matrix, Quaternion, Vector

from ..edm_format.types import AnimatingNode, TransformNode
from .import_context import _import_ctx
from .node_identity import _is_connector_transform, _transform_display_name


def _reset_transform_debug(options):
    options = options or {}
    enabled = bool(options.get("debug_transforms", False))
    raw_limit = options.get("debug_transform_limit", 200)
    try:
        limit = max(1, int(raw_limit))
    except Exception:
        limit = 200
    name_filter = options.get("debug_transform_filter", None)
    if isinstance(name_filter, str) and not name_filter.strip():
        name_filter = None
    _import_ctx.transform_debug = {
        "enabled": enabled,
        "filter": name_filter,
        "limit": limit,
        "emitted": 0,
        "log_path": None,
        "chain_dumped": set(),
    }
    if enabled:
        filter_str = " filter='{}'".format(name_filter) if name_filter else ""
        print("Info: Transform debug enabled{} limit={}".format(filter_str, limit))


def _reset_mesh_origin_mode(options):
    options = options or {}
    mode = str(options.get("mesh_origin_mode", "APPROX")).upper()
    if mode not in {"APPROX", "RAW"}:
        mode = "APPROX"
    _import_ctx.mesh_origin_mode = mode


def _debug_node_label(node):
    if getattr(node, "transform", None):
        tf = node.transform
        name = _transform_display_name(tf) or type(tf).__name__
        return "tf:{}<{}>".format(name, type(tf).__name__)
    if getattr(node, "render", None):
        rn = node.render
        name = getattr(rn, "name", "") or type(rn).__name__
        return "rn:{}<{}>".format(name, type(rn).__name__)
    if getattr(node, "blender", None):
        return "bl:{}<{}>".format(node.blender.name, node.blender.type)
    return "<ROOT>"


def _debug_node_path(node):
    parts = []
    current = node
    while current is not None:
        parts.append(_debug_node_label(current))
        current = getattr(current, "parent", None)
    return " / ".join(reversed(parts))


def _debug_fmt_vec3(vec):
    return "({:+.6f}, {:+.6f}, {:+.6f})".format(vec.x, vec.y, vec.z)


def _debug_fmt_rot_deg(rot):
    euler = rot.to_euler("XYZ")
    return "({:+.3f}, {:+.3f}, {:+.3f})".format(
        math.degrees(euler.x), math.degrees(euler.y), math.degrees(euler.z)
    )


def _debug_fmt_trs(mat):
    loc, rot, scale = mat.decompose()
    return _debug_fmt_vec3(loc), _debug_fmt_rot_deg(rot), _debug_fmt_vec3(scale)


def _debug_filter_terms():
    dbg = getattr(_import_ctx, "transform_debug", {}) or {}
    needle = dbg.get("filter")
    if not needle:
        return []
    if isinstance(needle, str):
        return [term.strip().lower() for term in needle.split(",") if term.strip()]
    return [str(needle).strip().lower()]


def _anim_vector_to_blender(v):
    return Vector(v)


def _anim_quaternion_to_blender(q):
    return q if hasattr(q, "to_matrix") else Quaternion(q)


def _is_neg90_x_basis_matrix(mat, eps=1e-3):
    try:
        target = (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        )
        for r in range(3):
            for c in range(3):
                if abs(float(mat[r][c]) - target[r][c]) > eps:
                    return False
        return True
    except Exception:
        return False


def _anim_scale_components(value):
    if len(value) < 3:
        return (1.0, 1.0, 1.0)
    return (value[0], value[1], value[2])


def _expected_local_matrix(tfnode, blender_obj=None):
    if tfnode is None:
        return None
    if isinstance(tfnode, TransformNode):
        is_connector = _is_connector_transform(tfnode, blender_obj)
        local_mat = Matrix(tfnode.matrix)
        if is_connector:
            local_mat = local_mat @ Matrix.Rotation(math.radians(-90.0), 4, "X")
        if (
            blender_obj is not None
            and blender_obj.parent is not None
            and _import_ctx.edm_version >= 10
        ):
            local_mat = blender_obj.parent.matrix_basis @ local_mat
        return local_mat
    if isinstance(tfnode, AnimatingNode):
        local_mat = None
        if hasattr(tfnode, "zero_transform_local_matrix"):
            local_mat = tfnode.zero_transform_local_matrix
        elif hasattr(tfnode, "zero_transform_matrix"):
            local_mat = tfnode.zero_transform_matrix
        elif hasattr(tfnode, "zero_transform"):
            loc, rot, scale = tfnode.zero_transform
            local_mat = Matrix.LocRotScale(loc, rot, scale)
        if (
            local_mat is not None
            and blender_obj is not None
            and blender_obj.parent is not None
            and _import_ctx.edm_version >= 10
        ):
            local_mat = blender_obj.parent.matrix_basis @ local_mat
        return local_mat
    return None


def _debug_filter_match(node, path):
    name_filter = _import_ctx.transform_debug.get("filter")
    if not name_filter:
        return True
    needle = name_filter.lower()
    candidates = [path]
    if getattr(node, "blender", None):
        candidates.append(node.blender.name)
    if getattr(node, "transform", None):
        candidates.append(getattr(node.transform, "name", ""))
    if getattr(node, "render", None):
        candidates.append(getattr(node.render, "name", ""))
    return any(needle in (item or "").lower() for item in candidates)


def _debug_dump_node_transform(node):
    if not _import_ctx.transform_debug.get("enabled"):
        return
    if _import_ctx.transform_debug["emitted"] >= _import_ctx.transform_debug["limit"]:
        return
    if not getattr(node, "blender", None):
        return

    path = _debug_node_path(node)
    if not _debug_filter_match(node, path):
        return

    obj = node.blender
    bpy.context.view_layer.update()

    edm_local = _expected_local_matrix(getattr(node, "transform", None), obj)
    blender_local = obj.matrix_local.copy()
    blender_basis = obj.matrix_basis.copy()
    blender_world = obj.matrix_world.copy()
    parent_name = obj.parent.name if obj.parent else "<none>"

    print("[iEDM][TFDBG] {}".format(path))
    print("  object={} type={} parent={}".format(obj.name, obj.type, parent_name))

    if edm_local is not None:
        loc, rot, scale = _debug_fmt_trs(edm_local)
        print("  edm_local  loc={} rot_deg={} scale={}".format(loc, rot, scale))
    else:
        print("  edm_local  <none>")

    loc, rot, scale = _debug_fmt_trs(blender_local)
    print("  bl_localM  loc={} rot_deg={} scale={}".format(loc, rot, scale))
    loc, rot, scale = _debug_fmt_trs(blender_basis)
    print("  bl_basis   loc={} rot_deg={} scale={}".format(loc, rot, scale))
    loc, rot, scale = _debug_fmt_trs(blender_world)
    print("  bl_world   loc={} rot_deg={} scale={}".format(loc, rot, scale))

    if edm_local is not None:
        ed_loc, ed_rot, ed_scale = edm_local.decompose()
        bl_loc, bl_rot, bl_scale = blender_basis.decompose()
        loc_error = (bl_loc - ed_loc).length
        scale_error = (bl_scale - ed_scale).length
        rot_error = math.degrees(ed_rot.rotation_difference(bl_rot).angle)
        print(
            "  local_err  loc={:.6g} rot_deg={:.6g} scale={:.6g}".format(
                loc_error, rot_error, scale_error
            )
        )

    _import_ctx.transform_debug["emitted"] += 1
