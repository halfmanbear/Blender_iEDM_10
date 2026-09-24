# Fragment: apply_node_transform — assigns the local matrix to a Blender object.
import math

import bpy
from mathutils import Matrix, Vector

from ..edm_format.types import (
    AnimatingNode,
    ArgAnimationNode,
    Connector,
    LightNode,
    TransformNode,
)
from ..utils import action_fcurves
from .animation import _is_pos90_x_basis_matrix, _normalize_euler_xyz
from .graph_pipeline import (
    _debug_filter_terms,
    _debug_fmt_rot_deg,
    _debug_fmt_vec3,
    _is_neg90_x_basis_matrix,
)
from .prelude import (
    _ROOT_BASIS_FIX,
    _import_ctx,
    _import_profile_flag,
    _is_child_of_file_root,
    _is_connector_object,
    _is_connector_transform,
    _log,
)


def _transform_uses_quaternion_rotation(tfnode, obj=None):
    try:
        if obj is not None and _is_connector_object(obj):
            return False
    except Exception as exc:
        _log.debug("Optional operation failed: {}".format(exc), level=2)
    try:
        action = getattr(getattr(obj, "animation_data", None), "action", None)
        if action is not None:
            if any(fc.data_path == "rotation_euler" for fc in action_fcurves(action)):
                return False
            if any(
                fc.data_path == "rotation_quaternion" for fc in action_fcurves(action)
            ):
                return True
    except Exception as exc:
        _log.debug("Optional operation failed: {}".format(exc), level=2)
    try:
        if isinstance(tfnode, ArgAnimationNode) and getattr(tfnode, "rotData", None):
            return True
    except Exception as exc:
        _log.debug("Optional operation failed: {}".format(exc), level=2)
    return False


def _debug_set_trace(tfnode, obj, source_label, local_mat):
    if not _import_ctx.transform_debug.get("enabled"):
        return
    node_name = getattr(tfnode, "name", "") or type(tfnode).__name__
    filter_terms = _debug_filter_terms()
    if filter_terms:
        if not any(
            term in node_name.lower() or term in obj.name.lower()
            for term in filter_terms
        ):
            return
    loc, rot, scale = local_mat.decompose()
    print(
        "[iEDM][TFSET] {} src={} obj={} loc={} rot_deg={} scale={}".format(
            node_name,
            source_label,
            obj.name,
            _debug_fmt_vec3(loc),
            _debug_fmt_rot_deg(rot),
            _debug_fmt_vec3(scale),
        )
    )


def _debug_dump_parent_chain(tnode, tfnode, obj, local_mat):
    if not _import_ctx.transform_debug.get("enabled"):
        return
    filter_terms = _debug_filter_terms()
    if not filter_terms:
        return
    node_name = getattr(tfnode, "name", "") or type(tfnode).__name__
    if not any(
        term in node_name.lower() or term in obj.name.lower() for term in filter_terms
    ):
        return
    dumped = _import_ctx.transform_debug.setdefault("chain_dumped", set())
    dump_key = "{}::{}".format(node_name, obj.name)
    if dump_key in dumped:
        return
    dumped.add(dump_key)

    print(
        "[iEDM][CHAIN] begin node={} obj={} graph_parent={} blender_parent={}".format(
            node_name,
            obj.name,
            getattr(
                getattr(getattr(tnode, "parent", None), "transform", None),
                "name",
                "<ROOT>",
            )
            if tnode is not None
            else "<ROOT>",
            getattr(getattr(obj, "parent", None), "name", None),
        )
    )
    if tnode is not None:
        cur = tnode
        graph_parts = []
        while cur is not None:
            tf_cur = getattr(cur, "transform", None)
            rn_cur = getattr(cur, "render", None)
            if tf_cur is not None:
                label = "{}<{}>".format(
                    getattr(tf_cur, "name", "") or type(tf_cur).__name__,
                    type(tf_cur).__name__,
                )
            elif rn_cur is not None:
                label = "{}<{}>".format(
                    getattr(rn_cur, "name", "") or type(rn_cur).__name__,
                    type(rn_cur).__name__,
                )
            else:
                label = "<ROOT>"
            graph_parts.append(label)
            cur = getattr(cur, "parent", None)
        print("[iEDM][CHAIN] graph_path={}".format(" / ".join(reversed(graph_parts))))
    try:
        loc, rot, scale = local_mat.decompose()
        print(
            "[iEDM][CHAIN] assigned_local loc={} rot_deg={} scale={}".format(
                _debug_fmt_vec3(loc),
                _debug_fmt_rot_deg(rot),
                _debug_fmt_vec3(scale),
            )
        )
    except Exception as exc:
        _log.debug("Optional operation failed: {}".format(exc), level=2)

    current = obj
    level = 0
    while current is not None:
        try:
            basis_loc, basis_rot, basis_scale = current.matrix_basis.decompose()
            world_loc, world_rot, world_scale = current.matrix_world.decompose()
            print(
                "[iEDM][CHAIN] level={} obj={} type={} parent={} basis_loc={} "
                "basis_rot_deg={} basis_scale={} world_loc={} world_rot_deg={} "
                "world_scale={}".format(
                    level,
                    current.name,
                    getattr(current, "type", ""),
                    getattr(getattr(current, "parent", None), "name", None),
                    _debug_fmt_vec3(basis_loc),
                    _debug_fmt_rot_deg(basis_rot),
                    _debug_fmt_vec3(basis_scale),
                    _debug_fmt_vec3(world_loc),
                    _debug_fmt_rot_deg(world_rot),
                    _debug_fmt_vec3(world_scale),
                )
            )
        except Exception as e:
            print(
                "[iEDM][CHAIN] level={} obj={} error={}".format(
                    level, getattr(current, "name", "<unknown>"), e
                )
            )
        current = getattr(current, "parent", None)
        level += 1
    print("[iEDM][CHAIN] end node={} obj={}".format(node_name, obj.name))


def _set_local_matrix(obj, local_mat, wants_quaternion_rotation):
    try:
        obj.matrix_basis = local_mat
        return
    except Exception as e:
        _log.warn(
            "matrix_basis assign for '{}': {}".format(
                getattr(obj, "name", "<unknown>"), e
            ),
            exc=e,
        )
    loc, rot, scale = local_mat.decompose()
    obj.location = loc
    if wants_quaternion_rotation:
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = rot
    else:
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = _normalize_euler_xyz(rot.to_euler("XYZ"))
    obj.scale = scale


def _compose_with_parent_if_enabled(obj, local_mat):
    if (
        _import_ctx.legacy_v10_parent_compose
        and obj.parent is not None
        and _import_ctx.edm_version >= 10
    ):
        try:
            return obj.parent.matrix_basis @ local_mat
        except Exception:
            return local_mat
    return local_mat


def _is_plain_root_connector_wrapper(graph_node, blender_obj):
    if not _import_profile_flag("plain_root_connector_basis_fix"):
        return False
    if _import_ctx.edm_version < 10:
        return False
    if getattr(_import_ctx, "use_scene_root_basis_object", True):
        return False
    if blender_obj is None or getattr(blender_obj, "parent", None) is not None:
        return False
    if getattr(blender_obj, "type", "") != "EMPTY":
        return False
    try:
        children = list(getattr(graph_node, "children", []) or [])
    except Exception:
        return False
    return any(
        isinstance(getattr(ch, "render", None), Connector)
        or _is_connector_object(getattr(ch, "blender", None))
        for ch in children
    )


def _is_plain_root_connector_child(graph_node, blender_obj):
    if not _import_profile_flag("plain_root_connector_child_basis_fix"):
        return False
    if _import_ctx.edm_version < 10:
        return False
    if getattr(_import_ctx, "use_scene_root_basis_object", True):
        return False
    if graph_node is None or blender_obj is None:
        return False
    if getattr(blender_obj, "parent", None) is None:
        return False
    if not _is_connector_object(blender_obj):
        return False
    return _is_child_of_file_root(graph_node)


def _frames_only_lights(graph_node, blender_obj):
    if getattr(blender_obj, "type", "") == "LIGHT":
        return True
    children = list(getattr(graph_node, "children", None) or [])
    return bool(children) and all(
        isinstance(getattr(child, "render", None), LightNode)
        and getattr(child, "transform", None) is None
        for child in children
    )


def _light_frame_keeping_beam(matrix):
    """Remove shear from a light frame without moving its beam axis.

    Blender objects cannot hold shear, and decomposing a sheared matrix spreads
    the error over every axis. The exporter's Ry(+90) light frame maps the
    Blender beam (-Z) to EDM +X, so keep +X exact and square up the others.
    """
    columns = [matrix.col[i].to_3d() for i in range(3)]
    lengths = [c.length for c in columns]
    if min(lengths) < 1e-12 or matrix.to_3x3().determinant() <= 0.0:
        return matrix  # mirrored frames already round-trip through decompose
    units = [c / n for c, n in zip(columns, lengths, strict=True)]
    if max(abs(units[i].dot(units[j])) for i, j in ((0, 1), (0, 2), (1, 2))) < 1e-5:
        return matrix
    x = units[0]
    y = columns[1] - columns[1].dot(x) * x
    if y.length < 1e-12:
        return matrix
    y.normalize()
    z = x.cross(y)
    rotation = Matrix((x, y, z)).transposed().to_4x4()
    return (
        Matrix.Translation(matrix.to_translation())
        @ rotation
        @ Matrix.Diagonal(Vector(lengths).to_4d())
    )


def _frames_meshes_or_connectors(graph_node):
    """Frames whose exported Transform is exactly Blender's matrix_local."""
    children = list(getattr(graph_node, "children", None) or [])
    return bool(children) and all(
        getattr(child, "render", None) is not None
        and not isinstance(child.render, LightNode)
        and getattr(child, "transform", None) is None
        for child in children
    )


def _remember_shear(obj, local_matrix):
    """Keep what a loc/rot/scale basis dropped from a sheared mesh frame."""
    residual = obj.matrix_basis.inverted_safe() @ local_matrix
    if max(abs(residual[r][c] - (r == c)) for r in range(4) for c in range(4)) > 1e-6:
        obj["_iedm_shear_residual"] = [v for row in residual for v in row]


def restore_sheared_frames():
    """Move remembered shear into matrix_parent_inverse once parenting is final.

    Blender bases cannot hold shear, but matrix_parent_inverse can and the
    exporter writes static objects from matrix_local, which includes it.
    """
    for obj in bpy.data.objects:
        flat = obj.get("_iedm_shear_residual")
        if flat is None:
            continue
        residual = Matrix([flat[i : i + 4] for i in range(0, 16, 4)])
        basis = obj.matrix_basis.copy()
        obj.matrix_parent_inverse = (
            obj.matrix_parent_inverse @ basis @ residual @ basis.inverted_safe()
        )
        del obj["_iedm_shear_residual"]


def _is_plain_root_light(blender_obj):
    return (
        getattr(blender_obj, "type", "") == "LIGHT"
        and blender_obj.parent is None
        and _import_ctx.edm_version >= 10
        and not getattr(_import_ctx, "use_scene_root_basis_object", True)
    )


def apply_node_transform(node, obj, used_shared_parent=False):
    """Assign a transform to a TranslationNode or raw EDM node."""
    tnode = node if hasattr(node, "transform") else None
    tfnode = tnode.transform if tnode else node

    render = tnode.render if tnode else getattr(node, "render", None)
    if tfnode is None and render is None:
        return

    wants_quaternion_rotation = (
        _transform_uses_quaternion_rotation(tfnode, obj)
        if tfnode is not None
        else False
    )
    obj.rotation_mode = "QUATERNION" if wants_quaternion_rotation else "XYZ"

    # 1. Base Transform from Node.transform
    final_local = (
        Matrix(node._local_bl)
        if (
            tfnode is None
            and hasattr(node, "_local_bl")
            and getattr(node, "_local_bl", None) is not None
        )
        else Matrix.Identity(4)
    )
    if tfnode is None and _is_connector_object(obj) and obj.parent is None:
        final_local = _connector_fallback_local_matrix(obj, final_local)
    if isinstance(tfnode, TransformNode):
        final_local = _transform_node_local_matrix(tnode, tfnode, obj)
    elif isinstance(tfnode, AnimatingNode):
        final_local = _animated_node_local_matrix(tnode, tfnode, obj, final_local)

    # 2. Inherit RenderNode local offset if present
    if render and tfnode is not None and not _is_connector_object(obj):
        final_local = _inherit_render_local_offset(final_local, render)

    _set_local_matrix(obj, final_local, wants_quaternion_rotation)
    if isinstance(tfnode, TransformNode) and _frames_meshes_or_connectors(tnode):
        _remember_shear(obj, final_local)
    _debug_set_trace(tfnode, obj, type(tfnode).__name__, final_local)
    _debug_dump_parent_chain(tnode, tfnode, obj, final_local)


def _connector_fallback_local_matrix(obj, final_local):
    try:
        flat = obj.get("_iedm_connector_tf_matrix", None)
        if flat is None or len(flat) != 16:
            return final_local
        matrix = Matrix(
            tuple(
                tuple(float(value) for value in flat[offset : offset + 4])
                for offset in (0, 4, 8, 12)
            )
        )
        location = matrix.to_translation()
        rotation = matrix.to_3x3().to_4x4()
        if _is_pos90_x_basis_matrix(rotation) or _is_neg90_x_basis_matrix(rotation):
            rotation = Matrix.Identity(4)
        if (
            _import_profile_flag("plain_root_connector_basis_fix")
            and obj.parent is None
            and _import_ctx.edm_version >= 10
            and not getattr(_import_ctx, "use_scene_root_basis_object", True)
        ):
            location = (_ROOT_BASIS_FIX @ Matrix.Translation(location)).to_translation()
        return Matrix.Translation(location) @ rotation
    except Exception as exc:
        _log.warn(
            "connector tf-matrix parse for '{}': {}".format(
                getattr(obj, "name", "<unknown>"), exc
            ),
            exc=exc,
        )
        return final_local


def _transform_node_local_matrix(node, transform, obj):
    is_connector = _is_connector_transform(transform, obj)
    is_wrapper = _is_plain_root_connector_wrapper(node, obj)
    is_child = _is_plain_root_connector_child(node, obj)
    raw_matrix = Matrix(transform.matrix)
    local_matrix = Matrix(raw_matrix)
    if _frames_only_lights(node, obj):
        local_matrix = _light_frame_keeping_beam(local_matrix)
    if _is_plain_root_light(obj):
        # Root-level light frames are Y-up like root meshes.
        return _ROOT_BASIS_FIX @ local_matrix
    if is_connector or is_wrapper or is_child:
        if (
            _import_profile_flag("plain_root_connector_basis_fix")
            and obj.parent is None
            and _import_ctx.edm_version >= 10
            and not getattr(_import_ctx, "use_scene_root_basis_object", True)
        ) or is_child:
            local_matrix = _ROOT_BASIS_FIX @ local_matrix
        elif is_wrapper and obj.parent is not None and _import_ctx.edm_version >= 10:
            if _is_neg90_x_basis_matrix(raw_matrix.to_3x3().to_4x4()):
                try:
                    parent_rotation = obj.parent.matrix_world.to_3x3().to_4x4()
                except Exception:
                    parent_rotation = Matrix.Identity(4)
                if _is_pos90_x_basis_matrix(parent_rotation):
                    local_matrix = Matrix.Translation(raw_matrix.to_translation())
        if (
            is_connector
            or not is_wrapper
            and not _is_neg90_x_basis_matrix(raw_matrix.to_3x3().to_4x4())
        ):
            local_matrix = local_matrix @ Matrix.Rotation(math.radians(-90.0), 4, "X")
    return _compose_with_parent_if_enabled(obj, local_matrix)


def _animated_node_local_matrix(node, transform, obj, final_local):
    if hasattr(transform, "zero_transform_local_matrix"):
        final_local = _compose_with_parent_if_enabled(
            obj, transform.zero_transform_local_matrix
        )
    elif hasattr(transform, "zero_transform_matrix"):
        final_local = _compose_with_parent_if_enabled(
            obj, transform.zero_transform_matrix
        )
    elif hasattr(transform, "zero_transform"):
        location, rotation, scale = transform.zero_transform
        final_local = _compose_with_parent_if_enabled(
            obj, Matrix.LocRotScale(location, rotation, scale)
        )
    wrapper_offset = getattr(transform, "wrapper_rest_translation_matrix", None)
    parent = getattr(getattr(node, "parent", None), "blender", None) if node else None
    if (
        wrapper_offset is not None
        and parent is not None
        and not getattr(transform, "_wrapper_rest_translation_applied", False)
    ):
        try:
            parent.matrix_basis = parent.matrix_basis @ wrapper_offset
            transform._wrapper_rest_translation_applied = True
        except Exception as exc:
            _log.warn("wrapper_rest_translation apply: {}".format(exc), exc=exc)
    return final_local


def _inherit_render_local_offset(final_local, render):
    if hasattr(render, "pos"):
        render_matrix = Matrix.Translation(Vector(render.pos[:3]))
    elif hasattr(render, "matrix"):
        render_matrix = Matrix(render.matrix)
    else:
        return final_local
    return final_local @ render_matrix if not render_matrix.is_identity else final_local
