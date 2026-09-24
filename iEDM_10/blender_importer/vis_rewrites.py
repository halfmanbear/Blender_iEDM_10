"""Rewrite visibility controls and retain skin visibility helper imports."""

import math as math

import bpy as bpy
from mathutils import Matrix as Matrix

from ..edm_format.types import AnimatingNode as AnimatingNode
from ..edm_format.types import ArgVisibilityNode as ArgVisibilityNode
from . import anim_actions as _anim_actions_compat
from . import skin_visibility as _skin_visibility_compat
from .anim_actions import (
    _clear_object_animation_tracks as _clear_object_animation_tracks,
)
from .anim_actions import (
    _visibility_source_for_graph_node as _visibility_source_for_graph_node,
)
from .anim_actions import get_actions_for_node as get_actions_for_node
from .animation import _is_pos90_x_basis_matrix as _is_pos90_x_basis_matrix
from .graph_diagnostics import _is_neg90_x_basis_matrix as _is_neg90_x_basis_matrix
from .graph_postprocess import _reparent_preserve_world as _reparent_preserve_world
from .import_context import _ROOT_BASIS_FIX as _ROOT_BASIS_FIX
from .import_context import _import_ctx as _import_ctx
from .import_context import _import_profile_flag as _import_profile_flag
from .import_context import _log as _log
from .node_identity import _assign_action as _assign_action
from .node_identity import _ob_local_is_identity as _ob_local_is_identity
from .skin_visibility import _flat_to_matrix as _flat_to_matrix
from .skin_visibility import _has_inverse_scale_child as _has_inverse_scale_child
from .skin_visibility import _is_identityish_basis as _is_identityish_basis
from .skin_visibility import _is_matching_skin_child as _is_matching_skin_child
from .skin_visibility import _matching_skin_children as _matching_skin_children
from .skin_visibility import (
    _matching_visibility_children as _matching_visibility_children,
)
from .skin_visibility import _matrix_close as _matrix_close
from .skin_visibility import (
    _reparent_visibility_skin_children as _reparent_visibility_skin_children,
)
from .skin_visibility import (
    _restore_skin_visibility_object as _restore_skin_visibility_object,
)
from .skin_visibility import _same_world_rotation as _same_world_rotation
from .skin_visibility import (
    _tag_visibility_skin_children as _tag_visibility_skin_children,
)

_collect_merged_transform_actions_for_graph_node = (
    _anim_actions_compat._collect_merged_transform_actions_for_graph_node
)

_fix_inverse_scaled_visibility_rest_offset = (
    _skin_visibility_compat._fix_inverse_scaled_visibility_rest_offset
)

_restore_skin_visibility_transform_basis = (
    _skin_visibility_compat._restore_skin_visibility_transform_basis
)


def _split_multi_arg_visibility_controls(graph):
    ctx = _import_ctx.bone_import_ctx or {}
    scene_collection = bpy.context.collection

    for node in getattr(graph, "nodes", []) or []:
        ob = getattr(node, "blender", None)
        if ob is None or getattr(ob, "type", "") in {"ARMATURE", "MESH"}:
            continue

        vis_source = _visibility_source_for_graph_node(node)
        if vis_source is None:
            continue
        vis_actions = get_actions_for_node(vis_source)
        if len(vis_actions) <= 1:
            continue

        transform_actions = _collect_merged_transform_actions_for_graph_node(node, ctx)
        if transform_actions:
            continue

        direct_children = [ch for ch in list(ob.children)]
        _clear_object_animation_tracks(ob)
        _assign_action(ob, vis_actions[0])

        parent_for_chain = ob
        created_helpers = []
        for action in vis_actions[1:]:
            helper = bpy.data.objects.new(ob.name, None)
            helper.empty_display_size = 0.1
            scene_collection.objects.link(helper)
            helper.parent = parent_for_chain
            helper.matrix_parent_inverse = Matrix.Identity(4)
            helper.matrix_basis = Matrix.Identity(4)
            _assign_action(helper, action)
            helper["_iedm_identity_passthrough"] = True
            helper["_iedm_narrow_identity_passthrough"] = True
            helper["_iedm_vis_passthrough"] = True
            created_helpers.append(helper)
            parent_for_chain = helper

        if created_helpers:
            target_parent = created_helpers[-1]
            for child in direct_children:
                if child in created_helpers:
                    continue
                if child.parent == ob:
                    # Identity visibility wrappers preserve the authored child local.
                    child.parent = target_parent


def _apply_plain_root_visibility_basis_fix(graph):
    if not _import_profile_flag("plain_root_visibility_basis_fix"):
        return
    if _import_ctx.edm_version < 10:
        return
    if getattr(_import_ctx, "use_scene_root_basis_object", True):
        return

    for node in getattr(graph, "nodes", []) or []:
        if not _is_plain_root_visibility_candidate(node):
            continue
        ob = node.blender
        try:
            ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
        except Exception as e:
            _log.warn("_apply_plain_root_visibility_basis_fix", exc=e)


def _is_plain_root_visibility_candidate(node):
    """Check whether an identity root ArgVis wrapper needs the basis fix."""
    ob = getattr(node, "blender", None)
    tf = getattr(node, "transform", None)
    if ob is None or type(tf).__name__ != "ArgVisibilityNode":
        return False
    if not _ob_local_is_identity(ob):
        return False
    if not getattr(getattr(node, "parent", None), "_is_graph_root", False):
        return False
    collapsed = getattr(node, "_collapsed_transforms", []) or []
    if any(type(extra_tf).__name__ != "ArgVisibilityNode" for extra_tf in collapsed):
        return False
    if type(getattr(node, "render", None)).__name__ == "SkinNode":
        return False

    descendants = []
    stack = list(getattr(node, "children", []) or [])
    while stack:
        current = stack.pop()
        descendants.append(current)
        stack.extend(list(getattr(current, "children", []) or []))
    # An animated or skinned descendant already handles its own basis conversion.
    if any(
        isinstance(getattr(child, "transform", None), AnimatingNode)
        and not isinstance(getattr(child, "transform", None), ArgVisibilityNode)
        for child in descendants
    ):
        return False
    if any(
        type(getattr(child, "render", None)).__name__ == "SkinNode"
        for child in descendants
    ):
        return False
    has_render = type(getattr(node, "render", None)).__name__ == "RenderNode" or any(
        type(getattr(child, "render", None)).__name__ == "RenderNode"
        for child in descendants
    )
    if not has_render:
        return False
    return not any(
        getattr(child, "type", "") == "EMPTY" and not _ob_local_is_identity(child)
        for child in list(getattr(ob, "children", []) or [])
    )


def _apply_visibility_pair_wrapper_object_basis_fix():
    if _import_ctx.edm_version < 10:
        return
    if getattr(_import_ctx, "use_scene_root_basis_object", True):
        return

    def _is_render_mesh(obj):
        return (
            getattr(obj, "type", "") == "MESH"
            and str(obj.get("_iedm_dbg_r_cls", "") or "") == "RenderNode"
        )

    for ob in list(getattr(bpy.data, "objects", []) or []):
        if getattr(ob, "type", "") != "EMPTY":
            continue
        if not bool(ob.get("_iedm_vis_passthrough")):
            continue
        if not _ob_local_is_identity(ob):
            continue

        children = list(getattr(ob, "children", []) or [])
        mesh_children = [child for child in children if _is_render_mesh(child)]
        nonidentity_empty_children = [
            child
            for child in children
            if getattr(child, "type", "") == "EMPTY"
            and not _ob_local_is_identity(child)
        ]

        if mesh_children:
            continue
        if len(nonidentity_empty_children) != 1:
            continue

        try:
            ob.matrix_basis = _ROOT_BASIS_FIX.inverted() @ ob.matrix_basis
        except Exception as e:
            _log.warn("_apply_visibility_pair_wrapper_object_basis_fix", exc=e)


def _is_basis_only_rotation(mat, eps=1e-4):
    try:
        loc = mat.to_translation()
        if loc.length > eps:
            return False
        scale = mat.to_scale()
        for s in scale:
            if abs(s - 1.0) > eps:
                return False
        return True
    except Exception:
        return False


def _get_root_name(obj):
    root = obj
    while getattr(root, "parent", None) is not None:
        root = root.parent
    return getattr(root, "name", "")


def _apply_argvis_chain_basis_fix():
    """Strip unwanted 90-degree X basis rotations from visibility wrappers.

    Some scene-root-authored v10 graphs create visibility wrapper chains where
    intermediate nodes carry only the coordinate system conversion rotation. This
    fix removes the rotation from visibility nodes that:
    1. Are ArgVisibilityNode with an ArgVisibilityNode parent
    2. The parent is a direct child of _EDMFileRoot (or a root-level wrapper)
    3. The node has a basis-only quarter-turn rotation.

    Also handles animated objects (ArgRotationNode children) under ArgVisibilityNode
    parents where the ArgVisibilityNode carries an unnecessary rotation.
    """
    if _import_ctx.edm_version < 10:
        return
    if not _import_profile_flag("argvis_chain_basis_fix"):
        return
    if not getattr(_import_ctx, "use_scene_root_basis_object", True):
        return

    bpy.context.view_layer.update()

    case1_fixed = 0
    case2_fixed = 0
    for ob in list(getattr(bpy.data, "objects", []) or []):
        case1_fixed += _fix_argvis_wrapper_rotation(ob)
        case2_fixed += _fix_argvis_child_rotation(ob)

    _log.debug(
        "_apply_argvis_chain_basis_fix: case1_fixed={}, case2_fixed={}".format(
            case1_fixed, case2_fixed
        ),
        level=1,
    )


def _fix_argvis_wrapper_rotation(ob):
    """Remove a basis-only rotation from a root visibility wrapper."""
    parent = getattr(ob, "parent", None)
    if getattr(ob, "type", "") != "EMPTY" or parent is None:
        return 0
    if (
        str(ob.get("_iedm_dbg_tf_cls", "") or "") != "ArgVisibilityNode"
        or str(parent.get("_iedm_dbg_tf_cls", "") or "") != "ArgVisibilityNode"
        or _get_root_name(ob) != "_EDMFileRoot"
    ):
        return 0
    matrix = ob.matrix_basis
    if not _is_basis_only_rotation(matrix):
        return 0
    try:
        if _is_neg90_x_basis_matrix(matrix):
            ob.matrix_basis = _ROOT_BASIS_FIX.inverted() @ matrix
            return 1
        if _is_pos90_x_basis_matrix(matrix):
            ob.matrix_basis = Matrix.Identity(4)
            return 1
    except Exception as exc:
        _log.warn("_apply_argvis_chain_basis_fix", exc=exc)
    return 0


def _fix_argvis_child_rotation(ob):
    """Correct a scaled child under a rotated root visibility wrapper."""
    parent = getattr(ob, "parent", None)
    if getattr(ob, "type", "") != "EMPTY" or parent is None:
        return 0
    if str(parent.get("_iedm_dbg_tf_cls", "") or "") != "ArgVisibilityNode":
        return 0
    if _get_root_name(ob) != "_EDMFileRoot":
        return 0
    parent_matrix = parent.matrix_basis
    if not _is_basis_only_rotation(parent_matrix) or ob.scale.length < 0.1:
        return 0
    try:
        if _is_neg90_x_basis_matrix(parent_matrix):
            ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
            return 1
        if _is_pos90_x_basis_matrix(parent_matrix):
            ob.matrix_basis = _ROOT_BASIS_FIX.inverted() @ ob.matrix_basis
            return 1
    except Exception as exc:
        _log.warn("_apply_argvis_chain_basis_fix animated", exc=exc)
    return 0
