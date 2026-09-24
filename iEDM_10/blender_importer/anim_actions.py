"""Create and combine EDM actions; retain legacy action helper imports."""

import bpy as bpy

from ..edm_format.mathtypes import Matrix as Matrix
from ..edm_format.mathtypes import MatrixScale as MatrixScale
from ..edm_format.mathtypes import Quaternion as Quaternion
from ..edm_format.mathtypes import Vector as Vector
from ..edm_format.types import AnimatingNode as AnimatingNode
from ..edm_format.types import ArgAnimationNode as ArgAnimationNode
from ..edm_format.types import ArgVisibilityNode as ArgVisibilityNode
from ..utils import action_fcurves as action_fcurves
from . import action_curves as _action_curves_compat
from .action_curves import _build_arganimation_action as _build_arganimation_action
from .action_curves import _clone_action_filtered as _clone_action_filtered
from .action_curves import _copy_fcurve_points_local as _copy_fcurve_points_local
from .action_values import _action_chain_sort_value as _action_chain_sort_value
from .action_values import (
    _compose_oriented_scale_matrix as _compose_oriented_scale_matrix,
)
from .action_values import _frame_value_components as _frame_value_components
from .action_values import _frame_values_close as _frame_values_close
from .action_values import (
    _has_nonidentity_scale_orientation_keys as _has_nonidentity_scale_orientation_keys,
)
from .action_values import (
    _is_plain_root_unit_interval_argrot as _is_plain_root_unit_interval_argrot,
)
from .action_values import (
    _needs_multi_arg_rotation_helper_split as _needs_multi_arg_rotation_helper_split,
)
from .action_values import (
    _plain_root_unit_interval_rot_sets as _plain_root_unit_interval_rot_sets,
)
from .action_values import (
    _scale_orientation_quaternion as _scale_orientation_quaternion,
)
from .action_values import (
    _sorted_transform_actions_for_execution as _sorted_transform_actions_for_execution,
)
from .animation import _arg_anim_vector_to_blender as _arg_anim_vector_to_blender
from .animation import (
    _finalize_authored_transform_action as _finalize_authored_transform_action,
)
from .animation import _quat_is_identity as _quat_is_identity
from .animation import add_position_fcurves as add_position_fcurves
from .animation import add_rotation_fcurves as add_rotation_fcurves
from .animation import add_scale_fcurves as add_scale_fcurves
from .graph_diagnostics import (
    _anim_quaternion_to_blender as _anim_quaternion_to_blender,
)
from .graph_pipeline import _get_action_argument as _get_action_argument
from .graph_pipeline import _merge_actions_by_argument as _merge_actions_by_argument
from .import_context import _ROOT_BASIS_FIX as _ROOT_BASIS_FIX
from .import_context import _import_ctx as _import_ctx
from .import_context import _log as _log
from .import_logging import _log_bone_debug_event as _log_bone_debug_event
from .import_logging import _matrix_trs_summary as _matrix_trs_summary
from .node_identity import _strip_anim_prefix as _strip_anim_prefix
from .visibility_graph import (
    _is_authored_argvis_control_pair as _is_authored_argvis_control_pair,
)
from .visibility_graph import _is_child_of_file_root as _is_child_of_file_root
from .visibility_graph import (
    _is_top_level_visibility_authored_pair as _is_top_level_visibility_authored_pair,
)
from .visibility_timeline import (
    _anim_frame_to_scene_frame as _anim_frame_to_scene_frame,
)
from .visibility_timeline import _visibility_scene_keys as _visibility_scene_keys

_create_scale_orientation_rotation_action = (
    _action_curves_compat._create_scale_orientation_rotation_action
)


def create_visibility_actions(visNode):
    """Creates visibility actions from an ArgVisibilityNode"""
    actions = []
    for arg, ranges in visNode.visData:
        vis_name = visNode.name or "node"
        if vis_name.startswith("v_"):
            vis_name = vis_name[2:]
        vis_name = _strip_anim_prefix(vis_name)
        action = bpy.data.actions.new("{}_{}_Visib".format(arg, vis_name))
        actions.append(action)
        if hasattr(action, "argument"):
            action.argument = arg
        curve_visible = action_fcurves(action).new(data_path="VISIBLE")
        curve_hide_vp = action_fcurves(action).new(data_path="hide_viewport")

        def _add_constant_key(curve, frame, value):
            curve.keyframe_points.add(1)
            key = curve.keyframe_points[-1]
            key.co = (frame, float(value))
            key.interpolation = "CONSTANT"

        def _add_vis_keys(
            frame, visible, curve_visible=curve_visible, curve_hide_vp=curve_hide_vp
        ):
            vis_val = 1.0 if visible else 0.0
            hide_val = 0.0 if visible else 1.0
            _add_constant_key(curve_visible, frame, vis_val)
            _add_constant_key(curve_hide_vp, frame, hide_val)

        for frame, visible in _visibility_scene_keys(ranges):
            _add_vis_keys(frame, visible)
        curve_visible.update()
        curve_hide_vp.update()
    return actions


def create_arganimation_actions(node):
    "Creates a set of actions to represent an ArgAnimationNode"
    # Deferred: orient_scale imports from this module, so this can only be
    # imported here, not at module load time.
    from .orient_scale import _reconstruct_edm_local_matrix

    actions = []
    _node_name = getattr(node, "name", "") or type(node).__name__

    local_edm, rot_scale_edm, base_mat, pos_vec = _reconstruct_edm_local_matrix(node)

    _acts_like_top_level_control = _is_child_of_file_root(
        node
    ) or _is_top_level_visibility_authored_pair(node)
    _top_level_root_basis_baked = (
        _import_ctx.edm_version >= 10
        and _acts_like_top_level_control
        and not getattr(_import_ctx, "use_scene_root_basis_object", True)
    )
    apply_root_basis = _top_level_root_basis_baked

    if apply_root_basis:
        # pos_vec is already Z-up (passthrough) — apply _ROOT_BASIS_FIX only to the
        # rotation/scale block so translation is not double-converted.
        rot_scale_bl = _ROOT_BASIS_FIX @ rot_scale_edm
        local_bl = Matrix.Translation(pos_vec) @ rot_scale_bl
    else:
        local_bl = local_edm

    dcLoc, dcRot, dcScale = local_bl.decompose()
    base_scale_vec = Vector(
        (node.base.scale[0], node.base.scale[1], node.base.scale[2])
    )
    q1_raw = (
        node.base.quat_1
        if hasattr(node.base.quat_1, "to_matrix")
        else Quaternion(node.base.quat_1)
    )
    q2_raw = (
        node.base.quat_2
        if hasattr(node.base.quat_2, "to_matrix")
        else Quaternion(node.base.quat_2)
    )
    _compose_oriented_scale_matrix(base_scale_vec, q2_raw)
    Matrix.Translation(_arg_anim_vector_to_blender(node, node.base.position))

    _bone_ctx = _import_ctx.bone_import_ctx or {}
    _is_armature_bone_source = node in (
        _bone_ctx.get("bone_name_by_transform", {}) or {}
    )

    zero_mat = local_bl

    bmat_inv = getattr(node, "bmat_inv", None)
    if bmat_inv is not None:
        try:
            zero_mat = zero_mat @ bmat_inv.inverted()
        except Exception as e:
            _log.warn("bmat_inv invert for bone zero-transform: {}".format(e), exc=e)

    node.zero_transform_local_matrix = zero_mat
    dcLoc, dcRot, dcScale = zero_mat.decompose()
    node.zero_transform_matrix = zero_mat
    node.zero_transform = dcLoc, dcRot, dcScale
    _log_bone_debug_event(
        "anim-zero",
        {
            "node_name": _node_name,
            "node_type": type(node).__name__,
            "is_armature_bone_source": bool(_is_armature_bone_source),
            "base_matrix": _matrix_trs_summary(node.base.matrix),
            "base_position": [round(float(v), 6) for v in Vector(node.base.position)],
            "base_quat_1": [round(float(v), 6) for v in q1_raw],
            "base_quat_2": [round(float(v), 6) for v in q2_raw],
            "zero_transform": _matrix_trs_summary(zero_mat),
            "top_level_root_basis_baked": bool(_top_level_root_basis_baked),
            "use_scene_root_basis_object": bool(
                getattr(_import_ctx, "use_scene_root_basis_object", True)
            ),
        },
        _node_name,
    )
    for arg in node.get_all_args():
        frame_mapper = _anim_frame_to_scene_frame
        actions.append(
            _build_arganimation_action(
                node,
                arg,
                local_bl,
                frame_mapper=frame_mapper,
            )
        )
    return actions


def get_actions_for_node(node):
    """Accepts a node and gets or creates actions to apply their animations"""
    if hasattr(node, "actions") and node.actions:
        actions = node.actions
    else:
        actions = []
        if isinstance(node, ArgVisibilityNode):
            actions = create_visibility_actions(node)
        if isinstance(node, ArgAnimationNode):
            actions = create_arganimation_actions(node)
        node.actions = actions
    return actions


def _clear_object_animation_tracks(ob):
    if ob is None:
        return
    ad = ob.animation_data
    if ad is None:
        return
    try:
        ad.action = None
    except Exception as exc:
        _log.debug(
            "Could not clear action from animation data: {}".format(exc), level=2
        )
    try:
        while ad.nla_tracks:
            ad.nla_tracks.remove(ad.nla_tracks[0])
    except Exception as exc:
        _log.debug("Could not clear NLA tracks: {}".format(exc), level=2)


def _action_has_visibility_curve(action):
    if action is None:
        return False
    try:
        return action_fcurves(action).find("VISIBLE") is not None
    except Exception:
        return False


def _build_nonarmature_action_plan(transform_actions, vis_actions):
    grouped = {}
    order = []
    for source in list(transform_actions or []) + list(vis_actions or []):
        arg = _get_action_argument(source)
        key = ("arg", arg) if arg is not None else ("name", getattr(source, "name", ""))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(source)
    planned = []
    for key in order:
        merged = _merge_actions_by_argument(grouped[key])
        if merged:
            planned.append(merged[0])
    return planned


def _collect_merged_transform_actions_for_graph_node(node, ctx=None):
    ctx = ctx or (_import_ctx.bone_import_ctx or {})
    if node is None or getattr(node, "blender", None) is None:
        return []
    tf = getattr(node, "transform", None)
    if tf is None:
        return []

    all_transforms = getattr(node, "_collapsed_transforms", [tf] if tf else [])
    anim_transforms = [t for t in all_transforms if isinstance(t, AnimatingNode)]
    if not anim_transforms:
        return []
    transform_anim_transforms = [
        t for t in anim_transforms if not isinstance(t, ArgVisibilityNode)
    ]
    if not transform_anim_transforms:
        return []

    bone_anim_source_nodes = ctx.get("bone_anim_source_nodes", set())
    bone_anim_source_transforms = ctx.get("bone_anim_source_transforms", set())
    skip_object_anim = node in bone_anim_source_nodes or any(
        t in bone_anim_source_transforms for t in transform_anim_transforms
    )
    if skip_object_anim:
        return []

    if isinstance(tf, ArgVisibilityNode):
        actions = []
        for extra_tf in transform_anim_transforms:
            actions.extend(get_actions_for_node(extra_tf))
    else:
        actions = (
            get_actions_for_node(tf) if not isinstance(tf, ArgVisibilityNode) else []
        )
        for extra_tf in transform_anim_transforms:
            if extra_tf is not tf:
                actions.extend(get_actions_for_node(extra_tf))
    return _merge_actions_by_argument(actions)


def _visibility_source_for_graph_node(node):
    tf = getattr(node, "transform", None)
    if isinstance(tf, ArgVisibilityNode):
        return tf
    parent = getattr(node, "parent", None)
    parent_tf = getattr(parent, "transform", None) if parent is not None else None
    if (
        parent is not None
        and isinstance(parent_tf, ArgVisibilityNode)
        and getattr(parent, "blender", None) == getattr(node, "blender", None)
    ):
        return parent_tf
    return None
