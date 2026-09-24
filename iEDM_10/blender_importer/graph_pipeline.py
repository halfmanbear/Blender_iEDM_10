"""Merge and attach actions; retain graph helper import paths."""

import math as math

import bpy as bpy
from mathutils import Matrix as Matrix
from mathutils import Quaternion as Quaternion
from mathutils import Vector as Vector

from ..edm_format.types import AnimatingNode as AnimatingNode
from ..edm_format.types import ArgVisibilityNode as ArgVisibilityNode
from ..edm_format.types import TransformNode as TransformNode
from ..utils import action_fcurves as action_fcurves
from .graph_collections import _SEMANTIC_NAME_MAP as _SEMANTIC_NAME_MAP
from .graph_collections import _assign_collections as _assign_collections
from .graph_collections import _categorize_file_root as _categorize_file_root
from .graph_collections import _classify_graph_nodes as _classify_graph_nodes
from .graph_collections import _create_lod_collections as _create_lod_collections
from .graph_collections import _get_or_create_child_col as _get_or_create_child_col
from .graph_collections import (
    _inherit_graph_collection_categories as _inherit_graph_collection_categories,
)
from .graph_collections import (
    _move_objects_to_categories as _move_objects_to_categories,
)
from .graph_diagnostics import (
    _anim_quaternion_to_blender as _anim_quaternion_to_blender,
)
from .graph_diagnostics import _anim_scale_components as _anim_scale_components
from .graph_diagnostics import _anim_vector_to_blender as _anim_vector_to_blender
from .graph_diagnostics import _debug_dump_node_transform as _debug_dump_node_transform
from .graph_diagnostics import _debug_filter_match as _debug_filter_match
from .graph_diagnostics import _debug_filter_terms as _debug_filter_terms
from .graph_diagnostics import _debug_fmt_rot_deg as _debug_fmt_rot_deg
from .graph_diagnostics import _debug_fmt_trs as _debug_fmt_trs
from .graph_diagnostics import _debug_fmt_vec3 as _debug_fmt_vec3
from .graph_diagnostics import _debug_node_label as _debug_node_label
from .graph_diagnostics import _debug_node_path as _debug_node_path
from .graph_diagnostics import _expected_local_matrix as _expected_local_matrix
from .graph_diagnostics import _is_neg90_x_basis_matrix as _is_neg90_x_basis_matrix
from .graph_diagnostics import _reset_mesh_origin_mode as _reset_mesh_origin_mode
from .graph_diagnostics import _reset_transform_debug as _reset_transform_debug
from .import_context import _import_ctx as _import_ctx
from .import_context import _log as _log
from .node_identity import _is_connector_transform as _is_connector_transform
from .node_identity import _transform_display_name as _transform_display_name


def _get_action_argument(action):
    if action is None:
        return None
    try:
        if hasattr(action, "argument"):
            arg = int(action.argument)
            if arg >= 0:
                return arg
    except Exception as e:
        _log.warn("action.argument read: {}".format(e), exc=e)
    try:
        prefix = str(getattr(action, "name", "")).split("_", 1)[0]
        if prefix and (
            prefix.isdigit() or (prefix.startswith("-") and prefix[1:].isdigit())
        ):
            return int(prefix)
    except Exception as e:
        _log.warn("action name prefix parse: {}".format(e), exc=e)
    return None


def _copy_fcurve_points(src_curve, dst_curve):
    for kp in src_curve.keyframe_points:
        try:
            frame = float(kp.co[0])
            value = float(kp.co[1])
        except Exception:
            continue
        new_kp = dst_curve.keyframe_points.insert(frame, value, options={"FAST"})
        try:
            new_kp.interpolation = kp.interpolation
            new_kp.handle_left_type = kp.handle_left_type
            new_kp.handle_right_type = kp.handle_right_type
            new_kp.easing = kp.easing
        except Exception as e:
            _log.debug("fcurve interpolation copy: {}".format(e), level=2)


def _merge_actions_by_argument(actions):
    """Merge multiple source actions into one action per EDM argument."""
    if not actions or len(actions) <= 1:
        return actions

    grouped = {}
    order = []
    for action in actions:
        arg = _get_action_argument(action)
        key = ("arg", arg) if arg is not None else ("name", getattr(action, "name", ""))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(action)

    if all(len(grouped[k]) == 1 for k in order):
        return actions

    merged_actions = []
    for key in order:
        src_actions = grouped[key]
        if len(src_actions) == 1:
            merged_actions.append(src_actions[0])
            continue

        base = src_actions[0]
        merged = bpy.data.actions.new(base.name)
        arg = key[1] if key[0] == "arg" else None
        if arg is not None and hasattr(merged, "argument"):
            merged.argument = int(arg)

        for src in src_actions:
            for src_curve in action_fcurves(src):
                dst_curve = action_fcurves(merged).find(
                    src_curve.data_path, index=src_curve.array_index
                )
                if dst_curve is None:
                    dst_curve = action_fcurves(merged).new(
                        data_path=src_curve.data_path, index=src_curve.array_index
                    )
                _copy_fcurve_points(src_curve, dst_curve)

        merged_actions.append(merged)

    return merged_actions


def _push_action_to_nla(ob, action):
    """Push action onto an NLA track when supported."""
    if ob is None:
        return False
    if not ob.animation_data:
        ob.animation_data_create()
    try:
        track = ob.animation_data.nla_tracks.new()
        track.name = action.name
        # A strip maps the action's first key onto its start frame; start there so
        # argument keys keep their authored scene frames.
        start = float(action.frame_range[0])
        strip = track.strips.new(action.name, int(start), action)
        if getattr(strip, "action_slot", False) is None and action.slots:
            strip.action_slot = action.slots[0]
        if abs(strip.frame_start - start) > 1e-6 and hasattr(strip, "frame_start_ui"):
            strip.frame_start_ui = start
        strip.extrapolation = "HOLD"
        if "Visib" in action.name:
            strip.blend_type = "COMBINED"
        else:
            strip.blend_type = "REPLACE"
        return True
    except Exception as e:
        _log.warn(
            "NLA push for '{}' on '{}': {}".format(
                getattr(action, "name", "<action>"), getattr(ob, "name", "<unknown>"), e
            ),
            exc=e,
        )
        return False
