"""Coordinate armature creation and skin binding; retain helper imports."""

import struct as struct

import bpy as bpy

from ...edm_format.mathtypes import Matrix as Matrix
from ...edm_format.mathtypes import Quaternion as Quaternion
from ...edm_format.mathtypes import Vector as Vector
from ...edm_format.types import AnimatingNode as AnimatingNode
from ...edm_format.types import ArgAnimatedBone as ArgAnimatedBone
from ...edm_format.types import ArgVisibilityNode as ArgVisibilityNode
from ...edm_format.types import Bone as Bone
from ...utils import action_fcurves as action_fcurves
from ...utils import new_grouped_fcurve as new_grouped_fcurve
from ..anim_actions import get_actions_for_node as get_actions_for_node
from ..export_skin_bind import mark_imported_rig as mark_imported_rig
from ..import_context import _ROOT_BASIS_FIX as _ROOT_BASIS_FIX
from ..import_context import _import_ctx as _import_ctx
from ..import_context import _import_profile_flag as _import_profile_flag
from ..import_context import _log as _log
from ..import_logging import _log_bone_debug_event as _log_bone_debug_event
from ..import_logging import _matrix_trs_summary as _matrix_trs_summary
from ..node_identity import _is_bone_transform as _is_bone_transform
from ..node_identity import _transform_display_name as _transform_display_name
from . import bone_actions as _bone_actions_compat
from .armature_build import _apply_armature_transforms as _apply_armature_transforms
from .armature_build import _build_edit_bones as _build_edit_bones
from .armature_build import _create_armature_object as _create_armature_object
from .armature_build import _finalize_bone_import_ctx as _finalize_bone_import_ctx
from .bone_actions import _action_has_fcurve as _action_has_fcurve
from .bone_actions import (
    _attach_retargeted_bone_actions as _attach_retargeted_bone_actions,
)
from .bone_actions import _copy_bone_rotation_curves as _copy_bone_rotation_curves
from .bone_actions import _copy_fcurve_to_action as _copy_fcurve_to_action
from .bone_actions import (
    _transfer_bone_actions_to_armature as _transfer_bone_actions_to_armature,
)
from .bone_rest import _bone_bind_matrix as _bone_bind_matrix
from .bone_rest import _bone_rest_matrix_for_node as _bone_rest_matrix_for_node
from .bone_rest import _create_edit_bone as _create_edit_bone
from .bone_rest import (
    _debug_bone_bind_matrix_summary as _debug_bone_bind_matrix_summary,
)
from .bone_rest import _edit_bone_length as _edit_bone_length
from .bone_rest import _effective_root_basis_fix as _effective_root_basis_fix
from .bone_rest import _unique_bone_name as _unique_bone_name
from .skin_binding import _assign_skin_vertex_weights as _assign_skin_vertex_weights
from .skin_binding import _attach_skin_mesh_to_parent as _attach_skin_mesh_to_parent
from .skin_binding import (
    _bake_skin_mesh_object_transforms as _bake_skin_mesh_object_transforms,
)
from .skin_binding import (
    _channel_slices_from_vertex_format as _channel_slices_from_vertex_format,
)
from .skin_binding import _choose_skin_bind_target as _choose_skin_bind_target
from .skin_binding import _decode_packed_bone_indices as _decode_packed_bone_indices
from .skin_binding import _find_skin_parent_candidate as _find_skin_parent_candidate
from .skin_binding import (
    _localize_skin_mesh_to_bind_target as _localize_skin_mesh_to_bind_target,
)
from .skin_binding import (
    _mesh_bounds_center_and_extent as _mesh_bounds_center_and_extent,
)
from .skin_binding import _skin_bone_weight_sums as _skin_bone_weight_sums
from .skin_binding import _skin_parent_candidate_score as _skin_parent_candidate_score

_merge_visibility_action_into_transform_action = (
    _bone_actions_compat._merge_visibility_action_into_transform_action
)


def _prepare_bone_import(graph, parent_obj=None):
    """Create a single armature for EDM Bone/ArgAnimatedBone nodes."""
    _import_ctx.bone_import_ctx = None

    bone_nodes = [n for n in graph.nodes if _is_bone_transform(n.transform)]
    if not bone_nodes:
        return

    # Only map actual Bone/ArgAnimatedBone nodes to the synthesized armature.
    # Swallowing non-bone ancestors (ArgVisibilityNode wrappers etc.) collapses
    # authored control/visibility transforms and causes DOT hierarchy drift.
    bone_chain_nodes = set(bone_nodes)

    arm_obj, arm_data, apply_bone_root_fix, arm_carries_basis_fix = (
        _create_armature_object(
            bone_nodes,
            parent_obj,
        )
    )
    node_to_bone_name = _build_edit_bones(
        arm_obj,
        arm_data,
        bone_nodes,
        apply_bone_root_fix,
    )
    _finalize_bone_import_ctx(
        arm_obj,
        node_to_bone_name,
        bone_chain_nodes,
        arm_carries_basis_fix,
        graph,
        apply_bone_root_fix,
    )
    _apply_armature_transforms(arm_obj)


def _bind_skin_object(mesh_obj, skin_node):
    """Attach a SkinNode mesh to imported armature with vertex groups."""
    ctx = _import_ctx.bone_import_ctx or {}
    arm_obj = ctx.get("armature")
    bone_name_by_transform = ctx.get("bone_name_by_transform", {})
    bone_rest_matrix_by_name = ctx.get("bone_rest_matrix_by_name", {})
    if arm_obj is None or mesh_obj is None or mesh_obj.type != "MESH":
        return

    palette = [bone_name_by_transform.get(b) for b in getattr(skin_node, "bones", [])]
    skin_bones = [name for name in palette if name]
    if not skin_bones:
        return
    # Vertex bone index i is palette[i + 1]; palette[0] is the control bone.
    control_bone = palette[0] or skin_bones[0]
    weight_bones = palette[1:]

    group_map = {}
    for bone_name in dict.fromkeys(skin_bones):
        vg = mesh_obj.vertex_groups.get(bone_name)
        if vg is None:
            vg = mesh_obj.vertex_groups.new(name=bone_name)
        group_map[bone_name] = vg

    channel_slices = _channel_slices_from_vertex_format(
        skin_node.material.vertex_format
    )

    arm_mod = mesh_obj.modifiers.get("Armature")
    if arm_mod is None:
        arm_mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    arm_mod.object = arm_obj

    bind_target_name, bind_target_loc = _choose_skin_bind_target(
        control_bone,
        weight_bones,
        bone_rest_matrix_by_name,
        mesh_obj,
        skin_node,
        channel_slices,
    )
    try:
        mesh_obj["_iedm_skin_bind_target_bone"] = str(bind_target_name or "")
        if bind_target_loc is not None:
            mesh_obj["_iedm_skin_bind_target_loc"] = [
                float(bind_target_loc.x),
                float(bind_target_loc.y),
                float(bind_target_loc.z),
            ]
    except Exception as exc:
        _log.debug("Optional operation failed: {}".format(exc), level=2)

    _attach_skin_mesh_to_parent(mesh_obj, arm_obj, skin_node, bind_target_loc)

    _localize_skin_mesh_to_bind_target(mesh_obj, bind_target_loc)

    if not mesh_obj.data.vertices:
        return
    _assign_skin_vertex_weights(
        mesh_obj, skin_node, control_bone, weight_bones, group_map, channel_slices
    )
    _bake_skin_mesh_object_transforms(mesh_obj)
