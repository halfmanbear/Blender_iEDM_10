"""Coordinate an EDM import and retain historical session helper imports."""

import os as os
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from typing import Mapping as Mapping

import bpy as bpy

from ..edm_format import EDMFile as EDMFile
from ..edm_format.mathtypes import Matrix as Matrix
from ..edm_format.types import AnimatingNode as AnimatingNode
from ..edm_format.types import LodNode as LodNode
from ..edm_format.types import TransformNode as TransformNode
from ..edm_format.types.render_special_nodes import SkinNode as SkinNode
from ..utils import action_fcurves as action_fcurves
from ..utils import chdir as chdir
from ..utils import print_edm_graph as print_edm_graph
from . import graph_postprocess as _graph_postprocess_compat
from . import session_visibility as _session_visibility_compat
from . import skin_rewrites as _skin_rewrites_compat
from . import skin_visibility as _skin_visibility_compat
from . import vis_rewrites as _vis_rewrites_compat
from .bbox_utils import (
    _cache_root_aabb_payloads_on_scene as _cache_root_aabb_payloads_on_scene,
)
from .bbox_utils import _create_light_box_from_root as _create_light_box_from_root
from .bbox_utils import _create_user_box_from_root as _create_user_box_from_root
from .bbox_utils import _has_special_box as _has_special_box
from .bbox_utils import create_bounding_box_from_root as create_bounding_box_from_root
from .ctrl_splits import (
    _rename_control_wrapper_mesh_pairs as _rename_control_wrapper_mesh_pairs,
)
from .ctrl_splits import (
    _split_multi_arg_rotation_controls as _split_multi_arg_rotation_controls,
)
from .graph_build import build_graph as build_graph
from .graph_collections import _assign_collections as _assign_collections
from .graph_diagnostics import _debug_filter_terms as _debug_filter_terms
from .graph_diagnostics import _debug_fmt_rot_deg as _debug_fmt_rot_deg
from .graph_diagnostics import _debug_fmt_vec3 as _debug_fmt_vec3
from .graph_diagnostics import _reset_mesh_origin_mode as _reset_mesh_origin_mode
from .graph_diagnostics import _reset_transform_debug as _reset_transform_debug
from .import_capabilities import (
    derive_import_capabilities as derive_import_capabilities,
)
from .import_capabilities import inspect_import_graph as inspect_import_graph
from .import_context import _ROOT_BASIS_FIX as _ROOT_BASIS_FIX
from .import_context import DEFAULT_PROFILE as DEFAULT_PROFILE
from .import_context import FRAME_SCALE as FRAME_SCALE
from .import_context import _import_capability_detail as _import_capability_detail
from .import_context import _import_capability_name as _import_capability_name
from .import_context import _import_ctx as _import_ctx
from .import_context import _import_profile_detail as _import_profile_detail
from .import_context import _import_profile_flag as _import_profile_flag
from .import_context import _import_profile_name as _import_profile_name
from .import_context import _log as _log
from .material_setup import create_material as create_material
from .node_identity import _assign_action as _assign_action
from .node_transform import apply_node_transform as apply_node_transform
from .nodes.armature import _prepare_bone_import as _prepare_bone_import
from .nodes.core import _apply_shadeless as _apply_shadeless
from .nodes.core import _process_lod_post_children as _process_lod_post_children
from .nodes.core import process_node as process_node
from .nodes.diagnostics import _print_import_diagnostics as _print_import_diagnostics
from .orient_scale import (
    _rewrite_oriented_scale_controls as _rewrite_oriented_scale_controls,
)
from .scene_root import _create_graph_root_object as _create_graph_root_object
from .scene_root import _create_import_scene_boxes as _create_import_scene_boxes
from .scene_root import (
    _detect_bonetransform_prefix_compound as _detect_bonetransform_prefix_compound,
)
from .session_postprocess import _debug_dump_stage_objects as _debug_dump_stage_objects
from .session_postprocess import _finalize_render_origins as _finalize_render_origins
from .session_postprocess import _finalize_skin_bind_space as _finalize_skin_bind_space
from .session_postprocess import (
    _finish_import_postprocess as _finish_import_postprocess,
)
from .session_postprocess import (
    _run_control_rewrite_postprocess as _run_control_rewrite_postprocess,
)
from .session_postprocess import _run_import_postprocess as _run_import_postprocess
from .session_postprocess import (
    _run_orientation_postprocess as _run_orientation_postprocess,
)
from .session_postprocess import (
    _run_render_offset_postprocess as _run_render_offset_postprocess,
)
from .session_postprocess import (
    _run_skin_transform_postprocess as _run_skin_transform_postprocess,
)
from .session_postprocess import (
    _run_visibility_basis_postprocess as _run_visibility_basis_postprocess,
)
from .session_setup import ImportOptions as ImportOptions
from .session_setup import SceneBoxOptions as SceneBoxOptions
from .session_setup import (
    _configure_blender_import_scene as _configure_blender_import_scene,
)
from .session_setup import _configure_mesh_origin_mode as _configure_mesh_origin_mode
from .session_setup import _create_import_materials as _create_import_materials
from .session_setup import _resolve_scene_box_options as _resolve_scene_box_options
from .session_setup import _start_import_session as _start_import_session
from .session_setup import _store_import_metadata as _store_import_metadata
from .session_visibility import (
    _add_visibility_hide_driver as _add_visibility_hide_driver,
)
from .session_visibility import _visibility_controller_for as _visibility_controller_for
from .session_visibility import (
    _visibility_frame_intervals as _visibility_frame_intervals,
)
from .vis_rewrites import _apply_argvis_chain_basis_fix as _apply_argvis_chain_basis_fix
from .vis_rewrites import (
    _apply_plain_root_visibility_basis_fix as _apply_plain_root_visibility_basis_fix,
)
from .vis_rewrites import (
    _split_multi_arg_visibility_controls as _split_multi_arg_visibility_controls,
)
from .visibility_timeline import _visibility_scene_keys as _visibility_scene_keys
from .visibility_timeline import _visibility_scene_ranges as _visibility_scene_ranges

_apply_root_visibility_pair_wrapper_basis_fix = (
    _graph_postprocess_compat._apply_root_visibility_pair_wrapper_basis_fix
)

_apply_static_root_visibility_wrapper_basis_fix = (
    _graph_postprocess_compat._apply_static_root_visibility_wrapper_basis_fix
)

_zero_render_child_mesh_locals_under_transform = (
    _graph_postprocess_compat._zero_render_child_mesh_locals_under_transform
)

_propagate_visibility_hide_to_render_nodes = (
    _session_visibility_compat._propagate_visibility_hide_to_render_nodes
)

_resolve_skin_parent_overrides_by_bind_rest = (
    _skin_rewrites_compat._resolve_skin_parent_overrides_by_bind_rest
)

_fix_inverse_scaled_visibility_rest_offset = (
    _skin_visibility_compat._fix_inverse_scaled_visibility_rest_offset
)

_restore_skin_visibility_transform_basis = (
    _skin_visibility_compat._restore_skin_visibility_transform_basis
)

_apply_visibility_pair_wrapper_object_basis_fix = (
    _vis_rewrites_compat._apply_visibility_pair_wrapper_object_basis_fix
)


def read_file(filename, options=None):
    options = ImportOptions.from_mapping(options)

    edm, features = _start_import_session(filename, options)
    _configure_mesh_origin_mode(options, features)
    _configure_blender_import_scene()
    _create_import_materials(filename, edm, options)
    _store_import_metadata(edm)

    _import_ctx.file_has_bones = bool(features.has_bones)

    _cache_root_aabb_payloads_on_scene(edm.root)

    graph = build_graph(edm)
    graph.print_tree()

    _create_graph_root_object(graph, options, features)
    _create_import_scene_boxes(edm.root, options.scene_boxes)

    has_skin_nodes = any(
        isinstance(getattr(n, "render", None), SkinNode) for n in graph.nodes
    )
    if has_skin_nodes:
        _prepare_bone_import(graph, graph.root.blender)

    graph.walk_tree(process_node)

    _run_import_postprocess(edm, graph, options)
