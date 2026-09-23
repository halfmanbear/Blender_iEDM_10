# Pipeline coordinator — all implementation lives in the fragments below.
# This module re-exports their public names for callers that only import
# import_pipeline (each fragment also imports directly from the others it
# needs; see docs/ARCHITECTURE.md).
#
# Fragment files:
#   anim_actions.py    — visibility + ArgAnimation action builders
#   orient_scale.py    — oriented scale decomposition and rewrite pass
#   vis_rewrites.py    — visibility graph basis-fix passes
#   skin_rewrites.py   — skin parent bind-rest resolution
#   orient_fixes.py    — mesh/empty world orientation post-passes
#   ctrl_splits.py     — multi-arg control splitting + mesh-pair renaming
#   node_transform.py  — apply_node_transform

from .animation import (
    _arg_anim_vector_to_blender,
    _finalize_authored_transform_action,
    _is_pos90_x_basis_matrix,
    _normalize_angle_radians,
    _normalize_euler_action_curves,
    _normalize_euler_xyz,
    _quat_is_identity,
    add_position_fcurves,
    add_rotation_fcurves,
    add_scale_fcurves,
)
from .graph_pipeline import (
    _anim_quaternion_to_blender,
    _debug_filter_terms,
    _debug_fmt_rot_deg,
    _debug_fmt_vec3,
    _get_action_argument,
    _reset_mesh_origin_mode,
    _reset_transform_debug,
)
from .graph_postprocess import (
    _apply_root_visibility_pair_wrapper_basis_fix,
    _apply_static_root_visibility_wrapper_basis_fix,
    _reparent_preserve_world,
    _zero_render_child_mesh_locals_under_transform,
)
from .material_setup import (
    _ANIMATED_UNIFORM_TO_EDMPROPS,
    _ensure_placeholder_texture_image,
    _find_texture_file,
    _infer_material_texture_roles,
    _map_animated_uniforms_to_edmprops,
    _material_animated_uniform_payload,
    _material_prop_scalar,
    _material_prop_value,
    _material_texture_payload,
    _material_uniform_payload,
    _preserve_material_payload,
    _preserve_object_material_args,
    create_material,
)
from .object_create import create_connector, create_object, create_segments
from .prelude import (
    _ROOT_BASIS_FIX,
    _SUFFIX_RE,
    DEFAULT_PROFILE,
    _is_authored_argvis_control_pair,
    _is_child_of_file_root,
    _is_nested_authored_argvis_control_pair,
    _is_root_visibility_chain_authored_pair,
    _is_top_level_visibility_authored_pair,
    _log_bone_debug_event,
    _matrix_trs_summary,
    _strip_anim_prefix,
    _transform_display_name,
)
