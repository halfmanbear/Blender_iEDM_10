# Pipeline coordinator — all implementation lives in the fragments below.
# All names resolved via the shared namespace injected by reader.py.
#
# Fragment files (loaded via reader.py _MODULES):
#   anim_actions.py    — visibility + ArgAnimation action builders
#   orient_scale.py    — oriented scale decomposition and rewrite pass
#   vis_rewrites.py    — visibility graph basis-fix passes
#   skin_rewrites.py   — skin parent bind-rest resolution
#   orient_fixes.py    — mesh/empty world orientation post-passes
#   ctrl_splits.py     — multi-arg control splitting + mesh-pair renaming
#   node_transform.py  — apply_node_transform
#
# Bottom imports trigger sub-module loading before reader.py builds _SHARED.

from .material_setup import (
  _find_texture_file,
  _ensure_placeholder_texture_image,
  create_material,
  _ANIMATED_UNIFORM_TO_EDMPROPS,
  _material_prop_scalar,
  _material_prop_value,
  _material_texture_payload,
  _infer_material_texture_roles,
  _material_uniform_payload,
  _material_animated_uniform_payload,
  _preserve_material_payload,
  _preserve_object_material_args,
  _map_animated_uniforms_to_edmprops,
)


from .prelude import (
  DEFAULT_PROFILE,
  _ROOT_BASIS_FIX,
  _SUFFIX_RE,
  _is_child_of_file_root,
  _is_top_level_visibility_authored_pair,
  _is_authored_argvis_control_pair,
  _is_nested_authored_argvis_control_pair,
  _is_root_visibility_chain_authored_pair,
  _matrix_trs_summary,
  _strip_anim_prefix,
  _transform_display_name,
  _log_bone_debug_event,
)
from .graph_pipeline import (
    _anim_quaternion_to_blender,
    _debug_filter_terms,
    _get_action_argument,
    _debug_fmt_vec3,
    _debug_fmt_rot_deg,
    _reset_transform_debug,
    _reset_mesh_origin_mode,
  )
from .object_create import create_connector, create_object, create_segments
from .animation import (
  _arg_anim_vector_to_blender,
  _normalize_angle_radians,
  _normalize_euler_xyz,
  _normalize_euler_action_curves,
  _finalize_authored_transform_action,
  add_position_fcurves,
  add_rotation_fcurves,
  add_scale_fcurves,
  _is_pos90_x_basis_matrix,
  _quat_is_identity,
)
from .graph_postprocess import (
  _apply_plain_root_visibility_mesh_basis_fix,
  _apply_plain_root_visibility_object_basis_fix,
  _apply_root_visibility_pair_wrapper_basis_fix,
  _apply_static_root_visibility_wrapper_basis_fix,
  _fix_owner_encoded_render_offsets,
  _reparent_preserve_world,
  _zero_render_child_mesh_locals_under_transform,
)
