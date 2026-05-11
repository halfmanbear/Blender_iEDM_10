# Session orchestration fragment.
# Functions here drive the top-level import sequence; the pipeline detail
# (graph construction, node processing, animations) lives in import_pipeline.py.

import os
from dataclasses import dataclass, field
from typing import Mapping

import bpy

from ..edm_format import EDMFile
from ..edm_format.mathtypes import Matrix
from ..edm_format.types import AnimatingNode, LodNode, TransformNode
from ..utils import chdir, print_edm_graph
from .bbox_utils import (
  _cache_root_aabb_payloads_on_scene,
  _create_light_box_from_root,
  _create_user_box_from_root,
  _has_special_box,
  create_bounding_box_from_root,
)
from .ctrl_splits import (
  _rename_control_wrapper_mesh_pairs,
  _split_multi_arg_nonarmature_controls,
  _split_multi_arg_rotation_controls,
)
from .graph_build import build_graph
from .graph_pipeline import (
  _assign_collections,
  _debug_filter_terms,
  _debug_fmt_rot_deg,
  _debug_fmt_vec3,
  _reset_mesh_origin_mode,
  _reset_transform_debug,
)
from .graph_postprocess import (
  _apply_plain_root_visibility_mesh_basis_fix,
  _apply_plain_root_visibility_object_basis_fix,
  _apply_root_visibility_pair_wrapper_basis_fix,
  _apply_static_root_visibility_wrapper_basis_fix,
  _fix_owner_encoded_render_offsets,
  _zero_render_child_mesh_locals_under_transform,
)
from .import_capabilities import derive_import_capabilities, inspect_import_graph
from .material_setup import create_material
from .node_transform import apply_node_transform
from .nodes.armature import _prepare_bone_import
from .nodes.core import _apply_shadeless, _process_lod_post_children, process_node
from .nodes.diagnostics import _print_import_diagnostics
from .orient_fixes import (
  _apply_collision_mesh_orientation_fix,
  _apply_scene_root_empty_orientation_fix,
  _apply_scene_root_mesh_orientation_fix,
)
from .orient_scale import _rewrite_oriented_scale_controls
from .prelude import (
  DEFAULT_PROFILE,
  FRAME_SCALE,
  _ROOT_BASIS_FIX,
  _import_capability_detail,
  _import_capability_name,
  _import_ctx,
  _import_profile_detail,
  _import_profile_flag,
  _import_profile_name,
  _log,
)
from .skin_rewrites import _resolve_skin_parent_overrides_by_bind_rest
from .vis_rewrites import (
  _apply_argvis_chain_basis_fix,
  _apply_plain_root_visibility_basis_fix,
  _apply_visibility_pair_wrapper_object_basis_fix,
  _fix_inverse_scaled_visibility_rest_offset,
  _restore_skin_visibility_transform_basis,
  _split_multi_arg_visibility_controls,
)


@dataclass(frozen=True)
class SceneBoxOptions:
  bounding_box: bool = True
  user_box: bool = True
  light_box: bool = True

  def any_enabled(self):
    return self.bounding_box or self.user_box or self.light_box


@dataclass(frozen=True)
class ImportOptions:
  raw: Mapping[str, object] = field(default_factory=dict)
  scene_boxes: SceneBoxOptions = field(default_factory=SceneBoxOptions)

  @classmethod
  def from_mapping(cls, options):
    if isinstance(options, cls):
      return options
    raw = dict(options or {})
    return cls(raw=raw, scene_boxes=_resolve_scene_box_options(raw))

  def get(self, key, default=None):
    return self.raw.get(key, default)


def _resolve_scene_box_options(options):
  """Resolve scene-box creation flags from legacy and per-box options."""
  has_individual_options = any(
    key in options
    for key in ("import_bounding_box", "import_user_box", "import_light_box")
  )

  if "preserve_scene_boxes" in options:
    default_enabled = bool(options.get("preserve_scene_boxes"))
  else:
    default_enabled = not has_individual_options

  return SceneBoxOptions(
    bounding_box=bool(options.get("import_bounding_box", default_enabled)),
    user_box=bool(options.get("import_user_box", default_enabled)),
    light_box=bool(options.get("import_light_box", default_enabled)),
  )


def _start_import_session(filename, options):
  # Re-initialize the global import context for this new import session
  _import_ctx.__init__()
  _reset_transform_debug(options)
  _import_ctx.render_split_debug = {"enabled": bool(options.get("debug_render_splits", False))}
  _reset_mesh_origin_mode(options)
  _import_ctx.bone_import_ctx = None
  _import_ctx.source_dir = os.path.dirname(os.path.abspath(filename))

  # Parse the EDM file
  edm = EDMFile(filename)

  if getattr(_import_ctx, "verbosity", 0) >= 1:
    print("[iEDM] Debug[1]: Raw file graph:")
    print_edm_graph(edm.transformRoot)

  # Store EDM version globally early so importer heuristics can use it before
  # addon-export patch toggles are configured.
  _import_ctx.edm_version = edm.version
  _import_ctx.import_profile = DEFAULT_PROFILE
  _import_ctx.import_capabilities = derive_import_capabilities(edm)
  features = inspect_import_graph(edm)

  _log.info("EDM import capabilities = {} ({})".format(
    _import_capability_name(),
    getattr(getattr(_import_ctx, "import_capabilities", None), "description", ""),
  ))
  _log.info("EDM import capability detail = {}".format(_import_capability_detail()))

  # Dump active flags so flag-related bugs are immediately visible
  caps = getattr(_import_ctx, "import_capabilities", None)
  flags = getattr(caps, "flags", None) or {}
  if flags:
    active = {k: v for k, v in flags.items() if v}
    if active:
      _log.debug("Active flags: {}".format(
        ", ".join("{}={}".format(k, v) for k, v in sorted(active.items()))
      ), level=1)

  return edm, features


def _configure_mesh_origin_mode(options, features):
  # Detect v10 owner-encoded split render graphs (shared-parent render chunks).
  has_bones = features.has_bones
  has_owner_encoded_split_renders = features.has_owner_encoded_split_renders
  has_generic_render_chunks = features.has_generic_render_chunks
  has_shell_nodes = features.has_shell_nodes
  has_segments_nodes = features.has_segments_nodes
  plain_root_v10 = features.plain_root_v10
  auto_geometry_safe_v10_split = bool(
    plain_root_v10
    and not has_bones
    and (
      has_owner_encoded_split_renders
      or (has_generic_render_chunks and has_shell_nodes)
    )
  )

  user_mesh_origin_mode = str(options.get("mesh_origin_mode", "") or "").upper()
  can_auto_override_mesh_origin = (user_mesh_origin_mode in {"", "APPROX"})
  if (
    _import_profile_flag("auto_raw_mesh_origin_v10_split")
    and auto_geometry_safe_v10_split
    and can_auto_override_mesh_origin
    and _import_ctx.mesh_origin_mode != "RAW"
  ):
    _import_ctx.mesh_origin_mode = "RAW"
    _log.info("Auto-selected mesh origin mode RAW for v10 split control-node asset (plain root, no bones).")
  elif (
    _import_profile_flag("auto_raw_mesh_origin_plain_root_non_skeletal")
    and
    can_auto_override_mesh_origin
    and _import_ctx.mesh_origin_mode != "RAW"
    and plain_root_v10
    and not has_bones
  ):
    _import_ctx.mesh_origin_mode = "RAW"
    _log.info("Auto-selected mesh origin mode RAW for v10 plain-root non-skeletal asset (preserve authored transforms).")


def _configure_blender_import_scene():
  bpy.context.preferences.edit.use_negative_frames = False
  bpy.context.scene.use_preview_range = False
  bpy.context.scene.frame_start = 0
  bpy.context.scene.frame_end = FRAME_SCALE
  bpy.context.scene.frame_set(0)


def _create_import_materials(filename, edm, options):
  with chdir(os.path.dirname(os.path.abspath(filename))):
    for material in edm.root.materials:
      material.blender_material = create_material(material)
      if material.blender_material and options.get("shadeless", False):
        _apply_shadeless(material.blender_material)

  if _import_ctx.mesh_origin_mode == "RAW":
    _log.info("Mesh origin mode RAW (no geometry-center recenter on render-only nodes)")


def _store_import_metadata(edm):
  bpy.context.scene.edm_version = edm.version
  try:
    bpy.context.scene["_iedm_import_profile"] = str(_import_profile_name())
    bpy.context.scene["_iedm_import_profile_detail"] = str(_import_profile_detail())
    bpy.context.scene["_iedm_import_capabilities"] = str(_import_capability_name())
    bpy.context.scene["_iedm_import_capability_detail"] = str(_import_capability_detail())
  except Exception as e:
    _log.warn("store import metadata", exc=e)


def _create_graph_root_object(graph, options, features):
  root_tf = getattr(graph.root, "transform", None)
  has_root_transform_payload = isinstance(root_tf, (TransformNode, AnimatingNode))
  has_shell_nodes = features.has_shell_nodes
  has_segments_nodes = features.has_segments_nodes
  wants_embedded_collision_root_basis_object = bool(
    _import_ctx.edm_version >= 10
    and _import_profile_flag("embedded_collision_scene_root_basis_object")
    and (has_shell_nodes or has_segments_nodes)
  )
  _import_ctx.collision_geometry_basis_fix = bool(
    _import_ctx.edm_version >= 10
    and _import_profile_flag("collision_geometry_basis_fix")
    and (has_shell_nodes or has_segments_nodes)
  )
  wants_lod_root_basis_object = bool(
    _import_ctx.edm_version >= 10
    and _import_profile_flag("lod_root_scene_basis_object")
    and isinstance(root_tf, LodNode)
  )

  force_root_obj = bool(options.get("force_root_object", False))

  wants_profile_root_basis_object = bool(
    _import_ctx.edm_version >= 10
    and _import_profile_flag("v10_root_object_basis_fix")
    and _import_profile_flag("implicit_scene_root_basis_object")
    and not has_root_transform_payload
  )

  _import_ctx.use_scene_root_basis_object = bool(
    force_root_obj or wants_embedded_collision_root_basis_object or wants_lod_root_basis_object or (
      _import_profile_flag("implicit_scene_root_basis_object")
      and has_root_transform_payload
    ) or wants_profile_root_basis_object
    or (
      has_root_transform_payload
      and _import_ctx.edm_version >= 10
      and _import_profile_flag("v10_root_object_basis_fix")
    )
  )

  if has_root_transform_payload or force_root_obj or wants_embedded_collision_root_basis_object or wants_lod_root_basis_object or wants_profile_root_basis_object:
    root_name = getattr(root_tf, "name", "") or ""
    if not root_name.strip():
      root_name = "_EDMFileRoot"

    root_obj = bpy.data.objects.new(root_name, None)
    root_obj.empty_display_size = 0.1
    bpy.context.collection.objects.link(root_obj)
    graph.root.blender = root_obj

    if _import_ctx.edm_version >= 10 and _import_profile_flag("v10_root_object_basis_fix"):
      m_root = Matrix.Identity(4)
      if hasattr(root_tf, "matrix"):
        m_root = Matrix(root_tf.matrix)
      elif hasattr(root_tf, "base") and hasattr(root_tf.base, "matrix"):
        m_root = Matrix(root_tf.base.matrix)
      root_obj.matrix_basis = _ROOT_BASIS_FIX @ m_root
    elif has_root_transform_payload:
      apply_node_transform(graph.root, graph.root.blender, used_shared_parent=False)
  else:
    graph.root.blender = None


def _create_import_scene_boxes(edm_root, box_options):
  if box_options.any_enabled():
    bbox_coord_fix = _ROOT_BASIS_FIX if _import_ctx.use_scene_root_basis_object else None
    if box_options.bounding_box and not _has_special_box("BOUNDING_BOX"):
      create_bounding_box_from_root(edm_root, coord_fix=bbox_coord_fix)
    if box_options.user_box:
      _create_user_box_from_root(edm_root, coord_fix=bbox_coord_fix)
    if box_options.light_box:
      _create_light_box_from_root(edm_root, coord_fix=bbox_coord_fix)


def _debug_dump_stage_objects(stage_name):
  dbg = getattr(_import_ctx, "transform_debug", {}) or {}
  if not dbg.get("enabled"):
    return
  filter_terms = _debug_filter_terms()
  if not filter_terms:
    return

  matches = []
  for ob in list(getattr(bpy.data, "objects", []) or []):
    name = getattr(ob, "name", "") or ""
    lowered = name.lower()
    if any(term in lowered for term in filter_terms):
      matches.append(ob)
  if not matches:
    return

  print("[iEDM][STAGE] begin stage={} matched={}".format(stage_name, len(matches)))
  for ob in sorted(matches, key=lambda item: getattr(item, "name", "")):
    try:
      basis_loc, basis_rot, basis_scale = ob.matrix_basis.decompose()
      world_loc, world_rot, world_scale = ob.matrix_world.decompose()
      action = getattr(getattr(ob, "animation_data", None), "action", None)
      curves = [fc.data_path for fc in getattr(action, "fcurves", [])] if action else []
      print(
        "[iEDM][STAGE] stage={} obj={} type={} parent={} action={} curves={} basis_loc={} basis_rot_deg={} basis_scale={} world_loc={} world_rot_deg={} world_scale={}".format(
          stage_name,
          ob.name,
          getattr(ob, "type", ""),
          getattr(getattr(ob, "parent", None), "name", None),
          getattr(action, "name", None),
          curves,
          _debug_fmt_vec3(basis_loc),
          _debug_fmt_rot_deg(basis_rot),
          _debug_fmt_vec3(basis_scale),
          _debug_fmt_vec3(world_loc),
          _debug_fmt_rot_deg(world_rot),
          _debug_fmt_vec3(world_scale),
        )
      )
    except Exception as e:
      print("[iEDM][STAGE] stage={} obj={} error={}".format(stage_name, getattr(ob, "name", "<unknown>"), e))
  print("[iEDM][STAGE] end stage={}".format(stage_name))


def _run_render_offset_postprocess(graph):
  if _import_profile_flag("owner_encoded_render_offset_fix"):
    _fix_owner_encoded_render_offsets(graph)
  _zero_render_child_mesh_locals_under_transform(graph)
  _debug_dump_stage_objects("after_process_node")


def _run_visibility_basis_postprocess(graph):
  _apply_plain_root_visibility_basis_fix(graph)
  _debug_dump_stage_objects("after_plain_root_visibility_basis_fix")
  _apply_plain_root_visibility_object_basis_fix()
  _debug_dump_stage_objects("after_plain_root_visibility_object_basis_fix")
  _apply_plain_root_visibility_mesh_basis_fix()
  _debug_dump_stage_objects("after_plain_root_visibility_mesh_basis_fix")
  _apply_static_root_visibility_wrapper_basis_fix(graph)
  _debug_dump_stage_objects("after_static_root_visibility_wrapper_basis_fix")
  _apply_root_visibility_pair_wrapper_basis_fix(graph)
  _debug_dump_stage_objects("after_root_visibility_pair_wrapper_basis_fix")


def _run_control_rewrite_postprocess(graph, options):
  _apply_argvis_chain_basis_fix()
  _debug_dump_stage_objects("after_argvis_chain_basis_fix")
  _rewrite_oriented_scale_controls(graph)
  _debug_dump_stage_objects("after_rewrite_oriented_scale_controls")
  graph.walk_tree(_process_lod_post_children)
  _debug_dump_stage_objects("after_lod_post_children")

  rewrite_multi_arg_helpers = bool(
    options.get("rewrite_multi_arg_helpers", _import_profile_flag("default_rewrite_multi_arg_helpers"))
  )
  if rewrite_multi_arg_helpers:
    _split_multi_arg_nonarmature_controls(graph)
    _split_multi_arg_visibility_controls(graph)
  else:
    _split_multi_arg_rotation_controls(graph)
  _debug_dump_stage_objects("after_multi_arg_rewrite")

  _apply_visibility_pair_wrapper_object_basis_fix()
  _debug_dump_stage_objects("after_visibility_pair_wrapper_object_basis_fix")
  _rename_control_wrapper_mesh_pairs(graph)
  _debug_dump_stage_objects("after_rename_control_wrapper_mesh_pairs")


def _run_orientation_postprocess():
  _apply_collision_mesh_orientation_fix()
  _debug_dump_stage_objects("after_collision_mesh_orientation_fix")
  _apply_scene_root_mesh_orientation_fix()
  _debug_dump_stage_objects("after_scene_root_mesh_orientation_fix")
  _apply_scene_root_empty_orientation_fix()
  _debug_dump_stage_objects("after_scene_root_empty_orientation_fix")
  _restore_skin_visibility_transform_basis()
  _debug_dump_stage_objects("after_restore_skin_visibility_transform_basis")
  _resolve_skin_parent_overrides_by_bind_rest()
  _debug_dump_stage_objects("after_resolve_skin_parent_overrides_by_bind_rest")
  _fix_inverse_scaled_visibility_rest_offset()
  _debug_dump_stage_objects("after_fix_inverse_scaled_visibility_rest_offset")


def _finish_import_postprocess(edm, graph, options):
  _print_import_diagnostics(edm, graph)

  if options.get("assign_collections", _import_profile_flag("default_assign_collections")):
    _assign_collections(graph)

  if _import_ctx.transform_debug.get("enabled"):
    if _import_ctx.transform_debug["emitted"] >= _import_ctx.transform_debug["limit"]:
      _log.info("Transform debug reached output limit of {}".format(_import_ctx.transform_debug["limit"]))
    _log.info("Transform debug emitted {} record(s)".format(_import_ctx.transform_debug["emitted"]))

  _propagate_visibility_hide_to_render_nodes()

  bpy.context.scene.frame_set(100)
  bpy.context.view_layer.update()


def _run_import_postprocess(edm, graph, options):
  _run_render_offset_postprocess(graph)
  _run_visibility_basis_postprocess(graph)
  _run_control_rewrite_postprocess(graph, options)
  _run_orientation_postprocess()
  _finish_import_postprocess(edm, graph, options)


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

  _prepare_bone_import(graph, graph.root.blender)

  graph.walk_tree(process_node)
  _run_import_postprocess(edm, graph, options)


def _propagate_visibility_hide_to_render_nodes():
  """Viewport-visibility propagation from ArgVisibilityNode empties to descendants.

  Not implemented: keyframe-based and driver-based approaches both have
  unavoidable side effects (see import_pipeline.py for full rationale).
  The ArgVisibilityNode empty already carries an animated hide_viewport fcurve.
  """
  pass
