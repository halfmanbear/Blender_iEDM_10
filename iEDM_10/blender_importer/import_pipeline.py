
def read_file(filename, options=None):
  options = options or {}
  # Re-initialize the global import context for this new import session
  _import_ctx.__init__()
  _reset_transform_debug(options)
  _import_ctx.render_split_debug = {"enabled": bool(options.get("debug_render_splits", False))}
  _reset_mesh_origin_mode(options)
  _import_ctx.bone_import_ctx = None
  _import_ctx.source_dir = os.path.dirname(os.path.abspath(filename))

  # Parse the EDM file
  edm = EDMFile(filename)

  print("Raw file graph:")
  print_edm_graph(edm.transformRoot)

  # Store EDM version globally early so importer heuristics can use it before
  # addon-export patch toggles are configured.
  _import_ctx.edm_version = edm.version
  _import_ctx.edm_exporter_class, _import_ctx.edm_exporter_detail = classify_edm_file(filename)
  _import_ctx.import_profile = get_legacy_import_profile(_import_ctx.edm_exporter_class)
  _import_ctx.import_capabilities = derive_import_capabilities(edm)
  features = inspect_import_graph(edm)

  print("Info: EDM import capabilities = {} ({})".format(
    _import_capability_name(),
    getattr(getattr(_import_ctx, "import_capabilities", None), "description", ""),
  ))
  print("Info: EDM import capability detail = {}".format(_import_capability_detail()))
  print("Info: Legacy EDM family classifier = {} ({})".format(
    _import_ctx.edm_exporter_class,
    _import_ctx.edm_exporter_detail,
  ))
  print("Info: Legacy EDM override profile = {} ({})".format(
    _import_profile_name(),
    _import_profile_detail(),
  ))

  # Detect v10 owner-encoded split render graphs (shared-parent render chunks).
  # These small control-node assets (e.g. Geometry+Animation+Collision.edm)
  # are sensitive to mesh recentering and plain-root basis suppression.
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

  # Preserve authored pivots on v10 split control-node assets by disabling
  # mesh-origin recentering. Blender operator defaults may pass
  # mesh_origin_mode='APPROX' explicitly, so treat APPROX as auto-overridable.
  user_mesh_origin_mode = str(options.get("mesh_origin_mode", "") or "").upper()
  can_auto_override_mesh_origin = (user_mesh_origin_mode in {"", "APPROX"})
  if (
    _import_profile_flag("auto_raw_mesh_origin_v10_split")
    and auto_geometry_safe_v10_split
    and can_auto_override_mesh_origin
    and _import_ctx.mesh_origin_mode != "RAW"
  ):
    _import_ctx.mesh_origin_mode = "RAW"
    print("Info: Auto-selected mesh origin mode RAW for v10 split control-node asset (plain root, no bones).")
  elif (
    _import_profile_flag("auto_raw_mesh_origin_plain_root_non_skeletal")
    and
    can_auto_override_mesh_origin
    and _import_ctx.mesh_origin_mode != "RAW"
    and plain_root_v10
    and not has_bones
  ):
    # F-117-style plain-root non-skeletal v10 assets preserve authored object
    # transforms only when mesh origin recentering is disabled. APPROX mode adds
    # per-mesh local offsets (e.g. F-117_paint.012 +0.007624 X / +0.438309 Z).
    _import_ctx.mesh_origin_mode = "RAW"
    print("Info: Auto-selected mesh origin mode RAW for v10 plain-root non-skeletal asset (preserve authored transforms).")

  # Set the required blender preferences
  bpy.context.preferences.edit.use_negative_frames = False
  bpy.context.scene.use_preview_range = False
  bpy.context.scene.frame_start = 0
  bpy.context.scene.frame_end = FRAME_SCALE
  bpy.context.scene.frame_set(0)



  # Convert the materials (cwd matters for search)
  with chdir(os.path.dirname(os.path.abspath(filename))):
    for material in edm.root.materials:
      material.blender_material = create_material(material)
      if material.blender_material and options.get("shadeless", False):
        _apply_shadeless(material.blender_material)

  if _import_ctx.mesh_origin_mode == "RAW":
    print("Info: Mesh origin mode RAW (no geometry-center recenter on render-only nodes)")

  # Store version in scene for export
  bpy.context.scene.edm_version = edm.version
  try:
    bpy.context.scene["_iedm_import_family"] = str(_import_ctx.edm_exporter_class)
    bpy.context.scene["_iedm_import_family_detail"] = str(_import_ctx.edm_exporter_detail)
    bpy.context.scene["_iedm_import_profile"] = str(_import_profile_name())
    bpy.context.scene["_iedm_import_profile_detail"] = str(_import_profile_detail())
    bpy.context.scene["_iedm_import_capabilities"] = str(_import_capability_name())
    bpy.context.scene["_iedm_import_capability_detail"] = str(_import_capability_detail())
  except Exception:
    pass

  _import_ctx.file_has_bones = bool(has_bones)

  # For ALL .edm files, we MUST apply _ROOT_BASIS_FIX (the inverse of the DCS 
  # Y-up coordinate swap) to the root children so that they import into Blender 
  # as Z-up. When re-exported, io_scene_edm will apply its ROOT_TRANSFORM_MATRIX 
  # (which is the forward DCS swap) and restore the file to its original state, 
  # whether it was originally exported from 3ds Max (with a root swap matrix) 
  # or from Blender (with optimized meshes in Y-up space).

  # Cache raw AABB payloads early so the export fallback has them regardless of
  # whether the scene-box empties are created.
  preserve_scene_boxes = bool(options.get("preserve_scene_boxes", True))
  _cache_root_aabb_payloads_on_scene(edm.root)

  # Build the translation graph with hierarchy simplification
  graph = build_graph(edm)
  graph.print_tree()

  # For plain model::Node roots, avoid forcing a Blender empty:
  # the official round-trip can preserve a non-transform root and forcing an
  # authored object turns it into `TransformNode '_'` on export.
  # Keep an explicit root object only when the file root actually carries
  # transform payload.
  root_tf = getattr(graph.root, "transform", None)
  has_root_transform_payload = isinstance(root_tf, (TransformNode, AnimatingNode))
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
  
  # The official exporter wraps scene space under ROOT_TRANSFORM_MATRIX.
  # Import must apply the inverse at the same scene level for v10 assets.
  force_root_obj = bool(options.get("force_root_object", False))

  # Also create a root basis object when the profile needs one for
  # coordinate correction, even if the file root has no transform payload
  # (e.g. blob-format skeletal files with plain Node '_' root).
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
  )

  if has_root_transform_payload or force_root_obj or wants_embedded_collision_root_basis_object or wants_lod_root_basis_object or wants_profile_root_basis_object:
    root_name = getattr(root_tf, "name", "") or ""
    if not root_name.strip():
      root_name = "_EDMFileRoot"

    root_obj = bpy.data.objects.new(root_name, None)
    root_obj.empty_display_size = 0.1
    bpy.context.collection.objects.link(root_obj)
    graph.root.blender = root_obj

    # For V10, apply the global basis fix to the root object.
    # If the root carries its own transform (e.g. 3ds Max/Blender_ioEDM swap),
    # multiply them so they cancel out to Identity in Blender space.
    if _import_ctx.edm_version >= 10 and _import_profile_flag("v10_root_object_basis_fix"):
      m_root = Matrix.Identity(4)
      if hasattr(root_tf, "matrix"):
        m_root = Matrix(root_tf.matrix)
      elif hasattr(root_tf, "base") and hasattr(root_tf.base, "matrix"):
        m_root = Matrix(root_tf.base.matrix)
      
      # Apply the fix: Blender_World = _ROOT_BASIS_FIX * EDM_Root_Matrix
      root_obj.matrix_basis = _ROOT_BASIS_FIX @ m_root
    elif has_root_transform_payload:
      apply_node_transform(graph.root, graph.root.blender, used_shared_parent=False)
  else:
    graph.root.blender = None

  # Create explicit root box empties from decoded v10 root metadata.
  # The official exporter recomputes AABB from all scene geometry, including
  # extreme-position connectors (e.g. particle views at z=-280m), producing a
  # huge bounding box.  These special empties carry the original raw AABB values
  # so the export AABB passthrough patch can restore them exactly.
  #
  # EDM bbox vectors are stored in EDM Y-up space (same as all other geometry).
  # When the scene has a root basis object carrying _ROOT_BASIS_FIX, all other
  # objects inherit that Y→Z rotation via their parent chain.  The box empties
  # are placed directly in world space (not under the root object), so we must
  # apply _ROOT_BASIS_FIX to the AABB corners explicitly to land them in the
  # correct Blender Z-up world position.
  if preserve_scene_boxes:
    bbox_coord_fix = _ROOT_BASIS_FIX if _import_ctx.use_scene_root_basis_object else None
    if not _has_special_box("BOUNDING_BOX"):
      create_bounding_box_from_root(edm.root, coord_fix=bbox_coord_fix)
    _create_user_box_from_root(edm.root, coord_fix=bbox_coord_fix)
    _create_light_box_from_root(edm.root, coord_fix=bbox_coord_fix)

  # Reconstruct armature data for Bone/ArgAnimatedBone transforms
  _prepare_bone_import(graph, graph.root.blender)

  # Walk through every node, and do the node processing
  graph.walk_tree(process_node)

  # Fix owner-encoded render chunks whose shared_parent differs from their graph
  # parent. At this point all transform empties exist and have matrix_basis set.
  if _import_profile_flag("owner_encoded_render_offset_fix"):
    _fix_owner_encoded_render_offsets(graph)

  # Plain-root v10 assets can carry static geometry under top-level
  # ArgVisibilityNode wrappers. Those wrappers do not pass through the normal
  # transform-application block, so apply the missing root basis at the end of
  # import after all object parenting/offset post-passes have run.
  _apply_plain_root_visibility_basis_fix(graph)
  _apply_plain_root_visibility_object_basis_fix()
  _apply_plain_root_visibility_mesh_basis_fix()
  _apply_static_root_visibility_wrapper_basis_fix(graph)
  _apply_root_visibility_pair_wrapper_basis_fix(graph)
  # Keep legacy 3ds Max wrapper-repair passes opt-in. The reader
  # preserves the authored local graph instead of applying world-space repair.
  _apply_argvis_chain_basis_fix()
  _rewrite_oriented_scale_controls(graph)
  # Post-children pass: apply LodNode properties after children are created.
  graph.walk_tree(_process_lod_post_children)

  # Helper-empty rewrites improve round-trip survivability but reduce authored
  # Blender scene parity, so keep them opt-in.
  rewrite_multi_arg_helpers = bool(
    options.get("rewrite_multi_arg_helpers", _import_profile_flag("default_rewrite_multi_arg_helpers"))
  )
  if rewrite_multi_arg_helpers:
    _split_multi_arg_nonarmature_controls(graph)
    _split_multi_arg_visibility_controls(graph)
  else:
    _split_multi_arg_rotation_controls(graph)

  _apply_visibility_pair_wrapper_object_basis_fix()
  _rename_control_wrapper_mesh_pairs(graph)

  # Fix mesh orientation in BLOB_COLLISION files where parent chain broke
  # the _ROOT_BASIS_FIX rotation inheritance.
  _apply_collision_mesh_orientation_fix()

  # Legacy 3ds Max world-orientation repair. This is intentionally profile-
  # gated so strict DLL-aligned imports preserve the authored graph as read.
  _apply_3dsmax_mesh_orientation_fix()
  _apply_3dsmax_empty_orientation_fix()
  _restore_skin_visibility_transform_basis()

  # Diagnostic: count EDM nodes vs created Blender objects
  _print_import_diagnostics(edm, graph)

  # Mirror the official authoring scene structure by default.
  if options.get("assign_collections", _import_profile_flag("default_assign_collections")):
    _assign_collections(graph)

  if _import_ctx.transform_debug.get("enabled"):
    if _import_ctx.transform_debug["emitted"] >= _import_ctx.transform_debug["limit"]:
      print("Info: Transform debug reached output limit of {}".format(_import_ctx.transform_debug["limit"]))
    print("Info: Transform debug emitted {} record(s)".format(_import_ctx.transform_debug["emitted"]))

  # Key hide_viewport/hide_render on descendant MESH objects (model::RenderNodes)
  # of each ArgVisibilityNode empty so the viewport reflects EDM subtree
  # visibility. Done last so all parenting is finalised.
  _propagate_visibility_hide_to_render_nodes()

  bpy.context.scene.frame_set(100)
  bpy.context.view_layer.update()

def _propagate_visibility_hide_to_render_nodes():
  """Viewport-visibility propagation from ArgVisibilityNode empties to their
  descendant MESH objects.

  Not implemented: both keyframe-based and driver-based approaches have
  unavoidable side effects. Keyframe insertion (keyframe_insert) creates an
  action on the mesh that the exporter reads, corrupting visibility animation
  data. Drivers (driver_add) trigger a depsgraph evaluation that exposes stale
  parent-chain matrices, causing incorrect mesh scaling.

  The ArgVisibilityNode empty already carries an animated hide_viewport fcurve
  (from create_visibility_actions) which correctly reflects the EDM visibility
  in the viewport. Descendant mesh objects remain visible in Blender's viewport
  even when the parent empty is hidden — this is a known Blender limitation
  where hide_viewport does not cascade through the parent hierarchy. The
  round-trip export is unaffected.
  """
  pass


def create_visibility_actions(visNode):
  """Creates visibility actions from an ArgVisibilityNode"""
  def _vis_arg_to_frame(value):
    # EDM visibility ranges are authored on the same normalized timeline as
    # other v10 animation channels.
    return int(round((value + 1.0) * FRAME_SCALE / 2.0))

  actions = []
  for (arg, ranges) in visNode.visData:
    # Creates a visibility animation track
    vis_name = visNode.name or "node"
    # Strip visibility "v_" prefix and animation prefixes from name
    if vis_name.startswith("v_"):
      vis_name = vis_name[2:]
    vis_name = _strip_anim_prefix(vis_name)
    action = bpy.data.actions.new("{}_{}_Visib".format(arg, vis_name))
    actions.append(action)
    if hasattr(action, "argument"):
      action.argument = arg
    # Create f-curves for the official exporter visibility property and for
    # Blender's viewport hide so the ArgVisibilityNode empty itself hides.
    # The exporter reads only VISIBLE; hide_viewport is ignored by the
    # exporter's collection-based traversal. hide_render is intentionally
    # omitted — empties don't render and we don't want "Disable in Renders"
    # toggling. Descendant MESH objects get hide_viewport via the separate
    # _propagate_visibility_hide_to_render_nodes post-pass.
    curve_visible = action.fcurves.new(data_path="VISIBLE")
    curve_hide_vp = action.fcurves.new(data_path="hide_viewport")

    def _add_constant_key(curve, frame, value):
      curve.keyframe_points.add(1)
      key = curve.keyframe_points[-1]
      key.co = (frame, float(value))
      key.interpolation = 'CONSTANT'

    def _add_vis_keys(frame, visible):
      vis_val = 1.0 if visible else 0.0
      hide_val = 0.0 if visible else 1.0
      _add_constant_key(curve_visible, frame, vis_val)
      _add_constant_key(curve_hide_vp, frame, hide_val)

    # Create the keyframe data
    # Visibility ranges in EDM are [start, end) where end is the first OFF
    # frame boundary. Reconstruct official scene key style:
    # Emit:
    #   0     -> 0  (if first range starts after frame 0)
    #   start -> 1
    #   end-1 -> 1
    #   end   -> 0
    # The initial off-key is required so the exporter's fcurves_animation
    # samples a 0 value before the first on-transition; without it, constant
    # extrapolation makes the object visible from frame 0 regardless of when
    # the first visible range begins.
    if ranges:
      first_frameStart = max(0, min(FRAME_SCALE, _vis_arg_to_frame(ranges[0][0])))
      if first_frameStart > 0:
        _add_vis_keys(0, False)
    for (start, end) in ranges:
      frameStart = max(0, min(FRAME_SCALE, _vis_arg_to_frame(start)))
      frameOff = FRAME_SCALE + 1 if end > 1.0 else _vis_arg_to_frame(end)
      if frameOff <= frameStart:
        frameOff = frameStart + 1
      frameEndVisible = frameOff - 1

      _add_vis_keys(frameStart, True)

      if frameOff <= FRAME_SCALE:
        _add_vis_keys(frameEndVisible, True)
        _add_vis_keys(frameOff, False)
  return actions


def _prefers_euler_rotation_curves(node):
  try:
    # `model::ArgAnimationNode::animate` always composes quaternion
    # rotations directly into matrices. There is no ArgRotationNode-specific
    # Euler evaluation path, so import quaternion curves unchanged.
    if isinstance(node, ArgRotationNode):
      return False
  except Exception:
    pass
  return False

def _is_pos90_x_basis_matrix(mat, eps=1e-3):
  try:
    target = (
      (1.0, 0.0, 0.0),
      (0.0, 0.0, -1.0),
      (0.0, 1.0, 0.0),
    )
    for r in range(3):
      for c in range(3):
        if abs(float(mat[r][c]) - target[r][c]) > eps:
          return False
    return True
  except Exception:
    return False


def _transform_uses_quaternion_rotation(tfnode, obj=None):
  try:
    if obj is not None and _is_connector_object(obj):
      return False
  except Exception:
    pass
  try:
    action = getattr(getattr(obj, "animation_data", None), "action", None)
    if action is not None:
      if any(fc.data_path == "rotation_euler" for fc in action.fcurves):
        return False
      if any(fc.data_path == "rotation_quaternion" for fc in action.fcurves):
        return True
  except Exception:
    pass
  try:
    if isinstance(tfnode, ArgAnimationNode) and getattr(tfnode, "rotData", None):
      return True
  except Exception:
    pass
  return False

def _outer_argrot_prefers_raw_q1(node, base_mat=None):
  """Return True for the mirrored Su-27-style outer ArgRotationNode branch.

  These top-level outer wing-control empties already have the correct local
  q1 handedness on the negative side of the aircraft. Applying the generic
  v10 ROOT_BASIS_FIX to q1 over-rotates the left branch (Dummy670 family),
  while the positive side still needs the fix (Dummy671 family).
  """
  if type(node).__name__ != "ArgRotationNode":
    return False
  if _is_authored_argvis_control_pair(node):
    return False
  if not (
    _is_child_of_file_root(node)
    or _is_top_level_visibility_authored_pair(node)
  ):
    return False
  children = getattr(node, "children", None) or []
  vis = children[0] if children else None
  if vis is None or not isinstance(vis, ArgVisibilityNode):
    return False
  vis_children = getattr(vis, "children", None) or []
  inner = vis_children[0] if vis_children else None
  if not _is_nested_authored_argvis_control_pair(inner):
    return False
  return True


def _plain_root_unit_interval_rot_sets(node):
  if type(node).__name__ != "ArgRotationNode":
    return None
  if _is_authored_argvis_control_pair(node):
    return None
  if not _is_child_of_file_root(node):
    return None
  children = getattr(node, "children", None) or []
  vis = children[0] if children else None
  if vis is not None and isinstance(vis, ArgVisibilityNode):
    return None
  rot_sets = [keys for _arg, keys in (getattr(node, "rotData", None) or []) if keys]
  if not rot_sets or getattr(node, "posData", None) or getattr(node, "scaleData", None):
    return None
  frames = [float(getattr(k, "frame", 0.0)) for keys in rot_sets for k in keys]
  if not frames:
    return None
  if min(frames) < -1e-6 or max(frames) > 1.0 + 1e-6:
    return None
  if min(frames) < 0.0 - 1e-6:
    return None
  return rot_sets


def _plain_root_unit_interval_frame_mapper(node):
  """Use [0..1] -> [FRAME_SCALE/2..FRAME_SCALE] for plain root ArgRotation controls.

  Official exports author these controls on scene frames 100..200. Keeping them
  on that interval preserves the expected rest pose through frame 100 instead of
  starting the motion early at frame 0.
  """
  if not _import_profile_flag("plain_root_unit_interval_frame_remap"):
    return None
  rot_sets = _plain_root_unit_interval_rot_sets(node)
  if not rot_sets:
    return None
  return lambda value: int(round((FRAME_SCALE / 2.0) + float(value) * FRAME_SCALE / 2.0))


def _is_plain_root_unit_interval_argrot(node):
  return _plain_root_unit_interval_rot_sets(node) is not None


def _plain_root_unit_interval_first_key(node):
  rot_sets = _plain_root_unit_interval_rot_sets(node)
  if not rot_sets:
    return None
  for keys in rot_sets:
    if keys:
      q = keys[0].value
      return q if hasattr(q, 'to_matrix') else Quaternion(q)
  return None


def _plain_root_unit_interval_identity_first(node):
  q = _plain_root_unit_interval_first_key(node)
  if q is None:
    return False
  return _quat_is_identity(q)


def _plain_root_identity_q1_argrot(node):
  if type(node).__name__ != "ArgRotationNode":
    return False
  if _is_authored_argvis_control_pair(node):
    return False
  if not _is_child_of_file_root(node):
    return False
  children = getattr(node, "children", None) or []
  vis = children[0] if children else None
  if vis is not None and isinstance(vis, ArgVisibilityNode):
    return False
  if getattr(node, "posData", None) or getattr(node, "scaleData", None):
    return False
  if not getattr(node, "rotData", None):
    return False
  if not _quat_is_identity(getattr(node.base, "quat_1", None)):
    return False
  if not _quat_is_identity(getattr(node.base, "quat_2", None)):
    return False
  return _is_neg90_x_basis_matrix(Matrix(node.base.matrix))


def _root_direct_visibility_argrot(node):
  if not _import_profile_flag("root_direct_visibility_argrot"):
    return False
  # This override is only valid for plain-root style imports where the file
  # root basis is baked directly into top-level animated nodes. When we create
  # an explicit scene root basis object, preserving raw q1 here over-rotates
  # nodes like FA-18 Dummy72/Dummy73: matQuat @ q1 should cancel to identity
  # locally, but the raw-q1 override leaves an extra +90 X on the control.
  if getattr(_import_ctx, "use_scene_root_basis_object", True):
    return False
  if type(node).__name__ != "ArgRotationNode":
    return False
  if not (
    _is_child_of_file_root(node)
    or _is_top_level_visibility_authored_pair(node)
  ):
    return False
  if getattr(node, "posData", None) or getattr(node, "scaleData", None):
    return False
  children = list(getattr(node, "children", None) or [])
  if not children:
    return False
  if not all(isinstance(child, ArgVisibilityNode) for child in children):
    return False
  if not _is_neg90_x_basis_matrix(Matrix(node.base.matrix)):
    return False
  q1 = getattr(node.base, "quat_1", None)
  q2 = getattr(node.base, "quat_2", None)
  if q1 is None or q2 is None:
    return False
  q1v = q1 if hasattr(q1, "to_matrix") else Quaternion(q1)
  if not _is_pos90_x_basis_matrix(q1v.to_matrix().to_4x4()):
    return False
  if not _quat_is_identity(q2):
    return False
  return True


def _plain_root_z_halfturn_argrot(node):
  if not _import_profile_flag("plain_root_z_halfturn_argrot"):
    return False
  if type(node).__name__ != "ArgRotationNode":
    return False
  if not _is_child_of_file_root(node):
    return False
  if getattr(_import_ctx, "use_scene_root_basis_object", True):
    return False
  if not _is_neg90_x_basis_matrix(Matrix(node.base.matrix)):
    return False
  if getattr(node, "posData", None) or getattr(node, "scaleData", None):
    return False
  if not _is_plain_root_unit_interval_argrot(node):
    return False
  q1 = getattr(node.base, "quat_1", None)
  if q1 is None or not _quat_is_halfturn_z(q1):
    return False
  return True


def _plain_root_negscale_rest_y_halfturn_argrot(node):
  if not _import_profile_flag("plain_root_negscale_rest_y_halfturn_argrot"):
    return False
  if type(node).__name__ != "ArgRotationNode":
    return False
  if getattr(_import_ctx, "use_scene_root_basis_object", True):
    return False
  if not _is_child_of_file_root(node):
    return False
  if getattr(node, "posData", None) or getattr(node, "scaleData", None):
    return False
  if not getattr(node, "rotData", None):
    return False
  if not _all_negative_base_scale(node):
    return False
  if not _is_neg90_x_basis_matrix(Matrix(node.base.matrix)):
    return False
  q1 = getattr(node.base, "quat_1", None)
  if q1 is None:
    return False
  if _quat_is_identity(q1) or _quat_is_halfturn_x(q1):
    return False
  children = list(getattr(node, "children", None) or [])
  if not children:
    return False
  has_visibility_child = any(isinstance(child, ArgVisibilityNode) for child in children)
  has_transform_child = any(isinstance(child, TransformNode) for child in children)
  return has_visibility_child and has_transform_child


def _quat_to_matrix4(value):
  quat = value if hasattr(value, "to_matrix") else Quaternion(value)
  return quat.to_matrix().to_4x4()


def _scale_orientation_quaternion(value):
  if hasattr(value, "to_matrix"):
    return value
  comps = tuple(float(v) for v in value[:4])
  # Scale orientation keys are stored as (x, y, z, w).
  return Quaternion((comps[3], comps[0], comps[1], comps[2]))


def _compose_oriented_scale_matrix(scale_vec, orientation_quat):
  scale_mat = MatrixScale(Vector((scale_vec[0], scale_vec[1], scale_vec[2])))
  orient = orientation_quat if hasattr(orientation_quat, "to_matrix") else Quaternion(orientation_quat)
  if _quat_is_identity(orient):
    return scale_mat
  orient_mat = orient.to_matrix().to_4x4()
  return orient_mat @ scale_mat @ orient_mat.inverted()


def _matrix3_is_orthonormal(mat, eps=1e-4):
  try:
    cols = [Vector((float(mat[0][i]), float(mat[1][i]), float(mat[2][i]))) for i in range(3)]
    for col in cols:
      if abs(col.length - 1.0) > eps:
        return False
    for i in range(3):
      for j in range(i + 1, 3):
        if abs(cols[i].dot(cols[j])) > eps:
          return False
    return True
  except Exception:
    return False


def _needs_top_reflection_parent(local_mat, eps=1e-4):
  """Detect improper orthonormal top wrappers Blender cannot represent exactly.

  Some oriented-scale ArgRotation controls reduce to an orthonormal matrix with
  negative determinant. Blender stores object transforms as loc/rot/scale, so
  assigning that matrix directly can decompose into a slightly wrong non-uniform
  scale. When that happens, move the constant reflection into a parent helper
  and animate the child in a proper rotation basis.
  """
  try:
    rot3 = local_mat.to_3x3()
    if not _matrix3_is_orthonormal(rot3, eps):
      return False
    det = float(rot3.determinant())
    if det >= 0.0:
      return False
    _loc, _rot, scale = local_mat.decompose()
    mags = [abs(float(scale[i])) for i in range(3)]
    return any(abs(m - 1.0) > eps for m in mags)
  except Exception:
    return False


def _left_reflection_x_matrix():
  return Matrix((
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
  ))


def _action_chain_sort_value(action, default=-1):
  if action is None:
    return default
  try:
    return int(action.get("_iedm_chain_sort", default))
  except Exception:
    return default


def _sorted_transform_actions_for_execution(actions):
  transform_actions = list(actions or [])
  if len(transform_actions) <= 1:
    return transform_actions
  return sorted(
    transform_actions,
    key=lambda action: (_action_chain_sort_value(action, -1), _get_action_argument(action) or -1),
    reverse=True,
  )


def _needs_multi_arg_rotation_helper_split(node):
  tf = getattr(node, "transform", None)
  if tf is None or not isinstance(tf, ArgAnimationNode):
    return False
  rot_args = [arg for arg, keys in (getattr(tf, "rotData", None) or []) if keys]
  return len(rot_args) > 1


def _has_nonidentity_scale_orientation_keys(keys4):
  for key in list(keys4 or []):
    try:
      quat = _scale_orientation_quaternion(key.value)
    except Exception:
      continue
    if not _quat_is_identity(quat):
      return True
  return False


def _render_local_matrix_for_graph_node(node):
  render = getattr(node, "render", None)
  if render is None:
    return Matrix.Identity(4)
  try:
    if hasattr(render, "pos"):
      return Matrix.Translation(Vector(render.pos[:3]))
    if hasattr(render, "matrix"):
      return Matrix(render.matrix)
  except Exception as e:
    print(f"Warning in blender_importer\\import_pipeline.py: {e}")
  return Matrix.Identity(4)


def _scale_orientation_source_for_graph_node(node):
  if node is None:
    return None
  transforms = list(getattr(node, "_collapsed_transforms", None) or [])
  if not transforms:
    tf = getattr(node, "transform", None)
    if tf is not None:
      transforms = [tf]
  for tf in transforms:
    if not isinstance(tf, ArgAnimationNode):
      continue
    scale_sets = list(getattr(tf, "scaleData", None) or [])
    has_scale_keys = any((entry[1][0] or entry[1][1]) for entry in scale_sets if isinstance(entry, (list, tuple)) and len(entry) == 2)
    q2 = getattr(getattr(tf, "base", None), "quat_2", None)
    if has_scale_keys or (q2 is not None and not _quat_is_identity(q2)):
      return tf
  return None


def _copy_fcurve_points_local(src_curve, dst_curve):
  for kp in src_curve.keyframe_points:
    try:
      frame = float(kp.co[0])
      value = float(kp.co[1])
    except Exception:
      continue
    new_kp = dst_curve.keyframe_points.insert(frame, value, options={'FAST'})
    try:
      new_kp.interpolation = kp.interpolation
      new_kp.handle_left_type = kp.handle_left_type
      new_kp.handle_right_type = kp.handle_right_type
      new_kp.easing = kp.easing
    except Exception:
      pass


def _clone_action_filtered(action, name_suffix="", exclude_paths=None, include_paths=None):
  if action is None:
    return None
  exclude_paths = set(exclude_paths or [])
  include_paths = set(include_paths or [])
  cloned = bpy.data.actions.new("{}{}".format(action.name, name_suffix))
  arg = _get_action_argument(action)
  if arg is not None and hasattr(cloned, "argument"):
    cloned.argument = int(arg)
  copied = 0
  for src_curve in action.fcurves:
    if include_paths and src_curve.data_path not in include_paths:
      continue
    if src_curve.data_path in exclude_paths:
      continue
    dst_curve = cloned.fcurves.new(data_path=src_curve.data_path, index=src_curve.array_index)
    _copy_fcurve_points_local(src_curve, dst_curve)
    copied += 1
  if copied == 0:
    try:
      bpy.data.actions.remove(cloned)
    except Exception:
      pass
    return None
  return cloned


def _create_scale_orientation_rotation_action(name, keys4, frame_mapper=None, invert=False):
  if not keys4:
    return None
  action = bpy.data.actions.new(name)
  curves = []
  for idx in range(4):
    curves.append(action.fcurves.new(data_path="rotation_quaternion", index=idx))
  frame_mapper = frame_mapper or _anim_frame_to_scene_frame
  previous_quat = None
  for framedata in keys4:
    quat = _scale_orientation_quaternion(framedata.value)
    if invert:
      quat = quat.conjugated()
    if previous_quat is not None and previous_quat.dot(quat) < 0.0:
      quat = -quat
    previous_quat = quat.copy()
    frame = frame_mapper(framedata.frame)
    for curve, component in zip(curves, quat):
      curve.keyframe_points.add(1)
      curve.keyframe_points[-1].co = (frame, float(component))
      curve.keyframe_points[-1].interpolation = 'LINEAR'
  for curve in curves:
    try:
      curve.update()
    except Exception:
      pass
  return action


def _object_collection_targets(ob):
  cols = list(getattr(ob, "users_collection", None) or [])
  if cols:
    return cols
  return [bpy.context.collection]


def _insert_parent_wrapper_object(child, name, local_matrix=None, reset_child_local=False):
  parent = getattr(child, "parent", None)
  helper = bpy.data.objects.new(name, None)
  helper.empty_display_size = 0.1
  for collection in _object_collection_targets(child):
    try:
      collection.objects.link(helper)
    except RuntimeError:
      pass
  helper.parent = parent
  helper.matrix_parent_inverse = Matrix.Identity(4)
  helper.matrix_basis = local_matrix.copy() if hasattr(local_matrix, "copy") else (local_matrix or Matrix.Identity(4))
  child.parent = helper
  child.matrix_parent_inverse = Matrix.Identity(4)
  if reset_child_local:
    child.matrix_basis = Matrix.Identity(4)
  return helper


def _promote_oriented_scale_top_name(node, top_wrapper, leaf):
  """Keep the authored EDM control name on the object that carries its action.

  For control-only empties, the top oriented-scale wrapper represents the real
  EDM node transform. The identity leaf exists only to realize Blender's lack
  of per-object oriented scale, so it should not keep the authored node name.
  """
  if node is None or top_wrapper is None or leaf is None:
    return
  if getattr(leaf, "type", "") != "EMPTY":
    return
  if getattr(node, "render", None) is not None:
    return

  original_name = getattr(leaf, "name", "") or ""
  if not original_name:
    return

  try:
    leaf.name = "{}_iedm_leaf".format(original_name)
  except Exception:
    return

  try:
    top_wrapper.name = original_name
  except Exception:
    return

  try:
    top_wrapper["_iedm_oriented_scale_node_name"] = original_name
    leaf["_iedm_oriented_scale_leaf_of"] = original_name
  except Exception:
    pass


def _reconstruct_edm_local_matrix(node):
  """Reconstruct the exact local matrix that ArgAnimationNode::animate() builds.
  
  Per format spec for `model::ArgAnimationNode::animate`, the static local part
  is assembled as:
    local = base.matrix @ T(base.position) @ R(quat_1) @ oriented_scale(base.scale, quat_2)
  
  Where `quat_2` is the scale-orientation quaternion used to build
  `R(quat_2) @ diag(scale) @ R(quat_2)^-1`.
  """
  base_mat = Matrix(node.base.matrix)
  pos_mat = Matrix.Translation(Vector((node.base.position[0], node.base.position[1], node.base.position[2])))
  base_scale_vec = Vector((node.base.scale[0], node.base.scale[1], node.base.scale[2]))
  q1 = node.base.quat_1 if hasattr(node.base.quat_1, "to_matrix") else Quaternion(node.base.quat_1)
  q2 = node.base.quat_2 if hasattr(node.base.quat_2, "to_matrix") else Quaternion(node.base.quat_2)
  
  q1m = q1.to_matrix().to_4x4()
  oriented_scale = _compose_oriented_scale_matrix(base_scale_vec, q2)
  local_edm = base_mat @ pos_mat @ q1m @ oriented_scale
  
  return local_edm, base_mat


def _edm_matrix_to_blender_basis(mat_edm, apply_root_basis=False):
  """Apply EDM-to-Blender basis conversion exactly once at matrix boundary.
  
  This is the inverse of _ROOT_BASIS_FIX that the exporter applies at write time.
  """
  if apply_root_basis:
    return _ROOT_BASIS_FIX @ mat_edm
  return mat_edm


def _arganimation_prerotation_basis_local(node):
  """Return the constant affine prefix before `quat_1` is applied."""
  base_mat = Matrix(node.base.matrix)
  pos_mat = Matrix.Translation(Vector((node.base.position[0], node.base.position[1], node.base.position[2])))

  acts_like_top_level_control = (
    _is_child_of_file_root(node)
    or _is_top_level_visibility_authored_pair(node)
  )
  apply_root_basis = bool(
    _import_ctx.edm_version >= 10
    and acts_like_top_level_control
    and not getattr(_import_ctx, "use_scene_root_basis_object", True)
  )
  return _edm_matrix_to_blender_basis(base_mat @ pos_mat, apply_root_basis)


def _arganimation_rotation_basis_local(node):
  """Return the full static basis up to and including `quat_1`.

  This is the legacy single-object basis used when Blender can store the full
  affine+rotation top transform without drift.
  """
  return _arganimation_prerotation_basis_local(node) @ _arganimation_key_rotation_basis_local(node)


def _arganimation_key_rotation_basis_local(node):
  """Return the exact static `quat_1` rotation basis.

  For oriented-scale wrapper rebuilds the animated control object should keep
  only the authored rotation basis. Any affine prefix from `base.matrix` and
  `base.position` can be lifted into a constant parent helper when Blender
  cannot round-trip the combined affine+rotation matrix exactly.
  """
  q1 = node.base.quat_1 if hasattr(node.base.quat_1, "to_matrix") else Quaternion(node.base.quat_1)
  return q1.to_matrix().to_4x4()


def _matrix_transform_roundtrip_error(local_mat):
  try:
    loc, rot, scale = local_mat.decompose()
    rebuilt = Matrix.Translation(loc) @ rot.to_matrix().to_4x4() @ MatrixScale(scale)
    return max(
      abs(float(local_mat[r][c]) - float(rebuilt[r][c]))
      for r in range(4) for c in range(4)
    )
  except Exception:
    return float("inf")


def _needs_prerotation_affine_split(top_local, prerotation_local, rotation_local, eps=1e-4):
  """Detect top wrappers Blender cannot store exactly as one object transform.

  Some ArgRotation/ArgAnimation controls carry a representable affine prefix in
  `base.matrix @ T(base.position)` and a representable authored `quat_1`
  rotation, but their product becomes a sheared affine transform that drifts
  when Blender decomposes it into loc/rot/scale. Split those cases into:

    prefix_helper @ animated_rotation_object

  so the named animated object stays on an exact rotation basis.
  """
  top_err = _matrix_transform_roundtrip_error(top_local)
  if top_err <= eps:
    return False
  pre_err = _matrix_transform_roundtrip_error(prerotation_local)
  rot_err = _matrix_transform_roundtrip_error(rotation_local)
  return pre_err <= eps and rot_err <= eps


def _build_arganimation_action(node, arg, basis_local, frame_mapper=None, include_scale=True, action_name=None, rotation_basis_local=None):
  """Build a single action for one ArgAnimationNode argument on a chosen basis.

  This is used both for the normal import path and for oriented-scale wrapper
  reconstruction, where the animated transform must be re-derived against the
  wrapper's actual local basis instead of cloning curves from the original node.
  """
  posData = [x[1] for x in node.posData if x[0] == arg]
  rotData = [x[1] for x in node.rotData if x[0] == arg]
  scaleData = [x[1] for x in node.scaleData if x[0] == arg] if include_scale else []

  action = bpy.data.actions.new(action_name or "{}_{}".format(arg, _strip_anim_prefix(node.name or "anim")))
  if hasattr(action, "argument"):
    action.argument = arg

  rot_arg_order = [entry_arg for entry_arg, keys in (getattr(node, "rotData", None) or []) if keys]
  rot_arg_index = {entry_arg: idx for idx, entry_arg in enumerate(rot_arg_order)}
  if arg in rot_arg_index and len(rot_arg_order) > 1:
    action["_iedm_chain_sort"] = int(rot_arg_index[arg])
    action["_iedm_rotation_accum_args"] = int(len(rot_arg_order))

  static_loc, static_rot, _static_scale = basis_local.decompose()
  if rotation_basis_local is not None:
    try:
      _rot_loc, static_rot, _rot_scale = rotation_basis_local.decompose()
    except Exception:
      pass
  leftRot = static_rot
  rightRot = Quaternion((1, 0, 0, 0))
  leftPos = Matrix.Translation(static_loc) if posData else Matrix.Identity(4)
  rightPos = Matrix.Identity(4)
  base_scale_vec = Vector((node.base.scale[0], node.base.scale[1], node.base.scale[2]))
  key_quat_to_blender = lambda q: _anim_quaternion_to_blender(q)

  for pos in posData:
    add_position_fcurves(action, pos, leftPos, rightPos, node=node, frame_mapper=frame_mapper)
  for rot in rotData:
    add_rotation_fcurves(
      action,
      rot,
      leftRot,
      rightRot,
      quat_to_blender=key_quat_to_blender,
      frame_mapper=frame_mapper,
      use_euler=_prefers_euler_rotation_curves(node),
    )
  for sca_pair in scaleData:
    keys3 = sca_pair[1] if isinstance(sca_pair, tuple) and len(sca_pair) > 1 else []
    add_scale_fcurves(action, keys3, frame_mapper=frame_mapper, base_scale=base_scale_vec)

  _log_bone_debug_event(
    "anim-action",
    {
      "node_name": getattr(node, "name", "") or type(node).__name__,
      "node_type": type(node).__name__,
      "argument": int(arg) if isinstance(arg, int) else arg,
      "action_name": action.name,
      "has_pos": bool(posData),
      "has_rot": bool(rotData),
      "has_scale": bool(scaleData),
      "left_rotation": [round(float(v), 6) for v in leftRot],
      "right_rotation": [round(float(v), 6) for v in rightRot],
      "left_position": _matrix_trs_summary(leftPos),
      "right_position": _matrix_trs_summary(rightPos),
    },
    getattr(node, "name", "") or type(node).__name__,
    action.name,
  )
  _finalize_authored_transform_action(action)
  return action


def create_arganimation_actions(node):
  "Creates a set of actions to represent an ArgAnimationNode"
  actions = []
  _node_name = getattr(node, "name", "") or type(node).__name__

  # Matrix-first reconstruction: exactly mirror what DLL's animate() does
  local_edm, base_mat = _reconstruct_edm_local_matrix(node)
  
  # Determine if we need to apply root basis conversion
  _acts_like_top_level_control = (
    _is_child_of_file_root(node)
    or _is_top_level_visibility_authored_pair(node)
  )
  _top_level_root_basis_baked = (
    _import_ctx.edm_version >= 10
    and _acts_like_top_level_control
    and not getattr(_import_ctx, "use_scene_root_basis_object", True)
  )
  apply_root_basis = _top_level_root_basis_baked
  
  # Apply EDM-to-Blender basis conversion exactly once at the boundary
  local_bl = _edm_matrix_to_blender_basis(local_edm, apply_root_basis)
  
  # Decompose once we have the correct local matrix
  dcLoc, dcRot, dcScale = local_bl.decompose()
  base_scale_vec = Vector((node.base.scale[0], node.base.scale[1], node.base.scale[2]))
  q1_raw = node.base.quat_1 if hasattr(node.base.quat_1, "to_matrix") else Quaternion(node.base.quat_1)
  q2_raw = node.base.quat_2 if hasattr(node.base.quat_2, "to_matrix") else Quaternion(node.base.quat_2)
  aabS = _compose_oriented_scale_matrix(base_scale_vec, q2_raw)
  aabT = Matrix.Translation(_arg_anim_vector_to_blender(node, node.base.position))
  
  # For non-armature imports, ArgAnimatedBone carries an extra inv_base_bone_matrix
  _bone_ctx = _import_ctx.bone_import_ctx or {}
  _is_armature_bone_source = node in (_bone_ctx.get("bone_name_by_transform", {}) or {})
  
  # Save zero transform - use the already-computed local_bl
  zero_mat = local_bl
  
  bmat_inv = getattr(node, "bmat_inv", None)
  if bmat_inv is not None:
    try:
      zero_mat = zero_mat @ bmat_inv.inverted()
    except Exception as e:
      print(f"Warning in blender_importer\import_pipeline.py: {e}")

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
      "use_scene_root_basis_object": bool(getattr(_import_ctx, "use_scene_root_basis_object", True)),
    },
    _node_name,
  )
  for arg in node.get_all_args():
    frame_mapper = _plain_root_unit_interval_frame_mapper(node)
    actions.append(_build_arganimation_action(node, arg, local_bl, frame_mapper=frame_mapper, include_scale=True))
  return actions


def get_actions_for_node(node):
  """Accepts a node and gets or creates actions to apply their animations"""
  
  # Don't do this twice
  if hasattr(node, "actions") and node.actions:
    actions = node.actions
  else:
    actions = []
    if isinstance(node, ArgVisibilityNode):
      actions = create_visibility_actions(node)
    if isinstance(node, ArgAnimationNode):
      actions = create_arganimation_actions(node)
    # Save these actions on the node
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
  except Exception:
    pass
  try:
    while ad.nla_tracks:
      ad.nla_tracks.remove(ad.nla_tracks[0])
  except Exception:
    pass


def _action_has_visibility_curve(action):
  if action is None:
    return False
  try:
    return action.fcurves.find("VISIBLE") is not None
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
  if node is None or getattr(node, 'blender', None) is None:
    return []
  tf = getattr(node, 'transform', None)
  if tf is None:
    return []

  all_transforms = getattr(node, '_collapsed_transforms', [tf] if tf else [])
  anim_transforms = [t for t in all_transforms if isinstance(t, AnimatingNode)]
  if not anim_transforms:
    return []
  transform_anim_transforms = [t for t in anim_transforms if not isinstance(t, ArgVisibilityNode)]
  if not transform_anim_transforms:
    return []

  bone_anim_source_nodes = ctx.get('bone_anim_source_nodes', set())
  bone_anim_source_transforms = ctx.get('bone_anim_source_transforms', set())
  skip_object_anim = (
    node in bone_anim_source_nodes
    or any(t in bone_anim_source_transforms for t in transform_anim_transforms)
  )
  if skip_object_anim:
    return []

  if isinstance(tf, ArgVisibilityNode):
    actions = []
    for extra_tf in transform_anim_transforms:
      actions.extend(get_actions_for_node(extra_tf))
  else:
    actions = get_actions_for_node(tf) if not isinstance(tf, ArgVisibilityNode) else []
    for extra_tf in transform_anim_transforms:
      if extra_tf is not tf:
        actions.extend(get_actions_for_node(extra_tf))
  return _merge_actions_by_argument(actions)


def _visibility_source_for_graph_node(node):
  tf = getattr(node, 'transform', None)
  if isinstance(tf, ArgVisibilityNode):
    return tf
  parent = getattr(node, 'parent', None)
  parent_tf = getattr(parent, 'transform', None) if parent is not None else None
  if (
    parent is not None
    and isinstance(parent_tf, ArgVisibilityNode)
      and getattr(parent, 'blender', None) == getattr(node, 'blender', None)
  ):
    return parent_tf
  return None


def _rewrite_oriented_scale_controls(graph):
  bone_ctx = _import_ctx.bone_import_ctx or {}
  bone_anim_source_nodes = bone_ctx.get('bone_anim_source_nodes', set())
  bone_anim_source_transforms = bone_ctx.get('bone_anim_source_transforms', set())

  for node in getattr(graph, 'nodes', []) or []:
    ob = getattr(node, 'blender', None)
    if ob is None or getattr(ob, 'type', '') == 'ARMATURE':
      continue

    source_tf = _scale_orientation_source_for_graph_node(node)
    if source_tf is None:
      continue
    if node in bone_anim_source_nodes or source_tf in bone_anim_source_transforms:
      continue
    if getattr(source_tf, "_iedm_oriented_scale_wrapped", False):
      continue

    bmat_inv = getattr(source_tf, "bmat_inv", None)
    if bmat_inv is not None:
      continue

    ad = getattr(ob, "animation_data", None)
    active_action = getattr(ad, "action", None) if ad is not None else None
    if ad is not None and len(list(getattr(ad, "nla_tracks", []) or [])) > 0:
      continue

    nonempty_scale_sets = []
    for entry in list(getattr(source_tf, "scaleData", None) or []):
      if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        continue
      arg, pair = entry
      if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        continue
      keys4 = list(pair[0] or [])
      keys3 = list(pair[1] or [])
      if keys4 or keys3:
        nonempty_scale_sets.append((arg, keys4, keys3))

    if len(nonempty_scale_sets) > 1:
      continue

    scale_arg = None
    keys4 = []
    keys3 = []
    if nonempty_scale_sets:
      scale_arg, keys4, keys3 = nonempty_scale_sets[0]
    action_arg = _get_action_argument(active_action)
    if active_action is not None and scale_arg is not None and action_arg is not None and int(scale_arg) != int(action_arg):
      continue

    has_anim_scale = bool(keys3)
    has_anim_orient = _has_nonidentity_scale_orientation_keys(keys4)
    q2_raw = getattr(getattr(source_tf, "base", None), "quat_2", None)
    q2_raw = q2_raw if hasattr(q2_raw, "to_matrix") else Quaternion(q2_raw or (1.0, 0.0, 0.0, 0.0))
    base_scale_vec = Vector(getattr(getattr(source_tf, "base", None), "scale", (1.0, 1.0, 1.0))[:3])
    has_base_scale = any(abs(float(base_scale_vec[i]) - 1.0) > 1e-6 for i in range(3))
    has_base_orient = (not _quat_is_identity(q2_raw)) and has_base_scale

    needs_oriented_scale_rewrite = (has_anim_scale and has_anim_orient) or has_base_orient
    if not needs_oriented_scale_rewrite:
      continue

    zero_transform = getattr(source_tf, "zero_transform_local_matrix", None)
    if zero_transform is None:
      continue

    static_scale_mat = _compose_oriented_scale_matrix(base_scale_vec, q2_raw) if has_base_scale else Matrix.Identity(4)
    try:
      top_local = Matrix(zero_transform) @ static_scale_mat.inverted()
    except Exception:
      continue
    prerotation_parent_local = None
    top_wrapper_local = top_local
    prerotation_basis_local = _arganimation_prerotation_basis_local(source_tf)
    split_rotation_basis_local = _arganimation_key_rotation_basis_local(source_tf)
    rotation_basis_local = _arganimation_rotation_basis_local(source_tf)
    if _needs_prerotation_affine_split(top_local, prerotation_basis_local, split_rotation_basis_local):
      prerotation_parent_local = prerotation_basis_local
      top_wrapper_local = split_rotation_basis_local
      rotation_basis_local = split_rotation_basis_local

    frame_mapper = _plain_root_unit_interval_frame_mapper(source_tf)

    top_action = None
    if active_action is not None and action_arg is not None:
      try:
        top_action = _build_arganimation_action(
          source_tf,
          int(action_arg),
          top_wrapper_local,
          frame_mapper=frame_mapper,
          include_scale=False,
          action_name="{}_iedm_tf".format(active_action.name),
          rotation_basis_local=rotation_basis_local,
        )
      except Exception:
        top_action = None
    if top_action is None and active_action is not None:
      top_action = _clone_action_filtered(
        active_action,
        "_iedm_tf",
        exclude_paths={"scale", "VISIBLE", "hide_viewport"},
      )
    leaf_vis_action = _clone_action_filtered(
      active_action,
      "_iedm_vis",
      include_paths={"VISIBLE", "hide_viewport"},
    ) if active_action is not None else None
    render_local = _render_local_matrix_for_graph_node(node)

    top_wrapper = _insert_parent_wrapper_object(ob, "{}_iedm_tf".format(ob.name), top_wrapper_local, reset_child_local=False)
    top_wrapper["_iedm_oriented_scale_helper"] = True
    top_wrapper["_iedm_oriented_scale_top"] = True
    if prerotation_parent_local is not None:
      prerotation_parent = _insert_parent_wrapper_object(
        top_wrapper,
        "{}_iedm_tprefix".format(ob.name),
        prerotation_parent_local,
        reset_child_local=False,
      )
      prerotation_parent["_iedm_oriented_scale_helper"] = True
      prerotation_parent["_iedm_oriented_scale_top_prefix"] = True
    if top_action is not None:
      top_wrapper.animation_data_create()
      top_wrapper.animation_data.action = top_action
      top_wrapper.rotation_mode = "QUATERNION" if _transform_uses_quaternion_rotation(source_tf, top_wrapper) else "XYZ"

    child = ob
    if has_base_scale:
      if not _quat_is_identity(q2_raw):
        post_base = _insert_parent_wrapper_object(child, "{}_iedm_sbase_post".format(ob.name), q2_raw.conjugated().to_matrix().to_4x4(), reset_child_local=False)
        post_base.rotation_mode = 'QUATERNION'
        post_base["_iedm_oriented_scale_helper"] = True
        child = post_base
      base_scale = _insert_parent_wrapper_object(child, "{}_iedm_sbase".format(ob.name), MatrixScale(base_scale_vec), reset_child_local=False)
      base_scale["_iedm_oriented_scale_helper"] = True
      child = base_scale
      if not _quat_is_identity(q2_raw):
        pre_base = _insert_parent_wrapper_object(child, "{}_iedm_sbase_pre".format(ob.name), q2_raw.to_matrix().to_4x4(), reset_child_local=False)
        pre_base.rotation_mode = 'QUATERNION'
        pre_base["_iedm_oriented_scale_helper"] = True
        child = pre_base

    _sanim_arg_pfx = "{}_".format(int(scale_arg)) if scale_arg is not None else ""
    if has_anim_scale:
      if has_anim_orient:
        post_anim = _insert_parent_wrapper_object(child, "{}_iedm_sanim_post".format(ob.name), Matrix.Identity(4), reset_child_local=False)
        post_anim.rotation_mode = 'QUATERNION'
        post_anim["_iedm_oriented_scale_helper"] = True
        post_action = _create_scale_orientation_rotation_action("{}{}_iedm_sanim_post".format(_sanim_arg_pfx, ob.name), keys4, frame_mapper=frame_mapper, invert=True)
        if post_action is not None:
          if scale_arg is not None and hasattr(post_action, "argument"):
            post_action.argument = int(scale_arg)
          post_anim.animation_data_create()
          post_anim.animation_data.action = post_action
        child = post_anim

      anim_scale = _insert_parent_wrapper_object(child, "{}_iedm_sanim".format(ob.name), Matrix.Identity(4), reset_child_local=False)
      anim_scale["_iedm_oriented_scale_helper"] = True
      anim_scale_action = bpy.data.actions.new("{}{}_iedm_sanim".format(_sanim_arg_pfx, ob.name))
      if scale_arg is not None and hasattr(anim_scale_action, "argument"):
        anim_scale_action.argument = int(scale_arg)
      add_scale_fcurves(anim_scale_action, keys3, frame_mapper=frame_mapper, base_scale=None)
      if len(anim_scale_action.fcurves):
        anim_scale.animation_data_create()
        anim_scale.animation_data.action = anim_scale_action
      else:
        try:
          bpy.data.actions.remove(anim_scale_action)
        except Exception:
          pass
      child = anim_scale

      if has_anim_orient:
        pre_anim = _insert_parent_wrapper_object(child, "{}_iedm_sanim_pre".format(ob.name), Matrix.Identity(4), reset_child_local=False)
        pre_anim.rotation_mode = 'QUATERNION'
        pre_anim["_iedm_oriented_scale_helper"] = True
        pre_action = _create_scale_orientation_rotation_action("{}{}_iedm_sanim_pre".format(_sanim_arg_pfx, ob.name), keys4, frame_mapper=frame_mapper, invert=False)
        if pre_action is not None:
          if scale_arg is not None and hasattr(pre_action, "argument"):
            pre_action.argument = int(scale_arg)
          pre_anim.animation_data_create()
          pre_anim.animation_data.action = pre_action
        child = pre_anim

    _clear_object_animation_tracks(ob)
    if leaf_vis_action is not None:
      ob.animation_data_create()
      ob.animation_data.action = leaf_vis_action
    ob.matrix_basis = render_local
    ob["_iedm_oriented_scale_leaf"] = True
    _promote_oriented_scale_top_name(node, top_wrapper, ob)
    source_tf._iedm_oriented_scale_wrapped = True


def _split_multi_arg_visibility_controls(graph):
  ctx = _import_ctx.bone_import_ctx or {}
  scene_collection = bpy.context.collection

  for node in getattr(graph, 'nodes', []) or []:
    ob = getattr(node, 'blender', None)
    if ob is None or getattr(ob, 'type', '') in {'ARMATURE', 'MESH'}:
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
    ob.animation_data_create()
    ob.animation_data.action = vis_actions[0]

    parent_for_chain = ob
    created_helpers = []
    for action in vis_actions[1:]:
      helper = bpy.data.objects.new(ob.name, None)
      helper.empty_display_size = 0.1
      scene_collection.objects.link(helper)
      helper.parent = parent_for_chain
      helper.matrix_parent_inverse = Matrix.Identity(4)
      helper.matrix_basis = Matrix.Identity(4)
      helper.animation_data_create()
      helper.animation_data.action = action
      helper['_iedm_identity_passthrough'] = True
      helper['_iedm_narrow_identity_passthrough'] = True
      helper['_iedm_vis_passthrough'] = True
      created_helpers.append(helper)
      parent_for_chain = helper

    if created_helpers:
      target_parent = created_helpers[-1]
      for child in direct_children:
        if child in created_helpers:
          continue
        if child.parent == ob:
          _reparent_preserve_world(child, target_parent)


def _fix_owner_encoded_render_offsets(graph):
  """Post-pass: fix owner-encoded render chunks whose shared_parent differs from
  their graph parent. During process_node the shared_parent empty may not exist yet
  (sibling ordering), so the offset is deferred. By now all empties have had
  zero_transform_local_matrix applied to matrix_basis, so we can compute the correct
  relative offset without relying on a depsgraph update."""
  def _accum_world_basis(obj):
    """Walk up Blender parent chain multiplying matrix_basis (matrix_parent_inverse is
    always Identity in this importer, so matrix_local == matrix_basis)."""
    mats = []
    o = obj
    while o is not None:
      mats.append(o.matrix_basis.copy())
      o = o.parent
    result = Matrix.Identity(4)
    for m in reversed(mats):
      result = result @ m
    return result

  for node in graph.nodes:
    if node.render is None or node.blender is None:
      continue
    if getattr(node.blender, "type", "") != "MESH":
      continue
    if node.transform is not None and not isinstance(node.transform, ArgVisibilityNode):
      continue
    if getattr(node.render, "split_owner_encoded", False):
      continue
    shared_parent = getattr(node.render, "shared_parent", None)
    if shared_parent is None:
      continue
    parent_transform = getattr(getattr(node, "parent", None), "transform", None)
    if shared_parent is parent_transform:
      continue  # same-parent: no correction needed
    sp_obj = getattr(shared_parent, "_blender_obj", None)
    parent_obj = getattr(getattr(node, "parent", None), "blender", None)
    if sp_obj is None or parent_obj is None:
      continue
    try:
      relative = _accum_world_basis(parent_obj).inverted() @ _accum_world_basis(sp_obj)
      current_local = node.blender.matrix_basis.copy()
      node.blender.matrix_basis = relative @ current_local
      node.blender.pop("_iedm_identity_passthrough", None)
      node.blender.pop("_iedm_narrow_identity_passthrough", None)
    except Exception as e:
      print(f"Warning in _fix_owner_encoded_render_offsets: {e}")


def _apply_plain_root_visibility_basis_fix(graph):
  if not _import_profile_flag("plain_root_visibility_basis_fix"):
    return
  if _import_ctx.edm_version < 10:
    return
  if getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  for node in getattr(graph, "nodes", []) or []:
    tf = getattr(node, "transform", None)
    ob = getattr(node, "blender", None)
    tf_cls_name = type(tf).__name__ if tf is not None else ""
    collapsed_cls_names = [type(extra_tf).__name__ for extra_tf in (getattr(node, "_collapsed_transforms", []) or [])]
    if ob is None or tf_cls_name != "ArgVisibilityNode":
      continue
    if not _ob_local_is_identity(ob):
      continue
    if not getattr(getattr(node, "parent", None), "_is_graph_root", False):
      continue
    if any(cls_name != "ArgVisibilityNode" for cls_name in collapsed_cls_names):
      continue
    if type(getattr(node, "render", None)).__name__ == "SkinNode":
      continue

    child_nodes = list(getattr(node, "children", []) or [])

    # Skip ArgVis nodes whose direct graph children include any animated
    # (non-ArgVis) transform. When use_scene_root_basis_object is False,
    # AnimatingNode children are "child of file root" and have _ROOT_BASIS_FIX
    # already baked into their zero_transform_local_matrix via
    # _top_level_root_basis_baked. Applying it again here to the ArgVis parent
    # would double-apply the basis fix and place those children at wrong locations.
    has_anim_child_in_graph = any(
      isinstance(getattr(ch, "transform", None), AnimatingNode)
      and not isinstance(getattr(ch, "transform", None), ArgVisibilityNode)
      for ch in child_nodes
    )
    if has_anim_child_in_graph:
      continue

    descendant_nodes = []
    stack = list(child_nodes)
    while stack:
      cur = stack.pop()
      descendant_nodes.append(cur)
      stack.extend(list(getattr(cur, "children", []) or []))

    if any(type(getattr(child, "render", None)).__name__ == "SkinNode" for child in descendant_nodes):
      continue

    has_render_descendant = (
      type(getattr(node, "render", None)).__name__ == "RenderNode"
      or any(type(getattr(child, "render", None)).__name__ == "RenderNode" for child in descendant_nodes)
    )
    if not has_render_descendant:
      continue

    direct_child_objects = list(getattr(ob, "children", []) or [])
    has_nonidentity_child_empty = any(
      getattr(child, "type", "") == "EMPTY" and not _ob_local_is_identity(child)
      for child in direct_child_objects
    )
    if has_nonidentity_child_empty:
      continue

    try:
      ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
    except Exception as e:
      print(f"Warning in _apply_plain_root_visibility_basis_fix: {e}")


def _apply_plain_root_visibility_object_basis_fix():
  if not _import_profile_flag("plain_root_visibility_object_basis_fix"):
    return
  if _import_ctx.edm_version < 10:
    return
  if getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  def _basis_is_identity(obj, eps=1e-6):
    try:
      m = obj.matrix_basis
      return all(
        abs(float(m[r][c]) - (1.0 if r == c else 0.0)) <= eps
        for r in range(4) for c in range(4)
      )
    except Exception:
      return False

  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "parent", None) is not None:
      continue
    if getattr(ob, "type", "") != "EMPTY":
      continue
    if not bool(ob.get("_iedm_vis_passthrough")):
      continue
    if not _basis_is_identity(ob):
      continue
    stack = list(getattr(ob, "children", []) or [])
    while stack:
      child = stack.pop()
      child_type = getattr(child, "type", "")
      if child_type == "EMPTY":
        if _basis_is_identity(child):
          stack.extend(list(getattr(child, "children", []) or []))
        continue
      if child_type != "MESH":
        continue
      if str(child.get("_iedm_dbg_r_cls", "") or "") != "RenderNode":
        continue
      if not _basis_is_identity(child):
        continue
      try:
        child.matrix_basis = _ROOT_BASIS_FIX @ child.matrix_basis
      except Exception as e:
        print(f"Warning in _apply_plain_root_visibility_object_basis_fix: {e}")


def _apply_plain_root_visibility_mesh_basis_fix():
  if not _import_profile_flag("plain_root_visibility_mesh_basis_fix"):
    return
  if _import_ctx.edm_version < 10:
    return
  if getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  def _basis_is_identity(obj, eps=1e-6):
    try:
      m = obj.matrix_basis
      return all(
        abs(float(m[r][c]) - (1.0 if r == c else 0.0)) <= eps
        for r in range(4) for c in range(4)
      )
    except Exception:
      return False

  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "parent", None) is not None:
      continue
    if getattr(ob, "type", "") != "MESH":
      continue
    if str(ob.get("_iedm_dbg_r_cls", "") or "") != "RenderNode":
      continue
    if not ob.get("_iedm_src_vis_chain"):
      continue
    if not _basis_is_identity(ob):
      continue
    try:
      ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
    except Exception as e:
      print(f"Warning in _apply_plain_root_visibility_mesh_basis_fix: {e}")


def _apply_static_root_visibility_wrapper_basis_fix(graph):
  # When the profile applies the basis fix directly to mesh children
  # (plain_root_visibility_object_basis_fix), the EMPTY vis-wrapper parents
  # must NOT also receive the fix.  The mesh children already carry +90 X
  # individually, so having the parent carry it too would produce a double
  # rotation on every child mesh.  For those profiles (e.g. BLENDER_DEF)
  # this pass is skipped entirely; the mesh-level fix handles orientation.
  if _import_profile_flag("plain_root_visibility_object_basis_fix"):
    return
  # When a scene root basis object (_EDMFileRoot) is present, every Blender
  # object in the hierarchy already inherits _ROOT_BASIS_FIX from it.  Applying
  # the fix again to a static ArgVisibility wrapper would produce a double 90°X
  # rotation (180°X) on every mesh child of that wrapper.  Skip this pass
  # entirely; the inheritance chain carries the correct orientation.
  if getattr(_import_ctx, "use_scene_root_basis_object", True):
    return
  seen_objects = set()
  for node in getattr(graph, "nodes", []) or []:
    if not _is_static_root_visibility_wrapper(node):
      continue
    ob = getattr(node, "blender", None)
    if ob is None:
      continue
    ob_key = ob.name_full if hasattr(ob, "name_full") else id(ob)
    if ob_key in seen_objects:
      continue
    seen_objects.add(ob_key)
    try:
      ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
    except Exception as e:
      print(f"Warning in _apply_static_root_visibility_wrapper_basis_fix: {e}")


def _apply_root_visibility_pair_wrapper_basis_fix(graph):
  seen = set()
  for node in getattr(graph, "nodes", []) or []:
    if not _is_root_visibility_pair_child(node):
      continue
    parent = getattr(node, "parent", None)
    ob = getattr(parent, "blender", None)
    if ob is None:
      continue
    key = ob.name_full if hasattr(ob, "name_full") else id(ob)
    if key in seen:
      continue
    seen.add(key)
    try:
      ob.matrix_basis = _ROOT_BASIS_FIX.inverted() @ ob.matrix_basis
    except Exception as e:
      print(f"Warning in _apply_root_visibility_pair_wrapper_basis_fix: {e}")


def _apply_visibility_pair_wrapper_object_basis_fix():
  if _import_ctx.edm_version < 10:
    return
  if getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  def _is_render_mesh(obj):
    return getattr(obj, "type", "") == "MESH" and str(obj.get("_iedm_dbg_r_cls", "") or "") == "RenderNode"

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
      child for child in children
      if getattr(child, "type", "") == "EMPTY" and not _ob_local_is_identity(child)
    ]

    if mesh_children:
      continue
    if len(nonidentity_empty_children) != 1:
      continue

    try:
      ob.matrix_basis = _ROOT_BASIS_FIX.inverted() @ ob.matrix_basis
    except Exception as e:
      print(f"Warning in _apply_visibility_pair_wrapper_object_basis_fix: {e}")


def _restore_skin_visibility_transform_basis():
  """Restore same-name transform helper basis for root visibility skin branches."""
  if _import_ctx.edm_version < 10:
    return

  def _flat_to_matrix(value):
    try:
      if hasattr(value, "to_list"):
        value = value.to_list()
      elif not isinstance(value, (list, tuple)):
        value = list(value)
    except Exception:
      return None
    if not isinstance(value, (list, tuple)) or len(value) != 16:
      return None
    try:
      return Matrix((
        tuple(float(v) for v in value[0:4]),
        tuple(float(v) for v in value[4:8]),
        tuple(float(v) for v in value[8:12]),
        tuple(float(v) for v in value[12:16]),
      ))
    except Exception:
      return None

  def _matrix_close(a, b, eps=1e-5):
    try:
      return all(abs(float(a[r][c]) - float(b[r][c])) <= eps for r in range(4) for c in range(4))
    except Exception:
      return False

  for vis_ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(vis_ob, "type", "") != "EMPTY":
      continue
    if not bool(vis_ob.get("_iedm_vis_passthrough")):
      continue

    vis_name = str(getattr(vis_ob, "name", "") or "")
    if not vis_name:
      continue

    transform_children = []
    vis_skin_children = []
    for child in list(getattr(vis_ob, "children", []) or []):
      if getattr(child, "type", "") == "EMPTY":
        if (
          str(child.get("_iedm_dbg_tf_cls", "") or "") == "TransformNode"
          and str(child.get("_iedm_dbg_tf_name", "") or "") == vis_name
        ):
          transform_children.append(child)
      elif getattr(child, "type", "") == "MESH":
        if (
          bool(child.get("_iedm_skin_parent_override"))
          and str(child.get("_iedm_src_render_cls", "") or "") == "SkinNode"
          and str(child.get("_iedm_src_render_name", "") or "") == vis_name
        ):
          vis_skin_children.append(child)

    if len(transform_children) != 1:
      continue

    helper = transform_children[0]
    helper_skin_children = [
      child for child in list(getattr(helper, "children", []) or [])
      if (
        getattr(child, "type", "") == "MESH"
        and bool(child.get("_iedm_skin_parent_override"))
        and str(child.get("_iedm_src_render_cls", "") or "") == "SkinNode"
        and str(child.get("_iedm_src_render_name", "") or "") == vis_name
      )
    ]
    if not vis_skin_children and not helper_skin_children:
      continue

    authored_local = _flat_to_matrix(helper.get("IEDM_LOCAL_BL_MAT"))
    if authored_local is None:
      continue
    if _matrix_close(authored_local, Matrix.Identity(4)):
      continue
    if not _ob_local_is_identity(helper):
      continue
    if not _matrix_close(vis_ob.matrix_basis, authored_local):
      continue

    try:
      vis_ob.matrix_basis = Matrix.Identity(4)
      helper.matrix_basis = authored_local
    except Exception as e:
      print(f"Warning in _restore_skin_visibility_transform_basis: {e}")
      continue

    for mesh_ob in vis_skin_children:
      try:
        _reparent_preserve_world(mesh_ob, helper)
        mesh_ob["_iedm_dbg_skin_helper_name"] = helper.name
      except Exception as e:
        print(f"Warning in _restore_skin_visibility_transform_basis reparent: {e}")

    for mesh_ob in helper_skin_children:
      try:
        mesh_ob["_iedm_dbg_skin_helper_name"] = helper.name
      except Exception as e:
        print(f"Warning in _restore_skin_visibility_transform_basis tag: {e}")


def _apply_argvis_chain_basis_fix():
  """Strip erroneous ±90 X basis rotation from intermediate ArgVisibilityNode wrappers.

  3ds Max exported files sometimes create visibility wrapper chains where intermediate
  nodes incorrectly carry the coordinate system conversion rotation. This fix removes
  the rotation from visibility nodes that:
  1. Are ArgVisibilityNode with an ArgVisibilityNode parent
  2. The parent is a direct child of _EDMFileRoot (or a root-level wrapper)
  3. The node's matrix is essentially just a ±90 X basis rotation

  Also handles animated objects (ArgRotationNode children) under ArgVisibilityNode
  parents where the ArgVisibilityNode carries an unnecessary rotation.

  This fix ONLY applies to 3ds Max exported files, not Blender DEF files where
  those rotations may be intentional.
  """
  if _import_ctx.edm_version < 10:
    return
  if not _import_profile_flag("legacy_3dsmax_argvis_chain_basis_fix"):
    return

  edm_class = getattr(_import_ctx, "edm_exporter_class", "")
  if edm_class not in ("3DSMAX_BANO", "3DSMAX_DEF"):
    return

  def _is_basis_only_rotation(mat, eps=1e-4):
    """Check if matrix is essentially just a basis rotation (no translation/scale)."""
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
    """Get the name of the topmost parent."""
    root = obj
    while getattr(root, 'parent', None) is not None:
      root = root.parent
    return getattr(root, 'name', '')

  bpy.context.view_layer.update()

  case1_fixed = 0
  case2_fixed = 0
  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "type", "") != "EMPTY":
      continue

    parent = getattr(ob, 'parent', None)
    if parent is None:
      continue

    ob_type = str(ob.get("_iedm_dbg_tf_cls", "") or "")
    parent_type = str(parent.get("_iedm_dbg_tf_cls", "") or "")

    root_name = _get_root_name(ob)

    # Case 1: ArgVisibilityNode -> ArgVisibilityNode chain (existing behavior)
    if ob_type == "ArgVisibilityNode" and parent_type == "ArgVisibilityNode":
      if root_name != "_EDMFileRoot":
        continue

      mat = ob.matrix_basis
      if not _is_basis_only_rotation(mat):
        continue

      if _is_neg90_x_basis_matrix(mat):
        try:
          ob.matrix_basis = _ROOT_BASIS_FIX.inverted() @ ob.matrix_basis
          case1_fixed += 1
        except Exception as e:
          print(f"Warning in _apply_argvis_chain_basis_fix: {e}")
      elif _is_pos90_x_basis_matrix(mat):
        try:
          ob.matrix_basis = Matrix.Identity(4)
          case1_fixed += 1
        except Exception as e:
          print(f"Warning in _apply_argvis_chain_basis_fix (pos90): {e}")

    # Case 2: Animated object (ArgRotationNode/ArgAnimationNode child) under
    # ArgVisibilityNode parent. If the parent carries a basis-only rotation,
    # compensate by adjusting the child's matrix_basis.
    elif parent_type == "ArgVisibilityNode":
      if root_name != "_EDMFileRoot":
        continue

      parent_mat = parent.matrix_basis
      if not _is_basis_only_rotation(parent_mat):
        continue

      scale = ob.scale
      if scale.length < 0.1:
        continue

      try:
        if _is_neg90_x_basis_matrix(parent_mat):
          ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
          case2_fixed += 1
        elif _is_pos90_x_basis_matrix(parent_mat):
          ob.matrix_basis = _ROOT_BASIS_FIX.inverted() @ ob.matrix_basis
          case2_fixed += 1
      except Exception as e:
        print(f"Warning in _apply_argvis_chain_basis_fix (animated): {e}")

  print(f"[DEBUG] _apply_argvis_chain_basis_fix: case1_fixed={case1_fixed}, case2_fixed={case2_fixed}")


def _apply_collision_mesh_orientation_fix():
  """Post-pass: fix mesh objects in BLOB_COLLISION files at wrong world orientation.

  When the root basis object carries _ROOT_BASIS_FIX (90° X), all mesh descendants
  should inherit that rotation. Meshes at identity world X (0°) indicate the parent
  chain broke the rotation inheritance (e.g. through a collapsed wrapper or
  double-applied connector X-fix). Apply _ROOT_BASIS_FIX directly to these meshes.
  """
  if _import_ctx.edm_version < 10:
    return
  edm_class = getattr(_import_ctx, "edm_exporter_class", "")
  # Embedded shell payloads also appear in BLENDER_DEF scenes where authored
  # transforms already match the reference .blend. The collision repair pass is
  # only valid for collision-only families whose mesh orientation depends on the
  # synthetic scene root basis object.
  if edm_class not in {"BLOB_COLLISION", "COLLISION_MESH"}:
    return
  if getattr(_import_ctx, "collision_geometry_basis_fix", False):
    return
  if not _import_profile_flag("embedded_collision_scene_root_basis_object"):
    return
  if not getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  bpy.context.view_layer.update()

  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "type", "") != "MESH":
      continue
    # Skip meshes with active animation (their matrix is driven by fcurves)
    ad = getattr(ob, "animation_data", None)
    if ad is not None and (getattr(ad, "action", None) is not None or len(getattr(ad, "nla_tracks", []) or []) > 0):
      continue
    try:
      world_mat = ob.matrix_world.copy()
      _, world_rot, _ = world_mat.decompose()
      world_x_deg = abs(math.degrees(world_rot.to_euler("XYZ").x))
      # Check if the world X rotation is near 0° (identity) or near ±180° (flipped)
      # instead of the expected ~90° from _ROOT_BASIS_FIX inheritance.
      if world_x_deg < 5.0 or world_x_deg > 175.0:
        ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
    except Exception as e:
      print(f"Warning in _apply_collision_mesh_orientation_fix: {e}")


def _apply_3dsmax_mesh_orientation_fix():
  """Post-pass: fix mesh objects in 3DSMAX files at wrong world orientation.

  When the root basis object carries _ROOT_BASIS_FIX (90° X), all mesh descendants
  should inherit that rotation. Meshes at identity world X (0°) need _ROOT_BASIS_FIX
  applied.

  For animated meshes with only visibility fcurves, the mesh's parent (ArgVisibilityNode
  empty) may be at identity rotation. We need to apply _ROOT_BASIS_FIX to the parent,
  not the mesh.
  """
  print("[DEBUG] _apply_3dsmax_mesh_orientation_fix: starting")
  if _import_ctx.edm_version < 10:
    print("[DEBUG] _apply_3dsmax_mesh_orientation_fix: skipped (version < 10)")
    return
  if not _import_profile_flag("legacy_3dsmax_world_orientation_postfix"):
    print("[DEBUG] _apply_3dsmax_mesh_orientation_fix: skipped (profile disabled)")
    return
  edm_class = getattr(_import_ctx, "edm_exporter_class", "")
  print(f"[DEBUG] _apply_3dsmax_mesh_orientation_fix: edm_class={edm_class}")
  if edm_class not in ("3DSMAX_BANO", "3DSMAX_DEF"):
    print("[DEBUG] _apply_3dsmax_mesh_orientation_fix: skipped (not 3DSMAX)")
    return
  use_root_basis = getattr(_import_ctx, "use_scene_root_basis_object", True)
  print(f"[DEBUG] _apply_3dsmax_mesh_orientation_fix: use_scene_root_basis_object={use_root_basis}")
  if not use_root_basis:
    print("[DEBUG] _apply_3dsmax_mesh_orientation_fix: skipped (no root basis)")
    return

  bpy.context.view_layer.update()

  def _mat_to_euler_x_deg(mat):
    """Get X rotation in degrees from matrix."""
    try:
      _, rot, _ = mat.decompose()
      return abs(math.degrees(rot.to_euler("XYZ").x))
    except:
      return 0.0

  def _has_rotation_fcurves(ob):
    """Check if object has rotation fcurves that override matrix."""
    ad = getattr(ob, "animation_data", None)
    if ad is None:
      return False
    action = getattr(ad, "action", None)
    if action is None:
      return False
    for fc in action.fcurves:
      if fc.data_path in ("rotation_euler", "rotation_quaternion"):
        return True
    return False

  def _has_only_visibility_fcurves(ob):
    """Check if object has only visibility fcurves (VISIBLE, hide_viewport)."""
    ad = getattr(ob, "animation_data", None)
    if ad is None:
      return False
    action = getattr(ad, "action", None)
    if action is None:
      return False
    for fc in action.fcurves:
      if fc.data_path not in ("VISIBLE", "hide_viewport"):
        return False
    return len(action.fcurves) > 0

  def _get_root_name(obj):
    """Get the name of the topmost parent."""
    root = obj
    while getattr(root, 'parent', None) is not None:
      root = root.parent
    return getattr(root, 'name', '')

  def _is_argvis_empty(ob):
    """Check if object is an ArgVisibilityNode empty."""
    return str(ob.get("_iedm_dbg_tf_cls", "")) == "ArgVisibilityNode"

  fixed_count = 0
  skipped_rotation_fcurves = 0
  still_broken = 0
  parent_fixed = 0
  visibility_only_count = 0
  cleared_anim_count = 0
  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "type", "") != "MESH":
      continue
    
    try:
      world_mat = ob.matrix_world.copy()
      world_x_deg = _mat_to_euler_x_deg(world_mat)
      
      if world_x_deg >= 5.0:
        continue
      
      has_rot_fcurves = _has_rotation_fcurves(ob)
      
      if has_rot_fcurves:
        parent = getattr(ob, 'parent', None)
        root_name = _get_root_name(ob)
        
        if parent is not None and _is_argvis_empty(parent) and root_name == "_EDMFileRoot":
          if not _has_rotation_fcurves(parent):
            parent.matrix_basis = _ROOT_BASIS_FIX @ parent.matrix_basis
            bpy.context.view_layer.update()
            world_x_deg2 = _mat_to_euler_x_deg(ob.matrix_world)
            if world_x_deg2 > 5.0:
              parent_fixed += 1
              fixed_count += 1
              print(f"[DEBUG] mesh_fix (rot_fcurves parent): {ob.name} world_x={world_x_deg:.2f} -> {world_x_deg2:.2f}deg")
              continue
            else:
              parent.matrix_basis = _ROOT_BASIS_FIX.inverted() @ parent.matrix_basis
        
        skipped_rotation_fcurves += 1
        continue
      
      parent = getattr(ob, 'parent', None)
      root_name = _get_root_name(ob)
      has_vis = _has_only_visibility_fcurves(ob)
      
      if has_vis:
        visibility_only_count += 1
      
      if has_vis and parent is not None and _is_argvis_empty(parent) and root_name == "_EDMFileRoot":
        if not _has_rotation_fcurves(parent):
          parent.matrix_basis = _ROOT_BASIS_FIX @ parent.matrix_basis
          bpy.context.view_layer.update()
          world_x_deg2 = _mat_to_euler_x_deg(ob.matrix_world)
          if world_x_deg2 > 5.0:
            parent_fixed += 1
            fixed_count += 1
            continue
          else:
            parent.matrix_basis = _ROOT_BASIS_FIX.inverted() @ parent.matrix_basis
      
      if has_vis and not _has_rotation_fcurves(ob):
        ad = getattr(ob, "animation_data", None)
        if ad is not None:
          action = getattr(ad, "action", None)
          nla_tracks = list(getattr(ad, "nla_tracks", []) or [])
          if action is not None:
            ob.animation_data_clear()
            bpy.context.view_layer.update()
            new_world = parent.matrix_world @ _ROOT_BASIS_FIX if parent else _ROOT_BASIS_FIX
            ob.matrix_world = new_world
            bpy.context.view_layer.update()
            world_x_deg2 = _mat_to_euler_x_deg(ob.matrix_world)
            if world_x_deg2 > 5.0:
              cleared_anim_count += 1
              fixed_count += 1
              print(f"[DEBUG] mesh_fix (cleared_anim): {ob.name} world_x={world_x_deg:.2f} -> {world_x_deg2:.2f}deg")
            else:
              parent_world_x = _mat_to_euler_x_deg(parent.matrix_world) if parent else 0
              print(f"[DEBUG] mesh_fix: {ob.name} FAILED world_x={world_x_deg:.2f} -> after_fix={world_x_deg2:.2f}deg (parent_world={parent_world_x:.2f})")
              still_broken += 1
              ob.animation_data_create()
              ad = ob.animation_data
              ad.action = action
              for track in nla_tracks:
                new_track = ad.nla_tracks.new()
                for strip in track.strips:
                  new_track.strips.new(strip.name, strip.frame_start, strip.action)
            continue
      
      if not has_vis:
        ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
        bpy.context.view_layer.update()
        world_x_deg2 = _mat_to_euler_x_deg(ob.matrix_world)
        if world_x_deg2 > 5.0:
          fixed_count += 1
          print(f"[DEBUG] mesh_fix (direct): {ob.name} world_x={world_x_deg:.2f} -> {world_x_deg2:.2f}deg")
        else:
          still_broken += 1
          print(f"[DEBUG] mesh_fix (direct): {ob.name} FAILED world_x={world_x_deg:.2f} -> {world_x_deg2:.2f}deg")
      else:
        still_broken += 1
    except Exception as e:
      print(f"Warning in _apply_3dsmax_mesh_orientation_fix: {e}")
  
  print(f"[DEBUG] _apply_3dsmax_mesh_orientation_fix: fixed {fixed_count} ({parent_fixed} via parent, {cleared_anim_count} cleared anim), skipped {skipped_rotation_fcurves} rotation, {still_broken} still broken, {visibility_only_count} visibility-only")


def _apply_3dsmax_empty_orientation_fix():
  """Post-pass: fix empty objects in 3DSMAX files at wrong world orientation.

  Empty objects at identity world X rotation need _ROOT_BASIS_FIX applied.
  Only objects with rotation fcurves are skipped.
  """
  if _import_ctx.edm_version < 10:
    return
  if not _import_profile_flag("legacy_3dsmax_world_orientation_postfix"):
    return
  edm_class = getattr(_import_ctx, "edm_exporter_class", "")
  if edm_class not in ("3DSMAX_BANO", "3DSMAX_DEF"):
    return
  if not getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  bpy.context.view_layer.update()

  def _mat_to_euler_x_deg(mat):
    """Get X rotation in degrees from matrix."""
    try:
      _, rot, _ = mat.decompose()
      return abs(math.degrees(rot.to_euler("XYZ").x))
    except:
      return 0.0

  def _has_rotation_fcurves(ob):
    """Check if object has rotation fcurves that override matrix."""
    ad = getattr(ob, "animation_data", None)
    if ad is None:
      return False
    action = getattr(ad, "action", None)
    if action is None:
      return False
    for fc in action.fcurves:
      if fc.data_path in ("rotation_euler", "rotation_quaternion"):
        return True
    return False

  fixed_count = 0
  skipped_rotation_fcurves = 0
  still_broken = 0
  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "type", "") != "EMPTY":
      continue
    
    try:
      world_mat = ob.matrix_world.copy()
      world_x_deg = _mat_to_euler_x_deg(world_mat)
      
      if world_x_deg < 5.0:
        if _has_rotation_fcurves(ob):
          skipped_rotation_fcurves += 1
          continue
        
        ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
        bpy.context.view_layer.update()
        world_x_deg2 = _mat_to_euler_x_deg(ob.matrix_world)
        
        if world_x_deg2 > 5.0:
          print(f"[DEBUG] 3dsmax_empty_fix: {ob.name} world_x={world_x_deg:.2f}deg -> AFTER FIX world_x={world_x_deg2:.2f}deg")
          fixed_count += 1
        else:
          still_broken += 1
    except Exception as e:
      print(f"Warning in _apply_3dsmax_empty_orientation_fix: {e}")
  
  print(f"[DEBUG] _apply_3dsmax_empty_orientation_fix: fixed {fixed_count}, skipped {skipped_rotation_fcurves} rotation, {still_broken} still broken")

def _split_multi_arg_nonarmature_controls(graph):
  ctx = _import_ctx.bone_import_ctx or {}
  scene_collection = bpy.context.collection

  for node in getattr(graph, 'nodes', []) or []:
    ob = getattr(node, 'blender', None)
    if ob is None or getattr(ob, 'type', '') == 'ARMATURE':
      continue

    vis_source = _visibility_source_for_graph_node(node)
    vis_actions = get_actions_for_node(vis_source) if vis_source else []
    transform_actions = _sorted_transform_actions_for_execution(
      _collect_merged_transform_actions_for_graph_node(node, ctx)
    )
    if getattr(ob, 'type', '') == 'MESH' and not transform_actions:
      continue
    planned_actions = _build_nonarmature_action_plan(transform_actions, vis_actions)
    if len(planned_actions) <= 1:
      continue

    direct_children = [ch for ch in list(ob.children)]
    _clear_object_animation_tracks(ob)
    ob.animation_data_create()
    ob.animation_data.action = planned_actions[0]

    parent_for_chain = ob
    created_helpers = []
    for action in planned_actions[1:]:
      helper = bpy.data.objects.new(ob.name, None)
      helper.empty_display_size = 0.1
      scene_collection.objects.link(helper)
      helper.parent = parent_for_chain
      helper.matrix_parent_inverse = Matrix.Identity(4)
      helper.matrix_basis = Matrix.Identity(4)
      helper.animation_data_create()
      helper.animation_data.action = action
      helper['_iedm_identity_passthrough'] = True
      helper['_iedm_narrow_identity_passthrough'] = True
      if _action_has_visibility_curve(action):
        helper['_iedm_vis_passthrough'] = True
      created_helpers.append(helper)
      parent_for_chain = helper

    if created_helpers:
      target_parent = created_helpers[-1]
      for child in direct_children:
        if child in created_helpers:
          continue
        if child.parent == ob:
          _reparent_preserve_world(child, target_parent)


def _split_multi_arg_rotation_controls(graph):
  ctx = _import_ctx.bone_import_ctx or {}
  scene_collection = bpy.context.collection

  for node in getattr(graph, 'nodes', []) or []:
    ob = getattr(node, 'blender', None)
    if ob is None or getattr(ob, 'type', '') == 'ARMATURE':
      continue
    if not _needs_multi_arg_rotation_helper_split(node):
      continue

    vis_source = _visibility_source_for_graph_node(node)
    vis_actions = get_actions_for_node(vis_source) if vis_source else []
    transform_actions = _sorted_transform_actions_for_execution(
      _collect_merged_transform_actions_for_graph_node(node, ctx)
    )
    if len(transform_actions) <= 1:
      continue

    planned_actions = _build_nonarmature_action_plan(transform_actions, vis_actions)
    if len(planned_actions) <= 1:
      continue

    direct_children = [ch for ch in list(ob.children)]
    _clear_object_animation_tracks(ob)
    ob.animation_data_create()
    ob.animation_data.action = planned_actions[0]
    ob["_iedm_multi_arg_rotation_split"] = True

    parent_for_chain = ob
    created_helpers = []
    for action in planned_actions[1:]:
      helper = bpy.data.objects.new(ob.name, None)
      helper.empty_display_size = 0.1
      scene_collection.objects.link(helper)
      helper.parent = parent_for_chain
      helper.matrix_parent_inverse = Matrix.Identity(4)
      helper.matrix_basis = Matrix.Identity(4)
      helper.animation_data_create()
      helper.animation_data.action = action
      helper['_iedm_identity_passthrough'] = True
      helper['_iedm_narrow_identity_passthrough'] = True
      if _action_has_visibility_curve(action):
        helper['_iedm_vis_passthrough'] = True
      created_helpers.append(helper)
      parent_for_chain = helper

    if created_helpers:
      target_parent = created_helpers[-1]
      for child in direct_children:
        if child in created_helpers:
          continue
        if child.parent == ob:
          _reparent_preserve_world(child, target_parent)


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
  _get_action_argument,
  _debug_fmt_vec3,
  _debug_fmt_rot_deg,
)
from .object_create import create_connector, create_segments
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
  _quat_is_halfturn_x,
  _quat_is_halfturn_z,
  _all_negative_base_scale,
)
from .graph_postprocess import _reparent_preserve_world


def _rename_control_wrapper_mesh_pairs(graph):
  """Keep visible meshes on the semantic base name instead of `.001` helpers.

  Official BLENDER_DEF scenes frequently contain a control empty with the same
  semantic name as its sole mesh child. Blender then auto-suffixes the child
  mesh (`tf_0466.001`), which is noisy in the scene and makes those helper
  meshes look like the "wrong" object to select. Rename the control wrapper to
  its authored control prefix so the visible mesh can reclaim the base name.
  """
  prefix_by_cls = {
    "ArgVisibilityNode": "v_",
    "ArgRotationNode": "ar_",
    "ArgPositionNode": "al_",
    "ArgScaleNode": "as_",
  }

  for node in getattr(graph, "nodes", []) or []:
    tf = getattr(node, "transform", None)
    ob = getattr(node, "blender", None)
    if tf is None or ob is None:
      continue
    if getattr(node, "render", None) is not None:
      continue
    if getattr(ob, "type", "") != "EMPTY":
      continue

    prefix = prefix_by_cls.get(type(tf).__name__, "")
    if not prefix:
      continue

    children = [child for child in (getattr(node, "children", []) or []) if getattr(child, "blender", None) is not None]
    if len(children) != 1:
      continue

    child = children[0]
    child_render = getattr(child, "render", None)
    child_ob = getattr(child, "blender", None)
    if child_render is None or child_ob is None:
      continue
    if getattr(child, "transform", None) is not None:
      continue
    if getattr(child_ob, "type", "") != "MESH":
      continue

    desired_mesh_name = str(getattr(child_render, "name", "") or "")
    desired_mesh_name = _strip_anim_prefix(desired_mesh_name)
    if desired_mesh_name.startswith("Empty_"):
      desired_mesh_name = desired_mesh_name[len("Empty_"):]
    if not desired_mesh_name:
      desired_mesh_name = _SUFFIX_RE.sub("", getattr(child_ob, "name", "") or "")
    if not desired_mesh_name or desired_mesh_name.lower() == "root":
      continue

    current_parent_name = str(getattr(ob, "name", "") or "")
    current_parent_base = _SUFFIX_RE.sub("", _strip_anim_prefix(current_parent_name))
    current_mesh_name = str(getattr(child_ob, "name", "") or "")
    desired_parent_name = prefix + desired_mesh_name

    # Only rewrite when the current names are a Blender duplicate split of the
    # same semantic base (`tf_0466` + `tf_0466.001`) or when the child already
    # carries the semantic base but the parent still steals it.
    if current_parent_base != desired_mesh_name and current_mesh_name != desired_mesh_name:
      continue

    try:
      ob.name = desired_parent_name
      child_ob.name = desired_mesh_name
    except Exception as e:
      print(f"Warning in _rename_control_wrapper_mesh_pairs: {e}")


def apply_node_transform(node, obj, used_shared_parent=False):
  """Assigns the transform to a given node. node can be a TranslationNode or raw EDM node."""
  # Unpack TranslationNode if provided
  tnode = node if hasattr(node, "transform") else None
  tfnode = tnode.transform if tnode else node
  
  render = tnode.render if tnode else getattr(node, "render", None)
  if tfnode is None and render is None:
    return

  wants_quaternion_rotation = _transform_uses_quaternion_rotation(tfnode, obj) if tfnode is not None else False
  obj.rotation_mode = "QUATERNION" if wants_quaternion_rotation else "XYZ"

  def _debug_set_trace(source_label, local_mat):
    if not _import_ctx.transform_debug.get("enabled"):
      return
    needle = _import_ctx.transform_debug.get("filter")
    node_name = getattr(tfnode, "name", "") or type(tfnode).__name__
    if needle:
      n = needle.lower()
      if n not in node_name.lower() and n not in obj.name.lower():
        return
    loc, rot, scale = local_mat.decompose()
    print("[iEDM][TFSET] {} src={} obj={} loc={} rot_deg={} scale={}".format(
      node_name, source_label, obj.name, _debug_fmt_vec3(loc), _debug_fmt_rot_deg(rot), _debug_fmt_vec3(scale)
    ))

  def _set_local_matrix(local_mat):
    try:
      obj.matrix_basis = local_mat
      return
    except Exception as e:
      print(f"Warning in blender_importer\import_pipeline.py: {e}")
    loc, rot, scale = local_mat.decompose()
    obj.location = loc
    if wants_quaternion_rotation:
      obj.rotation_mode = "QUATERNION"
      obj.rotation_quaternion = rot
    else:
      obj.rotation_mode = "XYZ"
      obj.rotation_euler = _normalize_euler_xyz(rot.to_euler('XYZ'))
    obj.scale = scale

  def _compose_with_parent_if_enabled(local_mat):
    if _import_ctx.legacy_v10_parent_compose and obj.parent is not None and _import_ctx.edm_version >= 10:
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

  # 1. Base Transform from Node.transform
  final_local = Matrix(node._local_bl) if (tfnode is None and hasattr(node, "_local_bl") and getattr(node, "_local_bl", None) is not None) else Matrix.Identity(4)
  if tfnode is None and _is_connector_object(obj) and obj.parent is None:
    try:
      flat = obj.get("_iedm_connector_tf_matrix", None)
      if flat is not None and len(flat) == 16:
        connector_local = Matrix((
          tuple(float(v) for v in flat[0:4]),
          tuple(float(v) for v in flat[4:8]),
          tuple(float(v) for v in flat[8:12]),
          tuple(float(v) for v in flat[12:16]),
        ))
        loc = connector_local.to_translation()
        rot_mat = connector_local.to_3x3().to_4x4()
        if _is_pos90_x_basis_matrix(rot_mat):
          rot_mat = Matrix.Identity(4)
        elif _is_neg90_x_basis_matrix(rot_mat):
          rot_mat = Matrix.Identity(4)
        if (
          _import_profile_flag("plain_root_connector_basis_fix")
          and obj.parent is None
          and _import_ctx.edm_version >= 10
          and not getattr(_import_ctx, "use_scene_root_basis_object", True)
        ):
          loc = (_ROOT_BASIS_FIX @ Matrix.Translation(loc)).to_translation()
        final_local = Matrix.Translation(loc) @ rot_mat
    except Exception as e:
      print(f"Warning in blender_importer\import_pipeline.py: {e}")
  if isinstance(tfnode, TransformNode):
    is_connector_obj = _is_connector_transform(tfnode, obj)
    is_connector_wrapper = _is_plain_root_connector_wrapper(tnode, obj)
    is_connector_child = _is_plain_root_connector_child(tnode, obj)
    raw_local_mat = Matrix(tfnode.matrix)
    local_mat = Matrix(raw_local_mat)
    if is_connector_obj or is_connector_wrapper or is_connector_child:
      if (
        _import_profile_flag("plain_root_connector_basis_fix")
        and obj.parent is None
        and _import_ctx.edm_version >= 10
        and not getattr(_import_ctx, "use_scene_root_basis_object", True)
      ):
        local_mat = _ROOT_BASIS_FIX @ local_mat
      elif is_connector_child:
        local_mat = _ROOT_BASIS_FIX @ local_mat
      elif (
        is_connector_wrapper
        and obj.parent is not None
        and _import_ctx.edm_version >= 10
        and _is_neg90_x_basis_matrix(raw_local_mat.to_3x3().to_4x4())
      ):
        try:
          parent_rot = obj.parent.matrix_world.to_3x3().to_4x4()
        except Exception:
          parent_rot = Matrix.Identity(4)
        if _is_pos90_x_basis_matrix(parent_rot):
          local_mat = Matrix.Translation(raw_local_mat.to_translation())
      if is_connector_obj:
        local_mat = local_mat @ Matrix.Rotation(math.radians(-90.0), 4, "X")
      elif not is_connector_wrapper and not _is_neg90_x_basis_matrix(raw_local_mat.to_3x3().to_4x4()):
        local_mat = local_mat @ Matrix.Rotation(math.radians(-90.0), 4, "X")
    final_local = _compose_with_parent_if_enabled(local_mat)
  elif isinstance(tfnode, AnimatingNode):
    if hasattr(tfnode, "zero_transform_local_matrix"):
      final_local = _compose_with_parent_if_enabled(tfnode.zero_transform_local_matrix)
    elif hasattr(tfnode, "zero_transform_matrix"):
      final_local = _compose_with_parent_if_enabled(tfnode.zero_transform_matrix)
    elif hasattr(tfnode, "zero_transform"):
      loc, rot, scale = tfnode.zero_transform
      final_local = _compose_with_parent_if_enabled(Matrix.LocRotScale(loc, rot, scale))

    wrapper_offset = getattr(tfnode, "wrapper_rest_translation_matrix", None)
    if wrapper_offset is not None and tnode is not None:
      parent_graph = getattr(tnode, "parent", None)
      parent_obj = getattr(parent_graph, "blender", None) if parent_graph is not None else None
      if parent_obj is not None and not getattr(tfnode, "_wrapper_rest_translation_applied", False):
        try:
          parent_obj.matrix_basis = parent_obj.matrix_basis @ wrapper_offset
          tfnode._wrapper_rest_translation_applied = True
        except Exception as e:
          print(f"Warning in blender_importer\import_pipeline.py: {e}")

  # 2. Inherit RenderNode local offset if present
  # In EDM, RenderNodes can have their own pos/matrix separate from the transform node.
  if render and not _is_connector_object(obj):
    m_rn = Matrix.Identity(4)
    if hasattr(render, "pos"):
      m_rn = Matrix.Translation(Vector(render.pos[:3]))
    elif hasattr(render, "matrix"):
      m_rn = Matrix(render.matrix)
    
    if not m_rn.is_identity:
      final_local = final_local @ m_rn

  _set_local_matrix(final_local)
  _debug_set_trace(type(tfnode).__name__, final_local)


from .mesh_create import _create_mesh
from .prelude import _ROOT_BASIS_FIX, _import_ctx


def create_object(node):
  """Accepts an edm renderable and returns a blender object"""

  if not isinstance(node, (RenderNode, ShellNode, NumberNode, SkinNode)):
    print("WARNING: Do not understand creating types {} yet".format(type(node)))
    return

  if isinstance(node, (RenderNode, NumberNode, SkinNode)):
    vertexFormat = node.material.vertex_format
  elif isinstance(node, ShellNode):
    vertexFormat = node.vertex_format

  # Skip empty split children (owner-encoded splits with no triangles)
  if not node.indexData:
    return None

  def _is_morph_node(render_node):
    if render_node is None:
      return False
    if hasattr(render_node, "_morph_payload"):
      return True
    source_type = str(getattr(render_node, "_source_type_name", "") or "")
    return source_type == "model::MorphNode" or type(render_node).__name__ == "MorphNode"

  def _preserve_morph_payload(blender_obj, render_node):
    payload = getattr(render_node, "_morph_payload", None)
    if not payload or blender_obj is None:
      return
    try:
      blender_obj["_iedm_translation_status"] = "preserve_only"
      blender_obj["_iedm_translation_source"] = "MorphNode"
      blender_obj["_iedm_morph_payload_size"] = int(len(payload))
      blender_obj["_iedm_morph_payload_sha1"] = __import__("hashlib").sha1(payload).hexdigest()
      blender_obj["_iedm_morph_payload_preview_hex"] = bytes(payload[:64]).hex()
      blender_obj["_iedm_morph_payload_vertex_count"] = int(len(getattr(render_node, "vertexData", None) or []))
      if len(payload) <= 2048:
        blender_obj["_iedm_morph_payload_b64"] = __import__("base64").b64encode(payload).decode("ascii")
        blender_obj["_iedm_morph_payload_storage"] = "inline_b64"
      else:
        safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in (blender_obj.name or "Morph"))
        text_name = "IEDM_MorphPayload_{}".format(safe_name[:48])
        text_block = bpy.data.texts.get(text_name)
        if text_block is None:
          text_block = bpy.data.texts.new(text_name)
        text_block.clear()
        text_block.write(__import__("base64").b64encode(payload).decode("ascii"))
        blender_obj["_iedm_morph_payload_text"] = text_name
        blender_obj["_iedm_morph_payload_storage"] = "text_b64"
    except Exception as e:
      print(f"Warning preserving MorphNode payload on {getattr(blender_obj, 'name', '')}: {e}")

  def _decode_morph_payload_to_shape_keys(blender_obj, render_node, transform):
    payload = getattr(render_node, "_morph_payload", None)
    if not payload or blender_obj is None or getattr(blender_obj, "type", "") != "MESH":
      return 0
    vertex_count = int(len(getattr(render_node, "vertexData", None) or []))
    if vertex_count <= 0 or len(payload) % 12 != 0:
      return 0
    if len(getattr(blender_obj.data, "vertices", []) or []) != vertex_count:
      return 0

    vec3_count = len(payload) // 12
    if vec3_count < vertex_count or (vec3_count % vertex_count) != 0:
      return 0

    morph_count = vec3_count // vertex_count
    if morph_count <= 0 or morph_count > 8:
      return 0

    try:
      floats = __import__("struct").unpack("<{}f".format(len(payload) // 4), payload)
    except Exception:
      return 0

    if any(not math.isfinite(v) for v in floats):
      return 0

    basis_positions = [vert.co.copy() for vert in blender_obj.data.vertices]
    if len(basis_positions) != vertex_count:
      return 0

    bbox_extent = 0.0
    for pos in basis_positions:
      bbox_extent = max(bbox_extent, abs(float(pos.x)), abs(float(pos.y)), abs(float(pos.z)))
    delta_limit = max(10.0, bbox_extent * 10.0, 1.0)

    delta_transform = None
    if transform is not None:
      try:
        delta_transform = Matrix(transform).to_3x3()
      except Exception:
        delta_transform = None

    decoded = 0
    if getattr(blender_obj.data, "shape_keys", None) is None:
      try:
        blender_obj.shape_key_add(name="Basis", from_mix=False)
      except Exception:
        return 0

    for morph_index in range(morph_count):
      start = morph_index * vertex_count * 3
      max_abs_delta = 0.0
      deltas = []
      for vertex_index in range(vertex_count):
        offset = start + (vertex_index * 3)
        delta = Vector((floats[offset], floats[offset + 1], floats[offset + 2]))
        if delta_transform is not None:
          try:
            delta = delta_transform @ delta
          except Exception:
            pass
        max_abs_delta = max(max_abs_delta, abs(float(delta.x)), abs(float(delta.y)), abs(float(delta.z)))
        deltas.append(delta)

      if max_abs_delta <= 1.0e-8 or max_abs_delta > delta_limit:
        continue

      try:
        key_block = blender_obj.shape_key_add(name="Morph_{:03d}".format(morph_index), from_mix=False)
      except Exception:
        continue
      for vertex_index, delta in enumerate(deltas):
        key_block.data[vertex_index].co = basis_positions[vertex_index] + delta
      decoded += 1

    if decoded:
      try:
        blender_obj["_iedm_translation_status"] = "approximate"
        blender_obj["_iedm_translation_source"] = "MorphNode"
        blender_obj["_iedm_morph_decode_mode"] = "raw_vec3_per_vertex"
        blender_obj["_iedm_morph_shape_key_count"] = int(decoded)
      except Exception:
        pass
    return decoded

  is_morph_node = _is_morph_node(node)

  # Create the mesh. Skinned meshes must NOT use compaction as their vertex
  # weights rely on the original EDM vertex indices. Morph payload decode also
  # needs the source EDM vertex ordering preserved.
  compact = not isinstance(node, SkinNode) and not is_morph_node
  
  mesh_transform = None
  if isinstance(node, SkinNode) and _import_ctx.edm_version >= 10:
    edm_class = getattr(_import_ctx, "edm_exporter_class", "")
    if edm_class in ("3DSMAX_BANO", "3DSMAX_DEF"):
      mesh_transform = _ROOT_BASIS_FIX
  elif isinstance(node, ShellNode) and getattr(_import_ctx, "collision_geometry_basis_fix", False):
    mesh_transform = _ROOT_BASIS_FIX
  
  mesh = _create_mesh(
    node.vertexData,
    node.indexData,
    vertexFormat,
    compact=compact,
    join_triangles=isinstance(node, ShellNode),
    transform=mesh_transform,
  )
  mesh.name = node.name

  # Owner-channel extraction needs index remap from source vertexData to mesh points.
  # Keep import stable for now; recover owner metadata in a dedicated mesh-mapping pass.

  # Create and link the object
  ob = bpy.data.objects.new(node.name, mesh)
  ob.edm.is_collision_shell = isinstance(node, ShellNode)
  ob.edm.is_renderable      = isinstance(node, (RenderNode, NumberNode, SkinNode))
  ob.edm.damage_argument = -1 if not isinstance(node, (RenderNode, NumberNode)) else node.damage_argument
  mat = getattr(node, "material", None)
  try:
    source_type = getattr(node, "_source_type_name", None)
    if source_type:
      ob["_iedm_source_node_type"] = str(source_type)
    layout_variant = getattr(node, "_layout_variant", None)
    if layout_variant:
      ob["_iedm_layout_variant"] = str(layout_variant)
    shell_family = getattr(node, "_shell_family", None)
    if shell_family:
      ob["_iedm_shell_family"] = str(shell_family)
  except Exception as e:
    print(f"Warning in blender_importer\\import_pipeline.py: {e}")
  if isinstance(node, ShellNode):
    _set_official_special_type(ob, 'COLLISION_SHELL')
    # Match original collision-shell viewport style from official scenes.
    ob.display_type = 'BOUNDS'
  elif isinstance(node, NumberNode):
    _set_official_special_type(ob, 'NUMBER_TYPE')
    payload = dict(getattr(node, "number_payload", None) or {})
    uv_params = payload.get("uv_params")
    raw = getattr(node, "number_params_raw", None)
    if uv_params and hasattr(ob, "EDMProps"):
      ob.EDMProps.NUMBER_UV_X_ARG = int(uv_params.get("x_arg", -1))
      ob.EDMProps.NUMBER_UV_X_SCALE = float(uv_params.get("x_scale", 0.0))
      ob.EDMProps.NUMBER_UV_Y_ARG = int(uv_params.get("y_arg", -1))
      ob.EDMProps.NUMBER_UV_Y_SCALE = float(uv_params.get("y_scale", 0.0))
    elif raw and hasattr(ob, "EDMProps"):
      # Legacy fallback for files parsed before structured number payloads
      # were attached.
      ob.EDMProps.NUMBER_UV_X_ARG = int(raw[1])
      ob.EDMProps.NUMBER_UV_X_SCALE = float(raw[2])
      ob.EDMProps.NUMBER_UV_Y_ARG = int(raw[3])
      ob.EDMProps.NUMBER_UV_Y_SCALE = float(raw[4])

    # Preserve the richer NumberNode payload on the Blender object so the
    # simplified EDMProps projection is not the only remaining representation.
    try:
      ob["IEDM_NUMBER_VALUE"] = float(getattr(node, "value", 0.0))
      if raw:
        ob["IEDM_NUMBER_PARAMS_RAW"] = [
          int(raw[0]), int(raw[1]), float(raw[2]), int(raw[3]), float(raw[4])
        ]
      if payload:
        payload_idprops = {}
        for key in ("unknown_start", "material", "parent", "damage_argument", "index_prefix"):
          if key in payload and payload[key] is not None:
            payload_idprops[key] = int(payload[key])
        uv_payload = payload.get("uv_params") or {}
        if uv_payload:
          payload_idprops["uv_params"] = {
            "unknown": int(uv_payload.get("unknown", 0)),
            "x_arg": int(uv_payload.get("x_arg", -1)),
            "x_scale": float(uv_payload.get("x_scale", 0.0)),
            "y_arg": int(uv_payload.get("y_arg", -1)),
            "y_scale": float(uv_payload.get("y_scale", 0.0)),
          }
        if payload_idprops:
          ob["IEDM_NUMBER_PAYLOAD"] = payload_idprops
    except Exception as e:
      print(f"Warning preserving NumberNode payload on {ob.name}: {e}")
  else:
    material_name = str(getattr(mat, "material_name", "") or "").lower()
    if material_name == "water_mask_material":
      _set_official_special_type(ob, 'WATER_MASK')
    else:
      _set_official_special_type(ob, 'UNKNOWN_TYPE')

  # Map damage_argument → EDMProps.DAMAGE_ARG for official exporter.
  if hasattr(ob, "EDMProps") and isinstance(node, (RenderNode, NumberNode)):
    dmg = getattr(node, "damage_argument", -1)
    if dmg is not None and dmg >= 0:
      ob.EDMProps.DAMAGE_ARG = int(dmg)
    try:
      ob.EDMProps.TWO_SIDED = bool(int(getattr(mat, "culling", 0)) != 0)
    except Exception:
      pass

  # Map animated uniforms from EDM material → EDMProps ARG fields.
  _map_animated_uniforms_to_edmprops(ob, node)

  bpy.context.collection.objects.link(ob)

  if is_morph_node:
    _preserve_morph_payload(ob, node)
    _decode_morph_payload_to_shape_keys(ob, node, mesh_transform)

  return ob
