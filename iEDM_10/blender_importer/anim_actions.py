# Fragment: animation action builders for visibility and ArgAnimation nodes.
# All names resolved via the shared namespace injected by reader.py.


def create_visibility_actions(visNode):
  """Creates visibility actions from an ArgVisibilityNode"""
  def _vis_arg_to_frame(value):
    return int(round((value + 1.0) * FRAME_SCALE / 2.0))

  actions = []
  for (arg, ranges) in visNode.visData:
    vis_name = visNode.name or "node"
    if vis_name.startswith("v_"):
      vis_name = vis_name[2:]
    vis_name = _strip_anim_prefix(vis_name)
    action = bpy.data.actions.new("{}_{}_Visib".format(arg, vis_name))
    actions.append(action)
    if hasattr(action, "argument"):
      action.argument = arg
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

    # Visibility ranges in EDM are [start, end) where end is the first OFF frame
    # boundary. Emit an initial off-key at frame 0 when the first range starts
    # after frame 0 so constant extrapolation doesn't bleed visible from the start.
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
    if isinstance(node, ArgRotationNode):
      return False
  except Exception as e:
    _log.debug("_prefers_euler_rotation_curves isinstance check: {}".format(e), level=2)
  return False


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
  """Map [0..1] -> [FRAME_SCALE/2..FRAME_SCALE] for plain root ArgRotation controls.

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


def _build_arganimation_action(node, arg, basis_local, frame_mapper=None, include_scale=True, action_name=None, rotation_basis_local=None):
  """Build a single action for one ArgAnimationNode argument on a chosen basis.

  Used both for the normal import path and for oriented-scale wrapper reconstruction,
  where the animated transform must be re-derived against the wrapper's actual local
  basis instead of cloning curves from the original node.
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
  pos_rot = static_rot.copy()
  if rotation_basis_local is not None:
    try:
      _rot_loc, static_rot, _rot_scale = rotation_basis_local.decompose()
    except Exception:
      pass
  leftRot = static_rot
  rightRot = Quaternion((1, 0, 0, 0))
  leftPos = (
    Matrix.Translation(static_loc) @ pos_rot.to_matrix().to_4x4()
    if posData else Matrix.Identity(4)
  )
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

  local_edm, rot_scale_edm, base_mat, pos_vec = _reconstruct_edm_local_matrix(node)

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

  if apply_root_basis:
    # pos_vec is already Z-up (passthrough) — apply _ROOT_BASIS_FIX only to the
    # rotation/scale block so translation is not double-converted.
    rot_scale_bl = _ROOT_BASIS_FIX @ rot_scale_edm
    local_bl = Matrix.Translation(pos_vec) @ rot_scale_bl
  else:
    local_bl = local_edm

  dcLoc, dcRot, dcScale = local_bl.decompose()
  base_scale_vec = Vector((node.base.scale[0], node.base.scale[1], node.base.scale[2]))
  q1_raw = node.base.quat_1 if hasattr(node.base.quat_1, "to_matrix") else Quaternion(node.base.quat_1)
  q2_raw = node.base.quat_2 if hasattr(node.base.quat_2, "to_matrix") else Quaternion(node.base.quat_2)
  aabS = _compose_oriented_scale_matrix(base_scale_vec, q2_raw)
  aabT = Matrix.Translation(_arg_anim_vector_to_blender(node, node.base.position))

  _bone_ctx = _import_ctx.bone_import_ctx or {}
  _is_armature_bone_source = node in (_bone_ctx.get("bone_name_by_transform", {}) or {})

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
