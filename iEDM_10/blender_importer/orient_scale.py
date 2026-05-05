# Fragment: oriented scale decomposition helpers and the rewrite pass.
# All names resolved via the shared namespace injected by reader.py.


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
    _log.warn("render local matrix read for '{}': {}".format(
      getattr(render, "name", type(render).__name__), e), exc=e)
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
  EDM node transform. The identity leaf exists only to realise Blender's lack of
  per-object oriented scale, so it should not keep the authored node name.
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
  """Return (local_edm, rot_scale_edm, base_mat, pos_vec) for an ArgAnimationNode.

  local_edm  = base_mat @ T(pos) @ q1m @ oriented_scale  (DLL formula verbatim)
  rot_scale_edm = base_mat @ q1m @ oriented_scale  (no translation)
  pos_vec       = Vector(base.position)              (Z-up passthrough)

  The DLL reads default_position and keyframe Vec3d values with the same readVec3d
  call — same coordinate space. Since vector_to_blender / _anim_vector_to_blender
  are both passthroughs for v10, base.position is already in Blender/Z-up space
  and must NOT receive _ROOT_BASIS_FIX.
  """
  base_mat = Matrix(node.base.matrix)
  pos_vec = Vector((node.base.position[0], node.base.position[1], node.base.position[2]))
  base_scale_vec = Vector((node.base.scale[0], node.base.scale[1], node.base.scale[2]))
  q1 = node.base.quat_1 if hasattr(node.base.quat_1, "to_matrix") else Quaternion(node.base.quat_1)
  q2 = node.base.quat_2 if hasattr(node.base.quat_2, "to_matrix") else Quaternion(node.base.quat_2)

  q1m = q1.to_matrix().to_4x4()
  oriented_scale = _compose_oriented_scale_matrix(base_scale_vec, q2)
  rot_scale_edm = base_mat @ q1m @ oriented_scale
  local_edm = base_mat @ Matrix.Translation(pos_vec) @ q1m @ oriented_scale

  return local_edm, rot_scale_edm, base_mat, pos_vec


def _arganimation_prerotation_basis_local(node):
  """Return the constant affine prefix before `quat_1` is applied."""
  base_mat = Matrix(node.base.matrix)
  pos_vec = Vector((node.base.position[0], node.base.position[1], node.base.position[2]))

  acts_like_top_level_control = (
    _is_child_of_file_root(node)
    or _is_top_level_visibility_authored_pair(node)
  )
  apply_root_basis = bool(
    _import_ctx.edm_version >= 10
    and acts_like_top_level_control
    and not getattr(_import_ctx, "use_scene_root_basis_object", True)
  )
  if apply_root_basis:
    return Matrix.Translation(pos_vec) @ (_ROOT_BASIS_FIX @ base_mat)
  return base_mat @ Matrix.Translation(pos_vec)


def _arganimation_rotation_basis_local(node):
  """Return the full static basis up to and including `quat_1`.

  Legacy single-object basis used when Blender can store the full affine+rotation
  top transform without drift.
  """
  return _arganimation_prerotation_basis_local(node) @ _arganimation_key_rotation_basis_local(node)


def _arganimation_key_rotation_basis_local(node):
  """Return the exact static `quat_1` rotation basis.

  For oriented-scale wrapper rebuilds the animated control object should keep only
  the authored rotation basis. Any affine prefix from `base.matrix` and
  `base.position` can be lifted into a constant parent helper when Blender cannot
  round-trip the combined affine+rotation matrix exactly.
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
  `base.matrix @ T(base.position)` and a representable authored `quat_1` rotation,
  but their product becomes a sheared affine transform that drifts when Blender
  decomposes it. Split those into:

    prefix_helper @ animated_rotation_object
  """
  top_err = _matrix_transform_roundtrip_error(top_local)
  if top_err <= eps:
    return False
  pre_err = _matrix_transform_roundtrip_error(prerotation_local)
  rot_err = _matrix_transform_roundtrip_error(rotation_local)
  return pre_err <= eps and rot_err <= eps


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
