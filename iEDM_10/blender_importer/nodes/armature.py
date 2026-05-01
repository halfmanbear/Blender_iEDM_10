import math

def _channel_slices_from_vertex_format(vertex_format):
  """Return channel->(start,end) slices based on vertex format packed layout."""
  offsets = {}
  if not vertex_format or not hasattr(vertex_format, "data"):
    return offsets
  cursor = 0
  for i, count in enumerate(vertex_format.data):
    n = int(count)
    if n <= 0:
      continue
    offsets[i] = (cursor, cursor + n)
    cursor += n
  return offsets


def _copy_fcurve_to_action(src_curve, dst_action, dst_path, action_group):
  dst_curve = dst_action.fcurves.new(
    data_path=dst_path,
    index=src_curve.array_index,
    action_group=action_group,
  )
  dst_curve.extrapolation = src_curve.extrapolation
  for key in src_curve.keyframe_points:
    new_key = dst_curve.keyframe_points.insert(key.co[0], key.co[1], options={"FAST"})
    new_key.interpolation = key.interpolation
    try:
      new_key.handle_left_type = key.handle_left_type
      new_key.handle_right_type = key.handle_right_type
      new_key.handle_left = key.handle_left
      new_key.handle_right = key.handle_right
    except Exception as e:
      print(f"Warning in blender_importer\nodes\armature.py: {e}")
  dst_curve.update()


def _action_has_fcurve(action, data_path, index=None):
  if action is None:
    return False
  for fcu in action.fcurves:
    if fcu.data_path != data_path:
      continue
    if index is None or fcu.array_index == index:
      return True
  return False


def _merge_visibility_action_into_transform_action(transform_action, vis_action, obj_name):
  """Clone a transform action and append VISIBLE fcurves from a visibility action."""
  if transform_action is None or vis_action is None:
    return transform_action
  if _action_has_fcurve(transform_action, "VISIBLE"):
    return transform_action

  vis_curves = [fcu for fcu in vis_action.fcurves if fcu.data_path == "VISIBLE"]
  if not vis_curves:
    return transform_action

  # Official exporter extracts the EDM argument from the action name prefix.
  # Preserve the visibility action's name prefix (e.g. "10_*") so VISIBLE
  # wrappers are emitted for merged visibility+transform actions.
  merged_name = (getattr(vis_action, "name", "") or "").strip() or transform_action.name
  merged = bpy.data.actions.new(merged_name)
  for src_curve in transform_action.fcurves:
    group_name = src_curve.group.name if getattr(src_curve, "group", None) else "Transform"
    _copy_fcurve_to_action(src_curve, merged, src_curve.data_path, group_name)
  for src_curve in vis_curves:
    if merged.fcurves.find("VISIBLE", index=src_curve.array_index) is not None:
      continue
    group_name = src_curve.group.name if getattr(src_curve, "group", None) else "Visibility"
    _copy_fcurve_to_action(src_curve, merged, "VISIBLE", group_name)
  return merged


def _transfer_bone_actions_to_armature(graph, arm_obj, node_to_bone_name):
  """Retarget per-node ArgAnimatedBone actions to armature pose-bone actions."""
  action_map = {}
  source_graph_nodes = set()
  source_transforms = set()
  bone_nodes = sorted(
    node_to_bone_name.keys(),
    key=lambda n: getattr(n.transform, "_graph_idx", 1 << 30),
  )
  for node in bone_nodes:
    bone_name = node_to_bone_name[node]
    # EDM bone animation is typically encoded on wrapper nodes above the Bone.
    seen_sources = set()
    current = node
    while current is not None and current.parent is not None and current.render is None and current.transform is not None:
      tfnode = current.transform
      if isinstance(tfnode, AnimatingNode) and not isinstance(tfnode, ArgVisibilityNode):
        if id(tfnode) not in seen_sources:
          seen_sources.add(id(tfnode))
          src_actions = get_actions_for_node(tfnode)
          if src_actions:
            source_graph_nodes.add(current)
            source_transforms.add(tfnode)
          for src_action in src_actions:
            _log_bone_debug_event(
              "retarget-source",
              {
                "graph_node": getattr(tfnode, "name", "") or type(tfnode).__name__,
                "graph_node_type": type(tfnode).__name__,
                "bone_name": bone_name,
                "action_name": src_action.name,
                "fcurves": [fcu.data_path for fcu in src_action.fcurves],
              },
              getattr(tfnode, "name", "") or type(tfnode).__name__,
              bone_name,
              src_action.name,
            )
            dst_action = action_map.get(src_action.name)
            if dst_action is None:
              dst_action = bpy.data.actions.new(src_action.name)
              action_map[src_action.name] = dst_action
            for src_curve in src_action.fcurves:
              dst_path = 'pose.bones["{}"].{}'.format(bone_name, src_curve.data_path)
              # Skip if this FCurve already exists (prevents crash on re-import)
              if dst_action.fcurves.find(dst_path, index=src_curve.array_index) is not None:
                continue
              _copy_fcurve_to_action(src_curve, dst_action, dst_path, bone_name)
      parent = current.parent
      if parent is None or _is_bone_transform(parent.transform):
        break
      current = parent

  if not action_map:
    return source_graph_nodes, source_transforms

  _log_bone_debug_event(
    "retarget-summary",
    {
      "armature": getattr(arm_obj, "name", None),
      "actions": sorted(action_map.keys()),
      "source_graph_nodes": sorted(
        (getattr(getattr(n, "transform", None), "name", "") or type(getattr(n, "transform", None)).__name__)
        for n in source_graph_nodes
      ),
    },
    getattr(arm_obj, "name", None),
  )

  arm_obj.animation_data_create()
  ad = arm_obj.animation_data
  ad.use_nla = True
  ad.action = None
  for track in list(ad.nla_tracks):
    ad.nla_tracks.remove(track)

  for action_name in sorted(action_map.keys()):
    action = action_map[action_name]
    track = ad.nla_tracks.new()
    track.name = action.name
    strip = track.strips.new(action.name, 1, action)
    strip.name = action.name

  return source_graph_nodes, source_transforms


def _prepare_bone_import(graph, parent_obj=None):
  """Create a single armature for EDM Bone/ArgAnimatedBone nodes."""
  _import_ctx.bone_import_ctx = None

  bone_nodes = [n for n in graph.nodes if _is_bone_transform(n.transform)]
  if not bone_nodes:
    return
  # Only map actual Bone/ArgAnimatedBone nodes to the synthesized armature.
  # Swallowing non-bone ancestors (especially ArgVisibilityNode wrappers)
  # collapses authored control/visibility transforms into `s_iEDM_Armature`
  # and causes major DOT hierarchy drift on round-trip.
  bone_chain_nodes = set(bone_nodes)

  arm_name = "iEDM_Armature"
  if bpy.data.objects.get(arm_name) is not None:
    i = 1
    while bpy.data.objects.get("{}.{:03d}".format(arm_name, i)) is not None:
      i += 1
    arm_name = "{}.{:03d}".format(arm_name, i)

  arm_data = bpy.data.armatures.new(arm_name)
  arm_obj = bpy.data.objects.new(arm_name, arm_data)
  bpy.context.collection.objects.link(arm_obj)

  # Determine how to apply the EDM Y-up → Blender Z-up basis correction.
  #
  # For 3DSMAX_BANO / 3DSMAX_DEF (v10_root_object_basis_fix = True):
  #   The parent scene root carries _ROOT_BASIS_FIX.  The armature cancels
  #   that parent (arm.world ≈ Identity).  Bone rest matrices are placed in
  #   Blender Z-up space so they match the skinned-mesh corrected space.
  #   (apply_bone_root_fix = True, _arm_carries_basis_fix = False)
  #
  # For BLOB_RENDER (v10_root_object_basis_fix = False):
  #   A root object may exist but it does NOT carry _ROOT_BASIS_FIX — it only
  #   holds the raw EDM node transform.  The armature carries _ROOT_BASIS_FIX so
  #   bone viewport display is correct.  Skinned meshes are NOT children of the
  #   armature (they end up parented to the root object via bone-node propagation);
  #   core.py sees arm_carries_basis_fix=True and applies _ROOT_BASIS_FIX to each
  #   SkinNode mesh's matrix_basis directly so their orientation matches non-skinned
  #   siblings.  Bone rests stay in EDM Y-up armature-local space.
  #   (apply_bone_root_fix = False, _arm_carries_basis_fix = True)
  _profile_needs_root_fix = _import_profile_flag("bone_rest_requires_root_basis_fix")
  _parent_carries_root_fix = (parent_obj is not None) and _import_profile_flag("v10_root_object_basis_fix")
  _arm_carries_basis_fix = _profile_needs_root_fix and not _parent_carries_root_fix
  apply_bone_root_fix = _profile_needs_root_fix and _parent_carries_root_fix

  if parent_obj is not None:
    arm_obj.parent = parent_obj
    arm_obj.matrix_parent_inverse = Matrix.Identity(4)
    if _arm_carries_basis_fix:
      # Parent exists but does NOT carry _ROOT_BASIS_FIX (BLOB_RENDER).
      # Cancel the parent's raw transform and apply _ROOT_BASIS_FIX so that
      # arm.matrix_world = _ROOT_BASIS_FIX, making skinned-mesh children and
      # bone viewport display both correctly Z-up.
      try:
        arm_obj.matrix_basis = parent_obj.matrix_basis.inverted() @ _ROOT_BASIS_FIX
      except Exception:
        arm_obj.matrix_basis = _ROOT_BASIS_FIX
    else:
      # Parent carries _ROOT_BASIS_FIX (3DSMAX_BANO / 3DSMAX_DEF).
      # Cancel the parent basis so arm.world ≈ Identity; bone rest matrices
      # are placed in Blender Z-up space (apply_bone_root_fix = True).
      try:
        arm_obj.matrix_basis = parent_obj.matrix_basis.inverted()
      except Exception:
        arm_obj.matrix_basis = Matrix.Identity(4)
  elif _arm_carries_basis_fix:
    # No parent at all.  Apply _ROOT_BASIS_FIX directly to the armature so
    # all skinned-mesh children inherit the Y-up→Z-up conversion.
    arm_obj.matrix_basis = _ROOT_BASIS_FIX
  _log_bone_debug_event(
    "armature-parent",
    {
      "armature": arm_name,
      "parent": getattr(parent_obj, "name", None) if parent_obj is not None else None,
      "armature_basis": _matrix_trs_summary(arm_obj.matrix_basis),
      "parent_basis": _matrix_trs_summary(getattr(parent_obj, "matrix_basis", None)) if parent_obj is not None else None,
    },
    arm_name,
    getattr(parent_obj, "name", None) if parent_obj is not None else None,
  )

  view_layer = bpy.context.view_layer
  prev_active = view_layer.objects.active
  try:
    for obj in bpy.context.selected_objects:
      obj.select_set(False)
    arm_obj.select_set(True)
    view_layer.objects.active = arm_obj
    if bpy.context.mode != "OBJECT":
      bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = arm_data.edit_bones
    node_to_bone_name = {}
    used_names = set()
    sorted_nodes = sorted(
      bone_nodes,
      key=lambda n: getattr(n.transform, "_graph_idx", 1 << 30),
    )

    def _unique_bone_name(base):
      name = base or "Bone"
      if name not in used_names:
        used_names.add(name)
        return name
      i = 1
      while True:
        candidate = "{}.{:03d}".format(name, i)
        if candidate not in used_names:
          used_names.add(candidate)
          return candidate
        i += 1

    for node in sorted_nodes:
      bone_name = _unique_bone_name(_transform_display_name(node.transform))
      edit_bones.new(bone_name)
      node_to_bone_name[node] = bone_name

    def _bone_bind_matrix(tfnode):
      """Extract the bone's own bind-pose matrix from EDM Bone or ArgAnimatedBone."""
      if isinstance(tfnode, Bone) and hasattr(tfnode, "bone_matrix"):
        return Matrix(tfnode.bone_matrix)
      if isinstance(tfnode, ArgAnimatedBone) and hasattr(tfnode, "inv_base_bone_matrix"):
        return Matrix(tfnode.inv_base_bone_matrix)
      return None

    def _bone_rest_matrix(node):
      """Compute a bone's full rest matrix in Blender/armature space.

      The exporter writes mat_inv = pbone.matrix.inverted() as the bind matrix:
        - Bone:            bone_matrix = mat_inv  (in addition to Bone.matrix)
        - ArgAnimatedBone: inv_base_bone_matrix = mat_inv  (separate field at node+488)
      Inverting gives pbone.matrix — the exact armature-space rest matrix we need
      for edit-bone placement.  This avoids precision loss from accumulating
      parent-chain matrix multiplications.

      For Blender-exported EDMs the bind matrix is already in Blender Z-up space,
      because io_scene_edm applies ROOT_TRANSFORM_MATRIX once at the scene root
      and stores all bone matrices in Blender's native coordinate system.

      For 3ds Max-exported EDMs (all variants) the bind matrix is in EDM Y-up
      space.  In those cases we apply _ROOT_BASIS_FIX (the inverse of
      ROOT_TRANSFORM_MATRIX) to convert Y-up → Blender Z-up before placing the
      edit bone, controlled by the bone_rest_requires_root_basis_fix profile flag.
      """
      apply_root_fix = apply_bone_root_fix  # pre-computed from enclosing scope
      tf = node.transform

      # Bone (non-animated): bone_matrix is the inverse bind matrix.
      # model::Bone serializes TransformNode.matrix plus a second matrixd for
      # bone_matrix. inv_base_bone_matrix does NOT exist on plain Bone —
      # it only appears on ArgAnimatedBone.
      if isinstance(tf, Bone) and not isinstance(tf, ArgAnimatedBone):
        if hasattr(tf, "bone_matrix"):
          inv_bind = Matrix(tf.bone_matrix)
          if not inv_bind.is_identity:
            try:
              rest = inv_bind.inverted()
              if apply_root_fix:
                rest = _ROOT_BASIS_FIX @ rest
              return rest
            except ValueError:
              pass
        # Fall back to accumulated world matrix
        world_mat = getattr(node, "_world_bl", None)
        if world_mat is None:
          world_mat = getattr(node, "_local_bl", Matrix.Identity(4))
        return world_mat

      # ArgAnimatedBone: inv_base_bone_matrix is the inverse bind matrix.
      if isinstance(tf, ArgAnimatedBone) and hasattr(tf, "inv_base_bone_matrix"):
        bone_bind = Matrix(tf.inv_base_bone_matrix)
        if not bone_bind.is_identity:
          try:
            rest = bone_bind.inverted()
            if apply_root_fix:
              rest = _ROOT_BASIS_FIX @ rest
            return rest
          except ValueError:
            pass
        # Fall back to accumulated world matrix
        world_mat = getattr(node, "_world_bl", None)
        if world_mat is None:
          world_mat = getattr(node, "_local_bl", Matrix.Identity(4))
        return world_mat

      world_mat = getattr(node, "_world_bl", None)
      if world_mat is None:
        world_mat = getattr(node, "_local_bl", Matrix.Identity(4))
      return world_mat

    # Debug: report bone bind matrix detection
    _bone_bind_found = 0
    _bone_bind_missing = 0
    for node in sorted_nodes:
      tf = node.transform
      bb = _bone_bind_matrix(tf)
      tf_type = type(tf).__name__
      tf_name = getattr(tf, "name", "?")
      if bb is not None:
        _bone_bind_found += 1
        if not bb.is_identity:
          rest = _bone_rest_matrix(node)
          print("  [bone-bind] {} '{}' -> non-identity bind matrix, rest translation=({:.3f},{:.3f},{:.3f})".format(
            tf_type, tf_name, rest[0][3], rest[1][3], rest[2][3]))
      else:
        _bone_bind_missing += 1
        print("  [bone-bind] {} '{}' -> NO bind matrix (type={}, has_bone_matrix={}, has_inv_base_bone_matrix={})".format(
          tf_type, tf_name, tf_type,
          hasattr(tf, "bone_matrix"),
          hasattr(tf, "inv_base_bone_matrix")))
    print("Info: Bone bind matrices: {} found, {} missing".format(_bone_bind_found, _bone_bind_missing))

    bone_node_set = set(sorted_nodes)
    for node in sorted_nodes:
      eb = edit_bones[node_to_bone_name[node]]
      bone_rest = _bone_rest_matrix(node)
      bone_bind = _bone_bind_matrix(node.transform)
      bone_name = node_to_bone_name[node]
      tf_name = getattr(node.transform, "name", "") or type(node.transform).__name__

      head = bone_rest.to_translation()
      rot3 = bone_rest.to_3x3()
      y_axis = rot3 @ Vector((0.0, 1.0, 0.0))
      z_axis = rot3 @ Vector((0.0, 0.0, 1.0))

      if y_axis.length < 1e-8:
        y_axis = Vector((0.0, 0.01, 0.0))
      y_axis.normalize()

      length = 0.05
      for child in node.children:
        if child in bone_node_set:
          child_rest = _bone_rest_matrix(child)
          child_head = child_rest.to_translation()
          dist = (child_head - head).length
          if dist > 1e-5:
            length = dist
            break

      eb.head = head
      eb.tail = head + y_axis * max(length, 0.01)
      if z_axis.length > 1e-8:
        eb.align_roll(z_axis)
      _log_bone_debug_event(
        "bone-bind-rest",
        {
          "bone_name": bone_name,
          "source_name": tf_name,
          "source_type": type(node.transform).__name__,
          "bind_matrix": _matrix_trs_summary(bone_bind) if bone_bind is not None else None,
          "rest_matrix": _matrix_trs_summary(bone_rest),
          "head_src": [round(float(v), 6) for v in head],
          "y_axis_src": [round(float(v), 6) for v in y_axis],
          "z_axis_src": [round(float(v), 6) for v in z_axis],
          "derived_length": round(float(max(length, 0.01)), 6),
          "parent_bone": node_to_bone_name.get(getattr(node, "parent", None)),
        },
        bone_name,
        tf_name,
      )
      _log_bone_debug_event(
        "bone-rest",
        {
          "bone_name": bone_name,
          "source_name": tf_name,
          "source_type": type(node.transform).__name__,
          "rest_matrix": _matrix_trs_summary(bone_rest),
          "head": [round(float(v), 6) for v in eb.head],
          "tail": [round(float(v), 6) for v in eb.tail],
          "parent_bone": node_to_bone_name.get(getattr(node, "parent", None)),
        },
        bone_name,
        tf_name,
      )

    for node in sorted_nodes:
      parent = node.parent
      while parent is not None and parent not in node_to_bone_name:
        parent = parent.parent
      if parent is None:
        continue
      eb = edit_bones[node_to_bone_name[node]]
      eb.parent = edit_bones[node_to_bone_name[parent]]

    for node in sorted_nodes:
      bone_name = node_to_bone_name[node]
      tf_name = getattr(node.transform, "name", "") or type(node.transform).__name__
      eb = edit_bones[bone_name]
      parent_matrix = eb.parent.matrix.copy() if eb.parent else None
      matrix_local = eb.matrix.copy()
      if parent_matrix is not None:
        try:
          matrix_local = parent_matrix.inverted() @ eb.matrix
        except Exception:
          matrix_local = eb.matrix.copy()
      _log_bone_debug_event(
        "edit-bone-final",
        {
          "bone_name": bone_name,
          "source_name": tf_name,
          "source_type": type(node.transform).__name__,
          "parent_bone": eb.parent.name if eb.parent else None,
          "matrix": _matrix_trs_summary(eb.matrix),
          "matrix_local": _matrix_trs_summary(matrix_local),
          "head": [round(float(v), 6) for v in eb.head],
          "tail": [round(float(v), 6) for v in eb.tail],
        },
        bone_name,
        tf_name,
      )
  finally:
    try:
      bpy.ops.object.mode_set(mode="OBJECT")
    except Exception as e:
      print(f"Warning in blender_importer\nodes\armature.py: {e}")
    if prev_active is not None:
      view_layer.objects.active = prev_active

  bone_name_by_transform = {}
  for tnode, bname in node_to_bone_name.items():
    if tnode.transform is not None:
      bone_name_by_transform[tnode.transform] = bname

  _import_ctx.bone_import_ctx = {
    "armature": arm_obj,
    "bone_name_by_node": node_to_bone_name,
    "bone_name_by_transform": bone_name_by_transform,
    "bone_chain_nodes": bone_chain_nodes,
    "bone_anim_source_nodes": set(),
    "bone_anim_source_transforms": set(),
    # True when armature carries _ROOT_BASIS_FIX itself (BLOB_RENDER path).
    # Used by core.py to apply the same fix to skinned mesh matrix_basis so
    # they appear Z-up even though their EDM parent is a bone (not the graph root).
    "arm_carries_basis_fix": _arm_carries_basis_fix,
  }

  # Preserve bind/rest transforms when exporter computes static bone matrices.
  # Seed the import context before retargeting so create_arganimation_actions()
  # can see that ArgAnimatedBone channels are being redirected into the shared
  # armature path and avoid double-applying inv_base_bone_matrix.
  arm_data.pose_position = "REST"
  _bone_anim_sources = _transfer_bone_actions_to_armature(graph, arm_obj, node_to_bone_name)
  bone_anim_source_nodes = set()
  bone_anim_source_transforms = set()
  if _bone_anim_sources:
    if isinstance(_bone_anim_sources, tuple) and len(_bone_anim_sources) == 2:
      bone_anim_source_nodes = _bone_anim_sources[0] or set()
      bone_anim_source_transforms = _bone_anim_sources[1] or set()
    else:
      bone_anim_source_nodes = _bone_anim_sources or set()

  _import_ctx.bone_import_ctx["bone_anim_source_nodes"] = bone_anim_source_nodes
  _import_ctx.bone_import_ctx["bone_anim_source_transforms"] = bone_anim_source_transforms


def _bind_skin_object(mesh_obj, skin_node):
  """Attach a SkinNode mesh to imported armature with vertex groups."""
  ctx = _import_ctx.bone_import_ctx or {}
  arm_obj = ctx.get("armature")
  bone_name_by_transform = ctx.get("bone_name_by_transform", {})
  if arm_obj is None or mesh_obj is None or mesh_obj.type != "MESH":
    return

  skin_bones = []
  for bone_node in getattr(skin_node, "bones", []):
    name = bone_name_by_transform.get(bone_node)
    if name:
      skin_bones.append(name)
  if not skin_bones:
    return

  group_map = {}
  for bone_name in skin_bones:
    vg = mesh_obj.vertex_groups.get(bone_name)
    if vg is None:
      vg = mesh_obj.vertex_groups.new(name=bone_name)
    group_map[bone_name] = vg

  arm_mod = mesh_obj.modifiers.get("Armature")
  if arm_mod is None:
    arm_mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
  arm_mod.object = arm_obj

  name_bonus = getattr(skin_node, "name", "") or ""
  def _skin_parent_candidates():
    if not name_bonus:
      return []
    matches = []
    for obj in list(getattr(bpy.data, "objects", []) or []):
      if obj in {mesh_obj, arm_obj}:
        continue
      if getattr(obj, "type", "") == "MESH":
        continue
      dbg_tf_name = str(obj.get("_iedm_dbg_tf_name", "") or "")
      dbg_tf_cls = str(obj.get("_iedm_dbg_tf_cls", "") or "")
      if dbg_tf_name != name_bonus and getattr(obj, "name", "") != name_bonus:
        continue
      score = 0
      if dbg_tf_cls == "TransformNode":
        score += 40
      elif dbg_tf_cls and dbg_tf_cls != "ArgVisibilityNode":
        score += 20
      elif dbg_tf_cls == "ArgVisibilityNode":
        score += 5
      if getattr(obj, "name", "") == name_bonus:
        score += 10
      matches.append((score, getattr(obj, "name", ""), obj))
    matches.sort(key=lambda item: (-item[0], item[1]))
    return [obj for _score, _name, obj in matches]

  candidate = None
  for obj in _skin_parent_candidates():
    candidate = obj
    break
  if (
    candidate
    and candidate not in {mesh_obj, arm_obj}
    and (
      mesh_obj.parent is None
      or mesh_obj.parent == arm_obj
    )
  ):
    mesh_obj.parent = candidate
    mesh_obj.matrix_parent_inverse = Matrix.Identity(4)
    mesh_obj["_iedm_skin_parent_override"] = True
  # Parity: Keep original hierarchy parent if one exists. Reparenting to the
  # armature root (the legacy path) breaks local transform inheritance from
  # ancestors like wings or pylons. Prefer the same-name wrapper override
  # first; otherwise the transient armature parent can leak a stale local
  # basis into the mesh before we reparent it back under the wrapper.
  elif mesh_obj.parent is None:
    mesh_obj.parent = arm_obj
  else:
    # Preserve previously inherited world transform if Blender already
    # attached this mesh to a wrapper object.
    mesh_obj.matrix_parent_inverse = Matrix.Identity(4)



  nverts = len(mesh_obj.data.vertices)
  if nverts == 0:
    return

  # For SkinNode, the mesh data was built using the full original vertexData 
  # pool (no compaction) to maintain 1:1 parity with the EDM bone weights.
  slice21 = _channel_slices_from_vertex_format(skin_node.material.vertex_format).get(21)
  per_vertex_group_count = [0] * nverts
  used_bone_indices = set()

  for vi in range(nverts):
    # Skinned meshes use the original EDM vertex index directly.
    if vi >= len(skin_node.vertexData):
      continue
    src = skin_node.vertexData[vi]
    weights = []
    if slice21:
      weights = [float(x) for x in src[slice21[0]:slice21[1]]]

    assigned = False
    for bi, weight in enumerate(weights[:4]):
      if bi >= len(skin_bones):
        break
      if weight <= 1e-6:
        continue
      group_map[skin_bones[bi]].add([vi], weight, "REPLACE")
      per_vertex_group_count[vi] += 1
      used_bone_indices.add(bi)
      assigned = True

    if not assigned:
      group_map[skin_bones[0]].add([vi], 1.0, "REPLACE")
      per_vertex_group_count[vi] += 1
      used_bone_indices.add(0)

  # Ensure every SkinNode bone appears at least once so exporter keeps palette.
  cursor = 0
  for bi, bone_name in enumerate(skin_bones):
    if bi in used_bone_indices:
      continue
    attempts = 0
    while attempts < nverts and per_vertex_group_count[cursor] >= 4:
      cursor = (cursor + 1) % nverts
      attempts += 1
    if attempts >= nverts:
      break
    group_map[bone_name].add([cursor], 0.01, "ADD")
    per_vertex_group_count[cursor] += 1
    cursor = (cursor + 1) % nverts



