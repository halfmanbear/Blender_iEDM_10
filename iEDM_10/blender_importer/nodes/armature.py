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
  if parent_obj is not None:
    basis = arm_obj.matrix_basis.copy()
    arm_obj.parent = parent_obj
    arm_obj.matrix_parent_inverse = Matrix.Identity(4)
    arm_obj.matrix_basis = basis

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
      if isinstance(tfnode, Bone) and hasattr(tfnode, "data") and len(tfnode.data) >= 1:
        return Matrix(tfnode.data[0])
      if isinstance(tfnode, ArgAnimatedBone) and hasattr(tfnode, "boneTransform"):
        return Matrix(tfnode.boneTransform)
      return None

    def _bone_rest_matrix(node):
      """Compute a bone's full rest matrix in Blender/armature space.

      The exporter writes mat_inv = pbone.matrix.inverted() as the bind matrix:
        - Bone:            data[1] = mat_inv
        - ArgAnimatedBone: boneTransform = mat_inv
      Inverting gives pbone.matrix — the exact armature-space rest matrix we need
      for edit-bone placement.  This avoids precision loss from accumulating
      parent-chain matrix multiplications.
      """
      tf = node.transform

      # Bone (non-animated): data[1] is the inverse bind matrix.
      if isinstance(tf, Bone) and not isinstance(tf, ArgAnimatedBone):
        if hasattr(tf, "data") and len(tf.data) >= 2:
          inv_bind = Matrix(tf.data[1])
          if not inv_bind.is_identity:
            try:
              return inv_bind.inverted()
            except ValueError:
              pass
        # Fall back to accumulated world matrix
        world_mat = getattr(node, "_world_bl", None)
        if world_mat is None:
          world_mat = getattr(node, "_local_bl", Matrix.Identity(4))
        return world_mat

      # ArgAnimatedBone: boneTransform is the inverse bind matrix.
      if isinstance(tf, ArgAnimatedBone) and hasattr(tf, "boneTransform"):
        bone_bind = Matrix(tf.boneTransform)
        if not bone_bind.is_identity:
          try:
            return bone_bind.inverted()
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
        print("  [bone-bind] {} '{}' -> NO bind matrix (type={}, has_data={}, has_boneTransform={})".format(
          tf_type, tf_name, tf_type,
          hasattr(tf, "data") and len(getattr(tf, "data", [])) >= 1,
          hasattr(tf, "boneTransform")))
    print("Info: Bone bind matrices: {} found, {} missing".format(_bone_bind_found, _bone_bind_missing))

    bone_node_set = set(sorted_nodes)
    for node in sorted_nodes:
      eb = edit_bones[node_to_bone_name[node]]
      bone_rest = _bone_rest_matrix(node)

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

    for node in sorted_nodes:
      parent = node.parent
      while parent is not None and parent not in node_to_bone_name:
        parent = parent.parent
      if parent is None:
        continue
      eb = edit_bones[node_to_bone_name[node]]
      eb.parent = edit_bones[node_to_bone_name[parent]]
  finally:
    try:
      bpy.ops.object.mode_set(mode="OBJECT")
    except Exception as e:
      print(f"Warning in blender_importer\nodes\armature.py: {e}")
    if prev_active is not None:
      view_layer.objects.active = prev_active

  # Preserve bind/rest transforms when exporter computes static bone matrices.
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
  bone_name_by_transform = {}
  for tnode, bname in node_to_bone_name.items():
    if tnode.transform is not None:
      bone_name_by_transform[tnode.transform] = bname

  _import_ctx.bone_import_ctx = {
    "armature": arm_obj,
    "bone_name_by_node": node_to_bone_name,
    "bone_name_by_transform": bone_name_by_transform,
    "bone_chain_nodes": bone_chain_nodes,
    "bone_anim_source_nodes": bone_anim_source_nodes,
    "bone_anim_source_transforms": bone_anim_source_transforms,
  }


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

  # Parity: Keep original hierarchy parent if one exists. Reparenting to the 
  # armature root (the legacy path) breaks local transform inheritance from 
  # ancestors like wings or pylons.
  if mesh_obj.parent is None:
    mesh_obj.parent = arm_obj
  else:
    # Preserve previously inherited world transform if Blender already
    # attached this mesh to a wrapper object.
    mesh_obj.matrix_parent_inverse = Matrix.Identity(4)
  
  name_bonus = getattr(skin_node, "name", "") or ""
  candidate = bpy.data.objects.get(name_bonus)
  if candidate and candidate not in {mesh_obj, arm_obj}:
    mesh_obj.parent = candidate
    mesh_obj.matrix_parent_inverse = Matrix.Identity(4)
    mesh_obj["_iedm_skin_parent_override"] = True

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

