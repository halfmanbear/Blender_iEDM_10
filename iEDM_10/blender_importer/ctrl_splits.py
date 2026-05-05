# Fragment: multi-arg control splitting and control/mesh-pair renaming.
# All names resolved via the shared namespace injected by reader.py.


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


def _rename_control_wrapper_mesh_pairs(graph):
  """Keep visible meshes on the semantic base name instead of `.001` helpers.

  Official plain-root visibility scenes frequently contain a control empty with the
  same semantic name as its sole mesh child. Blender auto-suffixes the child mesh
  (`tf_0466.001`), which is noisy. Rename the control wrapper to its authored control
  prefix so the visible mesh can reclaim the base name.
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

    # Only rewrite when the current names are a Blender duplicate split of the same
    # semantic base (`tf_0466` + `tf_0466.001`) or when the child already carries the
    # semantic base but the parent still steals it.
    if current_parent_base != desired_mesh_name and current_mesh_name != desired_mesh_name:
      continue

    try:
      ob.name = desired_parent_name
      child_ob.name = desired_mesh_name
    except Exception as e:
      print(f"Warning in _rename_control_wrapper_mesh_pairs: {e}")
