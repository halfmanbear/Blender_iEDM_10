# Fragment: visibility graph basis-fix passes.
# All names resolved via the shared namespace injected by reader.py.


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
    descendant_nodes = []
    stack = list(child_nodes)
    while stack:
      cur = stack.pop()
      descendant_nodes.append(cur)
      stack.extend(list(getattr(cur, "children", []) or []))

    # Skip ArgVis wrapper chains that hide any deeper animated transform.
    # The animated leaf already has its authored/root basis baked into its
    # zero_transform_local_matrix. Rotating the top visibility wrapper adds an
    # extra +90 X and mirrors the whole branch in world space.
    has_anim_descendant_in_graph = any(
      isinstance(getattr(ch, "transform", None), AnimatingNode)
      and not isinstance(getattr(ch, "transform", None), ArgVisibilityNode)
      for ch in descendant_nodes
    )
    if has_anim_descendant_in_graph:
      continue

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


def _fix_inverse_scaled_visibility_rest_offset():
  """Repair inverse-scaled visibility wrapper rest offsets structurally.

  Some legacy 3ds Max v10 assets encode render geometry under an identity
  ArgVisibilityNode wrapper, then under a tiny uniform-scale TransformNode. The
  visibility wrapper must stay under its animated parent so animation still
  drives the mesh, but its rest offset may need to resolve against the parent
  pivot's parent transform. Detect that pattern by structure only; do not key
  off object names or a specific EDM file.
  """
  def _has_inverse_scale_child(ob):
    for child in list(getattr(ob, "children", []) or []):
      try:
        loc, rot, scale = child.matrix_basis.decompose()
      except Exception:
        continue
      max_scale = max(abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z)))
      min_scale = min(abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z)))
      if max_scale <= 0.0 or max_scale >= 0.1:
        continue
      if min_scale / max_scale < 0.95:
        continue
      if abs(float(loc.x)) > 1e-4 or abs(float(loc.y)) > 1e-4 or abs(float(loc.z)) > 1e-4:
        continue
      try:
        if abs(rot.angle) > math.radians(1.0):
          continue
      except Exception:
        pass
      if any(getattr(desc, "type", "") == "MESH" for desc in getattr(child, "children_recursive", []) or []):
        return True
    return False

  def _is_identityish_basis(ob):
    try:
      loc, rot, scale = ob.matrix_basis.decompose()
      if abs(float(loc.x)) > 1e-4 or abs(float(loc.y)) > 1e-4 or abs(float(loc.z)) > 1e-4:
        return False
      if abs(float(scale.x) - 1.0) > 1e-4 or abs(float(scale.y) - 1.0) > 1e-4 or abs(float(scale.z) - 1.0) > 1e-4:
        return False
      return abs(rot.angle) <= math.radians(1.0)
    except Exception:
      return False

  def _same_world_rotation(a, b):
    try:
      _la, ra, _sa = a.matrix_world.decompose()
      _lb, rb, _sb = b.matrix_world.decompose()
      return abs(ra.rotation_difference(rb).angle) <= math.radians(1.0)
    except Exception:
      return False

  changed = False
  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "type", "") != "EMPTY":
      continue
    if str(ob.get("_iedm_dbg_tf_cls", "") or "") != "ArgVisibilityNode":
      continue
    if not _is_identityish_basis(ob):
      continue
    if not _has_inverse_scale_child(ob):
      continue

    parent = getattr(ob, "parent", None)
    candidate = getattr(parent, "parent", None) if parent is not None else None
    if parent is None or candidate is None:
      continue
    if getattr(parent, "type", "") != "EMPTY" or getattr(candidate, "type", "") != "EMPTY":
      continue
    if not _same_world_rotation(parent, candidate):
      continue

    try:
      parent_loc = parent.matrix_world.to_translation()
      candidate_loc = candidate.matrix_world.to_translation()
      delta = candidate_loc - parent_loc
    except Exception:
      continue

    if delta.length < 0.05 or delta.length > 1.0:
      continue

    try:
      old_world = ob.matrix_world.copy()
      old_loc, old_rot, old_scale = old_world.decompose()
      _ = old_loc
      target_loc = candidate.matrix_world.to_translation().copy()
      target_world = Matrix.LocRotScale(target_loc, old_rot, old_scale)
      target_local = parent.matrix_world.inverted_safe() @ target_world
      ob.matrix_parent_inverse = Matrix.Identity(4)
      ob.matrix_basis = target_local
      ob["_iedm_inverse_scaled_visibility_rest_offset_fix"] = True
      ob["_iedm_inverse_scaled_visibility_parent"] = getattr(parent, "name", "")
      ob["_iedm_inverse_scaled_visibility_candidate"] = getattr(candidate, "name", "")
      changed = True
    except Exception as e:
      print(f"Warning in _fix_inverse_scaled_visibility_rest_offset: {e}")

  if changed:
    try:
      bpy.context.view_layer.update()
    except Exception:
      pass


def _apply_argvis_chain_basis_fix():
  """Strip erroneous ±90 X basis rotation from intermediate ArgVisibilityNode wrappers.

  Some scene-root-authored v10 graphs create visibility wrapper chains where
  intermediate nodes carry only the coordinate system conversion rotation. This
  fix removes the rotation from visibility nodes that:
  1. Are ArgVisibilityNode with an ArgVisibilityNode parent
  2. The parent is a direct child of _EDMFileRoot (or a root-level wrapper)
  3. The node's matrix is essentially just a ±90 X basis rotation

  Also handles animated objects (ArgRotationNode children) under ArgVisibilityNode
  parents where the ArgVisibilityNode carries an unnecessary rotation.
  """
  if _import_ctx.edm_version < 10:
    return
  if not _import_profile_flag("argvis_chain_basis_fix"):
    return
  if not getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  def _is_basis_only_rotation(mat, eps=1e-4):
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

    # Case 1: ArgVisibilityNode -> ArgVisibilityNode chain
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

    # Case 2: Animated child under ArgVisibilityNode parent carrying a basis-only rotation.
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
