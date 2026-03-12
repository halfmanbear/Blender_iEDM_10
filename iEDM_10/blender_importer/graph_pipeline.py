def _reset_transform_debug(options):
  options = options or {}
  enabled = bool(options.get("debug_transforms", False))
  raw_limit = options.get("debug_transform_limit", 200)
  try:
    limit = max(1, int(raw_limit))
  except Exception:
    limit = 200
  name_filter = options.get("debug_transform_filter", None)
  if isinstance(name_filter, str) and not name_filter.strip():
    name_filter = None
  _import_ctx.transform_debug = {
    "enabled": enabled,
    "filter": name_filter,
    "limit": limit,
    "emitted": 0,
  }
  if enabled:
    filter_str = " filter='{}'".format(name_filter) if name_filter else ""
    print("Info: Transform debug enabled{} limit={}".format(filter_str, limit))


def _reset_mesh_origin_mode(options):
  options = options or {}
  mode = str(options.get("mesh_origin_mode", "APPROX")).upper()
  if mode not in {"APPROX", "RAW"}:
    mode = "APPROX"
  _import_ctx.mesh_origin_mode = mode


def _debug_node_label(node):
  if getattr(node, "transform", None):
    tf = node.transform
    name = _transform_display_name(tf) or type(tf).__name__
    return "tf:{}<{}>".format(name, type(tf).__name__)
  if getattr(node, "render", None):
    rn = node.render
    name = getattr(rn, "name", "") or type(rn).__name__
    return "rn:{}<{}>".format(name, type(rn).__name__)
  if getattr(node, "blender", None):
    return "bl:{}<{}>".format(node.blender.name, node.blender.type)
  return "<ROOT>"


def _debug_node_path(node):
  parts = []
  current = node
  while current is not None:
    parts.append(_debug_node_label(current))
    current = getattr(current, "parent", None)
  return " / ".join(reversed(parts))


def _debug_fmt_vec3(vec):
  return "({:+.6f}, {:+.6f}, {:+.6f})".format(vec.x, vec.y, vec.z)


def _debug_fmt_rot_deg(rot):
  euler = rot.to_euler("XYZ")
  return "({:+.3f}, {:+.3f}, {:+.3f})".format(
    math.degrees(euler.x), math.degrees(euler.y), math.degrees(euler.z)
  )


def _debug_fmt_trs(mat):
  loc, rot, scale = mat.decompose()
  return _debug_fmt_vec3(loc), _debug_fmt_rot_deg(rot), _debug_fmt_vec3(scale)

def _anim_vector_to_blender(v):
  return Vector(v)


def _anim_quaternion_to_blender(q):
  return q if hasattr(q, "to_matrix") else Quaternion(q)


def _is_neg90_x_basis_matrix(mat, eps=1e-3):
  try:
    target = (
      (1.0, 0.0, 0.0),
      (0.0, 0.0, 1.0),
      (0.0, -1.0, 0.0),
    )
    for r in range(3):
      for c in range(3):
        if abs(float(mat[r][c]) - target[r][c]) > eps:
          return False
    return True
  except Exception:
    return False


def _anim_base_quaternion_q1_to_blender(node, q):
  """Return the authored base/rest quaternion without an extra basis swap.

  FA-18 style v10 control-surface nodes (e.g. Stv_F1/Stv_F2/Stv_F3) already
  carry their rest quaternion in the same local basis as their mesh data and
  base.position. Applying an additional quaternion_to_blender conversion here
  breaks mirrored placement and pushes render children away from the airframe.
  """
  q_in = q if hasattr(q, "to_matrix") else Quaternion(q)
  return _anim_quaternion_to_blender(q_in)


def _anim_scale_components(value):
  if len(value) < 3:
    return (1.0, 1.0, 1.0)
  return (value[0], value[1], value[2])


def _expected_local_matrix(tfnode, blender_obj=None):
  if tfnode is None:
    return None
  if isinstance(tfnode, TransformNode):
    is_connector = _is_connector_transform(tfnode, blender_obj)
    local_mat = Matrix(tfnode.matrix)
    if is_connector:
      # Remove the exporter's fixed 90-degree X rotation instead of stripping all rotation.
      local_mat = local_mat @ Matrix.Rotation(math.radians(-90.0), 4, "X")
    if (
      blender_obj is not None
      and blender_obj.parent is not None
      and _import_ctx.edm_version >= 10
    ):
      local_mat = blender_obj.parent.matrix_basis @ local_mat
    return local_mat
  if isinstance(tfnode, AnimatingNode):
    local_mat = None
    if hasattr(tfnode, "zero_transform_local_matrix"):
      local_mat = tfnode.zero_transform_local_matrix
    elif hasattr(tfnode, "zero_transform_matrix"):
      local_mat = tfnode.zero_transform_matrix
    elif hasattr(tfnode, "zero_transform"):
      loc, rot, scale = tfnode.zero_transform
      local_mat = Matrix.LocRotScale(loc, rot, scale)
    if local_mat is not None and blender_obj is not None and blender_obj.parent is not None and _import_ctx.edm_version >= 10:
      local_mat = blender_obj.parent.matrix_basis @ local_mat
    return local_mat
  return None


def _debug_filter_match(node, path):
  name_filter = _import_ctx.transform_debug.get("filter")
  if not name_filter:
    return True
  needle = name_filter.lower()
  candidates = [path]
  if getattr(node, "blender", None):
    candidates.append(node.blender.name)
  if getattr(node, "transform", None):
    candidates.append(getattr(node.transform, "name", ""))
  if getattr(node, "render", None):
    candidates.append(getattr(node.render, "name", ""))
  return any(needle in (item or "").lower() for item in candidates)


def _debug_dump_node_transform(node):
  if not _import_ctx.transform_debug.get("enabled"):
    return
  if _import_ctx.transform_debug["emitted"] >= _import_ctx.transform_debug["limit"]:
    return
  if not getattr(node, "blender", None):
    return

  path = _debug_node_path(node)
  if not _debug_filter_match(node, path):
    return

  obj = node.blender
  # Ensure matrix_local/world reflect latest assignments before comparison.
  bpy.context.view_layer.update()

  edm_local = _expected_local_matrix(getattr(node, "transform", None), obj)
  blender_local = obj.matrix_local.copy()
  blender_basis = obj.matrix_basis.copy()
  blender_world = obj.matrix_world.copy()
  parent_name = obj.parent.name if obj.parent else "<none>"

  print("[iEDM][TFDBG] {}".format(path))
  print("  object={} type={} parent={}".format(obj.name, obj.type, parent_name))

  if edm_local is not None:
    loc, rot, scale = _debug_fmt_trs(edm_local)
    print("  edm_local  loc={} rot_deg={} scale={}".format(loc, rot, scale))
  else:
    print("  edm_local  <none>")

  loc, rot, scale = _debug_fmt_trs(blender_local)
  print("  bl_localM  loc={} rot_deg={} scale={}".format(loc, rot, scale))
  loc, rot, scale = _debug_fmt_trs(blender_basis)
  print("  bl_basis   loc={} rot_deg={} scale={}".format(loc, rot, scale))
  loc, rot, scale = _debug_fmt_trs(blender_world)
  print("  bl_world   loc={} rot_deg={} scale={}".format(loc, rot, scale))

  if edm_local is not None:
    ed_loc, ed_rot, ed_scale = edm_local.decompose()
    bl_loc, bl_rot, bl_scale = blender_basis.decompose()
    loc_error = (bl_loc - ed_loc).length
    scale_error = (bl_scale - ed_scale).length
    rot_error = math.degrees(ed_rot.rotation_difference(bl_rot).angle)
    print("  local_err  loc={:.6g} rot_deg={:.6g} scale={:.6g}".format(
      loc_error, rot_error, scale_error
    ))

  _import_ctx.transform_debug["emitted"] += 1

def iterate_renderNodes(edmFile):
  """Iterates all renderNodes in an edmFile - whilst ignoring any nesting
  due to e.g. renderNode splitting"""
  for node in edmFile.renderNodes:
    if hasattr(node, "children") and node.children:
      for child in node.children:
        yield child
    else:
      yield node

def iterate_all_objects(edmFile):
  for node in itertools.chain(iterate_renderNodes(edmFile), edmFile.connectors, edmFile.shellNodes, edmFile.lightNodes):
    yield node

def build_graph(edmFile):
  "Build a translation graph object from an EDM file"
  graph = TranslationGraph()
  # The first node is ALWAYS the root node
  graph.root.transform = edmFile.nodes[0]
  nodeLookup = {edmFile.nodes[0]: graph.root}

  # Add stable indices to transform nodes for deterministic fallback naming.
  for i, tfnode in enumerate(edmFile.nodes):
    tfnode._graph_idx = i

  # Add an entry for every other transform node
  for tfnode in edmFile.nodes[1:]:
    newNode = TranslationNode()
    newNode.transform = tfnode
    nodeLookup[tfnode] = newNode
    graph.nodes.append(newNode)

  # Make the parent/child links
  for node in graph.nodes:
    if node.transform.parent:
      node.parent = nodeLookup[node.transform.parent]
      node.parent.children.append(node)

  # Verify we only have one root
  assert len([x for x in graph.nodes if not x.parent]) == 1, "More than one root node after reading EDM"

  # Connect every renderNode to it's place in the chain
  for i, node in enumerate(iterate_all_objects(edmFile)):
    node._graph_render_idx = i
    if not hasattr(node, "parent") or node.parent is None:
      if (isinstance(node, (FakeOmniLightsNode, FakeSpotLightsNode, FakeALSNode))
          or type(node).__name__ in ("FakeOmniLightsNode", "FakeSpotLightsNode", "FakeALSNode")):
        owner = graph.root
      else:
        print("Warning: Node {} has no parent attribute; skipping".format(node))
        continue
    elif node.parent not in nodeLookup:
      print("Warning: Node {} parent {} not in lookup; skipping".format(node, node.parent))
      continue
    else:
      owner = nodeLookup[node.parent]
    newNode = TranslationNode()
    newNode.render = node
    graph.attach_node(newNode, owner)

  # Keep transform and render nodes separate in general, but absorb simple
  # connector-helper chains to avoid creating implementation-detail empties.
  def _absorb_connector_helper(node):
    return # DISABLED: absorbing the Connector render payload into its parent TransformNode
           # prevents _collapse_chains from recognising that node as a "Connector Transform"
           # dummy child, leaving a spurious model:TransformNode wrapping model:Connector
           # objects in the Blender scene. Leave disabled so _collapse_chains can handle it.
    if node.type != "TRANSFORM":
      return
    if node.render or len(node.children) != 1:
      return
    child = node.children[0]
    if child.type != "RENDER" or child.children:
      return
    if not isinstance(child.render, Connector):
      return
    node.render = child.render
    graph.remove_node(child)

  graph.walk_tree(_absorb_connector_helper)

  # Collapse redundant TransformNode -> RenderNode chains to reduce graph size.
  # This merges the render component up into the transform node.
  def _collapse_transform_render_chains(node):
    if node.type != "TRANSFORM":
      return

    if node.render or len(node.children) != 1:
      return
    child = node.children[0]
    if child.type != "RENDER":
      return
    # Don't absorb Connector render nodes — absorbing them gives the parent
    # TransformNode a render payload, which prevents _collapse_chains from
    # treating it as a collapsible "Connector Transform" dummy child, leaving
    # a spurious model:TransformNode wrapping model:Connector in the scene.
    if isinstance(child.render, Connector):
      return
    # Only collapse static TransformNodes.
    # Collapsing ArgAnimationNode changes the basis frame for animations if the
    # child RenderNode had a rotation (e.g. from Y-up to Z-up), leading to
    # incorrect orientation or animation axes.
    if not isinstance(node.transform, TransformNode):
      return
    # Keep direct render ownership explicit under argument-driven wrappers.
    parent_tf = getattr(getattr(node, "parent", None), "transform", None)
    if isinstance(parent_tf, (ArgVisibilityNode, ArgAnimationNode)):
      return

    # Move render payload to parent
    node.render = child.render
    # Remove child
    graph.remove_node(child)

  graph.walk_tree(_collapse_transform_render_chains)

  # --- Pre-compute local/world matrices in Blender space for graph simplification ---
  graph.root._is_graph_root = True
  
  # For V10+, the graph root carries the global basis fix.
  m_root_edm = Matrix.Identity(4)
  root_tf = graph.root.transform
  if root_tf:
    if hasattr(root_tf, "matrix"):
      m_root_edm = Matrix(root_tf.matrix)
    elif hasattr(root_tf, "base") and hasattr(root_tf.base, "matrix"):
      m_root_edm = Matrix(root_tf.base.matrix)

  if _import_ctx.edm_version >= 10:
    graph.root._local_bl = _ROOT_BASIS_FIX @ m_root_edm
  else:
    graph.root._local_bl = Matrix.Identity(4)
    
  graph.root._world_bl = graph.root._local_bl.copy()

  for node in graph.nodes:
    if node == graph.root:
      continue
    m_edm = Matrix.Identity(4)
    tf = node.transform
    if tf:
      if hasattr(tf, "matrix"):
        m_edm = Matrix(tf.matrix)
      elif hasattr(tf, "base") and hasattr(tf.base, "matrix"):
        m_edm = Matrix(tf.base.matrix)
      elif isinstance(tf, Bone) and hasattr(tf, "data") and len(tf.data) >= 1:
        m_edm = Matrix(tf.data[0])
    rn = node.render
    if rn:
      m_rn = Matrix.Identity(4)
      if hasattr(rn, "pos"):
        m_rn = Matrix.Translation(Vector(rn.pos[:3]))
      elif hasattr(rn, "matrix"):
        m_rn = Matrix(rn.matrix)
      
      if not m_rn.is_identity:
        m_edm = m_edm @ m_rn

    # Convert to Blender space for graph decisions.
    # For V10, no per-node axis conversion; V8 uses a full axis swap.
    node._local_bl = m_edm

  # Calculate world matrices recursively
  _visited_world = set()
  def _calculate_world_bl(node):
    if id(node) in _visited_world:
      return
    _visited_world.add(id(node))
    if node.parent and hasattr(node.parent, "_world_bl"):
      node._world_bl = node.parent._world_bl @ node._local_bl
    elif not hasattr(node, "_world_bl"):
      node._world_bl = node._local_bl.copy()
    for child in node.children:
      _calculate_world_bl(child)

  _calculate_world_bl(graph.root)

  # --- Mark skeleton nodes before collapsing ---
  for n in graph.nodes:
    n._is_skeleton = is_skeleton_node(n) or (n.render is not None)

  # --- Surgical elimination of exporter artifacts ---
  nodes_to_remove = []
  for n in graph.nodes:
    if n == graph.root:
      continue
    if n.render is not None:
      continue
    name = _transform_display_name(n.transform)
    # Skip identity placeholder "root" nodes directly under graph root
    if name in {"root", ""} and n.parent == graph.root and n._local_bl.is_identity:
      nodes_to_remove.append(n)
      continue
    # Skip "Connector Transform" and "Fake Light Transform" wrapper artifacts
    if name in {"Connector Transform", "Fake Light Transform"} and n.parent:
      nodes_to_remove.append(n)
      continue

  for n in nodes_to_remove:
    p = n.parent
    if p is None:
      continue
    insert_at = p.children.index(n) if n in p.children else len(p.children)
    if n in p.children:
      p.children.pop(insert_at)
    for child in list(n.children):
      child.parent = p
      p.children.insert(insert_at, child)
      insert_at += 1
      child._local_bl = n._local_bl @ child._local_bl
    if n in graph.nodes:
      graph.nodes.remove(n)

  # Recalculate world matrices after elimination
  _visited_world.clear()
  _calculate_world_bl(graph.root)

  # --- Collapse redundant single-child transform chains ---
  def _collapse_chains(node):
    if not node.children or node.render:
      return
    if not hasattr(node, "_collapsed_transforms"):
      node._collapsed_transforms = [node.transform] if node.transform else []

    if len(node.children) == 1:
      child = node.children[0]
      if child.transform and not child.render:
        # Don't collapse skeleton nodes (preserves Empties for animation)
        if getattr(node, "_is_skeleton", False) or getattr(child, "_is_skeleton", False):
          return
        name_self = _transform_display_name(node.transform)
        name_child = _transform_display_name(child.transform)
        is_dummy_child = name_child in {"Connector Transform", "Fake Light Transform", ""}
        if (name_self and name_self == name_child) or is_dummy_child:
          node._local_bl = node._local_bl @ child._local_bl
          child_tfs = getattr(child, "_collapsed_transforms", [child.transform] if child.transform else [])
          node._collapsed_transforms.extend(child_tfs)
          for grandchild in list(child.children):
            grandchild.parent = node
            node.children.append(grandchild)
          node.children.remove(child)
          if child in graph.nodes:
            graph.nodes.remove(child)
          _collapse_chains(node)

  graph.walk_tree(_collapse_chains)

  # Recalculate world matrices after chain collapse
  _visited_world.clear()
  _calculate_world_bl(graph.root)

  # --- Collapse single generic-named render child into parent transform ---
  def _collapse_to_mesh(node):
    if node.transform is None or node.render is not None:
      return
    if len(node.children) != 1:
      return
    child = node.children[0]
    child_name = getattr(child.render, "name", "")
    if not child.render:
      return
    tf = node.transform
    # Keep explicit control-node structure under argument-driven wrappers.
    # Collapsing generic render chunks here can drop one effective render child
    # from Dummy* visibility/animation controls (e.g. Dummy650/648/598).
    if isinstance(tf, (ArgVisibilityNode, ArgAnimationNode)):
      return

    render_cls_name = type(child.render).__name__
    is_mesh_like_render = render_cls_name in {"RenderNode", "ShellNode", "NumberNode"}
    
    # Structural check: collapse single render children into parent transform
    # if the child is a basic mesh and parent is not an argument-driven guard.
    is_anim_tf_for_generic_guard = isinstance(tf, AnimatingNode) and not isinstance(tf, ArgVisibilityNode)
    
    # We collapse if the mesh doesn't have a shared parent (to avoid breaking v10 split nodes)
    # and isn't bone-related.
    render_shared = getattr(child.render, "shared_parent", None) is not None
    bone_related = _is_bone_transform(tf)

    should_collapse = (
      is_mesh_like_render 
      and not is_anim_tf_for_generic_guard 
      and not render_shared 
      and not bone_related
    )

    if should_collapse:
      node.render = child.render
      node.children = child.children
      for c in node.children:
        c.parent = node
      if child in graph.nodes:
        graph.nodes.remove(child)
      _collapse_to_mesh(node)

  graph.walk_tree(_collapse_to_mesh)

  # Deterministic traversal ordering for round-trip parity:
  # preserve original transform stream ordering first, and only use render
  # stream ordering for render-only nodes that have no transform owner.
  def _child_sort_key(child):
    if child.transform is not None:
      tf_idx = getattr(child.transform, "_graph_idx", 1 << 30)
      tf_name = getattr(child.transform, "name", "") or ""
      return (0, tf_idx, tf_name)

    render_idx = 1 << 30
    if child.render is not None:
      render_idx = getattr(child.render, "_graph_render_idx", render_idx)
    render_name = getattr(child.render, "name", "") if child.render is not None else ""
    return (1, render_idx, render_name)

  def _sort_children(node):
    if node.children:
      node.children.sort(key=_child_sort_key)

  graph.walk_tree(_sort_children)

  return graph


_SEMANTIC_NAME_MAP = {}


def _get_action_argument(action):
  if action is None:
    return None
  try:
    if hasattr(action, "argument"):
      arg = int(getattr(action, "argument"))
      if arg >= 0:
        return arg
  except Exception as e:
    print(f"Warning in blender_importer\graph_pipeline.py: {e}")
  try:
    prefix = str(getattr(action, "name", "")).split("_", 1)[0]
    if prefix and (prefix.isdigit() or (prefix.startswith("-") and prefix[1:].isdigit())):
      return int(prefix)
  except Exception as e:
    print(f"Warning in blender_importer\graph_pipeline.py: {e}")
  return None


def _copy_fcurve_points(src_curve, dst_curve):
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
    except Exception as e:
      print(f"Warning in blender_importer\graph_pipeline.py: {e}")


def _merge_actions_by_argument(actions):
  """Merge multiple source actions into one action per EDM argument."""
  if not actions or len(actions) <= 1:
    return actions

  grouped = {}
  order = []
  for action in actions:
    arg = _get_action_argument(action)
    key = ("arg", arg) if arg is not None else ("name", getattr(action, "name", ""))
    if key not in grouped:
      grouped[key] = []
      order.append(key)
    grouped[key].append(action)

  if all(len(grouped[k]) == 1 for k in order):
    return actions

  merged_actions = []
  for key in order:
    src_actions = grouped[key]
    if len(src_actions) == 1:
      merged_actions.append(src_actions[0])
      continue

    base = src_actions[0]
    merged = bpy.data.actions.new(base.name)
    arg = key[1] if key[0] == "arg" else None
    if arg is not None and hasattr(merged, "argument"):
      merged.argument = int(arg)

    for src in src_actions:
      for src_curve in src.fcurves:
        dst_curve = merged.fcurves.find(src_curve.data_path, index=src_curve.array_index)
        if dst_curve is None:
          dst_curve = merged.fcurves.new(data_path=src_curve.data_path, index=src_curve.array_index)
        _copy_fcurve_points(src_curve, dst_curve)

    merged_actions.append(merged)

  return merged_actions


def _push_action_to_nla(ob, action):
  """Push action onto an NLA track when supported."""
  if ob is None:
    return False
  if not ob.animation_data:
    ob.animation_data_create()
  try:
    track = ob.animation_data.nla_tracks.new()
    track.name = action.name
    track.strips.new(action.name, 1, action)
    return True
  except Exception:
    return False


def _assign_collections(graph):
  """Create named import collections and assign Blender objects to them.

  Rules:
  - ShellNode / SegmentsNode meshes and their ancestor empties -> "Collision"
  - RenderNode meshes that have a Blender parent (animated) -> "Vehicle"
  - RenderNode meshes without a Blender parent (root-level) -> "Texture_Animation"
  - Other nodes (SkinNode, NumberNode, LightNode, FakeLight) stay in Scene Collection
  - Empties follow the category of their children
  """
  scene = bpy.context.scene

  def _get_or_create_col(name):
    col = bpy.data.collections.get(name)
    if col is None:
      col = bpy.data.collections.new(name)
      scene.collection.children.link(col)
    return col

  col_vehicle = _get_or_create_col("Vehicle")
  col_collision = _get_or_create_col("Collision")
  col_tex_anim = _get_or_create_col("Texture_Animation")

  obj_category = {}

  for n in graph.nodes:
    if not getattr(n, "_is_primary", False) or not n.blender:
      continue
    if n.render is None:
      obj_category[n.blender] = None
      continue
    rtype = type(n.render).__name__
    if rtype in ("ShellNode", "SegmentsNode"):
      obj_category[n.blender] = "collision"
    elif rtype == "RenderNode":
      if n.blender.parent is None:
        obj_category[n.blender] = "texture_anim"
      else:
        obj_category[n.blender] = "vehicle"
    else:
      obj_category[n.blender] = None

  # Resolve empties based on children categories
  changed = True
  while changed:
    changed = False
    for n in graph.nodes:
      if not getattr(n, "_is_primary", False) or not n.blender:
        continue
      if obj_category.get(n.blender) is not None:
        continue
      child_cats = set()
      for child_obj in n.blender.children:
        cat = obj_category.get(child_obj)
        if cat:
          child_cats.add(cat)
      if "collision" in child_cats and "vehicle" not in child_cats:
        obj_category[n.blender] = "collision"
        changed = True
      elif "vehicle" in child_cats or "texture_anim" in child_cats:
        obj_category[n.blender] = "vehicle"
        changed = True

  _col_map = {
    "collision": col_collision,
    "vehicle": col_vehicle,
    "texture_anim": col_tex_anim,
  }

  for obj, cat in obj_category.items():
    target = _col_map.get(cat)
    if target is None:
      continue
    for cur_col in list(obj.users_collection):
      try:
        cur_col.objects.unlink(obj)
      except Exception as e:
        print(f"Warning in blender_importer\graph_pipeline.py: {e}")
    try:
      target.objects.link(obj)
    except RuntimeError:
      pass


def create_bounding_box_from_root(root):
  """Create an empty cube representing the EDM bounding box."""
  if not hasattr(root, "boundingBoxMin"):
    return None
  bmin = vector_to_blender(root.boundingBoxMin)
  bmax = vector_to_blender(root.boundingBoxMax)
  center = (bmin + bmax) * 0.5
  dims = Vector((abs(bmax.x - bmin.x), abs(bmax.y - bmin.y), abs(bmax.z - bmin.z)))
  ob = bpy.data.objects.new("Bounding_Box", None)
  ob.empty_display_type = "CUBE"
  ob.location = center
  ob.scale = dims * 0.5
  ob.empty_display_size = 1.0
  try:
    ob["_iedm_raw_bbox"] = json.dumps([
      float(root.boundingBoxMin[0]), float(root.boundingBoxMin[1]), float(root.boundingBoxMin[2]),
      float(root.boundingBoxMax[0]), float(root.boundingBoxMax[1]), float(root.boundingBoxMax[2]),
    ], separators=(",", ":"))
  except Exception as e:
    print(f"Warning in blender_importer\graph_pipeline.py: {e}")
  _set_official_special_type(ob, "BOUNDING_BOX")
  bpy.context.collection.objects.link(ob)
  return ob


def _has_special_box(special_type):
  """Return True if a scene object already represents a given special box type."""
  for obj in bpy.data.objects:
    if hasattr(obj, "EDMProps") and getattr(obj.EDMProps, "SPECIAL_TYPE", "") == special_type:
      return True
  return False


def _get_special_box(special_type):
  for obj in bpy.data.objects:
    if hasattr(obj, "EDMProps") and getattr(obj.EDMProps, "SPECIAL_TYPE", "") == special_type:
      return obj
  return None


def _cache_root_aabb_payloads_on_scene(root):
  """Cache raw imported AABB payloads on the scene for export fallback."""
  scn = bpy.context.scene
  try:
    if hasattr(root, "boundingBoxMin") and hasattr(root, "boundingBoxMax"):
      scn["_iedm_raw_bbox"] = json.dumps([
        float(root.boundingBoxMin[0]), float(root.boundingBoxMin[1]), float(root.boundingBoxMin[2]),
        float(root.boundingBoxMax[0]), float(root.boundingBoxMax[1]), float(root.boundingBoxMax[2]),
      ], separators=(",", ":"))
  except Exception as e:
    print(f"Warning in blender_importer\graph_pipeline.py: {e}")
  try:
    user_box = getattr(root, "user_box", None)
    if user_box is not None:
      min_vec_edm = Vector(user_box[0])
      max_vec_edm = Vector(user_box[1])
      if not any(math.isinf(v) for v in min_vec_edm) and not any(math.isinf(v) for v in max_vec_edm):
        scn["_iedm_raw_user_box"] = json.dumps([
          float(min_vec_edm[0]), float(min_vec_edm[1]), float(min_vec_edm[2]),
          float(max_vec_edm[0]), float(max_vec_edm[1]), float(max_vec_edm[2]),
        ], separators=(",", ":"))
  except Exception as e:
    print(f"Warning in blender_importer\graph_pipeline.py: {e}")
  try:
    light_box = getattr(root, "light_box", None)
    if light_box is not None:
      min_vec_edm = Vector(light_box[0])
      max_vec_edm = Vector(light_box[1])
      if not any(math.isinf(v) for v in min_vec_edm) and not any(math.isinf(v) for v in max_vec_edm):
        scn["_iedm_raw_light_box"] = json.dumps([
          float(min_vec_edm[0]), float(min_vec_edm[1]), float(min_vec_edm[2]),
          float(max_vec_edm[0]), float(max_vec_edm[1]), float(max_vec_edm[2]),
        ], separators=(",", ":"))
  except Exception as e:
    print(f"Warning in blender_importer\\graph_pipeline.py: {e}")


def _create_special_box(name, special_type, min_vec_edm, max_vec_edm, raw_prop_name):
  min_vec = vector_to_blender(min_vec_edm)
  max_vec = vector_to_blender(max_vec_edm)
  center = (min_vec + max_vec) * 0.5
  dims = Vector((abs(max_vec.x - min_vec.x), abs(max_vec.y - min_vec.y), abs(max_vec.z - min_vec.z)))
  ob = _get_special_box(special_type)
  if ob is None:
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "CUBE"
    ob.empty_display_size = 1.0
    _set_official_special_type(ob, special_type)
    bpy.context.collection.objects.link(ob)
  ob.location = center
  ob.scale = dims * 0.5
  try:
    ob[raw_prop_name] = json.dumps([
      float(min_vec_edm[0]), float(min_vec_edm[1]), float(min_vec_edm[2]),
      float(max_vec_edm[0]), float(max_vec_edm[1]), float(max_vec_edm[2]),
    ], separators=(",", ":"))
  except Exception as e:
    print(f"Warning in blender_importer\\graph_pipeline.py: {e}")
  _set_official_special_type(ob, special_type)
  return ob


def _create_user_box_from_root(root):
  """Create an empty cube representing the EDM user box when distinct."""
  user_box = getattr(root, "user_box", None)
  if user_box is None:
    return None
  min_vec_edm = Vector(user_box[0])
  max_vec_edm = Vector(user_box[1])
  if any(math.isinf(v) for v in min_vec_edm) or any(math.isinf(v) for v in max_vec_edm):
    return None
  return _create_special_box("User_Box", "USER_BOX", min_vec_edm, max_vec_edm, "_iedm_raw_user_box")


def _create_light_box_from_root(root):
  """Create an empty cube representing the EDM light box when present."""
  light_box = getattr(root, "light_box", None)
  if light_box is None:
    return None
  min_vec_edm = Vector(light_box[0])
  max_vec_edm = Vector(light_box[1])
  if any(math.isinf(v) for v in min_vec_edm) or any(math.isinf(v) for v in max_vec_edm):
    return None
  return _create_special_box("Light_Box", "LIGHT_BOX", min_vec_edm, max_vec_edm, "_iedm_raw_light_box")


