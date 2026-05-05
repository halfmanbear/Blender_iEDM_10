# Fragment: EDM graph construction from raw EDM file data.
# All names resolved via the shared namespace injected by reader.py.


def iterate_renderNodes(edmFile):
  """Yields all renderNodes, flattening one level of children for split nodes."""
  for node in edmFile.renderNodes:
    if hasattr(node, "children") and node.children:
      for child in node.children:
        yield child
    else:
      yield node


def iterate_all_objects(edmFile):
  for node in itertools.chain(iterate_renderNodes(edmFile), edmFile.connectors, edmFile.shellNodes, edmFile.lightNodes):
    yield node


# ---------------------------------------------------------------------------
# Phase helpers — each handles one clearly-named step of graph construction
# ---------------------------------------------------------------------------

def _init_graph(edmFile):
  """Create TranslationNodes for all EDM transform nodes and wire the hierarchy.

  Returns (graph, nodeLookup) where nodeLookup maps EDM transform node -> TranslationNode.
  """
  graph = TranslationGraph()
  graph.root.transform = edmFile.nodes[0]
  try:
    edmFile.nodes[0]._translation_node = graph.root
  except Exception:
    pass
  nodeLookup = {edmFile.nodes[0]: graph.root}

  for i, tfnode in enumerate(edmFile.nodes):
    tfnode._graph_idx = i

  for tfnode in edmFile.nodes[1:]:
    newNode = TranslationNode()
    newNode.transform = tfnode
    try:
      tfnode._translation_node = newNode
    except Exception:
      pass
    nodeLookup[tfnode] = newNode
    graph.nodes.append(newNode)

  for node in graph.nodes:
    if node.transform.parent:
      if node.transform.parent not in nodeLookup:
        raise ValueError(f"Transform node has parent not in node list: {node.transform.parent}")
      node.parent = nodeLookup[node.transform.parent]
      node.parent.children.append(node)

  assert len([x for x in graph.nodes if not x.parent]) == 1, "More than one root node after reading EDM"

  return graph, nodeLookup


def _attach_render_nodes(graph, edmFile, nodeLookup):
  """Attach render/connector/light nodes to their transform parent in the graph."""
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


def _make_collapse_tf_render_visitor(graph):
  """Return a walk_tree visitor that merges single-child RENDER nodes up into their TRANSFORM parent."""
  def _collapse_transform_render_chains(node):
    if node.type != "TRANSFORM":
      return
    if node.render or len(node.children) != 1:
      return
    child = node.children[0]
    if child.type != "RENDER":
      return
    # Don't absorb Connector render nodes — absorbing them prevents
    # _collapse_chains from treating the parent as a collapsible dummy child.
    if isinstance(child.render, Connector):
      return
    # Only collapse static TransformNodes; collapsing AnimatingNodes changes
    # the basis frame for animations.
    if not isinstance(node.transform, TransformNode):
      return
    parent_tf = getattr(getattr(node, "parent", None), "transform", None)
    if isinstance(parent_tf, (ArgVisibilityNode, ArgAnimationNode)):
      return
    node.render = child.render
    graph.remove_node(child)

  return _collapse_transform_render_chains


def _mark_graph_root_local(graph):
  """Stamp _is_graph_root on the root node and compute its _local_bl."""
  graph.root._is_graph_root = True
  m_root_edm = Matrix.Identity(4)
  root_tf = graph.root.transform
  if root_tf:
    if hasattr(root_tf, "matrix"):
      m_root_edm = Matrix(root_tf.matrix)
    elif hasattr(root_tf, "base") and hasattr(root_tf.base, "matrix"):
      m_root_edm = Matrix(root_tf.base.matrix)
  if _import_ctx.edm_version >= 10 and _import_profile_flag("graph_root_v10_basis_fix"):
    graph.root._local_bl = _ROOT_BASIS_FIX @ m_root_edm
  else:
    graph.root._local_bl = Matrix.Identity(4)
  graph.root._world_bl = graph.root._local_bl.copy()


def _compute_node_local_matrices(graph):
  """Pre-compute _local_bl for every non-root graph node from its EDM transform and render offset."""
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
      elif isinstance(tf, Bone) and hasattr(tf, "bone_matrix"):
        m_edm = Matrix(tf.bone_matrix)
    rn = node.render
    if rn:
      m_rn = Matrix.Identity(4)
      if hasattr(rn, "pos"):
        m_rn = Matrix.Translation(Vector(rn.pos[:3]))
      elif hasattr(rn, "matrix"):
        m_rn = Matrix(rn.matrix)
      if not m_rn.is_identity:
        m_edm = m_edm @ m_rn
    node._local_bl = m_edm


def _compute_world_matrices(graph):
  """Recursively fill _world_bl = parent._world_bl @ _local_bl for all nodes."""
  visited = set()

  def _recurse(node):
    if id(node) in visited:
      return
    visited.add(id(node))
    if node.parent and hasattr(node.parent, "_world_bl"):
      node._world_bl = node.parent._world_bl @ node._local_bl
    elif not hasattr(node, "_world_bl"):
      node._world_bl = node._local_bl.copy()
    for child in node.children:
      _recurse(child)

  _recurse(graph.root)


def _mark_skeleton_nodes(graph):
  """Tag each node as a skeleton node (needs its own Blender EMPTY)."""
  for n in graph.nodes:
    n._is_skeleton = is_skeleton_node(n) or (n.render is not None)


def _eliminate_artifact_wrappers(graph):
  """Remove identity/dummy TransformNodes injected by the exporter."""
  nodes_to_remove = []
  for n in graph.nodes:
    if n == graph.root:
      continue
    if n.render is not None:
      continue
    name = _transform_display_name(n.transform)
    if name in {"root", ""} and n.parent == graph.root and n._local_bl.is_identity:
      nodes_to_remove.append(n)
      continue
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


def _make_collapse_chains_visitor(graph):
  """Return a walk_tree visitor that collapses dummy single-child transform chains."""
  def _collapse_chains(node):
    if not node.children or node.render:
      return
    if not hasattr(node, "_collapsed_transforms"):
      node._collapsed_transforms = [node.transform] if node.transform else []

    if len(node.children) == 1:
      child = node.children[0]
      if child.transform and not child.render:
        name_self = _transform_display_name(node.transform)
        name_child = _transform_display_name(child.transform)

        # Preserve authored same-name Visibility -> Animation pairs as explicit
        # two-object chains to match the official exporter's scene shape.
        is_authored_pair = (name_self and name_self != "_" and name_self == name_child)
        if is_authored_pair and _import_profile_flag("preserve_authored_argvis_control_pairs"):
          return

        if getattr(node, "_is_skeleton", False) or getattr(child, "_is_skeleton", False):
          return

        is_dummy_child = name_child in {"Connector Transform", "Fake Light Transform", ""}
        if is_dummy_child:
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

  return _collapse_chains


def _make_collapse_mesh_visitor(graph):
  """Return a walk_tree visitor that merges a single mesh-render child up into its parent."""
  def _collapse_to_mesh(node):
    if node.transform is None or node.render is not None:
      return
    if len(node.children) != 1:
      return
    child = node.children[0]
    if not child.render:
      return
    tf = node.transform

    if isinstance(tf, ArgAnimationNode):
      return
    if isinstance(tf, ArgVisibilityNode) and _import_profile_flag("collapse_single_argvis_mesh_wrapper"):
      pass
    elif isinstance(tf, ArgVisibilityNode):
      return

    render_cls_name = type(child.render).__name__
    is_mesh_like_render = render_cls_name in {"RenderNode", "ShellNode", "NumberNode"}
    is_anim_tf_for_generic_guard = isinstance(tf, AnimatingNode) and not isinstance(tf, ArgVisibilityNode)
    render_shared = getattr(child.render, "shared_parent", None) is not None
    bone_related = _is_bone_transform(tf)

    if is_mesh_like_render and not is_anim_tf_for_generic_guard and not render_shared and not bone_related:
      node.render = child.render
      node.children = child.children
      for c in node.children:
        c.parent = node
      if child in graph.nodes:
        graph.nodes.remove(child)
      _collapse_to_mesh(node)

  return _collapse_to_mesh


def _sort_graph_children(graph):
  """Sort each node's children for deterministic round-trip traversal order."""
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_graph(edmFile):
  """Build a translation graph from an EDM file."""
  graph, nodeLookup = _init_graph(edmFile)
  _attach_render_nodes(graph, edmFile, nodeLookup)

  # _absorb_connector_helper is intentionally disabled — see inline comment in
  # original code. walk_tree call kept as a placeholder for future re-enabling.
  graph.walk_tree(lambda node: None)

  graph.walk_tree(_make_collapse_tf_render_visitor(graph))

  _mark_graph_root_local(graph)
  _compute_node_local_matrices(graph)
  _compute_world_matrices(graph)

  _mark_skeleton_nodes(graph)

  if _import_profile_flag("eliminate_exporter_artifact_wrappers"):
    _eliminate_artifact_wrappers(graph)
    _compute_world_matrices(graph)

  if _import_profile_flag("collapse_dummy_transform_chains"):
    graph.walk_tree(_make_collapse_chains_visitor(graph))
    _compute_world_matrices(graph)

  if _import_profile_flag("collapse_single_mesh_render_to_parent"):
    graph.walk_tree(_make_collapse_mesh_visitor(graph))

  _sort_graph_children(graph)

  return graph
