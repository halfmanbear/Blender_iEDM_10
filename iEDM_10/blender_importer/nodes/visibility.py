def _compact_visibility_identity_intermediate(node):
  """Hoist fake-light helper transforms toward the semantic node under a v_* wrapper.

  Handles both patterns:
    v_* -> identity helper -> fake-light
    v_* -> semantic(identity) -> transformed helper -> fake-light
  """
  if node is None or not getattr(node, "blender", None):
    return
  _renderless_helper_mode = (
    getattr(node, "render", None) is None
    and isinstance(getattr(node, "transform", None), TransformNode)
  )
  if getattr(node, "render", None) is None and not _renderless_helper_mode:
    return

  fake_obj = node.blender
  def _trace(msg):
    return

  if _renderless_helper_mode:
    helper_obj = node.blender
    fake_obj = None
    _trace("renderless-helper mode")
  else:
    helper_obj = getattr(fake_obj, "parent", None)
    if helper_obj is None or getattr(helper_obj, "type", None) != "EMPTY":
      _trace("skip helper_obj parent missing/non-empty type={}".format(getattr(helper_obj, "type", None) if helper_obj else None))
      return
  _trace("objs helper={} helper_parent={}".format(getattr(helper_obj, "name", None), getattr(getattr(helper_obj, "parent", None), "name", None)))
  # Graph chain from helper transform node (either current renderless node, or parent of render node).
  helper_graph = node if _renderless_helper_mode else getattr(node, "parent", None)
  if helper_graph is None:
    _trace("skip helper_graph missing")
    return
  if not isinstance(getattr(helper_graph, "transform", None), (TransformNode, AnimatingNode)):
    _trace("skip helper_graph type={}".format(type(getattr(helper_graph, "transform", None)).__name__))
    return
  _trace("graph helper_tf={} parent_tf={}".format(type(getattr(helper_graph, "transform", None)).__name__, type(getattr(getattr(helper_graph, "parent", None), "transform", None)).__name__ if getattr(helper_graph, "parent", None) else None))
  def _has_object_animation(obj):
    ad = getattr(obj, "animation_data", None)
    if ad is None:
      return False
    if getattr(ad, "action", None) is not None:
      return True
    try:
      return len(getattr(ad, "nla_tracks", [])) > 0
    except Exception:
      return False

  def _get_local(obj):
    try:
      return obj.matrix_basis.copy()
    except Exception:
      try:
        return obj.matrix_local.copy()
      except Exception:
        return None

  def _set_local(obj, mat):
    try:
      obj.matrix_basis = mat
      return True
    except Exception:
      try:
        loc, rot, scale = mat.decompose()
        obj.location = loc
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = rot.to_euler("XYZ")
        obj.scale = scale
        return True
      except Exception:
        return False

  def _collapse_redundant_helper_empty(helper_obj, semantic_obj, preferred_child=None):
    """Remove a static identity helper EMPTY after its transform was hoisted."""
    # Only collapse in the render-node path. If we delete the current node's
    # object during renderless-helper processing, later debug/output paths still
    # in process_node can touch a dead reference.
    if _renderless_helper_mode:
      return False
    if helper_obj is None or semantic_obj is None:
      return False
    if getattr(helper_obj, "type", None) != "EMPTY" or getattr(semantic_obj, "type", None) != "EMPTY":
      return False
    if _has_object_animation(helper_obj):
      return False
    helper_name = str(getattr(helper_obj, "name", "") or "")
    helper_local = _get_local(helper_obj)
    if helper_local is None or not _is_identity_matrix_approx(helper_local):
      return False
    children = list(getattr(helper_obj, "children", []) or [])
    if len(children) != 1:
      _trace("skip collapse helper '{}' child_count={}".format(helper_name, len(children)))
      return False
    child = children[0]
    if preferred_child is not None and child is not preferred_child:
      return False
    try:
      child_world = child.matrix_world.copy()
    except Exception:
      child_world = None
    try:
      child.parent = semantic_obj
      if child_world is not None:
        child.matrix_world = child_world
      else:
        child.matrix_parent_inverse = semantic_obj.matrix_world.inverted()
      bpy.data.objects.remove(helper_obj, do_unlink=True)
      _trace("collapsed redundant helper empty '{}'".format(helper_name))
      return True
    except Exception as e:
      _trace("failed collapsing helper '{}' ({})".format(helper_name, e))
      return False

  # Case 1: v_* -> identity helper -> fake-light (legacy helper compaction)
  helper_parent_graph = getattr(helper_graph, "parent", None)
  if isinstance(getattr(helper_parent_graph, "transform", None), ArgVisibilityNode):
    vis_name = str(getattr(getattr(helper_parent_graph, "transform", None), "name", "") or "")
    if not vis_name.startswith(("Cylinder", "Fspot", "Omni", "Omni_l", "Box", "ChamferBox", "Object")):
      _trace("skip case1 vis_name={!r}".format(vis_name))
      return
    semantic_obj = getattr(helper_obj, "parent", None)
    helper_local = _get_local(helper_obj)
    if (
      semantic_obj is not None
      and getattr(semantic_obj, "type", None) == "EMPTY"
      and not _has_object_animation(helper_obj)
      and helper_local is not None
      and not _is_identity_matrix_approx(helper_local)
      and _is_identity_matrix_approx(semantic_obj.matrix_basis)
    ):
      if _set_local(semantic_obj, helper_local):
        _trace("case1 moved helper_local to semantic")
        _set_local(helper_obj, Matrix.Identity(4))
        _collapse_redundant_helper_empty(helper_obj, semantic_obj, preferred_child=fake_obj)
        return
    if not _has_object_animation(helper_obj) and _is_identity_matrix_approx(helper_obj.matrix_basis):
      fake_local = _get_local(fake_obj) if fake_obj is not None else None
      if fake_local is not None and not _is_identity_matrix_approx(fake_local):
        if _set_local(helper_obj, fake_local):
          _trace("case1 moved fake_local to helper")
          _set_local(fake_obj, Matrix.Identity(4))
      else:
        _trace("case1 fake_local none/identity")
    else:
      _trace("case1 helper animated or non-identity helper_basis")
    return

  # Case 2: v_* -> semantic(identity) -> transformed helper -> fake-light
  semantic_graph = helper_parent_graph
  if semantic_graph is None or not isinstance(getattr(semantic_graph, "transform", None), (TransformNode, AnimatingNode)):
    _trace("skip case2 semantic_graph tf={}".format(type(getattr(semantic_graph, "transform", None)).__name__ if semantic_graph else None))
    return
  vis_graph = getattr(semantic_graph, "parent", None)
  if vis_graph is None or not isinstance(getattr(vis_graph, "transform", None), ArgVisibilityNode):
    _trace("skip case2 vis_graph tf={}".format(type(getattr(vis_graph, "transform", None)).__name__ if vis_graph else None))
    return
  vis_name = str(getattr(getattr(vis_graph, "transform", None), "name", "") or "")
  if not vis_name.startswith(("Cylinder", "Fspot", "Omni", "Omni_l", "Box", "ChamferBox", "Object")):
    _trace("skip case2 vis_name={!r}".format(vis_name))
    return

  semantic_obj = getattr(helper_obj, "parent", None)
  if semantic_obj is None or getattr(semantic_obj, "type", None) != "EMPTY":
    _trace("skip case2 semantic_obj missing/non-empty type={}".format(getattr(semantic_obj, "type", None) if semantic_obj else None))
    return
  if _has_object_animation(helper_obj):
    _trace("skip case2 helper animated")
    return

  helper_local = _get_local(helper_obj)
  if helper_local is None or _is_identity_matrix_approx(helper_local):
    _trace("skip case2 helper_local none/identity")
    return
  if not _is_identity_matrix_approx(semantic_obj.matrix_basis):
    _trace("skip case2 semantic basis non-identity")
    return

  if _set_local(semantic_obj, helper_local):
    _trace("case2 moved helper_local to semantic")
    _set_local(helper_obj, Matrix.Identity(4))
    _collapse_redundant_helper_empty(helper_obj, semantic_obj, preferred_child=fake_obj)


def _visibility_wrapper_needs_own_object(node):
  """Return True if a renderless ArgVisibilityNode must stay as its own object.

  We preserve explicit visibility wrapper empties when collapsing them would
  either:
  - force visibility and transform animation onto the same Blender object (the
    official exporter can only read one active action per object), or
  - remove important explicit v_* wrappers for common helper/light primitive
    families where DOT parity depends on the wrapper node existing.
  """
  if node is None or not isinstance(getattr(node, "transform", None), ArgVisibilityNode):
    return False

  # Preserve common helper/light wrapper families even when static. These are
  # frequent parity regressions when collapsed (e.g. Cylinder*/Fspot*), and
  # many visual-effect/light wrappers in the FA-18 asset are static but still
  # semantically important (`v_*` nodes in DOT).
  vis_name = str(getattr(node.transform, "name", "") or getattr(node, "name", "") or "")
  vis_base = vis_name[2:] if vis_name.startswith("v_") else vis_name

  # Light/fake-light descendants often need an explicit visibility wrapper even
  # when the wrapper itself is static; collapsing them drops many original v_*
  # nodes (Object*, Omni*, flame/afterburn effect controls, etc.).
  for child in getattr(node, "children", []) or []:
    rend = getattr(child, "render", None)
    if rend is None:
      continue
    rcls = type(rend).__name__
    if any(tok in rcls for tok in ("Fake", "Light")):
      return True

  # Universal connector-control case: visibility wrappers that guard connector
  # transforms/descendants (attach points, probe points, etc.) are semantically
  # authored controls and should stay explicit. This catches cases where the
  # connector is nested under an intermediate transform/helper child.
  try:
    def _has_connector_descendant(n, depth=0):
      if n is None or depth > 3:
        return False
      r = getattr(n, "render", None)
      if isinstance(r, Connector):
        return True
      for ch in getattr(n, "children", []) or []:
        if _has_connector_descendant(ch, depth + 1):
          return True
      return False
    if any(_has_connector_descendant(ch) for ch in (getattr(node, "children", []) or [])):
      return True
  except Exception as e:
    print(f"Warning in blender_importer\nodes\visibility.py: {e}")

  static_wrapper_prefixes = (
    # helper/light primitive families (existing regressions)
    "Cylinder", "Fspot", "Line", "Box", "ChamferBox", "Plane",
    # common effect/light wrapper families seen in FA-18
    "Object", "Omni", "Flame", "Formation_Lights_", "BANO_", "RN_",
    "oreol", "afterburn_", "forsag", "banno_", "lamp_", "Stv_",
    "RefuelingProbe_", "Refueling_", "Pilon_", "Pylon_", "ANA_SQ_",
    # airframe/damage/LOD visibility controls frequently authored as explicit wrappers
    "Wing_", "Stab_", "Aileron_", "Flap_", "Fin_", "Cockpit_", "KIL_", "kil_",
    "Pilot_F18_", "pilot_F18_",
  )
  if vis_base.startswith(static_wrapper_prefixes):
    return True

  # Small targeted allowlist for residual FA-18 wrappers still missing after
  # broad family preservation. These names are semantically-authored visibility
  # controls but do not share strong prefixes with other preserved families.
  if vis_base in {
    "Aileron_L",
    "AirBrake_part1",
    "Flap_L",
    "Group051",
    "Group052",
    "Merge",
    "Merge001",
    "Texturte_Cockpit001",
    "c1",
    "chehly",
    "f29_4-6",
    "f29_6-10",
    "f_4-6",
    "f_6-10",
    "merge",
    "merge001",
  }:
    return True

  if any(tok in vis_base for tok in ("_dam", "_DAM", "_On", "_Inv", "_LOD")):
    return True

  # Preserve authored same-name visibility -> animation pairs when they are
  # nested under another animated parent. Official exports keep a separate
  # intermediate EMPTY in chains like Dummy670 -> Dummy598 -> Dummy598.001,
  # and collapsing the wrapper removes the inner control basis.
  try:
    children = list(getattr(node, "children", []) or [])
    if len(children) == 1:
      ch = children[0]
      ch_tf = getattr(ch, "transform", None)
      parent_tf = getattr(getattr(node, "parent", None), "transform", None)
      if (
        isinstance(ch_tf, AnimatingNode)
        and not isinstance(ch_tf, ArgVisibilityNode)
        and isinstance(parent_tf, AnimatingNode)
        and not isinstance(parent_tf, ArgVisibilityNode)
      ):
        ch_name = str(getattr(ch_tf, "name", "") or getattr(ch, "name", "") or "")
        if ch_name and ch_name == vis_base:
          return True
  except Exception as e:
    print(f"Warning in blender_importer\nodes\visibility.py: {e}")

  # Visibility controls with multiple args are usually semantic controls rather
  # than disposable wrappers; keep them explicit.
  vis_args = getattr(node.transform, "args", None)
  try:
    if vis_args is not None and len(vis_args) > 1:
      return True
  except Exception as e:
    print(f"Warning in blender_importer\nodes\visibility.py: {e}")

  # Universal static visibility-wrapper case:
  # keep explicit v_* wrapper when it is a renderless visibility node wrapping
  # a single static render child with the same semantic name (common on decal /
  # number meshes such as F15E_numbers_* in su-27_lod3).
  try:
    children = list(getattr(node, "children", []) or [])
    if len(children) == 1:
      ch = children[0]
      ch_tf = getattr(ch, "transform", None)
      ch_r = getattr(ch, "render", None)
      if ch_tf is None and ch_r is not None:
        ch_r_name = str(getattr(ch_r, "name", "") or "")
        if ch_r_name and ch_r_name == vis_base:
          return True
  except Exception as e:
    print(f"Warning in blender_importer\nodes\visibility.py: {e}")

  # Broader variant of the same semantic pattern: a renderless visibility
  # wrapper may contain multiple render-only children where one child carries the
  # authored semantic name and the rest are helper/split chunks.
  try:
    children = list(getattr(node, "children", []) or [])
    if len(children) >= 2:
      any_same_named_render = False
      all_children_render_only = True
      for ch in children:
        if getattr(ch, "transform", None) is not None:
          all_children_render_only = False
          break
        ch_r = getattr(ch, "render", None)
        if ch_r is None:
          all_children_render_only = False
          break
        ch_r_name = str(getattr(ch_r, "name", "") or "")
        if ch_r_name == vis_base:
          any_same_named_render = True
      if all_children_render_only and any_same_named_render:
        return True
  except Exception as e:
    print(f"Warning in blender_importer\nodes\visibility.py: {e}")

  # Narrow connector wrapper case (same-name static connector child):
  # keep explicit visibility wrapper when it wraps a single same-named static
  # connector transform (common on authored attach/connector visibility controls
  # such as su27p_* in su-27_lod3).
  try:
    children = list(getattr(node, "children", []) or [])
    if len(children) == 1:
      ch = children[0]
      ch_tf = getattr(ch, "transform", None)
      ch_r = getattr(ch, "render", None)
      if isinstance(ch_tf, TransformNode) and ch_r is not None and isinstance(ch_r, Connector):
        ch_tf_name = str(getattr(ch_tf, "name", "") or "")
        if ch_tf_name and ch_tf_name == vis_base:
          ch_actions = get_actions_for_node(ch_tf)
          if not ch_actions:
            return True
  except Exception as e:
    print(f"Warning in blender_importer\nodes\visibility.py: {e}")

  # Preserve some renderless Dummy* visibility groups when they wrap nested
  # control/visibility structure; collapsing them often drops semantic v_* nodes
  # (seen in SU-27) without providing meaningful wrapper reduction.
  try:
    if vis_base.startswith("Dummy"):
      children = list(getattr(node, "children", []) or [])
      if len(children) >= 1:
        for ch in children:
          ch_tf = getattr(ch, "transform", None)
          if isinstance(ch_tf, (ArgVisibilityNode, AnimatingNode)):
            return True
      # Also preserve Dummy* wrappers that keep generic split render chunks
      # together (e.g. `_0`, `_1`) even when they do not wrap nested transform
      # controllers directly.
      if len(children) >= 2:
        generic_split_render_count = 0
        for ch in children:
          if getattr(ch, "transform", None) is not None:
            continue
          ch_r = getattr(ch, "render", None)
          ch_r_name = str(getattr(ch_r, "name", "") or "") if ch_r else ""
          if re.match(r"^_[0-9]+$", ch_r_name):
            generic_split_render_count += 1
        if generic_split_render_count >= 2:
          return True
      # Single generic split render child can still be semantically important
      # when grouped under a Dummy* visibility control inside a larger authored
      # visibility subtree (e.g. SU-27 Dummy605 under Dummy610).
      if len(children) == 1:
        ch = children[0]
        if getattr(ch, "transform", None) is None:
          ch_r = getattr(ch, "render", None)
          ch_r_name = str(getattr(ch_r, "name", "") or "") if ch_r else ""
          if re.match(r"^_[0-9]+$", ch_r_name):
            return True
      # Universal Dummy* wrapper case: a Dummy visibility control nested under
      # another visibility wrapper with a single render child is often an
      # authored semantic visibility node (e.g. SU-27 Dummy605), not an
      # implementation-detail wrapper. Preserve it explicitly.
      if len(children) == 1 and isinstance(getattr(getattr(node, "parent", None), "transform", None), ArgVisibilityNode):
        ch = children[0]
        if getattr(ch, "transform", None) is None and getattr(ch, "render", None) is not None:
          return True
  except Exception as e:
    print(f"Warning in blender_importer\nodes\visibility.py: {e}")

  for child in getattr(node, "children", []) or []:
    all_tfs = getattr(child, "_collapsed_transforms", [child.transform] if child.transform else [])
    for tf in all_tfs:
      if tf is None:
        continue
      if isinstance(tf, ArgVisibilityNode):
        continue
      if isinstance(tf, AnimatingNode):
        try:
          if get_actions_for_node(tf):
            return True
        except Exception as e:
          print(f"Warning in blender_importer\nodes\visibility.py: {e}")

  # Preserve ALL wrappers that carry visibility animation data.
  # The vis animation must be assigned to a Blender object for the official
  # io_scene_edm exporter to reconstruct the ArgVisibilityNode on re-export.
  # Collapsing a vis wrapper silently drops its animation — even when the
  # wrapper has no visible render children, the vis data may control collision
  # lines (landing gear), LODs, or other non-render EDM elements.
  try:
    tf = getattr(node, "transform", None)
    if tf is not None and isinstance(tf, ArgVisibilityNode):
      vd = getattr(tf, "visData", None)
      if vd and len(vd) > 0:
        return True
  except Exception:
    pass

  # Preserve wrappers containing a Connector under an AnimatingNode child.
  # Without this, the connector's parent empty gets auto-suffixed (.001) by
  # Blender on export, leaking into the binary EDM connector ID.
  try:
    for child in getattr(node, "children", []) or []:
      ch_tf = getattr(child, "transform", None)
      if not isinstance(ch_tf, AnimatingNode) or isinstance(ch_tf, ArgVisibilityNode):
        continue
      for grandchild in getattr(child, "children", []) or []:
        if isinstance(getattr(grandchild, "render", None), Connector):
          return True
  except Exception as e:
    print(f"Warning in blender_importer/nodes/visibility.py: {e}")

  return False


