# Fragment: core node processing — creates Blender objects from the EDM graph.
# All names resolved via the shared namespace injected by reader.py.


def _is_narrow_safe_identity_helper_name(name):
  if not name:
    return False
  if name in {"Connector Transform", "Fake Light Transform"}:
    return True
  if name.startswith(("Connector_", "Light_Dir", "Omni_", "Fspot", "Point", "Dummy")):
    return True
  return False


def _idprop_sequence_value(value):
  def _convert(item, depth=0):
    if depth > 2:
      return None
    if isinstance(item, (int, float, bool, str)):
      return item
    if isinstance(item, (bytes, bytearray)):
      return bytes(item).hex()
    item_to_list = getattr(item, "tolist", None)
    if callable(item_to_list):
      try:
        return _convert(item_to_list(), depth)
      except Exception:
        return None
    if isinstance(item, (list, tuple)):
      if len(item) > 16:
        return None
      converted = [_convert(sub, depth + 1) for sub in item]
      if any(sub is None for sub in converted):
        return None
      return converted
    return None

  if not isinstance(value, (list, tuple)):
    return None
  if len(value) > 16:
    return None
  converted = _convert(value, 0)
  if converted is None:
    return None
  if all(isinstance(item, (int, float, bool)) for item in converted):
    return converted
  return __import__("json").dumps(converted, separators=(",", ":"))


def _idprop_diag_value(value):
  if isinstance(value, (int, float, bool, str)):
    return value
  if isinstance(value, (bytes, bytearray)):
    return bytes(value).hex()
  return _idprop_sequence_value(value)


def _control_wrapper_prefix(tfnode):
  if isinstance(tfnode, ArgVisibilityNode):
    return "v_"
  if isinstance(tfnode, ArgRotationNode):
    return "ar_"
  if isinstance(tfnode, ArgPositionNode):
    return "al_"
  if isinstance(tfnode, ArgScaleNode):
    return "as_"
  return ""


def _preferred_control_wrapper_name(node, base_name):
  """Disambiguate control empties from their sole visible child mesh.

  Plain-root visibility/control assets often import as:
    tf_0466   (ArgRotation EMPTY)
      -> tf_0466.001 (RenderNode MESH)

  because Blender auto-suffixes the child mesh after the control empty claims
  the semantic base name. Naming the control object with its authored control
  prefix keeps the visible mesh on the base name.
  """
  tf = getattr(node, "transform", None)
  if tf is None:
    return base_name

  prefix = _control_wrapper_prefix(tf)
  if not prefix or base_name.startswith(prefix):
    return base_name

  children = list(getattr(node, "children", []) or [])
  if len(children) != 1:
    return base_name

  child = children[0]
  if getattr(child, "transform", None) is not None or getattr(child, "render", None) is None:
    return base_name

  child_name = str(getattr(child.render, "name", "") or "")
  if child_name.startswith("v_") and isinstance(tf, ArgVisibilityNode):
    child_name = child_name[2:]
  child_name = _strip_anim_prefix(child_name)
  if child_name.startswith("Empty_"):
    child_name = child_name[len("Empty_"):]

  base_plain = _strip_anim_prefix(base_name)
  if base_plain.startswith("Empty_"):
    base_plain = base_plain[len("Empty_"):]

  if not child_name or _is_generic_render_name(child_name):
    child_name = base_plain

  if not child_name or child_name.lower() == "root":
    return base_name

  if child_name == base_plain or _is_generic_render_name(base_plain) or base_plain.startswith("tf_"):
    return prefix + child_name

  return base_name


def _skin_bbox_local_matrix(skin_node):
  bbox = list(getattr(skin_node, "bbox", None) or [])
  if len(bbox) != 6:
    return None

  try:
    min_vec = vector_to_blender(bbox[0:3])
    max_vec = vector_to_blender(bbox[3:6])
    if any(math.isinf(v) or math.isnan(v) for v in min_vec) or any(math.isinf(v) or math.isnan(v) for v in max_vec):
      return None

    if (
      getattr(_import_ctx, "edm_version", 0) >= 10
      and _import_profile_flag("skin_mesh_geometry_root_basis_fix")
    ):
      corners = [
        (_ROOT_BASIS_FIX @ Vector((x, y, z, 1.0))).to_3d()
        for x in (min_vec.x, max_vec.x)
        for y in (min_vec.y, max_vec.y)
        for z in (min_vec.z, max_vec.z)
      ]
      min_vec = Vector((
        min(c.x for c in corners),
        min(c.y for c in corners),
        min(c.z for c in corners),
      ))
      max_vec = Vector((
        max(c.x for c in corners),
        max(c.y for c in corners),
        max(c.z for c in corners),
      ))

    dims = max_vec - min_vec
    if dims.length <= 1.0e-9:
      return None

    center = (min_vec + max_vec) * 0.5
    scale = Vector((abs(dims.x) * 0.5, abs(dims.y) * 0.5, abs(dims.z) * 0.5))
    return Matrix.Translation(center) @ Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0))
  except Exception as e:
    print(f"Warning in blender_importer/nodes/core.py: {e}")
    return None


def _wrap_skin_object_with_skin_box(mesh_obj, skin_node):
  if mesh_obj is None or getattr(mesh_obj, "type", "") != "MESH":
    return None
  if bool(mesh_obj.get("_iedm_skin_box_wrapped")):
    return None

  bbox_local = _skin_bbox_local_matrix(skin_node)
  if bbox_local is None:
    return None

  try:
    old_parent = getattr(mesh_obj, "parent", None)
    old_local = mesh_obj.matrix_basis.copy()
    old_world = mesh_obj.matrix_world.copy()

    skin_box = bpy.data.objects.new(f"{mesh_obj.name}_Skin_Box", None)
    skin_box.empty_display_type = "CUBE"
    skin_box.empty_display_size = 1.0
    _set_official_special_type(skin_box, "SKIN_BOX")
    skin_box["_iedm_translation_status"] = "exact"
    skin_box["_iedm_translation_source"] = "SkinNode.bbox"
    try:
      skin_box["_iedm_raw_skin_box"] = json.dumps(
        [float(v) for v in (getattr(skin_node, "bbox", None) or [])],
        separators=(",", ":"),
      )
    except Exception:
      pass

    target_collections = list(getattr(mesh_obj, "users_collection", []) or [])
    if not target_collections:
      target_collections = [bpy.context.collection]
    for col in target_collections:
      try:
        col.objects.link(skin_box)
      except RuntimeError:
        pass

    skin_box.parent = old_parent
    skin_box.matrix_parent_inverse = Matrix.Identity(4)
    skin_box.matrix_basis = old_local @ bbox_local

    mesh_obj.parent = skin_box
    mesh_obj.matrix_parent_inverse = Matrix.Identity(4)
    mesh_obj.matrix_basis = bbox_local.inverted_safe()
    mesh_obj["_iedm_skin_box_wrapped"] = True

    try:
      mesh_obj.matrix_world = old_world
    except Exception:
      pass

    return skin_box
  except Exception as e:
    print(f"Warning in blender_importer/nodes/core.py: {e}")
    return None


# ---------------------------------------------------------------------------
# process_node helpers — each handles one clearly-named responsibility
# ---------------------------------------------------------------------------

def _create_node_object(node, ctx, is_skel):
  """Create the Blender object for a graph node and store it in node.blender."""
  node.render_blender = None
  if node.render:
    render_name = getattr(node.render, "name", "") or ""
    name_unknown = getattr(node.render, "name_unknown", False)
    if isinstance(node.render, NumberNode) and render_name.endswith("_Number"):
      name_unknown = True
    is_connector_render = isinstance(node.render, Connector)
    is_shell_render = isinstance(node.render, ShellNode)
    is_segments_render = isinstance(node.render, SegmentsNode)

    final_name = render_name
    parent_is_arg_wrapper = isinstance(getattr(getattr(node, "parent", None), "transform", None), (ArgVisibilityNode, ArgAnimationNode))

    def _is_generic_shell_name(name):
      if _is_generic_render_name(name):
        return True
      if not name:
        return True
      return name.startswith("Node") and name[4:].isdigit()

    def _nearest_shell_alias(node):
      cur = getattr(node, "parent", None)
      while cur is not None:
        tf = getattr(cur, "transform", None)
        if tf is None:
          cur = getattr(cur, "parent", None)
          continue
        candidate = _transform_display_name(tf)
        if isinstance(tf, ArgVisibilityNode) and candidate.startswith("v_"):
          candidate = candidate[2:]
        candidate = _strip_anim_prefix(candidate)
        if candidate.startswith("Empty_"):
          candidate = candidate[len("Empty_"):]
        if candidate and candidate.lower() != "root" and not _is_generic_render_name(candidate):
          return candidate
        cur = getattr(cur, "parent", None)
      ridx = getattr(node.render, "_graph_render_idx", None)
      if ridx is not None:
        return "shell_{:04d}".format(ridx)
      return "shell"

    def _nearest_collision_alias(node, suffix):
      base = _nearest_shell_alias(node)
      if base.endswith("_shell") or base.endswith("_segments"):
        return base
      return "{}_{}".format(base, suffix)

    if (is_shell_render and _is_generic_shell_name(final_name)) or (is_segments_render and _is_generic_shell_name(final_name)) or _is_generic_render_name(final_name) or name_unknown:
      if is_shell_render and _is_generic_shell_name(final_name):
        final_name = "{}_shell".format(_nearest_shell_alias(node))
      elif is_segments_render and _is_generic_shell_name(final_name):
        final_name = _nearest_collision_alias(node, "segments")

      if is_connector_render:
        candidate = ""
        if node.parent and node.parent.transform:
          candidate = _transform_display_name(node.parent.transform)
          if candidate.startswith("v_"):
            candidate = candidate[2:]
          candidate = _strip_anim_prefix(candidate)
          if candidate.startswith("Empty_"):
            candidate = candidate[len("Empty_"):]
        if candidate and candidate.lower() != "root":
          final_name = "Connector_{}".format(candidate)
        else:
          final_name = "Connector"

      if (_is_generic_render_name(final_name) or name_unknown) and isinstance(node.transform, ArgVisibilityNode):
        candidate = getattr(node.transform, "name", "") or ""
        if candidate.startswith("v_"):
          candidate = candidate[2:]
        if candidate.startswith("Empty_"):
          candidate = candidate[len("Empty_"):]
        if candidate and candidate.lower() != "root" and not _is_generic_render_name(candidate):
          final_name = candidate

      if (_is_generic_render_name(final_name) or name_unknown) and node.parent and node.parent.transform:
        candidate = _transform_display_name(node.parent.transform)
        if isinstance(node.parent.transform, ArgVisibilityNode) and candidate.startswith("v_"):
          candidate = candidate[2:]
        if candidate.startswith("Empty_"):
          candidate = candidate[len("Empty_"):]
        if candidate and candidate.lower() != "root" and not _is_generic_render_name(candidate):
          final_name = candidate

    if (_is_generic_render_name(final_name) or name_unknown) and not parent_is_arg_wrapper:
      mat_name = ""
      if hasattr(node.render, "material") and node.render.material:
        mat_name = getattr(node.render.material, "name", "") or ""
      if mat_name:
        final_name = _SEMANTIC_NAME_MAP.get(mat_name, mat_name)

    if not parent_is_arg_wrapper:
      final_name = _SEMANTIC_NAME_MAP.get(final_name, final_name)

    if _is_generic_render_name(final_name) and not parent_is_arg_wrapper:
      ridx = getattr(node.render, "_graph_render_idx", None)
      if ridx is not None:
        final_name = "rn_{:04d}".format(ridx)

    if final_name:
      node.render.name = final_name

    if isinstance(node.render, Connector):
      node.blender = create_connector(node.render)
      try:
        _connector_edm_name = str(getattr(node.render, "name", "") or "")
        if _connector_edm_name and node.blender.name != _connector_edm_name:
          _conflict = bpy.data.objects.get(_connector_edm_name)
          if _conflict is not None and _conflict is not node.blender:
            if not _is_connector_object(_conflict):
              _conflict.name = _connector_edm_name + "_tf"
              node.blender.name = _connector_edm_name
          else:
            node.blender.name = _connector_edm_name
        if _connector_edm_name:
          node.blender["_iedm_connector_name"] = _connector_edm_name
      except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
      try:
        node.blender["_iedm_connector_apply_xfix"] = False
        parent_tf = getattr(getattr(node, "parent", None), "transform", None)
        if (
          node.transform is None
          and isinstance(parent_tf, TransformNode)
          and not isinstance(parent_tf, AnimatingNode)
        ):
          _ptf_name = getattr(parent_tf, "name", "") or ""
          if hasattr(parent_tf, "matrix"):
            try:
              _pm = Matrix(parent_tf.matrix)
              pbl = getattr(node.parent, "blender", None)
              if pbl is not None:
                pbl["_iedm_raw_transform_matrix"] = [float(v) for row in _pm for v in row]
                if _ptf_name:
                  pbl["_iedm_raw_transform_name"] = _ptf_name
              node.blender["_iedm_connector_tf_matrix"] = [float(v) for row in _pm for v in row]
            except Exception as e:
              print(f"Warning in blender_importer/nodes/core.py: {e}")
          if _ptf_name == "Connector Transform":
            try:
              node.blender["_iedm_connector_apply_xfix"] = _is_neg90_x_basis_matrix(_pm)
            except Exception:
              node.blender["_iedm_connector_apply_xfix"] = True
          if _ptf_name and _ptf_name != "Connector Transform":
            node.blender["_iedm_connector_tf_name"] = _ptf_name

        if (node.transform is not None
            and not isinstance(node.transform, AnimatingNode)):
          _ctf_name = getattr(node.transform, "name", "") or ""
          if hasattr(node.transform, "matrix"):
            try:
              _m = Matrix(node.transform.matrix)
              node.blender["_iedm_connector_tf_matrix"] = [float(v) for row in _m for v in row]
            except Exception as e:
              print(f"Warning in blender_importer/nodes/core.py: {e}")
          if _ctf_name == "Connector Transform":
            try:
              node.blender["_iedm_connector_apply_xfix"] = _is_neg90_x_basis_matrix(_m)
            except Exception:
              node.blender["_iedm_connector_apply_xfix"] = True
          if _ctf_name and _ctf_name != "Connector Transform":
            node.blender["_iedm_connector_tf_name"] = _ctf_name
        if "_iedm_connector_tf_matrix" not in node.blender and getattr(node, "_local_bl", None) is not None:
          try:
            _ml = Matrix(node._local_bl)
            node.blender["_iedm_connector_tf_matrix"] = [float(v) for row in _ml for v in row]
          except Exception as e:
            print(f"Warning in blender_importer/nodes/core.py: {e}")
        if "_iedm_connector_tf_name" not in node.blender:
          _fallback_name = str(getattr(getattr(node, "render", None), "name", "") or "")
          if _fallback_name:
            _fallback_name = _SUFFIX_RE.sub("", _fallback_name)
            node.blender["_iedm_connector_tf_name"] = _fallback_name
      except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
    elif isinstance(node.render, (RenderNode, ShellNode, NumberNode, SkinNode)):
      node.blender = create_object(node.render)
    elif isinstance(node.render, SegmentsNode):
      node.blender = create_segments(node.render)
      if (
        node.blender is not None
        and getattr(_import_ctx, "collision_geometry_basis_fix", False)
        and getattr(node.blender, "data", None) is not None
      ):
        try:
          node.blender.data.transform(_ROOT_BASIS_FIX)
          node.blender.data.update()
        except Exception as e:
          print(f"Warning in blender_importer/nodes/core.py: {e}")
    elif isinstance(node.render, LightNode):
      node.blender = create_lamp(node.render)
    elif isinstance(node.render, BillboardNode) or type(node.render).__name__ == "BillboardNode":
      node.blender = create_billboard(node.render)
    elif isinstance(node.render, FakeOmniLightsNode) or type(node.render).__name__ == "FakeOmniLightsNode":
      node.blender = create_fake_omni_lights(node.render)
    elif isinstance(node.render, FakeSpotLightsNode) or type(node.render).__name__ == "FakeSpotLightsNode":
      node.blender = create_fake_spot_lights(node.render)
    elif isinstance(node.render, FakeALSNode) or type(node.render).__name__ == "FakeALSNode":
      node.blender = create_fake_als_lights(node.render)
    else:
      print("Warning: No case yet for object node {}".format(node.render))

  elif is_skel:
    empty_name = _transform_display_name(node.transform) or type(node.transform).__name__
    _node_children_skel = getattr(node, "children", []) or []
    if (
      len(_node_children_skel) == 1
      and isinstance(getattr(_node_children_skel[0], "render", None), Connector)
      and getattr(_node_children_skel[0], "transform", None) is None
    ):
      empty_name = empty_name + "_tf"
    ob = bpy.data.objects.new(empty_name, None)
    ob.empty_display_size = 0.1
    bpy.context.collection.objects.link(ob)
    node.blender = ob
  else:
    tf_name = _transform_display_name(node.transform) or type(node.transform).__name__
    tf_name = _preferred_control_wrapper_name(node, tf_name)
    _node_children_ns = getattr(node, "children", []) or []
    if (
      len(_node_children_ns) == 1
      and isinstance(getattr(_node_children_ns[0], "render", None), Connector)
      and getattr(_node_children_ns[0], "transform", None) is None
    ):
      tf_name = tf_name + "_tf"
    ob = bpy.data.objects.new(tf_name, None)
    ob.empty_display_size = 0.1
    bpy.context.collection.objects.link(ob)
    node.blender = ob


def _parent_node_object(node, ctx):
  """Establish the Blender parent chain for node.blender."""
  # Skin: look for a same-name sibling transform to use as parent
  try:
    if isinstance(getattr(node, "render", None), SkinNode):
      parent_tf = getattr(getattr(node, "parent", None), "transform", None)
      node.blender["_iedm_dbg_skin_parent_tf_cls"] = str(type(parent_tf).__name__ or "")
      node.blender["_iedm_dbg_skin_parent_tf_name"] = str(getattr(parent_tf, "name", "") or "")
      node.blender["_iedm_dbg_skin_render_name"] = str(getattr(node.render, "name", "") or "")
      sib_names = []
      for sib in list(getattr(getattr(node, "parent", None), "children", []) or []):
        sib_tf = getattr(sib, "transform", None)
        sib_r = getattr(sib, "render", None)
        sib_names.append("{}:{}:{}".format(
          type(sib_tf).__name__ if sib_tf is not None else "",
          getattr(sib_tf, "name", "") if sib_tf is not None else "",
          getattr(sib_r, "name", "") if sib_r is not None else "",
        ))
      if sib_names:
        node.blender["_iedm_dbg_skin_siblings"] = json.dumps(sib_names, separators=(",", ":"))
  except Exception as e:
    print(f"Warning in blender_importer/nodes/core.py: {e}")

  if not node.blender.get("_iedm_skin_parent_override"):
    sibling_skin_helper = None
    try:
      if (
        isinstance(getattr(node, "render", None), SkinNode)
        and getattr(node, "transform", None) is None
        and node.parent is not None
        and isinstance(getattr(node.parent, "transform", None), ArgVisibilityNode)
      ):
        render_name = str(getattr(node.render, "name", "") or "")
        for sib in list(getattr(node.parent, "children", []) or []):
          if sib is node:
            continue
          sib_tf = getattr(sib, "transform", None)
          sib_bl = getattr(sib, "blender", None)
          if sib_tf is None or sib_bl is None or sib_bl == node.blender:
            continue
          if isinstance(sib_tf, ArgVisibilityNode):
            continue
          sib_name = str(getattr(sib_tf, "name", "") or "")
          if sib_name and sib_name == render_name:
            sibling_skin_helper = sib_bl
            break
        try:
          node.blender["_iedm_dbg_skin_helper_name"] = getattr(sibling_skin_helper, "name", "") if sibling_skin_helper is not None else ""
        except Exception:
          pass
    except Exception as e:
      print(f"Warning in blender_importer/nodes/core.py: {e}")

    if sibling_skin_helper is not None:
      node.blender.parent = sibling_skin_helper
      node.blender.matrix_parent_inverse = Matrix.Identity(4)
    elif node.parent and node.parent.blender and node.parent.blender != node.blender:
      node.blender.parent = node.parent.blender
      node.blender.matrix_parent_inverse = Matrix.Identity(4)

  # Reparent render-only duplicate chunks away from mesh parents
  try:
    ob_name = getattr(node.blender, "name", "") or ""
    parent_bl = getattr(getattr(node, "parent", None), "blender", None)
    grandparent_bl = getattr(getattr(getattr(node, "parent", None), "parent", None), "blender", None)
    is_render_only_dup = (
      node.render is not None
      and node.transform is None
      and len(ob_name) > 4
      and ob_name[-4] == "."
      and ob_name[-3:].isdigit()
    )
    parent_is_render_mesh = (
      parent_bl is not None
      and getattr(parent_bl, "type", "") == "MESH"
      and getattr(getattr(node, "parent", None), "render", None) is not None
    )
    if is_render_only_dup and parent_is_render_mesh and grandparent_bl is not None:
      try:
        gp_world_inv = grandparent_bl.matrix_world.inverted_safe()
        node.blender.parent = grandparent_bl
        node.blender.matrix_parent_inverse = Matrix.Identity(4)
        node.blender.matrix_local = gp_world_inv @ parent_bl.matrix_world
      except Exception:
        preserved_local = parent_bl.matrix_local.copy()
        _loc, _rot, _ = preserved_local.decompose()
        node.blender.parent = grandparent_bl
        node.blender.matrix_parent_inverse = Matrix.Identity(4)
        node.blender.matrix_local = Matrix.Translation(_loc) @ _rot.to_matrix().to_4x4()
  except Exception as e:
    print(f"Warning in blender_importer/nodes/core.py: {e}")


def _stamp_node_properties(node, ctx):
  """Write EDM metadata and debug IDprops to node.blender."""
  # Preserve transform -> blender object mapping
  if node.transform and node.blender:
    try:
      node.transform._blender_obj = node.blender
    except Exception as e:
      print(f"Warning in blender_importer/nodes/core.py: {e}")
    if isinstance(node.transform, ArgVisibilityNode):
      try:
        vis_alias = getattr(node.transform, "name", "") or ""
        for _pfx in ("ar_", "al_", "as_"):
          if vis_alias.startswith(_pfx):
            vis_alias = vis_alias[len(_pfx):]
            break
        if vis_alias:
          node.blender["_iedm_vis_export_name"] = vis_alias
      except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
      try:
        vis_payload = []
        for entry in getattr(node.transform, "visData", []) or []:
          if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
          try:
            varg = int(entry[0])
          except Exception:
            continue
          vranges = []
          src_ranges = entry[1] if isinstance(entry[1], (list, tuple)) else []
          for rng in src_ranges:
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
              continue
            try:
              vranges.append([float(rng[0]), float(rng[1])])
            except Exception:
              continue
          vis_payload.append([varg, vranges])
        if vis_payload:
          node.blender["_iedm_vis_raw_args"] = json.dumps(vis_payload, separators=(",", ":"))
      except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
      try:
        if node.render is None and getattr(node.blender, "type", None) == 'EMPTY':
          node.blender["_iedm_vis_passthrough"] = True
      except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
    else:
      # Mark importer-created static helper empties that only contribute an
      # identity transform (Blender duplicate-name splitting artefacts).
      try:
        ob = node.blender
        tf_name = getattr(node.transform, "name", "") or ""
        ob_name = getattr(ob, "name", "") or ""
        is_static_tf = not isinstance(node.transform, AnimatingNode)
        is_renderless_empty = (node.render is None and getattr(ob, "type", None) == 'EMPTY')
        has_one_child = len(getattr(node, "children", []) or []) == 1
        has_blender_dup_suffix = (len(ob_name) > 4 and ob_name[-4] == "." and ob_name[-3:].isdigit())
        not_bone_related = not isinstance(node.transform, (Bone, ArgAnimatedBone))
        if (
          is_static_tf
          and is_renderless_empty
          and has_one_child
          and has_blender_dup_suffix
          and not_bone_related
          and _ob_local_is_identity(ob)
        ):
          ob["_iedm_identity_passthrough"] = True
          if _is_narrow_safe_identity_helper_name(ob_name) or _is_narrow_safe_identity_helper_name(tf_name):
            ob["_iedm_narrow_identity_passthrough"] = True
        elif (
          is_static_tf
          and is_renderless_empty
          and has_one_child
          and not_bone_related
          and isinstance(getattr(node.parent, "transform", None), ArgVisibilityNode)
          and _ob_local_is_identity(ob)
        ):
          try:
            child0 = (getattr(node, "children", []) or [None])[0]
            child_render_cls = type(getattr(child0, "render", None)).__name__ if child0 is not None else None
            child_tf_name = str(getattr(getattr(child0, "transform", None), "name", "") or "")
          except Exception:
            child_render_cls = None
            child_tf_name = ""
          if child_render_cls == "LightNode" or child_tf_name == "Fake Light Transform":
            ob["_iedm_identity_passthrough"] = True
            ob["_iedm_narrow_identity_passthrough"] = True
      except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")

  if node.render is not None and node.blender is not None:
    vis_chain = []
    try:
      vis_chain = _node_visibility_chain_args(node)
      if vis_chain:
        node.blender["_iedm_src_vis_chain"] = json.dumps(vis_chain, separators=(",", ":"))
        node.blender["_iedm_src_vis_nearest"] = int(vis_chain[0])
    except Exception as e:
      print(f"Warning in blender_importer/nodes/core.py: {e}")
    try:
      render_cls_name = type(node.render).__name__
      node.blender["_iedm_src_render_cls"] = render_cls_name
      render_name = str(getattr(node.render, "name", "") or "")
      if render_name:
        node.blender["_iedm_src_render_name"] = render_name
    except Exception as e:
      print(f"Warning in blender_importer/nodes/core.py: {e}")
    try:
      dbg = getattr(_import_ctx, "visibility_debug", {}) or {}
      if dbg.get("enabled"):
        tracked_args = {38, 147, 183, 187, 188, 214, 215, 224, 225}
        hit_args = [arg for arg in vis_chain if arg in tracked_args]
        if hit_args:
          _debug_log_event(
            "[iEDM][VISOBJ] obj={!r} render_cls={} render_name={!r} vis_chain={} parent={!r}".format(
              getattr(node.blender, "name", None),
              type(node.render).__name__,
              getattr(node.render, "name", "") or "",
              hit_args,
              getattr(getattr(node.blender, "parent", None), "name", None),
            )
          )
    except Exception as e:
      print(f"Warning in blender_importer/nodes/core.py: {e}")

  # Original EDM animation-node name passthrough (no al_/ar_/as_ prefix)
  if (node.transform is not None
      and isinstance(node.transform, AnimatingNode)
      and not isinstance(node.transform, (Bone, ArgAnimatedBone))):
    try:
      orig_anim_name = getattr(node.transform, "name", "") or ""
      if (orig_anim_name
          and not orig_anim_name.startswith("al_")
          and not orig_anim_name.startswith("ar_")
          and not orig_anim_name.startswith("as_")):
        node.blender["_iedm_orig_anim_name"] = orig_anim_name
    except Exception as e:
      print(f"Warning in blender_importer/nodes/core.py: {e}")

  # Raw ArgAnimation payload for in-Blender inspection
  if isinstance(node.transform, ArgAnimationNode) and node.blender is not None:
    try:
      raw = {}
      base = getattr(node.transform, "base", None)
      if base is not None:
        try:
          bm = Matrix(base.matrix)
          raw["base_matrix"] = [float(v) for row in bm for v in row]
        except Exception as e:
          print(f"Warning in blender_importer/nodes/core.py: {e}")
        try:
          raw["base_position"] = [float(x) for x in base.position[:3]]
        except Exception as e:
          print(f"Warning in blender_importer/nodes/core.py: {e}")
        try:
          q1 = base.quat_1 if hasattr(base.quat_1, "__iter__") else Quaternion(base.quat_1)
          raw["base_quat_1"] = [float(x) for x in q1]
        except Exception as e:
          print(f"Warning in blender_importer/nodes/core.py: {e}")
        try:
          q2 = base.quat_2 if hasattr(base.quat_2, "__iter__") else Quaternion(base.quat_2)
          raw["base_quat_2"] = [float(x) for x in q2]
        except Exception as e:
          print(f"Warning in blender_importer/nodes/core.py: {e}")
        try:
          raw["base_scale"] = [float(x) for x in base.scale[:3]]
        except Exception as e:
          print(f"Warning in blender_importer/nodes/core.py: {e}")

      pos_payload = []
      for entry in (getattr(node.transform, "posData", None) or []):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
          continue
        arg, keys = entry
        try:
          arg_i = int(arg)
        except Exception:
          continue
        key_rows = []
        for k in (keys or []):
          try:
            key_rows.append([float(k.frame), float(k.value[0]), float(k.value[1]), float(k.value[2])])
          except Exception:
            continue
        pos_payload.append([arg_i, key_rows])
      if pos_payload:
        raw["pos_data"] = pos_payload

      rot_payload = []
      for entry in (getattr(node.transform, "rotData", None) or []):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
          continue
        arg, keys = entry
        try:
          arg_i = int(arg)
        except Exception:
          continue
        key_rows = []
        for k in (keys or []):
          try:
            qv = k.value if hasattr(k.value, "__iter__") else Quaternion(k.value)
            key_rows.append([float(k.frame), float(qv[0]), float(qv[1]), float(qv[2]), float(qv[3])])
          except Exception:
            continue
        rot_payload.append([arg_i, key_rows])
      if rot_payload:
        raw["rot_data"] = rot_payload

      try:
        bmat_inv = getattr(getattr(node, "transform", None), "bmat_inv", None)
        if bmat_inv is None:
          bmat_inv = getattr(getattr(getattr(node, "transform", None), "base", None), "bmat_inv", None)
        if bmat_inv is not None:
          raw["bmat_inv"] = [float(v) for row in Matrix(bmat_inv) for v in row]
      except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")

      try:
        zmat = getattr(node.transform, "zero_transform_local_matrix", None)
        if zmat is not None:
          raw["zero_transform_local_matrix"] = [float(v) for row in Matrix(zmat) for v in row]
      except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")

      if raw:
        node.blender["_iedm_raw_arganim_payload"] = json.dumps(raw, separators=(",", ":"))
    except Exception as e:
      print(f"Warning in blender_importer/nodes/core.py: {e}")

  # Debug stamp: correlates Blender objects back to EDM transform/render classes
  try:
    if node.blender:
      node.blender["_iedm_dbg_tf_cls"] = type(node.transform).__name__ if node.transform is not None else ""
      node.blender["_iedm_dbg_r_cls"] = type(node.render).__name__ if node.render is not None else ""
      if node.transform is not None:
        node.blender["_iedm_dbg_tf_name"] = str(getattr(node.transform, "name", "") or "")
      if node.render is not None:
        node.blender["_iedm_dbg_r_name"] = str(getattr(node.render, "name", "") or "")
      try:
        src_m = None
        tf = node.transform
        if tf is not None:
          if hasattr(tf, "matrix"):
            src_m = Matrix(tf.matrix)
          elif hasattr(getattr(tf, "base", None), "matrix"):
            src_m = Matrix(tf.base.matrix)
          elif hasattr(getattr(tf, "base", None), "transform"):
            src_m = Matrix(getattr(tf.base, "transform"))
        if src_m is not None:
          src_rows = []
          for r in range(3):
            src_rows.append([round(float(src_m[r][c]), 4) for c in range(3)])
          node.blender["_iedm_dbg_src_rows"] = str(src_rows)
      except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
  except Exception as e:
    print(f"Warning in blender_importer/nodes/core.py: {e}")


def _apply_render_positioning(node):
  """Apply local matrix for render-only nodes; tag identity passthrough helpers.

  Returns _used_shared_parent_fallback (bool) for use in the animation hookup.
  """
  _used_shared_parent_fallback = False

  _is_render_only_positioning = (
    node.render and node.blender and node.blender.type == "MESH"
    and (not node.transform or isinstance(node.transform, ArgVisibilityNode))
  )

  if isinstance(getattr(node, "render", None), Connector) and getattr(node, "transform", None) is None and getattr(node, "blender", None) is not None:
    try:
      apply_node_transform(node, node.blender, used_shared_parent=False)
    except Exception as e:
      print(f"Warning in blender_importer/nodes/core.py: {e}")

  if _is_render_only_positioning:
    shared_parent = getattr(node.render, "shared_parent", None)
    parent_transform = getattr(getattr(node, "parent", None), "transform", None)
    parent_is_real_transform = (
      parent_transform is not None
      and not isinstance(parent_transform, ArgVisibilityNode)
    )
    applied_local_bl = False
    has_parent_obj = bool(node.parent and node.parent.blender)
    shared_parent_obj = getattr(shared_parent, "_blender_obj", None) if shared_parent is not None else None

    if shared_parent is not None and not has_parent_obj and not parent_is_real_transform:
      apply_node_transform(shared_parent, node.blender, used_shared_parent=True)
      _used_shared_parent_fallback = True

    if shared_parent is not None and has_parent_obj and not getattr(node.render, "split_owner_encoded", False):
      src = shared_parent_obj
      dst = node.parent.blender
      if src is not None:
        try:
          node.blender.matrix_local = dst.matrix_world.inverted() @ src.matrix_world
        except Exception as e:
          print(f"Warning in blender_importer/nodes/core.py: {e}")
      else:
        try:
          if (
            not parent_is_real_transform
            and getattr(shared_parent, "category", None) is NodeCategory.transform
          ):
            apply_node_transform(shared_parent, node.blender, used_shared_parent=True)
            _used_shared_parent_fallback = True
        except Exception as e:
          print(f"Warning in blender_importer/nodes/core.py: {e}")
    local_bl = getattr(node, "_local_bl", None)
    if local_bl is not None and not _used_shared_parent_fallback:
      try:
        has_render_local = hasattr(node.render, "matrix") or hasattr(node.render, "pos")
        if has_render_local:
          apply_node_transform(node, node.blender, used_shared_parent=False)
        else:
          matrix_to_apply = local_bl.copy() if hasattr(local_bl, "copy") else local_bl
          _prrbf_base = (
            _import_profile_flag("plain_root_render_local_basis_fix")
            and _import_ctx.edm_version >= 10
            and node.transform is None
            and not getattr(_import_ctx, "use_scene_root_basis_object", True)
            and str(getattr(node.render, "name", "") or "")
          )
          _parent_node = getattr(node, "parent", None)
          _needs_render_basis_fix = _prrbf_base and (
            getattr(_parent_node, "_is_graph_root", False)
            or (
              isinstance(getattr(_parent_node, "transform", None), ArgVisibilityNode)
              and getattr(getattr(_parent_node, "parent", None), "_is_graph_root", False)
              and any(
                isinstance(getattr(sib, "transform", None), AnimatingNode)
                and not isinstance(getattr(sib, "transform", None), ArgVisibilityNode)
                for sib in list(getattr(_parent_node, "children", []) or [])
              )
            )
          )
          if (
            not _needs_render_basis_fix
            and isinstance(getattr(node, "render", None), SkinNode)
            and not bool(node.blender.get("_iedm_skin_parent_override"))
          ):
            _bone_ctx = getattr(_import_ctx, "bone_import_ctx", None) or {}
            if bool(_bone_ctx.get("arm_carries_basis_fix")):
              _needs_render_basis_fix = True
          if _needs_render_basis_fix:
            if not (
              isinstance(getattr(node, "render", None), SkinNode)
              and bool(node.blender.get("_iedm_skin_parent_override"))
            ):
              matrix_to_apply = _ROOT_BASIS_FIX @ matrix_to_apply
          node.blender.matrix_basis = matrix_to_apply
        applied_local_bl = True
      except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")

    if _import_ctx.edm_version >= 10 and _import_profile_flag("argvis_parent_mesh_alignment_fix"):
      if (
        node.parent and isinstance(node.parent.transform, ArgVisibilityNode) and node.parent.blender
        and not _used_shared_parent_fallback
      ):
        parent_loc = node.parent.blender.matrix_basis.decompose()[0]
        if not applied_local_bl:
          _offset_mesh_world(node.blender, parent_loc)

    if (
      _import_ctx.mesh_origin_mode == "APPROX"
      and node.parent and node.parent.blender
      and not isinstance(node.parent.transform, (ArgVisibilityNode, AnimatingNode))
      and not isinstance(getattr(node, "render", None), SkinNode)
    ):
      if node.blender.location.length < 1e-9:
        _recenter_mesh_object_to_geometry(node.blender)
        if _import_ctx.edm_version >= 10:
          node.blender.location = (node.parent.blender.matrix_basis @ node.blender.location.to_4d()).to_3d()

    if isinstance(getattr(node, "render", None), SkinNode):
      _wrap_skin_object_with_skin_box(node.blender, node.render)

  # Tag identity passthrough for render-only nodes
  try:
    if node.blender and node.render is not None and node.transform is None:
      ob = node.blender
      ob_name = getattr(ob, "name", "") or ""
      has_parent = getattr(ob, "parent", None) is not None
      has_blender_dup_suffix = (len(ob_name) > 4 and ob_name[-4] == "." and ob_name[-3:].isdigit())
      parent_is_vis = isinstance(getattr(node.parent, "transform", None), ArgVisibilityNode)
      render_cls_name = type(node.render).__name__
      is_fake_light_render = render_cls_name in {"FakeOmniLightsNode", "FakeSpotLightsNode"}
      if (
        has_parent
        and _ob_local_is_identity(ob)
        and (
          has_blender_dup_suffix
          or (parent_is_vis and is_fake_light_render)
        )
      ):
        ob["_iedm_identity_passthrough"] = True
        if has_blender_dup_suffix and _is_narrow_safe_identity_helper_name(ob_name):
          ob["_iedm_narrow_identity_passthrough"] = True
        if parent_is_vis and is_fake_light_render:
          ob["_iedm_narrow_identity_passthrough"] = True
  except Exception as e:
    print(f"Warning in blender_importer/nodes/core.py: {e}")

  # Combined transform+render fake-light helpers under visibility wrapper
  try:
    if node.blender and node.render is not None and node.transform is not None:
      ob = node.blender
      parent_is_vis = isinstance(getattr(node.parent, "transform", None), ArgVisibilityNode)
      is_anim_tf = isinstance(node.transform, AnimatingNode)
      render_cls_name = type(node.render).__name__
      is_fake_light_render = render_cls_name in {"FakeOmniLightsNode", "FakeSpotLightsNode"}
      if parent_is_vis and is_fake_light_render and (not is_anim_tf) and _ob_local_is_identity(ob):
        ob["_iedm_identity_passthrough"] = True
        ob["_iedm_narrow_identity_passthrough"] = True
  except Exception as e:
    print(f"Warning in blender_importer/nodes/core.py: {e}")

  return _used_shared_parent_fallback


def _hookup_node_animations(node, ctx, vis_actions, used_shared_parent):
  """Connect animation actions and apply the rest transform."""
  _all_collapsed = getattr(node, "_collapsed_transforms", [])
  _has_collapsed_anim = any(isinstance(tf, AnimatingNode) for tf in _all_collapsed)
  if node.transform and (not isinstance(node.transform, ArgVisibilityNode) or _has_collapsed_anim):
    all_transforms = getattr(node, "_collapsed_transforms", [node.transform] if node.transform else [])
    anim_transforms = [tf for tf in all_transforms if isinstance(tf, AnimatingNode)]
    bone_anim_source_nodes = ctx.get("bone_anim_source_nodes", set())
    bone_anim_source_transforms = ctx.get("bone_anim_source_transforms", set())
    skip_object_anim = (
      node in bone_anim_source_nodes
      or any(tf in bone_anim_source_transforms for tf in anim_transforms)
    )

    if anim_transforms and not skip_object_anim:
      if isinstance(node.transform, ArgVisibilityNode):
        actions = []
        for extra_tf in anim_transforms:
          actions.extend(get_actions_for_node(extra_tf))
      else:
        actions = get_actions_for_node(node.transform)
        for extra_tf in anim_transforms:
          if extra_tf is not node.transform:
            actions.extend(get_actions_for_node(extra_tf))
      actions = _merge_actions_by_argument(actions)

      if actions:
        if vis_actions:
          vis_by_arg = {}
          vis_no_arg = []
          for vis_action in vis_actions:
            vis_arg = _get_action_argument(vis_action)
            if vis_arg is None:
              vis_no_arg.append(vis_action)
            elif vis_arg not in vis_by_arg:
              vis_by_arg[vis_arg] = vis_action

          merged_actions = []
          for action in actions:
            vis_action = vis_by_arg.pop(_get_action_argument(action), None)
            if vis_action is not None:
              action = _merge_visibility_action_into_transform_action(action, vis_action, node.blender.name)
            merged_actions.append(action)
          actions = merged_actions + list(vis_by_arg.values()) + vis_no_arg

        node.blender.animation_data_create()
        if len(actions) == 1:
          node.blender.animation_data.action = actions[0]
        else:
          nla_pushed = 0
          for action in actions:
            if _push_action_to_nla(node.blender, action):
              nla_pushed += 1
          if nla_pushed > 0:
            node.blender.animation_data.action = None
          else:
            node.blender.animation_data.action = actions[0]

    # Apply rest transform: for authored-pair collapse use the AnimatingNode's
    # zero_transform_local_matrix rather than the ArgVisibilityNode (identity).
    if isinstance(node.transform, ArgVisibilityNode) and anim_transforms:
      _tf_to_apply = next(
        (tf for tf in anim_transforms if not isinstance(tf, ArgVisibilityNode)),
        anim_transforms[0]
      )
      apply_node_transform(_tf_to_apply, node.blender, used_shared_parent=used_shared_parent)
      if _tf_to_apply is not node.transform:
        try:
          _tf_to_apply._blender_obj = node.blender
        except Exception:
          pass
    else:
      apply_node_transform(node, node.blender, used_shared_parent=used_shared_parent)

    if node.blender.type == "EMPTY":
      distFromScale = node.blender.scale - Vector((1, 1, 1))
      if distFromScale.length < 0.01:
        node.blender.empty_display_size = 0.01
    elif vis_actions:
      node.blender.animation_data_create()
      if len(vis_actions) == 1:
        node.blender.animation_data.action = vis_actions[0]
      else:
        nla_pushed = 0
        for action in vis_actions:
          if _push_action_to_nla(node.blender, action):
            nla_pushed += 1
        if nla_pushed > 0:
          node.blender.animation_data.action = None
        else:
          node.blender.animation_data.action = vis_actions[0]

  elif vis_actions:
    node.blender.animation_data_create()
    if len(vis_actions) == 1:
      node.blender.animation_data.action = vis_actions[0]
    else:
      nla_pushed = 0
      for action in vis_actions:
        if _push_action_to_nla(node.blender, action):
          nla_pushed += 1
      if nla_pushed > 0:
        node.blender.animation_data.action = None
      else:
        node.blender.animation_data.action = vis_actions[0]


def _dump_node_diagnostics(node):
  """Dump all numeric/vector EDM node attributes to Blender IDprops."""
  if not node.blender:
    return

  def _dump_diag(target, prefix="EDM_RAW_"):
    for attr in dir(target):
      if attr.startswith("_") or attr in ("blender", "children", "parent"):
        continue
      try:
        val = getattr(target, attr)
      except Exception:
        continue
      idprop_val = _idprop_diag_value(val)
      if idprop_val is None:
        continue
      try:
        node.blender[prefix + attr] = idprop_val
      except Exception:
        pass

  if node.transform:
    _dump_diag(node.transform, "EDM_TF_")
    if hasattr(node.transform, "base"):
      _dump_diag(node.transform.base, "EDM_BASE_")
  if node.render:
    _dump_diag(node.render, "EDM_RN_")

  if hasattr(node, "_local_bl"):
    node.blender["IEDM_LOCAL_BL_MAT"] = [float(v) for row in node._local_bl for v in row]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_node(node):
  """Processes a single node of the transform graph."""
  if node.parent is None:
    return

  node._is_primary = False

  # Collapse ArgVisibilityNode wrappers onto the child object when possible.
  # Preserve separate objects when the wrapped child carries transform animation.
  if node.render is None and isinstance(node.transform, ArgVisibilityNode):
    _collapsed_has_anim = any(
      isinstance(tf, AnimatingNode)
      for tf in getattr(node, "_collapsed_transforms", [])
    )
    if not _collapsed_has_anim and not _visibility_wrapper_needs_own_object(node):
      tf = getattr(node, "transform", None)
      child_count = len(getattr(node, "children", []) or [])
      _debug_log_event(
        "[iEDM][VISDBG] collapse-to-parent idx={} name={!r} children={} parent_obj={!r}".format(
          getattr(tf, "_graph_idx", None),
          getattr(tf, "name", "") if tf is not None else "",
          child_count,
          getattr(getattr(node.parent, "blender", None), "name", None),
        )
      )
      node.blender = node.parent.blender
      return
    tf = getattr(node, "transform", None)
    child_count = len(getattr(node, "children", []) or [])
    _debug_log_event(
      "[iEDM][VISDBG] keep-own-object idx={} name={!r} children={}".format(
        getattr(tf, "_graph_idx", None),
        getattr(tf, "name", "") if tf is not None else "",
        child_count,
      )
    )

  # EDM bone-control chains map to a single armature object
  ctx = _import_ctx.bone_import_ctx or {}
  arm_obj = ctx.get("armature")
  bone_chain_nodes = ctx.get("bone_chain_nodes", set())
  if arm_obj is not None and node.render is None and node in bone_chain_nodes:
    node.blender = arm_obj
    return

  is_skel = getattr(node, "_is_skeleton", False)

  _create_node_object(node, ctx, is_skel)

  if node.blender and isinstance(node.render, SkinNode):
    _bind_skin_object(node.blender, node.render)

  if not node.blender:
    node.blender = node.parent.blender if node.parent else None
    return

  node._is_primary = True

  _parent_node_object(node, ctx)
  _stamp_node_properties(node, ctx)

  _used_shared_parent_fallback = _apply_render_positioning(node)

  # Visibility source: own transform or collapsed parent wrapper
  _vis_source = None
  if isinstance(node.transform, ArgVisibilityNode):
    _vis_source = node.transform
  elif (
    node.parent
    and isinstance(node.parent.transform, ArgVisibilityNode)
    and node.parent.blender == node.blender
  ):
    _vis_source = node.parent.transform

  vis_actions = get_actions_for_node(_vis_source) if _vis_source else []

  # Material assignment
  material_target = node.render_blender if node.render_blender else node.blender
  if (
    hasattr(node.render, "material")
    and node.render.material
    and hasattr(node.render.material, "blender_material")
    and node.render.material.blender_material
    and material_target
    and hasattr(material_target, "data")
    and material_target.data
  ):
    render_cls_name = type(node.render).__name__
    if render_cls_name not in {"FakeOmniLightsNode", "FakeSpotLightsNode", "FakeALSNode"}:
      material_target.data.materials.append(node.render.material.blender_material)

  _hookup_node_animations(node, ctx, vis_actions, _used_shared_parent_fallback)

  _compact_visibility_identity_intermediate(node)

  _dump_node_diagnostics(node)

  _debug_dump_node_transform(node)

  if isinstance(node.transform, LodNode):
    node._lod_post_children = True


def _process_lod_post_children(node):
  """Apply LodNode properties after children have been processed."""
  if not getattr(node, "_lod_post_children", False):
    return
  assert node.blender.type == "EMPTY"
  node.blender.edm.is_lod_root = True
  try:
    node.blender["IEDM_LOD_LEVELS"] = [
      [float(start), float(end)] for start, end in getattr(node.transform, "level", [])
    ]
  except Exception:
    pass
  for (start, end), child in zip(node.transform.level, node.children):
    child.blender.edm.lod_min_distance = start
    child.blender.edm.lod_max_distance = end
    child.blender.edm.nouse_lod_distance = end > 1e6


def _apply_shadeless(mat):
  """Make a material shadeless by routing Base Color to Emission Color."""
  if not mat.use_nodes:
    return
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  principled = None
  for node in nodes:
    if node.type == 'BSDF_PRINCIPLED':
      principled = node
      break
  if not principled:
    return
  base_color_input = principled.inputs['Base Color']
  if base_color_input.links:
    source_socket = base_color_input.links[0].from_socket
    links.new(source_socket, principled.inputs['Emission Color'])
    principled.inputs['Emission Strength'].default_value = 1.0
