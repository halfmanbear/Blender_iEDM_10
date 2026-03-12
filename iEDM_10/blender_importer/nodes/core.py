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


def process_node(node):
  """Processes a single node of the transform graph"""
  _used_shared_parent_fallback = False
  # Root node has no processing
  if node.parent is None:
    return

  node._is_primary = False

  # Usually collapse visibility wrappers into the child object (tighter export
  # shape), but preserve them when the wrapped child carries transform animation.
  # In that case, keeping a separate wrapper avoids the exporter's one-action-per-
  # object limitation (visibility and transform args can differ).
  if node.render is None and isinstance(node.transform, ArgVisibilityNode):
    if not _visibility_wrapper_needs_own_object(node):
      node.blender = node.parent.blender
      return

  # EDM bone-control chains are represented by a single imported armature.
  ctx = _import_ctx.bone_import_ctx or {}
  arm_obj = ctx.get("armature")
  bone_chain_nodes = ctx.get("bone_chain_nodes", set())
  if arm_obj is not None and node.render is None and node in bone_chain_nodes:
    node.blender = arm_obj
    return

  is_skel = getattr(node, "_is_skeleton", False)

  # --- Create Blender object for this node ---
  node.render_blender = None
  if node.render:
    render_name = getattr(node.render, "name", "") or ""
    name_unknown = getattr(node.render, "name_unknown", False)
    if isinstance(node.render, NumberNode) and render_name.endswith("_Number"):
      name_unknown = True
    is_connector_render = isinstance(node.render, Connector)

    final_name = render_name
    parent_is_arg_wrapper = isinstance(getattr(getattr(node, "parent", None), "transform", None), (ArgVisibilityNode, ArgAnimationNode))

    if _is_generic_render_name(final_name) or name_unknown:
      # Avoid consuming semantic base names (e.g. Aileron_L / Flap_L) for generic
      # connector objects. Visibility wrappers later export as `v_ + obj.name`, so
      # if a connector grabs the base first Blender suffixes the wrapper (.001/.002).
      # Connector object names are not semantically important for parity here.
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
      # Fix: Blender deduplicates names within a session, so if a wrapper
      # TransformNode Empty was already created with the same name as this
      # Connector (the normal case in original 3ds Max EDMs), the Connector
      # object gets auto-suffixed to ".001"/".002"/etc.  The unpatched exporter
      # uses obj.name as the Connector ID, so ".001" leaks into the binary and
      # breaks all external attach-point references.  Resolve by renaming the
      # conflicting Empty (which is only needed as a hierarchy parent; its own
      # Blender name is irrelevant for export) so the Connector object can
      # reclaim the correct EDM name.
      try:
        _connector_edm_name = str(getattr(node.render, "name", "") or "")
        if _connector_edm_name and node.blender.name != _connector_edm_name:
          _conflict = bpy.data.objects.get(_connector_edm_name)
          if _conflict is not None and _conflict is not node.blender:
            if not _is_connector_object(_conflict):
              # Rename the conflicting object (wrapper Empty) out of the way.
              # Append "_tf" to signal it's a transform-hierarchy-only placeholder.
              # With pre-naming active, this path only fires if the TF wrapper was
              # not pre-named (e.g. a connector under an animating parent).
              _conflict.name = _connector_edm_name + "_tf"
              node.blender.name = _connector_edm_name
            # else: another connector already owns the base name — leave both as-is;
            # the export uses _iedm_connector_name, not the Blender object name.
          else:
            node.blender.name = _connector_edm_name
        if _connector_edm_name:
          node.blender["_iedm_connector_name"] = _connector_edm_name
      except Exception as e:
        print(f"Warning in blender_importer\nodes\core.py: {e}")
      # Store the original wrapper TransformNode name so the export patch can
      # produce TransformNode "BANO_0" instead of "Connector Transform" etc.
      # In original EDMs the wrapper transform has the connector's own name.
      try:
        node.blender["_iedm_connector_apply_xfix"] = False
        parent_tf = getattr(getattr(node, "parent", None), "transform", None)
        # Connector render nodes are often parented under a static wrapper
        # TransformNode. For parity, take the wrapper matrix/name as the
        # connector control transform source and bypass wrapper transform export.
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
            except Exception as e:
              print(f"Warning in blender_importer\nodes\core.py: {e}")
          if _ptf_name == "Connector Transform":
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
              print(f"Warning in blender_importer\nodes\core.py: {e}")
          if _ctf_name == "Connector Transform":
            node.blender["_iedm_connector_apply_xfix"] = True
          if _ctf_name and _ctf_name != "Connector Transform":
            node.blender["_iedm_connector_tf_name"] = _ctf_name
        if "_iedm_connector_tf_matrix" not in node.blender and getattr(node, "_local_bl", None) is not None:
          try:
            _ml = Matrix(node._local_bl)
            node.blender["_iedm_connector_tf_matrix"] = [float(v) for row in _ml for v in row]
          except Exception as e:
            print(f"Warning in blender_importer\nodes\core.py: {e}")
        if "_iedm_connector_tf_name" not in node.blender:
          _fallback_name = str(getattr(getattr(node, "render", None), "name", "") or "")
          if _fallback_name:
            _fallback_name = _SUFFIX_RE.sub("", _fallback_name)
            node.blender["_iedm_connector_tf_name"] = _fallback_name
      except Exception as e:
        print(f"Warning in blender_importer\nodes\core.py: {e}")
    elif isinstance(node.render, (RenderNode, ShellNode, NumberNode, SkinNode)):
      node.blender = create_object(node.render)
    elif isinstance(node.render, SegmentsNode):
      node.blender = create_segments(node.render)
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
    # Pre-name TF wrappers that are sole parents of a Connector with "_tf" suffix,
    # so duplicate chains get _tf/_tf.001/... rather than cascading connector renames.
    _node_children_skel = getattr(node, "children", []) or []
    if (
      not isinstance(node.transform, AnimatingNode)
      and len(_node_children_skel) == 1
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
    ob = bpy.data.objects.new(tf_name, None)
    ob.empty_display_size = 0.1
    bpy.context.collection.objects.link(ob)
    node.blender = ob

  if node.blender and isinstance(node.render, SkinNode):
    _bind_skin_object(node.blender, node.render)

  if not node.blender:
    node.blender = node.parent.blender if node.parent else None
    return

  node._is_primary = True

  # --- (2) Parent in a “parity-safe” way: identity parent inverse ---
  if not node.blender.get("_iedm_skin_parent_override"):
    if node.parent and node.parent.blender and node.parent.blender != node.blender:
      node.blender.parent = node.parent.blender
      node.blender.matrix_parent_inverse = Matrix.Identity(4)

  # Preserve mapping from EDM transform node -> created blender object
  if node.transform and node.blender:
    try:
      node.transform._blender_obj = node.blender
    except Exception as e:
      print(f"Warning in blender_importer\nodes\core.py: {e}")
    if isinstance(node.transform, ArgVisibilityNode):
      try:
        vis_alias = getattr(node.transform, "name", "") or ""
        # Store the FULL original EDM name (do not strip "v_").
        # The visibility export patch (_extract_visibility_animation) uses the
        # alias as-is, so "v_Gas_Tanks" → exports "v_Gas_Tanks" and plain
        # "Dummy001" → exports "Dummy001" (no extra "v_" prepended).
        # Only strip ar_/al_/as_ prefixes which are never part of the intended
        # visibility node name in a well-formed EDM.
        for _pfx in ("ar_", "al_", "as_"):
          if vis_alias.startswith(_pfx):
            vis_alias = vis_alias[len(_pfx):]
            break
        if vis_alias:
          node.blender["_iedm_vis_export_name"] = vis_alias
      except Exception as e:
        print(f"Warning in blender_importer\nodes\core.py: {e}")
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
        print(f"Warning in blender_importer\nodes\core.py: {e}")
      try:
        if node.render is None and getattr(node.blender, "type", None) == 'EMPTY':
          node.blender["_iedm_vis_passthrough"] = True
      except Exception as e:
        print(f"Warning in blender_importer\nodes\core.py: {e}")
    else:
      # Mark importer-created static helper empties that only contribute an
      # identity transform and exist due Blender duplicate-name splitting
      # (`Foo.001`, `Foo.002`, ...). The exporter otherwise emits no-op
      # TransformNodes for these helpers.
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
        # Also allow a narrow fake-light helper case with no Blender suffix:
        # v_* -> Omni_r001 (identity EMPTY) -> LightNode
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
        print(f"Warning in blender_importer\nodes\core.py: {e}")

  # Store the original EDM animation-node name so the export name-passthrough
  # patch can emit the plain name (e.g. "Dummy001", "pilot_SU_helmet_glass001")
  # instead of the prefixed name emitted by io_scene_edm ("ar_Dummy001").
  # Applies to all animated nodes — both pure empties AND animated mesh objects
  # — when the original EDM name has no al_/ar_/as_ prefix.  The proxy only
  # fires when the property is present, so prefixed names in the original are
  # never mis-renamed.
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
      print(f"Warning in blender_importer\nodes\core.py: {e}")
  # Preserve raw ArgAnimation payload on imported objects so problematic nodes
  # can be inspected directly in Blender without re-reading the EDM.
  if isinstance(node.transform, ArgAnimationNode) and node.blender is not None:
    try:
      raw = {}
      base = getattr(node.transform, "base", None)
      if base is not None:
        try:
          bm = Matrix(base.matrix)
          raw["base_matrix"] = [float(v) for row in bm for v in row]
        except Exception as e:
          print(f"Warning in blender_importer\nodes\core.py: {e}")
        try:
          raw["base_position"] = [float(x) for x in base.position[:3]]
        except Exception as e:
          print(f"Warning in blender_importer\nodes\core.py: {e}")
        try:
          q1 = base.quat_1 if hasattr(base.quat_1, "__iter__") else Quaternion(base.quat_1)
          raw["base_quat_1"] = [float(x) for x in q1]
        except Exception as e:
          print(f"Warning in blender_importer\nodes\core.py: {e}")
        try:
          q2 = base.quat_2 if hasattr(base.quat_2, "__iter__") else Quaternion(base.quat_2)
          raw["base_quat_2"] = [float(x) for x in q2]
        except Exception as e:
          print(f"Warning in blender_importer\nodes\core.py: {e}")
        try:
          raw["base_scale"] = [float(x) for x in base.scale[:3]]
        except Exception as e:
          print(f"Warning in blender_importer\nodes\core.py: {e}")

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
        zmat = getattr(node.transform, "zero_transform_local_matrix", None)
        if zmat is not None:
          raw["zero_transform_local_matrix"] = [float(v) for row in Matrix(zmat) for v in row]
      except Exception as e:
        print(f"Warning in blender_importer\nodes\core.py: {e}")

      if raw:
        node.blender["_iedm_raw_arganim_payload"] = json.dumps(raw, separators=(",", ":"))
    except Exception as e:
      print(f"Warning in blender_importer\nodes\core.py: {e}")

  # Temporary importer debug stamp: helps correlate Blender objects back to source
  # EDM transform/render classes while diagnosing cross-asset basis issues.
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
        print(f"Warning in blender_importer\nodes\core.py: {e}")
  except Exception as e:
    print(f"Warning in blender_importer\nodes\core.py: {e}")

  # --- Render-only positioning & v10 basis handling ---
  _is_render_only_positioning = (
    node.render and node.blender and node.blender.type == "MESH"
    and (not node.transform or isinstance(node.transform, ArgVisibilityNode))
  )

  if _is_render_only_positioning:
    shared_parent = getattr(node.render, "shared_parent", None)
    parent_transform = getattr(getattr(node, "parent", None), "transform", None)
    parent_is_real_transform = (
      parent_transform is not None
      and not isinstance(parent_transform, ArgVisibilityNode)
    )
    _used_shared_parent_fallback = False
    applied_local_bl = False
    has_parent_obj = bool(node.parent and node.parent.blender)
    shared_parent_obj = getattr(shared_parent, "_blender_obj", None) if shared_parent is not None else None

    # Owner-encoded split chunks can be authored in a shared control-space.
    # If we have no explicit owner wrapper to compensate against, fall back to
    # the shared parent transform directly.
    if shared_parent is not None and not has_parent_obj and not parent_is_real_transform:
      apply_node_transform(shared_parent, node.blender, used_shared_parent=True)
      _used_shared_parent_fallback = True

    if shared_parent is not None and has_parent_obj:
      src = shared_parent_obj
      dst = node.parent.blender
      if src is not None:
        try:
          node.blender.matrix_local = dst.matrix_world.inverted() @ src.matrix_world
        except Exception as e:
          print(f"Warning in blender_importer\nodes\core.py: {e}")
      else:
        # If the shared parent transform was collapsed (no Blender object was created
        # for it), apply that EDM transform directly to the render-only mesh object so
        # the exporter control node does not become an identity wrapper under v_*.
        try:
          if (
            not parent_is_real_transform
            and getattr(shared_parent, "category", None) is NodeCategory.transform
          ):
            apply_node_transform(shared_parent, node.blender, used_shared_parent=True)
            _used_shared_parent_fallback = True
        except Exception as e:
          print(f"Warning in blender_importer\nodes\core.py: {e}")
    local_bl = getattr(node, "_local_bl", None)
    if local_bl is not None and not _used_shared_parent_fallback:
      try:
        matrix_to_apply = local_bl.copy() if hasattr(local_bl, "copy") else local_bl
        node.blender.matrix_basis = matrix_to_apply
        applied_local_bl = True
      except Exception as e:
        print(f"Warning in blender_importer\nnodes\core.py: {e}")

    if _import_ctx.edm_version >= 10:
      # If a visibility wrapper is acting as a control node, keep mesh aligned.
      if (
        node.parent and isinstance(node.parent.transform, ArgVisibilityNode) and node.parent.blender
        and not _used_shared_parent_fallback
      ):
        parent_loc = node.parent.blender.matrix_basis.decompose()[0]
        if not applied_local_bl:
          _offset_mesh_world(node.blender, parent_loc)

    # Optional pivot recentering (kept as you had it)
    if (
      _import_ctx.mesh_origin_mode == "APPROX"
      and node.parent and node.parent.blender
      and not isinstance(node.parent.transform, (ArgVisibilityNode, AnimatingNode))
    ):
      if node.blender.location.length < 1e-9:
        _recenter_mesh_object_to_geometry(node.blender)
        if _import_ctx.edm_version >= 10:
          node.blender.location = (node.parent.blender.matrix_basis @ node.blender.location.to_4d()).to_3d()

  # Mark render-only Blender-suffixed duplicate helpers that ended up as
  # identity transforms after importer positioning. Exporter can skip their
  # no-op object transform wrapper safely.
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
    print(f"Warning in blender_importer\nodes\core.py: {e}")

  # Also handle combined transform+render fake-light helper nodes that are
  # identity under a visibility wrapper (e.g. v_* -> Omni_r001 -> Fake Light Transform).
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
    print(f"Warning in blender_importer\nodes\core.py: {e}")

  # --- Visibility animation hookup ---
  _vis_source = None
  if isinstance(node.transform, ArgVisibilityNode):
    _vis_source = node.transform
  elif (
    node.parent
    and isinstance(node.parent.transform, ArgVisibilityNode)
    and node.parent.blender == node.blender
  ):
    # Only inherit visibility from parent wrapper if that wrapper was collapsed
    # onto this same Blender object.
    _vis_source = node.parent.transform

  vis_actions = get_actions_for_node(_vis_source) if _vis_source else []

  # --- Materials ---
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

  # --- Apply transform animation / base transforms (non-visibility transforms only) ---
  if node.transform and not isinstance(node.transform, ArgVisibilityNode):
    all_transforms = getattr(node, "_collapsed_transforms", [node.transform] if node.transform else [])
    anim_transforms = [tf for tf in all_transforms if isinstance(tf, AnimatingNode)]
    bone_anim_source_nodes = ctx.get("bone_anim_source_nodes", set())
    bone_anim_source_transforms = ctx.get("bone_anim_source_transforms", set())
    skip_object_anim = (
      node in bone_anim_source_nodes
      or any(tf in bone_anim_source_transforms for tf in anim_transforms)
    )

    if anim_transforms and not skip_object_anim:
      actions = get_actions_for_node(node.transform)
      for extra_tf in anim_transforms:
        if extra_tf is not node.transform:
          actions.extend(get_actions_for_node(extra_tf))
      actions = _merge_actions_by_argument(actions)

      if actions:
        if vis_actions and len(actions) == 1:
          actions = list(actions)
          actions[0] = _merge_visibility_action_into_transform_action(actions[0], vis_actions[0], node.blender.name)
        node.blender.animation_data_create()
        if len(actions) == 1:
          node.blender.animation_data.action = actions[0]
        else:
          nla_pushed = 0
          for action in actions:
            if _push_action_to_nla(node.blender, action):
              nla_pushed += 1
          # Keep a primary active action on non-armature objects; importer
          # export hook reads extra strip actions explicitly.
          if getattr(node.blender, "type", "") == "ARMATURE":
            node.blender.animation_data.action = None if nla_pushed > 0 else actions[0]
          else:
            node.blender.animation_data.action = actions[0]

    apply_node_transform(node, node.blender, used_shared_parent=_used_shared_parent_fallback)

    if node.blender.type == "EMPTY":
      distFromScale = node.blender.scale - Vector((1, 1, 1))
      if distFromScale.length < 0.01:
          node.blender.empty_display_size = 0.01
    elif vis_actions:
      node.blender.animation_data_create()
      node.blender.animation_data.action = vis_actions[0]

  elif vis_actions:
    node.blender.animation_data_create()
    node.blender.animation_data.action = vis_actions[0]

  _compact_visibility_identity_intermediate(node)

  # Diagnostic: Dump all numeric/vector attributes of the EDM node to Blender
  # This allows the user to see what is being ignored by the parser.
  if node.blender:
    def _dump_diag(target, prefix="EDM_RAW_"):
      for attr in dir(target):
        if attr.startswith("_") or attr == "blender" or attr == "children" or attr == "parent":
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
    
    # Store the importer's calculated Blender local matrix
    if hasattr(node, "_local_bl"):
      node.blender["IEDM_LOCAL_BL_MAT"] = [float(v) for row in node._local_bl for v in row]

  _debug_dump_node_transform(node)

  if isinstance(node.transform, LodNode):
    node._lod_post_children = True


def _process_lod_post_children(node):
  """Apply LodNode properties after children have been processed."""
  if not getattr(node, "_lod_post_children", False):
    return
  assert node.blender.type == "EMPTY"
  node.blender.edm.is_lod_root = True
  for (start, end), child in zip(node.transform.level, node.children):
    child.blender.edm.lod_min_distance = start
    child.blender.edm.lod_max_distance = end
    child.blender.edm.nouse_lod_distance = end > 1e6


def _apply_shadeless(mat):
  """Make a material 'shadeless' by routing the Base Color texture to Emission Color
  on the Principled BSDF node, so it renders without lighting."""
  if not mat.use_nodes:
    return
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  # Find the Principled BSDF node
  principled = None
  for node in nodes:
    if node.type == 'BSDF_PRINCIPLED':
      principled = node
      break
  if not principled:
    return
  # Find what's connected to Base Color and duplicate that link to Emission Color
  base_color_input = principled.inputs['Base Color']
  if base_color_input.links:
    source_socket = base_color_input.links[0].from_socket
    links.new(source_socket, principled.inputs['Emission Color'])
    principled.inputs['Emission Strength'].default_value = 1.0


