# Fragment: late-stage skin parent binding via bind-rest world position.
# All names resolved via the shared namespace injected by reader.py.


def _resolve_skin_parent_overrides_by_bind_rest():
  """Late-stage deterministic skin parent selection using settled world transforms."""
  ctx = getattr(_import_ctx, "bone_import_ctx", None) or {}
  bone_rest_matrix_by_name = ctx.get("bone_rest_matrix_by_name", {}) or {}
  if not bone_rest_matrix_by_name:
    return

  def _read_bind_target_loc(mesh_ob):
    bone_name = str(mesh_ob.get("_iedm_skin_bind_target_bone", "") or "")
    if bone_name:
      mat = bone_rest_matrix_by_name.get(bone_name)
      if mat is not None:
        try:
          return bone_name, mat.to_translation().copy()
        except Exception:
          pass
    raw_loc = mesh_ob.get("_iedm_skin_bind_target_loc")
    if isinstance(raw_loc, (list, tuple)) and len(raw_loc) == 3:
      try:
        return bone_name, Vector((float(raw_loc[0]), float(raw_loc[1]), float(raw_loc[2])))
      except Exception:
        return bone_name, None
    return bone_name, None

  def _iter_candidates(mesh_ob, render_name, bind_target_name):
    seen = set()

    def _add(obj, reason):
      if obj is None or obj == mesh_ob:
        return
      if getattr(obj, "type", "") == "MESH":
        return
      key = obj.name_full if hasattr(obj, "name_full") else getattr(obj, "name", "")
      if key in seen:
        return
      seen.add(key)
      yield (obj, reason)

    current = getattr(mesh_ob, "parent", None)
    while current is not None:
      for item in _add(current, "ancestor"):
        yield item
      current = getattr(current, "parent", None)

    for obj in list(getattr(bpy.data, "objects", []) or []):
      obj_name = str(getattr(obj, "name", "") or "")
      dbg_tf_name = str(obj.get("_iedm_dbg_tf_name", "") or "")
      if render_name and (obj_name == render_name or dbg_tf_name == render_name):
        for item in _add(obj, "render_name"):
          yield item
      if bind_target_name and (obj_name == bind_target_name or dbg_tf_name == bind_target_name):
        for item in _add(obj, "bind_name"):
          yield item

  bpy.context.view_layer.update()

  for mesh_ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(mesh_ob, "type", "") != "MESH":
      continue
    if not bool(mesh_ob.get("_iedm_skin_parent_override")):
      continue
    if str(mesh_ob.get("_iedm_src_render_cls", "") or "") != "SkinNode":
      continue

    render_name = str(mesh_ob.get("_iedm_src_render_name", "") or mesh_ob.get("_iedm_dbg_skin_render_name", "") or "")
    bind_target_name, bind_target_loc = _read_bind_target_loc(mesh_ob)

    rows = []
    for obj, reason in _iter_candidates(mesh_ob, render_name, bind_target_name):
      try:
        world_loc = obj.matrix_world.to_translation().copy()
      except Exception:
        world_loc = None
      distance = None
      if bind_target_loc is not None and world_loc is not None:
        try:
          distance = (world_loc - bind_target_loc).length
        except Exception:
          distance = None
      rows.append((distance if distance is not None else float("inf"), obj.name, obj, reason, world_loc, distance))

    rows.sort(key=lambda item: (item[0], item[1]))

    current_parent = getattr(mesh_ob, "parent", None)
    current_distance = None
    if bind_target_loc is not None and current_parent is not None:
      try:
        current_distance = (current_parent.matrix_world.to_translation() - bind_target_loc).length
      except Exception:
        current_distance = None

    if not rows:
      continue

    best_obj = rows[0][2]
    best_distance = rows[0][5]
    if (
      bind_target_loc is None
      or best_obj is None
      or best_distance is None
      or best_obj == current_parent
    ):
      continue

    if current_distance is not None and best_distance + 1e-5 >= current_distance:
      continue

    try:
      mesh_ob.parent = best_obj
      mesh_ob.matrix_parent_inverse = Matrix.Identity(4)
      mesh_ob.matrix_basis = Matrix.Identity(4)
      mesh_ob["_iedm_dbg_skin_helper_name"] = getattr(best_obj, "name", "")
      mesh_ob["_iedm_skin_parent_override_late"] = True
    except Exception as e:
      print(f"Warning in _resolve_skin_parent_overrides_by_bind_rest: {e}")
