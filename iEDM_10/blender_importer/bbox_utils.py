# Fragment: EDM bounding box and special-box creation utilities.
# All names resolved via the shared namespace injected by reader.py.


def _transform_aabb_to_blender(fix, bmin, bmax):
  """Transform an EDM AABB by an orthogonal matrix into Blender space.

  Applies *fix* to all 8 corners of the axis-aligned box and returns the new
  (bmin, bmax) in the target coordinate frame. Required because a pure rotation
  remaps which axis is min/max.
  """
  corners = [
    fix @ Vector((x, y, z))
    for x in (bmin.x, bmax.x)
    for y in (bmin.y, bmax.y)
    for z in (bmin.z, bmax.z)
  ]
  return (
    Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners))),
    Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners))),
  )


def create_bounding_box_from_root(root, coord_fix=None):
  """Create an empty cube representing the EDM bounding box.

  *coord_fix* is an optional Matrix applied to the raw EDM AABB corners before
  placement (use _ROOT_BASIS_FIX when a scene root basis object carries the
  Y→Z conversion for the rest of the scene).
  """
  if not hasattr(root, "boundingBoxMin"):
    return None
  bmin = vector_to_blender(root.boundingBoxMin)
  bmax = vector_to_blender(root.boundingBoxMax)
  if coord_fix is not None:
    bmin, bmax = _transform_aabb_to_blender(coord_fix, bmin, bmax)
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
    _log.warn("bounding_box raw_bbox prop: {}".format(e), exc=e)
  _set_official_special_type(ob, "BOUNDING_BOX")
  bpy.context.collection.objects.link(ob)
  return ob


def _has_special_box(special_type):
  """Return True if a scene object already represents the given special box type."""
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
    _log.warn("cache scene bbox: {}".format(e), exc=e)
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
    _log.warn("cache scene user_box: {}".format(e), exc=e)
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
    _log.warn("cache scene light_box: {}".format(e), exc=e)


def _create_special_box(name, special_type, min_vec_edm, max_vec_edm, raw_prop_name, coord_fix=None):
  min_vec = vector_to_blender(min_vec_edm)
  max_vec = vector_to_blender(max_vec_edm)
  if coord_fix is not None:
    min_vec, max_vec = _transform_aabb_to_blender(coord_fix, min_vec, max_vec)
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
    _log.warn("special box raw prop '{}': {}".format(raw_prop_name, e), exc=e)
  _set_official_special_type(ob, special_type)
  return ob


def _create_user_box_from_root(root, coord_fix=None):
  """Create an empty cube representing the EDM user box when distinct."""
  user_box = getattr(root, "user_box", None)
  if user_box is None:
    return None
  min_vec_edm = Vector(user_box[0])
  max_vec_edm = Vector(user_box[1])
  if any(math.isinf(v) for v in min_vec_edm) or any(math.isinf(v) for v in max_vec_edm):
    return None
  return _create_special_box("User_Box", "USER_BOX", min_vec_edm, max_vec_edm, "_iedm_raw_user_box", coord_fix=coord_fix)


def _create_light_box_from_root(root, coord_fix=None):
  """Create an empty cube representing the EDM light box when present."""
  light_box = getattr(root, "light_box", None)
  if light_box is None:
    return None
  min_vec_edm = Vector(light_box[0])
  max_vec_edm = Vector(light_box[1])
  if any(math.isinf(v) for v in min_vec_edm) or any(math.isinf(v) for v in max_vec_edm):
    return None
  return _create_special_box("Light_Box", "LIGHT_BOX", min_vec_edm, max_vec_edm, "_iedm_raw_light_box", coord_fix=coord_fix)
