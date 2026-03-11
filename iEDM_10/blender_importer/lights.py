def _extract_light_property(prop_value):
  """Return (argument, keyframes, static_value) for a light property payload."""
  argument = -1
  keys = None
  static_value = None

  if hasattr(prop_value, "keys") and hasattr(prop_value, "argument"):
    try:
      argument = int(prop_value.argument)
    except Exception:
      argument = -1
    keys = list(getattr(prop_value, "keys", []) or [])
    if keys:
      static_value = getattr(keys[0], "value", None)
  elif isinstance(prop_value, list):
    keys = prop_value
    if keys:
      static_value = getattr(keys[0], "value", None)
  else:
    static_value = prop_value

  return argument, keys, static_value


def _get_prop_any(props, *names):
  """Fetch first present key from a light/property dict."""
  if not props:
    return None
  for name in names:
    if name in props:
      return props.get(name)
  return None


def _to_float(value, default=0.0):
  try:
    return float(value)
  except Exception:
    return float(default)


def _to_vec3(value, default=(1.0, 1.0, 1.0)):
  if value is None:
    return default
  try:
    return (float(value[0]), float(value[1]), float(value[2]))
  except Exception:
    return default


def _edm_light_brightness_to_blender_energy(brightness_value, light_type):
  """
  Invert the official exporter conversion path from Blender light energy to EDM
  Brightness property, so importing then exporting preserves values.
  """
  brightness = max(0.0, _to_float(brightness_value, 0.0))
  if brightness <= 0.0:
    return 0.0

  base = max(0.0, brightness) ** (1.0 / _BLENDER_LAMP_WEAK_COEFFICIENT)
  denom = _PBR_WATTS_TO_LUMENS * _BLENDER_LAMP_ENERGY_COEFFICIENT
  if denom <= 0.0:
    return 0.0
  energy = base / denom
  if light_type != 'SUN':
    energy *= (4.0 * math.pi)
  return max(0.0, energy)


def _set_edmprop(obj, prop_name, value):
  if not hasattr(obj, "EDMProps"):
    return
  edm_props = obj.EDMProps
  if hasattr(edm_props, prop_name):
    try:
      # Clamp integers to 32-bit signed limit for Blender's IntProperty
      if isinstance(value, int):
        value = min(2147483647, max(-2147483648, value))
      setattr(edm_props, prop_name, value)
    except (OverflowError, ValueError):
      pass
    except Exception as e:
      print(f"Warning in blender_importer\lights.py: {e}")


def _new_fcurve(action, data_path, array_index=None):
  idx = 0 if array_index is None else int(array_index)
  existing = action.fcurves.find(data_path, index=idx)
  if existing is not None:
    return existing
  if array_index is None:
    return action.fcurves.new(data_path=data_path)
  return action.fcurves.new(data_path=data_path, index=int(array_index))


def _add_light_keyframes(action, data_path, keys, value_fn, array_index=None):
  curve = _new_fcurve(action, data_path, array_index)
  for framedata in keys:
    frame = _anim_frame_to_scene_frame(getattr(framedata, "frame", 0.0))
    value = value_fn(getattr(framedata, "value", 0.0))
    key = curve.keyframe_points.insert(frame, float(value), options={'FAST'})
    key.interpolation = 'LINEAR'
  return curve


def _import_light_properties(node, obj, light_data, light_type):
  props = getattr(node, "lightProps", None) or {}
  if not props:
    return

  color_arg, color_keys, color_static = _extract_light_property(_get_prop_any(props, "Color", "color"))
  bright_arg, bright_keys, bright_static = _extract_light_property(_get_prop_any(props, "Brightness", "brightness"))
  dist_arg, dist_keys, dist_static = _extract_light_property(_get_prop_any(props, "Distance", "distance"))
  phi_arg, phi_keys, phi_static = _extract_light_property(_get_prop_any(props, "Phi", "phi"))
  theta_arg, theta_keys, theta_static = _extract_light_property(_get_prop_any(props, "Theta", "theta"))
  spec_arg, spec_keys, spec_static = _extract_light_property(_get_prop_any(props, "Specular", "specularAmount", "specular"))
  soft_arg, soft_keys, soft_static = _extract_light_property(_get_prop_any(props, "", "softness", "Softness"))
  vol_radius_arg, vol_radius_keys, vol_radius_static = _extract_light_property(_get_prop_any(props, "VolumeRadiusFactor", "radiusFactor"))
  vol_density_arg, vol_density_keys, vol_density_static = _extract_light_property(_get_prop_any(props, "VolumeDensityFactor", "densityFactor"))
  vol_near_arg, vol_near_keys, vol_near_static = _extract_light_property(_get_prop_any(props, "VolumeNearDistance", "nearDistance"))
  vol_type_arg, vol_type_keys, vol_type_static = _extract_light_property(_get_prop_any(props, "VolumeType", "volumeType"))

  if color_static is not None:
    light_data.color = _to_vec3(color_static, light_data.color)
  if bright_static is not None:
    light_data.energy = _edm_light_brightness_to_blender_energy(bright_static, light_type)
  if dist_static is not None:
    light_data.use_custom_distance = True
    light_data.cutoff_distance = max(0.0, _to_float(dist_static, 0.0))
  if spec_static is not None and hasattr(light_data, "specular_factor"):
    light_data.specular_factor = max(0.0, _to_float(spec_static, light_data.specular_factor))

  if light_type == 'SPOT':
    if phi_static is not None:
      light_data.spot_size = min(math.radians(170.0), max(0.0, _to_float(phi_static, light_data.spot_size)))
    if theta_static is not None:
      light_data.spot_blend = min(1.0, max(0.0, _to_float(theta_static, light_data.spot_blend)))

  # Map EDM animated-property arguments into official exporter EDMProps fields.
  if color_arg >= 0:
    _set_edmprop(obj, "LIGHT_COLOR_ARG", int(color_arg))
  if bright_arg >= 0:
    _set_edmprop(obj, "LIGHT_POWER_ARG", int(bright_arg))
  if dist_arg >= 0:
    _set_edmprop(obj, "LIGHT_DISTANCE_ARG", int(dist_arg))
  if spec_arg >= 0:
    _set_edmprop(obj, "LIGHT_SPECULAR_ARG", int(spec_arg))
  if soft_static is not None:
    _set_edmprop(obj, "LIGHT_SOFTNESS", max(0.0, _to_float(soft_static, 0.0)))
  if vol_radius_static is not None:
    _set_edmprop(obj, "LIGHT_VOLUME_RADIUS_FACTOR", min(1.0, max(0.0, _to_float(vol_radius_static, 0.0))))
  if vol_density_static is not None:
    _set_edmprop(obj, "LIGHT_VOLUME_DENSITY_FACTOR", min(1.0, max(0.0, _to_float(vol_density_static, 0.0))))
  if vol_near_static is not None:
    _set_edmprop(obj, "LIGHT_VOLUME_NEAR_DISTANCE", max(0.0, _to_float(vol_near_static, 0.0)))
  if vol_type_static is not None:
    try:
      vol_type_int = int(round(_to_float(vol_type_static, 4)))
      volume_types = {0: "LANDING", 1: "NAV", 2: "TAXI", 3: "BANO"}
      _set_edmprop(obj, "LIGHT_VOLUME_TYPE", volume_types.get(vol_type_int, "NONE"))
    except Exception as e:
      print(f"Warning in blender_importer\lights.py: {e}")
  if light_type == 'SPOT':
    spot_arg = phi_arg if phi_arg >= 0 else theta_arg
    if spot_arg >= 0:
      _set_edmprop(obj, "LIGHT_SPOT_SHAPE_ARG", int(spot_arg))

  # Recreate light-data animation curves for exporter parity.
  has_anim = any(keys for keys in (color_keys, bright_keys, dist_keys, phi_keys, theta_keys, spec_keys))
  if not has_anim:
    return

  action = bpy.data.actions.new("Light_{}".format(obj.name))
  if hasattr(action, "argument"):
    first_arg = next((a for a in (color_arg, bright_arg, dist_arg, phi_arg, theta_arg, spec_arg, soft_arg) if a >= 0), -1)
    if first_arg >= 0:
      action.argument = int(first_arg)

  if color_keys:
    for idx in range(3):
      _add_light_keyframes(action, "color", color_keys, lambda v, c=idx: _to_vec3(v)[c], array_index=idx)
  if bright_keys:
    _add_light_keyframes(
      action,
      "energy",
      bright_keys,
      lambda v: _edm_light_brightness_to_blender_energy(v, light_type),
    )
  if dist_keys:
    light_data.use_custom_distance = True
    _add_light_keyframes(action, "cutoff_distance", dist_keys, lambda v: max(0.0, _to_float(v, 0.0)))
  if spec_keys and hasattr(light_data, "specular_factor"):
    _add_light_keyframes(action, "specular_factor", spec_keys, lambda v: max(0.0, _to_float(v, 0.0)))
  if light_type == 'SPOT':
    if phi_keys:
      _add_light_keyframes(
        action,
        "spot_size",
        phi_keys,
        lambda v: min(math.radians(170.0), max(0.0, _to_float(v, 0.0))),
      )
    if theta_keys:
      _add_light_keyframes(
        action,
        "spot_blend",
        theta_keys,
        lambda v: min(1.0, max(0.0, _to_float(v, 0.0))),
      )

  anim_data = light_data.animation_data_create()
  anim_data.action = action


def create_lamp(node):
  """Creates a blender lamp from an edm renderable LampNode"""
  light_props = getattr(node, "lightProps", None) or {}
  has_spot_keys = any(k in light_props for k in ("Phi", "Theta", "phi", "theta"))
  light_type = 'SPOT' if has_spot_keys else 'POINT'
  light_data = bpy.data.lights.new(name=node.name, type=light_type)
  obj = bpy.data.objects.new(name=node.name, object_data=light_data)
  _import_light_properties(node, obj, light_data, light_type)
  bpy.context.collection.objects.link(obj)
  return obj


def _create_fake_light_material(obj_name, kind="fake_omni"):
  """Create a Blender material with the correct EDM node group for fake lights.

  Args:
    obj_name: Name for the material (will be used as-is).
    kind: "fake_omni" or "fake_spot" — determines which node group to create.

  Returns:
    A bpy.types.Material with the correct EDM node group attached, or None.
  """
  bridge = _ensure_official_material_bridge()
  if not bridge.get("available"):
    return None

  names = bridge.get("names", {})
  official_name = names.get(kind, names.get("fake_omni", "EDM_Fake_Omni_Material"))

  mat = bpy.data.materials.new(name=obj_name)
  mat.use_nodes = True
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links

  # Clear default nodes
  for node in list(nodes):
    nodes.remove(node)

  # Create Material Output
  output_node = nodes.new('ShaderNodeOutputMaterial')
  output_node.location = (400, 0)

  # Create the official EDM group node
  group_node = _create_official_group_node(nodes, bridge, kind, official_name)
  if group_node is None:
    # Last resort: create a bare ShaderNodeGroup with a named node tree
    try:
      group_node = nodes.new("ShaderNodeGroup")
      group_node.node_tree = bpy.data.node_groups.get(official_name)
      if group_node.node_tree is None:
        group_node.node_tree = bpy.data.node_groups.new(official_name, "ShaderNodeTree")
    except Exception:
      return mat  # Return material without group — better than nothing

  group_node.location = (0, 0)

  # Link group output to material output (if there's an output socket)
  if group_node.outputs:
    try:
      links.new(group_node.outputs[0], output_node.inputs['Surface'])
    except Exception as e:
      print(f"Warning in blender_importer\lights.py: {e}")

  return mat


def _create_fake_light_mesh(name, positions):
  """Create a mesh with one vertex per light position (already in Blender coords)."""
  mesh = bpy.data.meshes.new(name)
  mesh.vertices.add(len(positions))
  for i, pos in enumerate(positions):
    mesh.vertices[i].co = pos
  mesh.update()
  return mesh


def _unpack_uv_double(dval):
  """Decode a double that packs two floats (UV pair) via reinterpretation."""
  import struct
  raw = struct.pack('<d', dval)
  return struct.unpack('<2f', raw)


def create_fake_omni_lights(node):
  """Create a mesh object for FakeOmniLightsNode with one vertex per light."""
  name = node.name or "FakeOmniLights"

  # Each entry: 6 doubles = [x, y, z, size, uv_lb_packed, uv_rt_packed]
  # The last two doubles each pack two floats (UV coordinates).
  positions = []
  sizes = []
  uv_lb = None
  uv_rt = None
  for entry in node.data:
    pos_edm = (entry[0], entry[1], entry[2])
    positions.append(vector_to_blender(pos_edm))
    sizes.append(entry[3])
    if uv_lb is None:
      uv_lb = _unpack_uv_double(entry[4])
      uv_rt = _unpack_uv_double(entry[5])

  if not positions:
    # No light entries — create an empty placeholder
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "SPHERE"
    ob.empty_display_size = 0.1
    _set_official_special_type(ob, 'FAKE_LIGHT')
    bpy.context.collection.objects.link(ob)
    return ob

  mesh = _create_fake_light_mesh(name, positions)
  ob = bpy.data.objects.new(name, mesh)
  _set_official_special_type(ob, 'FAKE_LIGHT')

  # Assign dedicated fake omni material with correct EDM node group
  fake_mat = _create_fake_light_material(name, kind="fake_omni")
  if fake_mat is not None:
    mesh.materials.append(fake_mat)

  # Set EDMProps for fake light export
  if hasattr(ob, "EDMProps"):
    ob.EDMProps.SIZE = float(sizes[0]) if sizes else 3.0
    if uv_lb is not None:
      ob.EDMProps.UV_LB = (float(uv_lb[0]), float(uv_lb[1]))
      ob.EDMProps.UV_RT = (float(uv_rt[0]), float(uv_rt[1]))

  bpy.context.collection.objects.link(ob)
  return ob


def create_fake_spot_lights(node):
  """Create a mesh object for FakeSpotLightsNode.

  We reconstruct these as surface-mode quads (one quad per light) so the
  official exporter emits the same v10 FakeSpotLightsNode layout as reference
  assets (no extra trailing direction vector).
  """
  name = node.name or "FakeSpotLights"

  positions = []
  dirs_bl = []
  sizes = []
  for i, entry in enumerate(node.data):
    pos_edm = entry['position']
    positions.append(vector_to_blender(pos_edm))

    # Direction is more reliable in parentData than the currently-partial raw
    # fake-spot entry parser.
    dir_edm = None
    if i < len(getattr(node, "parentData", [])):
      pd = node.parentData[i]
      if len(pd) >= 3:
        cand = pd[2]
        try:
          if all(math.isfinite(float(v)) for v in cand):
            dir_edm = tuple(float(v) for v in cand)
        except Exception:
          dir_edm = None
    if dir_edm is None:
      cand = getattr(node, "trailing_direction", None)
      if cand is not None:
        try:
          if all(math.isfinite(float(v)) for v in cand):
            dir_edm = tuple(float(v) for v in cand)
        except Exception:
          dir_edm = None
    if dir_edm is None:
      cand = entry.get('direction')
      try:
        if cand is not None and all(math.isfinite(float(v)) for v in cand):
          dir_edm = tuple(float(v) for v in cand)
      except Exception:
        dir_edm = None
    if dir_edm is None:
      dir_edm = (1.0, 0.0, 0.0)

    dvec = vector_to_blender(dir_edm)
    if dvec.length <= 1e-8:
      dvec = Vector((1.0, 0.0, 0.0))
    else:
      dvec.normalize()
    dirs_bl.append(dvec)

    s = entry.get('size', 0.0)
    try:
      s = abs(float(s))
    except Exception:
      s = 0.0
    if not math.isfinite(s) or s <= 1e-6:
      s = 0.1
    sizes.append(s)

  if not positions:
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "SPHERE"
    ob.empty_display_size = 0.1
    _set_official_special_type(ob, 'FAKE_LIGHT')
    bpy.context.collection.objects.link(ob)
    return ob

  # Build one quad per light center, oriented by the light direction. Exporter
  # surface-mode fake spot parsing expects faces + UVs.
  verts = []
  faces = []
  uv_quads = []
  for i, (center, normal, size_val) in enumerate(zip(positions, dirs_bl, sizes)):
    up = Vector((0.0, 0.0, 1.0))
    if abs(normal.dot(up)) > 0.999:
      up = Vector((0.0, 1.0, 0.0))
    tangent = normal.cross(up)
    if tangent.length <= 1e-8:
      tangent = Vector((1.0, 0.0, 0.0))
    tangent.normalize()
    bitangent = normal.cross(tangent)
    if bitangent.length <= 1e-8:
      bitangent = Vector((0.0, 1.0, 0.0))
    bitangent.normalize()

    # parse_faces derives "light size" from the max distance from vertex0 to the
    # other quad vertices (diagonal dominates), so choose side length so the
    # resulting value is approximately size_val.
    half_side = max(0.001, float(size_val) / (2.0 * math.sqrt(2.0)))
    quad = [
      center - tangent * half_side - bitangent * half_side,
      center + tangent * half_side - bitangent * half_side,
      center + tangent * half_side + bitangent * half_side,
      center - tangent * half_side + bitangent * half_side,
    ]
    base = len(verts)
    verts.extend([tuple(v) for v in quad])
    faces.append((base + 0, base + 1, base + 2, base + 3))
    uv_quads.append(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))

  mesh = bpy.data.meshes.new(name)
  mesh.from_pydata(verts, [], faces)
  mesh.update()
  if faces:
    try:
      uv_layer = mesh.uv_layers.new(name="UVMap")
      loop_uv = uv_layer.data
      for poly in mesh.polygons:
        q = uv_quads[poly.index]
        for li, uv in zip(poly.loop_indices, q):
          loop_uv[li].uv = uv
    except Exception as e:
      print(f"Warning in blender_importer\lights.py: {e}")

  ob = bpy.data.objects.new(name, mesh)
  _set_official_special_type(ob, 'FAKE_LIGHT')

  # Assign dedicated fake spot material with correct EDM node group
  fake_mat = _create_fake_light_material(name, kind="fake_spot")
  if fake_mat is not None:
    mesh.materials.append(fake_mat)

  if hasattr(ob, "EDMProps"):
    ob.EDMProps.SURFACE_MODE = True
    ob.EDMProps.TWO_SIDED = False
    ob.EDMProps.SIZE = float(sizes[0]) if sizes else 0.1

  bpy.context.collection.objects.link(ob)
  return ob


def create_fake_als_lights(node):
  """Create a mesh object for FakeALSNode with one vertex per light."""
  name = node.name or "FakeALSLights"

  positions = []
  for entry in node.data:
    pos_edm = entry['position']
    positions.append(vector_to_blender(pos_edm))

  if not positions:
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "SPHERE"
    ob.empty_display_size = 0.1
    _set_official_special_type(ob, 'FAKE_LIGHT')
    bpy.context.collection.objects.link(ob)
    return ob

  mesh = _create_fake_light_mesh(name, positions)
  ob = bpy.data.objects.new(name, mesh)
  _set_official_special_type(ob, 'FAKE_LIGHT')

  # ALS lights map to fake_omni material kind
  fake_mat = _create_fake_light_material(name, kind="fake_omni")
  if fake_mat is not None:
    mesh.materials.append(fake_mat)

  bpy.context.collection.objects.link(ob)
  return ob
