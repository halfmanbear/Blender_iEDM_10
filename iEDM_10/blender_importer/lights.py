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


def _add_edmprop_keyframes(action, data_path, keys, value_fn):
  curve = _new_fcurve(action, data_path)
  for framedata in keys:
    frame = _anim_frame_to_scene_frame(getattr(framedata, "frame", 0.0))
    value = value_fn(getattr(framedata, "value", 0.0))
    key = curve.keyframe_points.insert(frame, float(value), options={'FAST'})
    key.interpolation = 'LINEAR'
  return curve


def _push_object_action_to_nla(obj, action):
  if obj is None or action is None:
    return False
  anim_data = obj.animation_data_create()
  try:
    track = anim_data.nla_tracks.new()
    track.name = action.name
    strip = track.strips.new(action.name, 0, action)
    strip.extrapolation = 'HOLD'
    strip.blend_type = 'REPLACE'
    return True
  except Exception as e:
    print(f"Warning in blender_importer/lights.py: {e}")
    return False


def _import_light_properties(node, obj, light_data, light_type):
  props = getattr(node, "lightProps", None) or {}
  if not props:
    return

  color_arg, color_keys, color_static = _extract_light_property(_get_prop_any(props, "Color", "color"))
  bright_arg, bright_keys, bright_static = _extract_light_property(_get_prop_any(props, "Brightness", "brightness"))
  dist_arg, dist_keys, dist_static = _extract_light_property(_get_prop_any(props, "Distance", "distance"))
  phi_arg, phi_keys, phi_static = _extract_light_property(_get_prop_any(props, "Phi", "phi"))
  theta_arg, theta_keys, theta_static = _extract_light_property(_get_prop_any(props, "Theta", "theta"))
  spec_arg, spec_keys, spec_static = _extract_light_property(_get_prop_any(props, "specularAmount", "specular_amount", "specular"))
  soft_arg, soft_keys, soft_static = _extract_light_property(_get_prop_any(props, "softness", "Softness"))
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
  if has_anim:
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

  edmprop_anim = any(keys for keys in (soft_keys, vol_radius_keys, vol_density_keys, vol_near_keys))
  if edmprop_anim and hasattr(obj, "EDMProps"):
    prop_action = bpy.data.actions.new("LightProps_{}".format(obj.name))
    if hasattr(prop_action, "argument"):
      first_arg = next((a for a in (soft_arg, vol_radius_arg, vol_density_arg, vol_near_arg, vol_type_arg) if a >= 0), -1)
      if first_arg >= 0:
        prop_action.argument = int(first_arg)

    if soft_keys:
      _add_edmprop_keyframes(
        prop_action,
        "EDMProps.LIGHT_SOFTNESS",
        soft_keys,
        lambda v: max(0.0, _to_float(v, 0.0)),
      )
    if vol_radius_keys:
      _add_edmprop_keyframes(
        prop_action,
        "EDMProps.LIGHT_VOLUME_RADIUS_FACTOR",
        vol_radius_keys,
        lambda v: min(1.0, max(0.0, _to_float(v, 0.0))),
      )
    if vol_density_keys:
      _add_edmprop_keyframes(
        prop_action,
        "EDMProps.LIGHT_VOLUME_DENSITY_FACTOR",
        vol_density_keys,
        lambda v: min(1.0, max(0.0, _to_float(v, 0.0))),
      )
    if vol_near_keys:
      _add_edmprop_keyframes(
        prop_action,
        "EDMProps.LIGHT_VOLUME_NEAR_DISTANCE",
        vol_near_keys,
        lambda v: max(0.0, _to_float(v, 0.0)),
      )

    # Keep object-level EDMProps animation separate from later transform /
    # visibility action assignment on the same object.
    if not _push_object_action_to_nla(obj, prop_action):
      obj_anim_data = obj.animation_data_create()
      obj_anim_data.action = prop_action


def create_lamp(node):
  """Creates a blender lamp from an edm renderable LampNode"""
  if getattr(node, "texture", None) is not None:
    surrogate = _create_textured_light_surrogate(node)
    if surrogate is not None:
      return surrogate

  light_props = getattr(node, "lightProps", None) or {}
  has_spot_keys = any(k in light_props for k in ("Phi", "Theta", "phi", "theta"))
  light_type = 'SPOT' if has_spot_keys else 'POINT'
  light_data = bpy.data.lights.new(name=node.name, type=light_type)
  obj = bpy.data.objects.new(name=node.name, object_data=light_data)
  _import_light_properties(node, obj, light_data, light_type)
  bpy.context.collection.objects.link(obj)
  return obj


def _store_textured_light_metadata(obj, node, surrogate_kind):
  if obj is None:
    return
  obj["_iedm_translation_status"] = "approximate"
  obj["_iedm_translation_source"] = "LightNode.Texture2dProperties"
  obj["_iedm_light_surrogate_kind"] = str(surrogate_kind or "")

  tex = getattr(node, "texture", None)
  if tex is None:
    return

  try:
    obj["_iedm_light_texture_name"] = str(getattr(tex, "name", "") or "")
    obj["_iedm_light_texture_index"] = int(getattr(tex, "index", 0) or 0)
    obj["_iedm_light_texture_wrap_s"] = int(getattr(tex, "wrap_s", 0) or 0)
    obj["_iedm_light_texture_wrap_t"] = int(getattr(tex, "wrap_t", 0) or 0)
    obj["_iedm_light_texture_mag_filter"] = int(getattr(tex, "mag_filter", 0) or 0)
    obj["_iedm_light_texture_min_filter"] = int(getattr(tex, "min_filter", 0) or 0)
  except Exception:
    pass

  uv_transform = getattr(tex, "uv_transform", None)
  if uv_transform is not None:
    try:
      obj["_iedm_light_texture_uv_matrix"] = [float(v) for row in uv_transform for v in row]
    except Exception:
      pass


def _apply_textured_light_material(material, node):
  if material is None:
    return
  tex = getattr(node, "texture", None)
  tex_name = str(getattr(tex, "name", "") or "") if tex is not None else ""
  if not tex_name:
    return
  group_node = _find_fake_light_group_node(material)
  if group_node is None:
    return
  tex_node = _find_or_create_emissive_texture_node(material, tex_name)
  if tex_node is None:
    return
  try:
    _link_texture_to_group_input(material.node_tree.links, tex_node, group_node, "Emissive")
  except Exception:
    pass


def _create_textured_light_surrogate(node):
  light_props = getattr(node, "lightProps", None) or {}
  has_spot_keys = any(k in light_props for k in ("Phi", "Theta", "phi", "theta"))
  name = node.name or "TexturedLight"

  if has_spot_keys:
    mesh = _create_fake_light_mesh(name, [Vector((0.0, 0.0, 0.0))])
    obj = bpy.data.objects.new(name, mesh)
    _set_official_special_type(obj, 'FAKE_LIGHT')
    if hasattr(obj, "EDMProps"):
      obj.EDMProps.SURFACE_MODE = False
      obj.EDMProps.TWO_SIDED = False
      obj.EDMProps.SIZE = 1.0
      front_lb, front_rt, back_lb, back_rt = _default_fake_spot_uvs(False)
      obj.EDMProps.UV_LB = front_lb
      obj.EDMProps.UV_RT = front_rt
      obj.EDMProps.UV_LB_BACK = back_lb
      obj.EDMProps.UV_RT_BACK = back_rt
    fake_mat = _create_fake_light_material(name, kind="fake_spot")
    if fake_mat is not None:
      mesh.materials.clear()
      mesh.materials.append(fake_mat)
      _apply_textured_light_material(fake_mat, node)
    _store_textured_light_metadata(obj, node, "fake_spot")
    _add_fake_spot_direction_child(obj, Vector((1.0, 0.0, 0.0)), distance=1.0)
    preview_light = bpy.data.lights.new(name=f"{name}_Preview", type='SPOT')
    try:
      _import_light_properties(node, obj, preview_light, 'SPOT')
    finally:
      bpy.data.lights.remove(preview_light)
    bpy.context.collection.objects.link(obj)
    return obj

  mesh, location = _create_fake_omni_mesh(name, [Vector((0.0, 0.0, 0.0))])
  obj = bpy.data.objects.new(name, mesh)
  obj.location = location
  _set_official_special_type(obj, 'FAKE_LIGHT')
  if hasattr(obj, "EDMProps"):
    obj.EDMProps.SIZE = 1.0
    obj.EDMProps.SURFACE_MODE = False
  fake_mat = _create_fake_light_material(name, kind="fake_omni")
  if fake_mat is not None:
    mesh.materials.clear()
    mesh.materials.append(fake_mat)
    _apply_textured_light_material(fake_mat, node)
  _store_textured_light_metadata(obj, node, "fake_omni")
  bpy.context.collection.objects.link(obj)
  return obj


def _create_official_render_material(obj_name, kind="default"):
  bridge = _ensure_official_material_bridge()
  if not bridge.get("available"):
    return None

  names = bridge.get("names", {})
  official_name = names.get(kind, names.get("default", "EDM_Default_Material"))

  mat = bpy.data.materials.new(name=obj_name)
  mat.use_nodes = True
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links

  for node in list(nodes):
    nodes.remove(node)

  output_node = nodes.new('ShaderNodeOutputMaterial')
  output_node.location = (400, 0)

  group_node = _create_official_group_node(nodes, bridge, kind, official_name)
  if group_node is None:
    return None

  group_node.location = (0, 0)
  if group_node.outputs:
    try:
      links.new(group_node.outputs[0], output_node.inputs['Surface'])
    except Exception as e:
      print(f"Warning in blender_importer\\lights.py: {e}")
  return mat


def _billboard_plane_axes(axis_name):
  axis = str(axis_name or "all")
  if axis in {"x", "along_x"}:
    return Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0))
  if axis in {"y", "along_y"}:
    return Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))
  return Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0))


def _create_billboard_surrogate_mesh(name, axis_name, matrix_value, pivot_value):
  pivot = Vector((0.0, 0.0, 0.0))
  if pivot_value is not None:
    try:
      pivot = Vector((float(pivot_value[0]), float(pivot_value[1]), float(pivot_value[2])))
    except Exception:
      pivot = Vector((0.0, 0.0, 0.0))

  axis_u, axis_v = _billboard_plane_axes(axis_name)
  half_size = 0.5
  local_verts = [
    pivot + ((-half_size) * axis_u) + ((-half_size) * axis_v),
    pivot + ((half_size) * axis_u) + ((-half_size) * axis_v),
    pivot + ((half_size) * axis_u) + ((half_size) * axis_v),
    pivot + ((-half_size) * axis_u) + ((half_size) * axis_v),
  ]

  transform = Matrix.Identity(4)
  if matrix_value is not None:
    try:
      transform = Matrix(matrix_value)
    except Exception:
      transform = Matrix.Identity(4)

  verts = []
  for vert in local_verts:
    try:
      verts.append(tuple((transform @ vert.to_4d()).xyz))
    except Exception:
      verts.append(tuple(vert))

  mesh = bpy.data.meshes.new(name)
  mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
  mesh.update()

  try:
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_values = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for loop_index, uv in enumerate(uv_values):
      uv_layer.data[loop_index].uv = uv
  except Exception as e:
    print(f"Warning in blender_importer\\lights.py: {e}")

  return mesh


def create_billboard(node):
  """Create an exporter-compatible approximate mesh surrogate for BillboardNode."""

  # Type mapping based on edm_plugin_ref.html
  type_map = {0: "direction", 1: "point"}
  axis_map = {
    0: "all",
    1: "x",
    2: "y",
    3: "z",
    4: "along_x",
    5: "along_y",
    6: "along_z"
  }

  billboard_type = type_map.get(getattr(node, "billboard_type", 0), "point")
  billboard_axis = axis_map.get(getattr(node, "billboard_axis", 0), "all")
  mesh = _create_billboard_surrogate_mesh(
    node.name or "Billboard",
    billboard_axis,
    getattr(node, "matrix", None),
    getattr(node, "pivot", None),
  )
  obj = bpy.data.objects.new(name=node.name or "Billboard", object_data=mesh)
  _set_official_special_type(obj, 'UNKNOWN_TYPE')
  obj.edm.billboard_type = billboard_type
  obj.edm.billboard_axis = billboard_axis
  obj["_iedm_translation_status"] = "approximate"
  obj["_iedm_translation_source"] = "BillboardNode"
  obj["_iedm_billboard_surrogate"] = True

  if getattr(node, "matrix", None) is not None:
    obj["_iedm_billboard_matrix"] = [float(v) for row in node.matrix for v in row]
  if getattr(node, "pivot", None) is not None:
    obj["_iedm_billboard_pivot"] = [float(v) for v in node.pivot]

  if mesh is not None and not mesh.materials:
    fallback_mat = _create_official_render_material(f"{obj.name}_Billboard", kind="default")
    if fallback_mat is not None:
      mesh.materials.append(fallback_mat)

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


def _material_for_fake_light(node, obj_name, kind):
  edm_mat = getattr(node, "material", None)
  blender_mat = getattr(edm_mat, "blender_material", None) if edm_mat else None
  if blender_mat is not None and _material_matches_fake_light_kind(blender_mat, kind):
    _apply_fake_light_material_payload(blender_mat, edm_mat, kind)
    return blender_mat
  fake_mat = _create_fake_light_material(obj_name, kind=kind)
  if fake_mat is not None:
    _apply_fake_light_material_payload(fake_mat, edm_mat, kind)
    return fake_mat
  return blender_mat


def _create_fake_light_mesh(name, positions):
  """Create a mesh with one vertex per light position (already in Blender coords)."""
  mesh = bpy.data.meshes.new(name)
  mesh.vertices.add(len(positions))
  for i, pos in enumerate(positions):
    mesh.vertices[i].co = pos
  mesh.update()
  return mesh


def _fake_light_world_from_edm(pos_edm):
  """
  Fake-light point payloads are exported through the official addon's
  ROOT_TRANSFORM_MATRIX. Invert that here so importer-created Blender objects
  match the reference .blend topology and object placement.
  """
  return Vector((float(pos_edm[0]), -float(pos_edm[2]), float(pos_edm[1])))


def _create_fake_omni_mesh(name, world_positions):
  if not world_positions:
    return _create_fake_light_mesh(name, []), Vector((0.0, 0.0, 0.0))

  center = Vector((0.0, 0.0, 0.0))
  for pos in world_positions:
    center += pos
  center /= float(len(world_positions))

  local_positions = [pos - center for pos in world_positions]
  unique_positions = {
    tuple(round(float(c), 6) for c in pos)
    for pos in local_positions
  }
  if len(unique_positions) == 8:
    xs = sorted({round(pos[0], 6) for pos in unique_positions})
    ys = sorted({round(pos[1], 6) for pos in unique_positions})
    zs = sorted({round(pos[2], 6) for pos in unique_positions})
    if len(xs) == len(ys) == len(zs) == 2:
      verts = [
        (xs[0], ys[0], zs[0]),
        (xs[0], ys[0], zs[1]),
        (xs[0], ys[1], zs[0]),
        (xs[0], ys[1], zs[1]),
        (xs[1], ys[0], zs[0]),
        (xs[1], ys[0], zs[1]),
        (xs[1], ys[1], zs[0]),
        (xs[1], ys[1], zs[1]),
      ]
      faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
      ]
      mesh = bpy.data.meshes.new(name)
      mesh.from_pydata(verts, [], faces)
      mesh.update()
      return mesh, center

  return _create_fake_light_mesh(name, local_positions), center


def _axis_box_layout(positions):
  unique_positions = {
    tuple(round(float(c), 6) for c in pos)
    for pos in positions
  }
  if len(unique_positions) != 8:
    return None, None
  xs = sorted({round(pos[0], 6) for pos in unique_positions})
  ys = sorted({round(pos[1], 6) for pos in unique_positions})
  zs = sorted({round(pos[2], 6) for pos in unique_positions})
  if len(xs) != 2 or len(ys) != 2 or len(zs) != 2:
    return None, None
  verts = [
    (xs[0], ys[0], zs[0]),
    (xs[0], ys[0], zs[1]),
    (xs[0], ys[1], zs[0]),
    (xs[0], ys[1], zs[1]),
    (xs[1], ys[0], zs[0]),
    (xs[1], ys[0], zs[1]),
    (xs[1], ys[1], zs[0]),
    (xs[1], ys[1], zs[1]),
  ]
  faces = [
    (0, 1, 3, 2),
    (2, 3, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (2, 6, 4, 0),
    (7, 3, 1, 5),
  ]
  return verts, faces


def _create_axis_box_mesh(name, positions):
  verts, faces = _axis_box_layout(positions)
  if verts is None:
    return None
  mesh = bpy.data.meshes.new(name)
  mesh.from_pydata(verts, [], faces)
  mesh.update()
  return mesh


def _classify_fake_spot_mode(positions, dirs_bl):
  if not positions:
    return "surface"
  if not dirs_bl:
    return "surface"
  ref = None
  uniform = True
  for dvec in dirs_bl:
    if dvec.length <= 1.0e-8:
      continue
    cur = dvec.normalized()
    if ref is None:
      ref = cur
      continue
    if ref.dot(cur) < 0.999:
      uniform = False
      break
  if not uniform:
    return "surface"
  verts, _faces = _axis_box_layout(positions)
  if verts is not None:
    return "non_surface_box"
  return "non_surface_points"


def _default_fake_spot_uvs(two_sided):
  front_lb = (0.0, 0.5)
  front_rt = (0.25, 1.0)
  back_lb = (0.75, 0.5) if two_sided else (0.0, 0.0)
  back_rt = (1.0, 1.0)
  return front_lb, front_rt, back_lb, back_rt


def _add_fake_spot_direction_child(parent_obj, direction, distance=1.0):
  if parent_obj is None:
    return None
  dvec = Vector(direction)
  if dvec.length <= 1.0e-8:
    dvec = Vector((1.0, 0.0, 0.0))
  else:
    dvec.normalize()
  child = bpy.data.objects.new("Light_Dir", None)
  child.empty_display_type = "PLAIN_AXES"
  child.empty_display_size = 0.1
  child.parent = parent_obj
  child.location = tuple(dvec * float(distance))
  quat = Vector((1.0, 0.0, 0.0)).rotation_difference(dvec)
  child.rotation_mode = 'QUATERNION'
  child.rotation_quaternion = quat
  bpy.context.collection.objects.link(child)
  return child


def _unpack_uv_double(dval):
  """Decode a double that packs two floats (UV pair) via reinterpretation."""
  import struct
  raw = struct.pack('<d', dval)
  return struct.unpack('<2f', raw)


def _safe_float(value, default=0.0):
  try:
    return float(value)
  except Exception:
    return float(default)


def _safe_vec(value, default=(0.0, 0.0, 0.0)):
  try:
    return tuple(float(v) for v in value)
  except Exception:
    return tuple(float(v) for v in default)


def _find_fake_light_group_node(material):
  if material is None or not getattr(material, "use_nodes", False) or not getattr(material, "node_tree", None):
    return None
  bridge = _ensure_official_material_bridge()
  valid_node_types = {
    bridge.get("node_types", {}).get("fake_omni"),
    bridge.get("node_types", {}).get("fake_spot"),
  }
  valid_tree_names = {
    bridge.get("names", {}).get("fake_omni"),
    bridge.get("names", {}).get("fake_spot"),
  }
  nodes = material.node_tree.nodes
  for node in nodes:
    node_tree = getattr(node, "node_tree", None)
    node_tree_name = getattr(node_tree, "name", None)
    if getattr(node, "bl_idname", None) in valid_node_types:
      return node
    if node.bl_idname == "ShaderNodeGroup" and node_tree_name in valid_tree_names:
      return node
    if hasattr(node, "inputs") and any(getattr(sock, "name", None) == "Emissive" for sock in getattr(node, "inputs", ())):
      return node
  return None


def _material_matches_fake_light_kind(material, kind):
  if material is None or not getattr(material, "use_nodes", False) or not getattr(material, "node_tree", None):
    return False
  bridge = _ensure_official_material_bridge()
  official_name = bridge.get("names", {}).get(kind)
  official_node_type = bridge.get("node_types", {}).get(kind)
  for node in material.node_tree.nodes:
    node_tree = getattr(node, "node_tree", None)
    node_tree_name = getattr(node_tree, "name", None)
    if official_node_type and getattr(node, "bl_idname", None) == official_node_type:
      return True
    if official_name and node_tree_name == official_name:
      return True
  return False


def _find_or_create_emissive_texture_node(material, texture_name):
  if material is None or not texture_name or not getattr(material, "node_tree", None):
    return None
  nodes = material.node_tree.nodes
  for node in nodes:
    if node.bl_idname != "ShaderNodeTexImage":
      continue
    image = getattr(node, "image", None)
    if image and os.path.splitext(os.path.basename(image.filepath))[0].lower() == str(texture_name).lower():
      return node
  filename = None
  source_dir = getattr(_import_ctx, "source_dir", None)
  if source_dir:
    try:
      with chdir(source_dir):
        filename = _find_texture_file(texture_name)
    except Exception:
      filename = None
  if not filename:
    filename = _find_texture_file(texture_name)
  if not filename:
    try:
      image = bpy.data.images.get(texture_name)
      if image is None:
        image = bpy.data.images.new(name=str(texture_name), width=1, height=1, alpha=True)
      tex_image = nodes.new('ShaderNodeTexImage')
      tex_image.image = image
      tex_image.location = (-350, 0)
      return tex_image
    except Exception as e:
      print(f"Warning in blender_importer/lights.py: {e}")
      return None
  try:
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(filename)
    tex_image.image.colorspace_settings.name = 'sRGB'
    tex_image.location = (-350, 0)
    return tex_image
  except Exception as e:
    print(f"Warning in blender_importer/lights.py: {e}")
    return None


def _set_material_group_input(group_node, socket_name, value):
  if not group_node:
    return
  for socket in group_node.inputs:
    if socket.name != socket_name or not hasattr(socket, "default_value"):
      continue
    try:
      socket.default_value = value
    except Exception as e:
      print(f"Warning in blender_importer/lights.py: {e}")
    return


def _apply_fake_light_material_payload(material, edm_material, kind):
  if material is None or edm_material is None:
    return
  group_node = _find_fake_light_group_node(material)
  if group_node is None:
    return

  textures = getattr(edm_material, "textures", None) or []
  if textures:
    tex_name = getattr(textures[0], "name", None)
    tex_node = _find_or_create_emissive_texture_node(material, tex_name)
    if tex_node is not None:
      try:
        _link_texture_to_group_input(material.node_tree.links, tex_node, group_node, "Emissive")
      except Exception:
        pass

  uniforms = getattr(edm_material, "uniforms", None) or {}
  anim_uniforms = getattr(edm_material, "animated_uniforms", None) or {}

  luminance_prop = anim_uniforms.get("luminance", uniforms.get("luminance"))
  luminance_val = None
  if hasattr(luminance_prop, "keys") and getattr(luminance_prop, "keys", None):
    luminance_val = getattr(luminance_prop.keys[0], "value", None)
  elif luminance_prop is not None:
    luminance_val = luminance_prop
  if luminance_val is not None:
    _set_material_group_input(group_node, "Luminance", _safe_float(luminance_val, 1.0))

  shift = uniforms.get("shiftToCamera")
  if shift is not None:
    _set_material_group_input(group_node, "ShiftToCamera", _safe_float(shift, 0.0))

  size_factors = uniforms.get("sizeFactors")
  size_vec = _safe_vec(size_factors, default=(4.0, 1000.0, 0.0))
  if len(size_vec) >= 1:
    _set_material_group_input(group_node, "MinSizePixels", _safe_float(size_vec[0], 4.0))
  if len(size_vec) >= 2:
    _set_material_group_input(group_node, "MaxDistance", _safe_float(size_vec[1], 1000.0))

  if kind == "fake_spot":
    cone_setup = uniforms.get("coneSetup")
    cone_vec = _safe_vec(cone_setup, default=())
    if len(cone_vec) >= 2:
      try:
        theta_deg = math.degrees(math.acos(max(-1.0, min(1.0, cone_vec[0]))))
        phi_deg = math.degrees(math.acos(max(-1.0, min(1.0, cone_vec[1]))))
        _set_material_group_input(group_node, "Inner Angle", theta_deg)
        _set_material_group_input(group_node, "Outer Angle", phi_deg)
      except Exception:
        pass
    specular = uniforms.get("specularAmount")
    if specular is not None:
      _set_material_group_input(group_node, "SpecularAmount", _safe_float(specular, 0.0))


def _apply_fake_light_animation_payload(obj, edm_material):
  if obj is None or not hasattr(obj, "EDMProps") or edm_material is None:
    return
  anim_uniforms = getattr(edm_material, "animated_uniforms", None) or {}
  luminance_prop = anim_uniforms.get("luminance")
  if luminance_prop is None or not hasattr(luminance_prop, "keys"):
    return

  keys = list(getattr(luminance_prop, "keys", []) or [])
  if not keys:
    return
  arg = getattr(luminance_prop, "argument", None)
  base_luminance = _safe_float(getattr(keys[0], "value", 1.0), 1.0)
  if abs(base_luminance) <= 1.0e-8:
    base_luminance = 1.0
  try:
    obj.EDMProps.ANIMATED_BRIGHTNESS = 1.0
  except Exception:
    pass

  try:
    action_name = "FakeLight_{}".format(obj.name)
    if arg is not None and int(arg) >= 0:
      action_name = "{}_{}".format(int(arg), obj.name)
    action = bpy.data.actions.new(action_name)
    if hasattr(action, "argument") and arg is not None and int(arg) >= 0:
      action.argument = int(arg)
    anim_data = obj.animation_data_create()
    anim_data.action = action
    for framedata in keys:
      frame = _anim_frame_to_scene_frame(getattr(framedata, "frame", 0.0))
      value = _safe_float(getattr(framedata, "value", 0.0), 0.0) / base_luminance
      obj.EDMProps.ANIMATED_BRIGHTNESS = value
      obj.keyframe_insert(data_path='EDMProps.ANIMATED_BRIGHTNESS', frame=frame)
    curve = action.fcurves.find('EDMProps.ANIMATED_BRIGHTNESS')
    if curve is not None:
      for key in curve.keyframe_points:
        key.interpolation = 'LINEAR'
  except Exception as e:
    print(f"Warning in blender_importer/lights.py: {e}")


def _has_animated_fake_omni_payload(node):
  return (
    getattr(node, "anim_arg_handle", None) is not None
    and getattr(node, "anim_arg_handle", 0xFFFFFFFF) != 0xFFFFFFFF
    and getattr(node, "anim_data_count", 0) > 0
    and getattr(node, "anim_data_raw", b"")
  )


def _apply_animated_fake_omni_brightness(ob, node, light_count):
  """Reconstruct per-light brightness animation from AnimatedFakeOmniLightsNode payload.

  The binary stores lightCount * 128 float32 brightness values — each light's
  curve presampled at 128 evenly-spaced arg positions. The official exporter
  expects: one vertex per light, a vertex group whose weight encodes each
  light's timing delay, and an action on EDMProps.ANIMATED_BRIGHTNESS named
  "{arg_handle}_{object_name}" with the master brightness curve as keyframes.
  """
  import struct as _struct

  arg_handle = getattr(node, "anim_arg_handle", None)
  data_count = getattr(node, "anim_data_count", 0)
  anim_data_raw = getattr(node, "anim_data_raw", b"")

  if not anim_data_raw or data_count == 0 or light_count <= 0:
    return

  # Format spec: anim_sample_rate must be 128. Use the explicit field when
  # available and consistent; fall back to data_count // light_count otherwise.
  explicit_rate = getattr(node, "anim_sample_rate", None)
  n_samples = data_count // light_count
  if explicit_rate is not None and explicit_rate > 0:
    if explicit_rate != n_samples:
      print(
        "Warning: animated fake light '{}': anim_sample_rate={} but data_count/light_count={}; "
        "using explicit rate".format(getattr(node, "name", ""), explicit_rate, n_samples)
      )
    n_samples = int(explicit_rate)
  if n_samples == 0:
    return

  all_floats = _struct.unpack_from("<{}f".format(data_count), anim_data_raw)
  lights_samples = [
    all_floats[i * n_samples:(i + 1) * n_samples]
    for i in range(light_count)
  ]
  master = lights_samples[0]

  # Recover per-light delay by finding circular shift vs master curve.
  # Shift of k samples → delay = k/n_samples * 2.0 in EDM time units,
  # stored as vertex group weight (exporter reads weight directly as delay).
  delays = [0.0]
  for j in range(1, light_count):
    samp = lights_samples[j]
    best_shift, best_score = 0, float("inf")
    for shift in range(n_samples):
      score = sum((master[i] - samp[(i + shift) % n_samples]) ** 2 for i in range(n_samples))
      if score < best_score:
        best_score = score
        best_shift = shift
    delays.append(min(best_shift / n_samples * 2.0, 1.0))

  # Assign delay weights — one vertex group, one weight per vertex.
  delay_group = ob.vertex_groups.new(name="brightness_delay")
  for vi, delay in enumerate(delays):
    if vi < len(ob.data.vertices):
      delay_group.add([vi], float(delay), 'REPLACE')

  # Build brightness action from master curve.
  # 128 samples span Blender frames [0, 200]; frame_i = i * 200 / n_samples.
  try:
    ob.EDMProps.ANIMATED_BRIGHTNESS = 1.0
  except Exception:
    pass

  action_name = "{}_{}".format(int(arg_handle), ob.name)
  action = bpy.data.actions.new(action_name)
  try:
    if hasattr(action, "argument"):
      action.argument = int(arg_handle)
  except Exception:
    pass

  anim_data = ob.animation_data_create()
  anim_data.action = action

  frame_step = 200.0 / n_samples
  for i, brightness in enumerate(master):
    frame = i * frame_step
    try:
      ob.EDMProps.ANIMATED_BRIGHTNESS = float(brightness)
      ob.keyframe_insert(data_path='EDMProps.ANIMATED_BRIGHTNESS', frame=frame)
    except Exception:
      pass

  curve = action.fcurves.find('EDMProps.ANIMATED_BRIGHTNESS')
  if curve is not None:
    for kp in curve.keyframe_points:
      kp.interpolation = 'LINEAR'
    try:
      curve.extrapolation = 'CONSTANT'
    except Exception:
      pass


def create_fake_omni_lights(node):
  """Create a non-surface fake omni object compatible with the official exporter."""
  name = node.name or "FakeOmniLights"

  positions = []
  decoded_entries = list(getattr(node, "decoded_data", []) or [])
  if not decoded_entries:
    try:
      from ..edm_format.types.lights import decode_fake_omni_entry
      decoded_entries = [decode_fake_omni_entry(entry) for entry in getattr(node, "data", [])]
    except Exception:
      decoded_entries = []

  size = None
  uv_lb = None
  uv_rt = None
  for decoded in decoded_entries:
    pos_edm = decoded.get("position", (0.0, 0.0, 0.0))
    positions.append(_fake_light_world_from_edm(pos_edm))
    if size is None:
      size = decoded.get("size")
      uv_lb = decoded.get("uv_lb")
      uv_rt = decoded.get("uv_rt")
  if not positions:
    for entry in getattr(node, "data", []) or []:
      positions.append(_fake_light_world_from_edm(entry[0:3]))

  if not positions:
    # No light entries — create an empty placeholder
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "SPHERE"
    ob.empty_display_size = 0.1
    _set_official_special_type(ob, 'FAKE_LIGHT')
    bpy.context.collection.objects.link(ob)
    return ob

  mesh, location = _create_fake_omni_mesh(name, positions)
  ob = bpy.data.objects.new(name, mesh)
  ob.location = location
  _set_official_special_type(ob, 'FAKE_LIGHT')

  # Assign dedicated fake omni material with correct EDM node group
  fake_mat = _material_for_fake_light(node, name, kind="fake_omni")
  if fake_mat is not None:
    mesh.materials.clear()
    mesh.materials.append(fake_mat)

  # Set EDMProps for fake light export
  if hasattr(ob, "EDMProps"):
    ob.EDMProps.SIZE = float(size) if size is not None else 3.0
    ob.EDMProps.SURFACE_MODE = False
    if uv_lb is not None:
      ob.EDMProps.UV_LB = (float(uv_lb[0]), float(uv_lb[1]))
    if uv_rt is not None:
      ob.EDMProps.UV_RT = (float(uv_rt[0]), float(uv_rt[1]))

  if _has_animated_fake_omni_payload(node):
    _apply_animated_fake_omni_brightness(ob, node, len(positions))
  else:
    _apply_fake_light_animation_payload(ob, getattr(node, "material", None))

  bpy.context.collection.objects.link(ob)
  return ob


def create_fake_spot_lights(node):
  """Create a mesh object for FakeSpotLightsNode."""
  name = node.name or "FakeSpotLights"

  positions = []
  dirs_bl = []
  sizes = []
  two_sided = False
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
    if entry.get('back_side') is True:
      two_sided = True
    elif 'flag' in entry:
      try:
        two_sided = two_sided or bool(int(entry.get('flag', 0)) & 0x1)
      except Exception:
        pass

  if not positions:
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "SPHERE"
    ob.empty_display_size = 0.1
    _set_official_special_type(ob, 'FAKE_LIGHT')
    bpy.context.collection.objects.link(ob)
    return ob

  mode = _classify_fake_spot_mode(positions, dirs_bl)
  if mode == "non_surface_box":
    mesh = _create_axis_box_mesh(name, positions)
    if mesh is None:
      mesh = _create_fake_light_mesh(name, positions)
  elif mode == "non_surface_points":
    mesh = _create_fake_light_mesh(name, positions)
  else:
    verts = []
    faces = []
    uv_quads = []
    for center, normal, size_val in zip(positions, dirs_bl, sizes):
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
        print(f"Warning in blender_importer/lights.py: {e}")

  ob = bpy.data.objects.new(name, mesh)
  _set_official_special_type(ob, 'FAKE_LIGHT')

  # Assign dedicated fake spot material with correct EDM node group
  fake_mat = _material_for_fake_light(node, name, kind="fake_spot")
  if fake_mat is not None:
    mesh.materials.clear()
    mesh.materials.append(fake_mat)

  if hasattr(ob, "EDMProps"):
    ob.EDMProps.SURFACE_MODE = (mode == "surface")
    ob.EDMProps.TWO_SIDED = bool(two_sided)
    ob.EDMProps.SIZE = float(sizes[0]) if sizes else 0.1
    if mode != "surface":
      front_lb, front_rt, back_lb, back_rt = _default_fake_spot_uvs(bool(two_sided))
      ob.EDMProps.UV_LB = front_lb
      ob.EDMProps.UV_RT = front_rt
      ob.EDMProps.UV_LB_BACK = back_lb
      ob.EDMProps.UV_RT_BACK = back_rt
  if _has_animated_fake_omni_payload(node):
    _apply_animated_fake_omni_brightness(ob, node, len(node.data))
  else:
    _apply_fake_light_animation_payload(ob, getattr(node, "material", None))

  bpy.context.collection.objects.link(ob)
  if mode != "surface":
    direction = dirs_bl[0] if dirs_bl else Vector((1.0, 0.0, 0.0))
    _add_fake_spot_direction_child(ob, direction, distance=max(1.0, float(sizes[0]) * 0.5 if sizes else 1.0))
  return ob


def create_fake_als_lights(node):
  """Create a mesh object for FakeALSNode with one vertex per light."""
  def _preserve_fake_als_metadata(obj, als_node):
    if obj is None or als_node is None:
      return
    try:
      obj["_iedm_translation_status"] = "approximate"
      obj["_iedm_translation_source"] = "FakeALSNode"
      obj["_iedm_fake_als_extra_preserved"] = True
      obj["_iedm_fake_als_count"] = int(len(getattr(als_node, "data", None) or []))
      header = list(getattr(als_node, "als_header", ()) or ())
      if header:
        obj["_iedm_fake_als_header"] = [int(v) for v in header[:3]]

      payload = {
        "header": [int(v) for v in header[:3]],
        "entries": [
          {
            "position": [float(v) for v in (entry.get("position") or ())[:3]],
            "extra": [float(v) for v in (entry.get("extra") or ())[:7]],
          }
          for entry in (getattr(als_node, "data", None) or [])
        ],
      }
      encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
      if len(encoded) <= 60000:
        obj["_iedm_fake_als_payload"] = encoded
        obj["_iedm_fake_als_payload_storage"] = "inline_json"
      else:
        safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in (obj.name or "FakeALS"))
        text_name = "IEDM_FakeALS_{}".format(safe_name[:48])
        text_block = bpy.data.texts.get(text_name)
        if text_block is None:
          text_block = bpy.data.texts.new(text_name)
        text_block.clear()
        text_block.write(encoded)
        obj["_iedm_fake_als_payload_text"] = text_name
        obj["_iedm_fake_als_payload_storage"] = "text_json"
    except Exception as e:
      print(f"Warning preserving FakeALSNode payload on {getattr(obj, 'name', '')}: {e}")

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
    _preserve_fake_als_metadata(ob, node)
    bpy.context.collection.objects.link(ob)
    return ob

  mesh = _create_fake_light_mesh(name, positions)
  ob = bpy.data.objects.new(name, mesh)
  _set_official_special_type(ob, 'FAKE_LIGHT')

  # ALS lights map to fake_omni material kind
  fake_mat = _material_for_fake_light(node, name, kind="fake_omni")
  if fake_mat is not None:
    mesh.materials.clear()
    mesh.materials.append(fake_mat)
  _apply_fake_light_animation_payload(ob, getattr(node, "material", None))
  _preserve_fake_als_metadata(ob, node)

  bpy.context.collection.objects.link(ob)
  return ob
