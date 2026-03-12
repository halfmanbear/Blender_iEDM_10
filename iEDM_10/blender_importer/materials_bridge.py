def _map_edm_material_to_official_kind(edm_material_name):
  mat = (edm_material_name or "").lower()
  if mat in {"glass_material"}:
    return "glass"
  if mat in {"mirror_material"}:
    return "mirror"
  # Keep BANO materials on regular render meshes unless the source node itself
  # is explicitly a Fake* light node. Mapping bano_material to fake_omni causes
  # RenderNode -> FakeOmniLightsNode type drift on round-trip.
  if mat in {"fake_omni_lights", "fake_omni_lights2", "fake_als_lights"}:
    return "fake_omni"
  if mat in {"fake_spot_lights"}:
    return "fake_spot"
  if mat in {"deck_material"}:
    return "deck"
  return "default"


def _create_official_group_node(nodes, bridge, official_kind, official_name):
  node_tree = _resolve_official_material_tree(bridge, official_name)
  custom_id = bridge.get("node_types", {}).get(official_kind)
  desc = bridge.get("material_descs", {}).get(official_name)

  if custom_id:
    try:
      custom_node = nodes.new(custom_id)
      # Custom official nodes need post_init to bind their EDM node tree.
      if desc is not None and hasattr(custom_node, "post_init"):
        custom_node.post_init(desc)
      elif node_tree is not None and hasattr(custom_node, "node_tree"):
        custom_node.node_tree = node_tree
      if getattr(custom_node, "node_tree", None) is not None:
        return custom_node
    except Exception as e:
      print(f"Warning in blender_importer\materials_bridge.py: {e}")

  if node_tree is None:
    return None
  try:
    fallback_node = nodes.new("ShaderNodeGroup")
    fallback_node.node_tree = node_tree
    return fallback_node
  except Exception:
    return None


def _set_group_enum_property(group_node, prop_name, value):
  if not group_node or not prop_name:
    return
  if not hasattr(group_node, prop_name):
    return
  try:
    setattr(group_node, prop_name, value)
  except Exception as e:
    print(f"Warning in blender_importer\materials_bridge.py: {e}")


def _map_edm_material_to_official_name(edm_material_name, bridge):
  names = bridge.get("names", {})
  kind = _map_edm_material_to_official_kind(edm_material_name)
  return names.get(kind, names.get("default", "EDM_Default_Material"))


def _official_transparency_enum(blending):
  try:
    value = int(blending)
  except Exception:
    value = 0
  return {
    0: "OPAQUE",
    1: "ALPHA_BLENDING",
    2: "ALPHA_TEST",
    3: "SUM_BLENDING",
  }.get(value, "OPAQUE")


def _official_shadow_enum(shadows):
  if shadows is None:
    return "SHADOW_CASTER_YES"
  if getattr(shadows, "cast_only", False):
    return "SHADOW_CASTER_ONLY"
  if getattr(shadows, "cast", False):
    return "SHADOW_CASTER_YES"
  return "SHADOW_CASTER_NO"


def _set_group_socket_default(group_node, socket_name, value):
  if not group_node:
    return
  for socket in group_node.inputs:
    if socket.name != socket_name:
      continue
    if not hasattr(socket, "default_value"):
      return
    try:
      socket.default_value = value
    except Exception as e:
      print(f"Warning in blender_importer\materials_bridge.py: {e}")
    return


def _find_group_input_socket(group_node, socket_name):
  if not group_node:
    return None
  for socket in group_node.inputs:
    if socket.name == socket_name:
      return socket
  return None


def _link_texture_to_group_input(links, texture_node, group_node, input_name, output_name="Color"):
  if not texture_node or not group_node:
    return
  from_socket = texture_node.outputs.get(output_name)
  to_socket = _find_group_input_socket(group_node, input_name)
  if not from_socket or not to_socket:
    return
  for link in to_socket.links:
    if link.from_socket == from_socket:
      return
  try:
    links.new(from_socket, to_socket)
  except Exception as e:
    print(f"Warning in blender_importer\materials_bridge.py: {e}")


def _ensure_uv_map_node(nodes):
  for node in nodes:
    if node.bl_idname == "ShaderNodeUVMap":
      return node
  try:
    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.name = "iEDM_UVMap"
    uv_node.label = "UV Map"
    uv_node.location = (-700, 200)
    return uv_node
  except Exception:
    return None


def _link_uv_to_texture_vector(links, uv_node, texture_node):
  if not uv_node or not texture_node:
    return
  from_socket = uv_node.outputs.get("UV")
  to_socket = texture_node.inputs.get("Vector")
  if not from_socket or not to_socket:
    return
  for link in to_socket.links:
    if link.from_socket == from_socket:
      return
  try:
    for link in list(to_socket.links):
      links.remove(link)
  except Exception as e:
    print(f"Warning in blender_importer\materials_bridge.py: {e}")
  try:
    links.new(from_socket, to_socket)
  except Exception as e:
    print(f"Warning in blender_importer\materials_bridge.py: {e}")


def _uv_map_name_from_channel(channel_index):
  if channel_index is None:
    return "UVMap"
  try:
    channel = int(channel_index)
  except Exception:
    channel = 0
  if channel <= 0:
    return "UVMap"
  return "UVMap.{:03d}".format(channel)


def _ensure_uv_map_node_for_channel(nodes, channel_index):
  uv_map_name = _uv_map_name_from_channel(channel_index)
  try:
    ch_idx = max(0, int(channel_index))
  except Exception:
    ch_idx = 0
  for node in nodes:
    if node.bl_idname != "ShaderNodeUVMap":
      continue
    if getattr(node, "uv_map", "") == uv_map_name:
      return node
  try:
    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = uv_map_name
    uv_node.name = "iEDM_UVMap_{}".format(uv_map_name.replace(".", "_"))
    uv_node.label = uv_map_name
    uv_node.location = (-700, 200 - 140 * ch_idx)
    return uv_node
  except Exception:
    return None


def _texture_uv_channel_map(edm_material):
  mapping = {}
  if edm_material is None:
    return mapping
  channels = getattr(edm_material, "texture_coordinates_channels", None) or []
  for tex_idx, uv_channel in enumerate(channels):
    try:
      uv_idx = int(uv_channel)
    except Exception:
      continue
    if uv_idx >= 0:
      mapping[int(tex_idx)] = uv_idx
  return mapping


def _find_active_material_output(nodes):
  first_output = None
  for node in nodes:
    if node.bl_idname != "ShaderNodeOutputMaterial":
      continue
    if first_output is None:
      first_output = node
    if getattr(node, "is_active_output", False):
      return node
  return first_output


def _link_group_surface_to_output(links, group_node, material_output):
  if not group_node or not material_output:
    return
  to_socket = material_output.inputs.get("Surface")
  if not to_socket:
    return

  from_socket = None
  for socket in group_node.outputs:
    if getattr(socket, "type", None) == "SHADER":
      from_socket = socket
      break
  if not from_socket:
    from_socket = group_node.outputs.get("Shader")
  if not from_socket:
    from_socket = group_node.outputs.get("BSDF")
  if not from_socket:
    return

  for link in list(to_socket.links):
    try:
      links.remove(link)
    except Exception as e:
      print(f"Warning in blender_importer\materials_bridge.py: {e}")
  try:
    links.new(from_socket, to_socket)
  except Exception as e:
    print(f"Warning in blender_importer\materials_bridge.py: {e}")


def _resolve_official_material_tree(bridge, official_material_name):
  descs = bridge.get("material_descs", {})
  desc = descs.get(official_material_name)
  if desc is not None:
    try:
      return desc.create()
    except Exception as e:
      print(f"Warning in blender_importer\materials_bridge.py: {e}")

  node_tree = bpy.data.node_groups.get(official_material_name)
  if node_tree is not None:
    return node_tree

  try:
    return bpy.data.node_groups.new(official_material_name, "ShaderNodeTree")
  except Exception:
    return None


def _attach_official_material_bridge(mat, edm_material, texture_nodes):
  """
  Add an official-EDM-compatible material group node so meshes imported by iEDM
  can be exported by the official exporter addon.
  """
  bridge = _ensure_official_material_bridge()
  if not bridge.get("available"):
    return False

  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  names = bridge.get("names", {})
  official_kind = _map_edm_material_to_official_kind(edm_material.material_name)
  official_name = _map_edm_material_to_official_name(edm_material.material_name, bridge)

  # Reuse existing compatible node if one is already present.
  group_node = None
  for node in nodes:
    node_tree = getattr(node, "node_tree", None)
    if node_tree and node_tree.name == official_name:
      group_node = node
      break

  if group_node is None:
    group_node = _create_official_group_node(nodes, bridge, official_kind, official_name)
    if group_node is None:
      return False
    group_node.location = (320, -360)
    group_node.width = 320

  trans_mode = _official_transparency_enum(getattr(edm_material, "blending", 0))
  shadow_mode = _official_shadow_enum(getattr(edm_material, "shadows", None))

  # Set custom node enums (preferred path for official exporter parser).
  _set_group_enum_property(group_node, "transparency", trans_mode)
  _set_group_enum_property(group_node, "deck_transparency", trans_mode)
  _set_group_enum_property(group_node, "glass_transparency", trans_mode)
  _set_group_enum_property(group_node, "shadow_caster", shadow_mode)
  _set_group_enum_property(group_node, "glass_shadow_caster", shadow_mode)

  if official_name in {names["default"], names["deck"], names["glass"]}:
    _set_group_socket_default(group_node, "Transparency", trans_mode)
  if official_name in {names["default"], names["glass"]}:
    _set_group_socket_default(group_node, "Shadow Caster", shadow_mode)

  # Map common EDM texture channels to official group inputs.
  tex0 = texture_nodes.get(0)  # diffuse/base
  tex1 = texture_nodes.get(1)  # normal
  tex2 = texture_nodes.get(2)  # spec/roughmet-ish
  tex3 = texture_nodes.get(3)  # decal/number atlas in many default materials
  tex_uv_channels = _texture_uv_channel_map(edm_material)

  # Route explicit EDM UV-channel assignments to texture vector inputs so the
  # official exporter can recover texture_coord_channels deterministically.
  for tex_idx, tex_node in texture_nodes.items():
    uv_channel = tex_uv_channels.get(tex_idx, None)
    if uv_channel is None:
      continue
    if int(uv_channel) <= 0:
      continue
    uv_node = _ensure_uv_map_node_for_channel(nodes, uv_channel)
    _link_uv_to_texture_vector(links, uv_node, tex_node)

  if official_name == names["default"]:
    if 3 not in tex_uv_channels and tex3:
      uv_node = _ensure_uv_map_node(nodes)
      if uv_node:
        _link_uv_to_texture_vector(links, uv_node, tex3)
    _link_texture_to_group_input(links, tex0, group_node, "Base Color")
    _link_texture_to_group_input(links, tex3, group_node, "Decal Color")
    _link_texture_to_group_input(links, tex3, group_node, "Decal Alpha*", "Alpha")
    _link_texture_to_group_input(links, tex1, group_node, "Normal (Non-Color)")
    _link_texture_to_group_input(links, tex2, group_node, "RoughMet (Non-Color)")
  elif official_name == names["deck"]:
    _link_texture_to_group_input(links, tex0, group_node, "Tiled Base Color")
    _link_texture_to_group_input(links, tex0, group_node, "Base_Alpha", "Alpha")
    _link_texture_to_group_input(links, tex0, group_node, "Decal Base")
    _link_texture_to_group_input(links, tex0, group_node, "Decal Alpha", "Alpha")
    _link_texture_to_group_input(links, tex1, group_node, "Tiled Normal (Non-Color)")
    _link_texture_to_group_input(links, tex2, group_node, "Tiled RoughMet (Non-Color)")
  elif official_name == names["glass"]:
    _link_texture_to_group_input(links, tex0, group_node, "Diffuse Color (Dirt)")
    _link_texture_to_group_input(links, tex0, group_node, "Diffuse Alpha*", "Alpha")
    _link_texture_to_group_input(links, tex0, group_node, "Glass Color (Color Filter)")
    _link_texture_to_group_input(links, tex0, group_node, "Glass Alpha*", "Alpha")
    _link_texture_to_group_input(links, tex1, group_node, "Normal (Non-Color)")
    _link_texture_to_group_input(links, tex2, group_node, "RoughMet (Non-Color)")
  elif official_name == names["mirror"]:
    _link_texture_to_group_input(links, tex0, group_node, "Base Color")
    _link_texture_to_group_input(links, tex1, group_node, "Normal (Non-Color)")
  else:
    _link_texture_to_group_input(links, tex0, group_node, "Emissive")

  material_output = _find_active_material_output(nodes)
  _link_group_surface_to_output(links, group_node, material_output)
  return True


