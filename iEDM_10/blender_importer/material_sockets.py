"""Resolve material sockets, UV channels, and exporter enum values."""

from .material_mapping import _map_edm_material_to_official_kind


def _set_group_enum_property(group_node, prop_name, value):
    if not group_node or not prop_name:
        return
    if not hasattr(group_node, prop_name):
        return
    try:
        setattr(group_node, prop_name, value)
    except Exception as e:
        print(f"Warning in blender_importer/materials_bridge.py: {e}")


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
        2: "Z_TEST",
        3: "SUM_BLENDING",
        4: "SUM_BLENDING_SI",
        6: "SHADOWED_BLENDING",
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
            print(f"Warning in blender_importer/materials_bridge.py: {e}")
        return


def _material_scalar(edm_material, *names, default=None):
    if edm_material is None:
        return default
    uniforms = getattr(edm_material, "uniforms", None) or {}
    anim_uniforms = getattr(edm_material, "animated_uniforms", None) or {}
    for name in names:
        value = uniforms.get(name, None)
        if value is None:
            value = anim_uniforms.get(name, None)
            if value is not None and hasattr(value, "keys"):
                keys = list(getattr(value, "keys", []) or [])
                value = getattr(keys[0], "value", None) if keys else None
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return default


def _find_group_input_socket(group_node, socket_name):
    if not group_node:
        return None
    for socket in group_node.inputs:
        if socket.name == socket_name:
            return socket
    return None


def _route_explicit_uv_channels(nodes, links, texture_nodes, uv_channels):
    """Preserve explicit EDM texture coordinate channel assignments."""
    for tex_idx, tex_node in texture_nodes.items():
        uv_channel = uv_channels.get(tex_idx)
        if uv_channel is None or int(uv_channel) <= 0:
            continue
        uv_node = _ensure_uv_map_node_for_channel(nodes, uv_channel)
        _link_uv_to_texture_vector(links, uv_node, tex_node)


def _link_texture_to_group_input(
    links, texture_node, group_node, input_name, output_name="Color"
):
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
        print(f"Warning in blender_importer/materials_bridge.py: {e}")


def _link_texture_to_any_group_input(
    links, texture_node, group_node, input_names, output_name="Color"
):
    if not texture_node or not group_node:
        return False
    for input_name in input_names:
        if _find_group_input_socket(group_node, input_name) is None:
            continue
        _link_texture_to_group_input(
            links, texture_node, group_node, input_name, output_name
        )
        return True
    return False


def _find_texture_node_by_name_substring(texture_nodes, needle):
    if not texture_nodes or not needle:
        return None
    needle = str(needle).lower()
    for tex_node in texture_nodes.values():
        label = str(getattr(tex_node, "label", "") or "")
        name = str(getattr(tex_node, "name", "") or "")
        image_name = ""
        try:
            image_name = str(
                getattr(getattr(tex_node, "image", None), "name", "") or ""
            )
        except Exception:
            image_name = ""
        haystack = " ".join((label, name, image_name)).lower()
        if needle in haystack:
            return tex_node
    return None


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
        print(f"Warning in blender_importer/materials_bridge.py: {e}")
    try:
        links.new(from_socket, to_socket)
    except Exception as e:
        print(f"Warning in blender_importer/materials_bridge.py: {e}")


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
            print(f"Warning in blender_importer/materials_bridge.py: {e}")
    try:
        links.new(from_socket, to_socket)
    except Exception as e:
        print(f"Warning in blender_importer/materials_bridge.py: {e}")
