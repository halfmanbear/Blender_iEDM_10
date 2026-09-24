"""Build official exporter material groups and retain bridge helper imports."""

import copy as copy

import bpy as bpy

from .exporter_properties import (
    _ensure_official_material_bridge as _ensure_official_material_bridge,
)
from .material_mapping import _SELF_ILLUM_MATERIALS as _SELF_ILLUM_MATERIALS
from .material_mapping import (
    _map_edm_material_to_official_kind as _map_edm_material_to_official_kind,
)
from .material_sockets import _ensure_uv_map_node as _ensure_uv_map_node
from .material_sockets import (
    _ensure_uv_map_node_for_channel as _ensure_uv_map_node_for_channel,
)
from .material_sockets import (
    _find_active_material_output as _find_active_material_output,
)
from .material_sockets import _find_group_input_socket as _find_group_input_socket
from .material_sockets import (
    _find_texture_node_by_name_substring as _find_texture_node_by_name_substring,
)
from .material_sockets import (
    _link_group_surface_to_output as _link_group_surface_to_output,
)
from .material_sockets import (
    _link_texture_to_any_group_input as _link_texture_to_any_group_input,
)
from .material_sockets import (
    _link_texture_to_group_input as _link_texture_to_group_input,
)
from .material_sockets import _link_uv_to_texture_vector as _link_uv_to_texture_vector
from .material_sockets import (
    _map_edm_material_to_official_name as _map_edm_material_to_official_name,
)
from .material_sockets import _material_scalar as _material_scalar
from .material_sockets import _official_shadow_enum as _official_shadow_enum
from .material_sockets import _official_transparency_enum as _official_transparency_enum
from .material_sockets import _route_explicit_uv_channels as _route_explicit_uv_channels
from .material_sockets import _set_group_enum_property as _set_group_enum_property
from .material_sockets import _set_group_socket_default as _set_group_socket_default
from .material_sockets import _texture_uv_channel_map as _texture_uv_channel_map
from .material_sockets import _uv_map_name_from_channel as _uv_map_name_from_channel


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
            print(f"Warning in blender_importer/materials_bridge.py: {e}")

    if node_tree is None:
        return None
    try:
        fallback_node = nodes.new("ShaderNodeGroup")
        fallback_node.node_tree = node_tree
        return fallback_node
    except Exception:
        return None


def _prepare_official_material_node(mat, edm_material):
    bridge = _ensure_official_material_bridge()
    if not bridge.get("available"):
        return None
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    names = bridge.get("names", {})
    official_kind = _map_edm_material_to_official_kind(edm_material.material_name)
    official_name = _map_edm_material_to_official_name(
        edm_material.material_name, bridge
    )
    group_node = next(
        (
            node
            for node in nodes
            if getattr(getattr(node, "node_tree", None), "name", None) == official_name
        ),
        None,
    )
    if group_node is None:
        group_node = _create_official_group_node(
            nodes, bridge, official_kind, official_name
        )
        if group_node is None:
            return None
        group_node.location = (320, -360)
        group_node.width = 320
    return bridge, nodes, links, names, official_name, group_node


def _resolve_official_material_tree(bridge, official_material_name):
    descs = bridge.get("material_descs", {})
    desc = descs.get(official_material_name)
    if desc is not None:
        try:
            if bpy.app.version >= (5, 0, 0):
                # Older exporter descriptors contain removed RGB nodes. Use
                # plain RNA identifiers: the exporter's Enum replacements can
                # fail nodes.new() in Blender 5.2/Python 3.13. Copy descriptors
                # so the exporter's cached definitions are not modified.
                desc = copy.copy(desc)
                desc.nodes = [copy.copy(node) for node in desc.nodes]
                replacements = {
                    "ShaderNodeSeparateRGB": "ShaderNodeSeparateColor",
                    "ShaderNodeCombineRGB": "ShaderNodeCombineColor",
                }
                for node in desc.nodes:
                    replacement = replacements.get(node.bl_idname)
                    if replacement:
                        node.bl_idname = replacement
                        node.mode = "RGB"
                        node.attrs = list(node.attrs)
                        if "mode" not in node.attrs:
                            node.attrs.append("mode")
            return desc.create()
        except Exception as e:
            print(f"Warning in blender_importer/materials_bridge.py: {e}")

    node_tree = bpy.data.node_groups.get(official_material_name)
    if node_tree is not None:
        return node_tree

    try:
        return bpy.data.node_groups.new(official_material_name, "ShaderNodeTree")
    except Exception:
        return None


def _attach_official_material_bridge(mat, edm_material, texture_nodes):
    """Attach an official-compatible material group for round-trip export."""
    prepared = _prepare_official_material_node(mat, edm_material)
    if prepared is None:
        return False
    bridge, nodes, links, names, official_name, group_node = prepared

    trans_mode = _official_transparency_enum(getattr(edm_material, "blending", 0))
    mat_lower = (getattr(edm_material, "material_name", "") or "").lower()
    if mat_lower.startswith("additive_self_illum"):
        # The exporter only emits additive_self_illum_* for SUM_BLENDING_SI;
        # the file itself stores these with plain sum blending (3).
        trans_mode = "SUM_BLENDING_SI"
    shadow_mode = _official_shadow_enum(getattr(edm_material, "shadows", None))

    _set_group_enum_property(group_node, "transparency", trans_mode)
    _set_group_enum_property(group_node, "deck_transparency", trans_mode)
    _set_group_enum_property(group_node, "glass_transparency", trans_mode)
    _set_group_enum_property(group_node, "shadow_caster", shadow_mode)
    _set_group_enum_property(group_node, "glass_shadow_caster", shadow_mode)

    if official_name in {names["default"], names["deck"], names["glass"]}:
        _set_group_socket_default(group_node, "Transparency", trans_mode)
    if official_name in {names["default"], names["glass"]}:
        _set_group_socket_default(group_node, "Shadow Caster", shadow_mode)
    if official_name in {names["default"], names["deck"]}:
        _set_group_socket_default(
            group_node, "DecalId", int(getattr(edm_material, "decal", 0) or 0)
        )

    opacity_value = _material_scalar(edm_material, "opacityValue", default=1.0)
    emissive_value = _material_scalar(
        edm_material, "selfIlluminationValue", "emissiveValue", default=0.0
    )
    ao_value = _material_scalar(edm_material, "aoValue", "lightMapValue", default=0.0)
    _set_group_socket_default(group_node, "Opacity Value", opacity_value)
    _set_group_socket_default(group_node, "Opacity Value*", opacity_value)
    _set_group_socket_default(group_node, "Emissive Value", emissive_value)
    _set_group_socket_default(group_node, "AO Value*", ao_value)
    _set_group_socket_default(group_node, "LightMap Value*", ao_value)
    _set_group_socket_default(group_node, "LightMap Value", ao_value)

    tex0 = texture_nodes.get(0)  # diffuse/base
    tex1 = texture_nodes.get(1)  # normal
    tex2 = texture_nodes.get(2) or texture_nodes.get(13)  # spec/roughmet-ish
    tex3 = texture_nodes.get(3)  # decal/number atlas in many default materials
    tex4 = texture_nodes.get(4)  # deck decal roughmet
    tex5 = texture_nodes.get(5)  # default/glass damage base or deck wet map
    tex7 = texture_nodes.get(7)  # deck damage base
    tex8 = texture_nodes.get(8)  # emissive/light texture
    tex9 = texture_nodes.get(9)  # lightmap or deck damage mask
    tex10 = texture_nodes.get(10) or tex1  # alternate normal slot
    # slot 1 is the normal map; slot 10 the damage normal (fa-18c f18c2 has both)
    tex_normal = tex1 or texture_nodes.get(10)
    tex14 = texture_nodes.get(14)  # glass filter
    tex18 = texture_nodes.get(18)  # damage mask
    tex_flir = _find_texture_node_by_name_substring(texture_nodes, "flir")
    tex_uv_channels = _texture_uv_channel_map(edm_material)

    _route_explicit_uv_channels(nodes, links, texture_nodes, tex_uv_channels)

    if official_name == names["default"]:
        from .material_default_links import link_default_material

        link_default_material(
            nodes, links, group_node, edm_material, texture_nodes, tex_uv_channels
        )
    elif official_name == names["deck"]:
        _link_texture_to_group_input(links, tex0, group_node, "Tiled Base Color")
        _link_texture_to_group_input(links, tex0, group_node, "Base_Alpha", "Alpha")
        _link_texture_to_group_input(links, tex3, group_node, "Decal Base")
        _link_texture_to_group_input(links, tex3, group_node, "Decal Alpha", "Alpha")
        _link_texture_to_group_input(
            links, tex4, group_node, "Decal RoughMetAO (Non-Color)"
        )
        _link_texture_to_group_input(
            links, tex10, group_node, "Tiled Normal (Non-Color)"
        )
        _link_texture_to_group_input(
            links, tex2, group_node, "Tiled RoughMet (Non-Color)"
        )
        _link_texture_to_group_input(links, tex7, group_node, "Damage Base")
        _link_texture_to_group_input(links, tex9, group_node, "Damage Map (Non-Color)")
        _link_texture_to_group_input(
            links, tex9, group_node, "Damage Map Alpha", "Alpha"
        )
        _link_texture_to_group_input(links, tex5, group_node, "Wet Map (Non-Color)")
    elif official_name == names["glass"]:
        _link_texture_to_group_input(links, tex0, group_node, "Diffuse Color (Dirt)")
        _link_texture_to_group_input(links, tex0, group_node, "Diffuse Alpha*", "Alpha")
        if tex14:
            _link_texture_to_group_input(
                links, tex14, group_node, "Glass Color (Color Filter)"
            )
            _link_texture_to_group_input(
                links, tex14, group_node, "Glass Alpha*", "Alpha"
            )
        else:
            _set_group_socket_default(
                group_node, "Glass Color (Color Filter)", (1.0, 1.0, 1.0, 1.0)
            )
            _set_group_socket_default(group_node, "Glass Alpha*", 1.0)
        _link_texture_to_group_input(
            links, tex_normal, group_node, "Normal (Non-Color)"
        )
        _link_texture_to_group_input(links, tex2, group_node, "RoughMet (Non-Color)")
        _link_texture_to_group_input(links, tex_flir, group_node, "Flir")
        _link_texture_to_group_input(links, tex5, group_node, "Damage Base")
        _link_texture_to_group_input(links, tex18, group_node, "Damage Map (Non-Color)")
        _link_texture_to_group_input(
            links, tex18, group_node, "Damage Map Alpha", "Alpha"
        )
        glass_type_val = (
            "GLASS_COCKPIT" if mat_lower == "glass_material" else "GLASS_INSTRUMENTAL"
        )
        _set_group_enum_property(group_node, "glass_type", glass_type_val)
    elif official_name == names["mirror"]:
        _link_texture_to_group_input(links, tex0, group_node, "Base Color")
        _link_texture_to_group_input(
            links, tex_normal, group_node, "Normal (Non-Color)"
        )
    else:
        _link_texture_to_group_input(links, tex0, group_node, "Emissive")
        _link_texture_to_any_group_input(
            links, tex8, group_node, ("Emissive", "Emission")
        )

    material_output = _find_active_material_output(nodes)
    _link_group_surface_to_output(links, group_node, material_output)
    return True
