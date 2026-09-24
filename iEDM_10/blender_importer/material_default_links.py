"""Texture wiring for the official EDM default material group."""

from .materials_bridge import (
    _SELF_ILLUM_MATERIALS,
    _ensure_uv_map_node,
    _find_texture_node_by_name_substring,
    _link_texture_to_group_input,
    _link_uv_to_texture_vector,
    _set_group_socket_default,
)

# The exporter picks the emissive type from which sockets are linked:
# additive_self_illum(_color) has no albedo texture, self_illum and
# additive_self_illum_color have no emissive texture (colour-driven).
_SELF_ILLUM_NO_ALBEDO = {
    "additive_self_illum_material",
    "additive_self_illum_color_material",
}
_SELF_ILLUM_COLOR_DRIVEN = {
    "self_illum_material",
    "additive_self_illum_color_material",
}


def _self_illum_color(edm_material):
    color = getattr(edm_material, "uniforms", {}).get("selfIlluminationColor")
    if color is None:
        return (1.0, 1.0, 1.0, 1.0)
    rgb = [float(c) for c in list(color)[:3]]
    return (*rgb, 1.0)


def _link_emissive(links, group_node, edm_material, mat_lower, tex0, tex8):
    if mat_lower in _SELF_ILLUM_COLOR_DRIVEN:
        _set_group_socket_default(
            group_node, "Emissive", _self_illum_color(edm_material)
        )
    elif tex8 is None and mat_lower in _SELF_ILLUM_MATERIALS:
        # Self-illum materials carry their glow texture in the base slot.
        _link_texture_to_group_input(links, tex0, group_node, "Emissive")
    else:
        _link_texture_to_group_input(links, tex8, group_node, "Emissive")
    _link_texture_to_group_input(links, tex8, group_node, "Emissive Mask", "Alpha")


def link_default_material(
    nodes, links, group_node, edm_material, texture_nodes, tex_uv_channels
):
    """Link imported texture slots to the default material group inputs."""
    mat_lower = (getattr(edm_material, "material_name", "") or "").lower()
    tex0 = texture_nodes.get(0)  # diffuse/base
    tex2 = texture_nodes.get(2) or texture_nodes.get(13)  # spec/roughmet-ish
    tex3 = texture_nodes.get(3)  # decal/number atlas
    tex5 = texture_nodes.get(5)  # damage base
    tex8 = texture_nodes.get(8)  # emissive/light texture
    tex9 = texture_nodes.get(9)  # lightmap
    tex10 = texture_nodes.get(10) or texture_nodes.get(1)  # normal
    tex18 = texture_nodes.get(18)  # damage mask
    tex_flir = _find_texture_node_by_name_substring(texture_nodes, "flir")

    if 3 not in tex_uv_channels and tex3:
        uv_node = _ensure_uv_map_node(nodes)
        if uv_node:
            _link_uv_to_texture_vector(links, uv_node, tex3)
    base_tex = None if mat_lower in _SELF_ILLUM_NO_ALBEDO else tex0
    link = _link_texture_to_group_input
    link(links, base_tex, group_node, "Base Color")
    link(links, base_tex, group_node, "Base Alpha*", "Alpha")
    link(links, tex3, group_node, "Decal Color")
    link(links, tex3, group_node, "Decal Alpha*", "Alpha")
    link(links, tex10, group_node, "Normal (Non-Color)")
    link(links, tex2, group_node, "RoughMet (Non-Color)")
    link(links, tex9, group_node, "LightMap (Non-Color)")
    _link_emissive(links, group_node, edm_material, mat_lower, tex0, tex8)
    link(links, tex_flir, group_node, "Flir")
    link(links, tex5, group_node, "Damage Base")
    link(links, tex10, group_node, "Damage Normal (Non-Color)")
    link(links, tex18, group_node, "Damage Map (Non-Color)")
    link(links, tex18, group_node, "Damage Map Alpha", "Alpha")
