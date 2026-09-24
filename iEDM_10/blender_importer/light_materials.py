import logging
import math
import os

import bpy

from ..utils import chdir
from .material_setup import _find_texture_file
from .materials_bridge import (
    _create_official_group_node,
    _link_texture_to_group_input,
)
from .prelude import (
    _ensure_official_material_bridge,
    _import_ctx,
)

_logger = logging.getLogger(__name__)


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
    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    # Create the official EDM group node
    group_node = _create_official_group_node(nodes, bridge, kind, official_name)
    if group_node is None:
        # Last resort: create a bare ShaderNodeGroup with a named node tree
        try:
            group_node = nodes.new("ShaderNodeGroup")
            group_node.node_tree = bpy.data.node_groups.get(official_name)
            if group_node.node_tree is None:
                group_node.node_tree = bpy.data.node_groups.new(
                    official_name, "ShaderNodeTree"
                )
        except Exception:
            return mat  # Return material without group — better than nothing

    group_node.location = (0, 0)

    # Link group output to material output (if there's an output socket)
    if group_node.outputs:
        try:
            links.new(group_node.outputs[0], output_node.inputs["Surface"])
        except Exception as e:
            print(f"Warning in blender_importer/lights.py: {e}")

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
    if (
        material is None
        or not getattr(material, "use_nodes", False)
        or not getattr(material, "node_tree", None)
    ):
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
        if hasattr(node, "inputs") and any(
            getattr(sock, "name", None) == "Emissive"
            for sock in getattr(node, "inputs", ())
        ):
            return node
    return None


def _material_matches_fake_light_kind(material, kind):
    if (
        material is None
        or not getattr(material, "use_nodes", False)
        or not getattr(material, "node_tree", None)
    ):
        return False
    bridge = _ensure_official_material_bridge()
    official_name = bridge.get("names", {}).get(kind)
    official_node_type = bridge.get("node_types", {}).get(kind)
    for node in material.node_tree.nodes:
        node_tree = getattr(node, "node_tree", None)
        node_tree_name = getattr(node_tree, "name", None)
        if (
            official_node_type
            and getattr(node, "bl_idname", None) == official_node_type
        ):
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
        if (
            image
            and os.path.splitext(os.path.basename(image.filepath))[0].lower()
            == str(texture_name).lower()
        ):
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
                image = bpy.data.images.new(
                    name=str(texture_name), width=1, height=1, alpha=True
                )
            tex_image = nodes.new("ShaderNodeTexImage")
            tex_image.image = image
            tex_image.location = (-350, 0)
            return tex_image
        except Exception as e:
            print(f"Warning in blender_importer/lights.py: {e}")
            return None
    try:
        tex_image = nodes.new("ShaderNodeTexImage")
        tex_image.image = bpy.data.images.load(filename)
        tex_image.image.colorspace_settings.name = "sRGB"
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
                _link_texture_to_group_input(
                    material.node_tree.links, tex_node, group_node, "Emissive"
                )
            except Exception:
                _logger.debug("Ignoring optional operation failure", exc_info=True)

    uniforms = getattr(edm_material, "uniforms", None) or {}
    anim_uniforms = getattr(edm_material, "animated_uniforms", None) or {}

    luminance_prop = anim_uniforms.get("luminance", uniforms.get("luminance"))
    luminance_val = None
    if hasattr(luminance_prop, "keys") and getattr(luminance_prop, "keys", None):
        luminance_val = getattr(luminance_prop.keys[0], "value", None)
    elif luminance_prop is not None:
        luminance_val = luminance_prop
    if luminance_val is not None:
        _set_material_group_input(
            group_node, "Luminance", _safe_float(luminance_val, 1.0)
        )

    shift = uniforms.get("shiftToCamera")
    if shift is not None:
        _set_material_group_input(group_node, "ShiftToCamera", _safe_float(shift, 0.0))

    size_factors = uniforms.get("sizeFactors")
    size_vec = _safe_vec(size_factors, default=(4.0, 1000.0, 0.0))
    if len(size_vec) >= 1:
        _set_material_group_input(
            group_node, "MinSizePixels", _safe_float(size_vec[0], 4.0)
        )
    if len(size_vec) >= 2:
        _set_material_group_input(
            group_node, "MaxDistance", _safe_float(size_vec[1], 1000.0)
        )

    if kind == "fake_spot":
        _apply_fake_spot_cone_uniforms(group_node, uniforms)
        specular = uniforms.get("specularAmount")
        if specular is not None:
            _set_material_group_input(
                group_node, "SpecularAmount", _safe_float(specular, 0.0)
            )


def _apply_fake_spot_cone_uniforms(group_node, uniforms):
    """Apply optional fake spot cone angles from the EDM material payload."""
    cone_vec = _safe_vec(uniforms.get("coneSetup"), default=())
    if len(cone_vec) < 2:
        return
    try:
        inner_angle = math.degrees(math.acos(max(-1.0, min(1.0, cone_vec[0]))))
        outer_angle = math.degrees(math.acos(max(-1.0, min(1.0, cone_vec[1]))))
        _set_material_group_input(group_node, "Inner Angle", inner_angle)
        _set_material_group_input(group_node, "Outer Angle", outer_angle)
    except Exception:
        _logger.debug("Ignoring optional operation failure", exc_info=True)
