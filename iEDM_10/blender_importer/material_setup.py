"""Create PBR materials and coordinate textures, animation, and metadata."""

import fnmatch as fnmatch
import glob as glob
import json as json
import logging as logging
import os as os
import re as re

import bpy as bpy

from ..utils import action_fcurves as action_fcurves
from .export_damage_mask import DAMAGE_MASK_PROP as DAMAGE_MASK_PROP
from .export_damage_mask import damage_mask_payload as damage_mask_payload
from .material_animation import _ANIM_UNIFORM_TO_SOCKET as _ANIM_UNIFORM_TO_SOCKET
from .material_animation import (
    _ANIMATED_UNIFORM_TO_EDMPROPS as _ANIMATED_UNIFORM_TO_EDMPROPS,
)
from .material_animation import (
    _HEURISTIC_ANIM_UNIFORM_TARGETS as _HEURISTIC_ANIM_UNIFORM_TARGETS,
)
from .material_animation import _UV_ANIM_KEYWORDS as _UV_ANIM_KEYWORDS
from .material_animation import (
    _create_material_socket_animations as _create_material_socket_animations,
)
from .material_animation import (
    _create_material_uv_animations as _create_material_uv_animations,
)
from .material_animation import (
    _heuristic_uniform_category as _heuristic_uniform_category,
)
from .material_animation import _keyframe_material_socket as _keyframe_material_socket
from .material_animation import _logger as _logger
from .material_animation import (
    _looks_like_uv_anim_uniform as _looks_like_uv_anim_uniform,
)
from .material_animation import (
    _mat_ensure_node_tree_action as _mat_ensure_node_tree_action,
)
from .material_animation import _mat_set_linear_on_path as _mat_set_linear_on_path
from .material_animation import _normalized_uniform_name as _normalized_uniform_name
from .material_animation import (
    _resolve_animated_uniform_target as _resolve_animated_uniform_target,
)
from .material_creation import _create_material_node_tree as _create_material_node_tree
from .material_creation import _create_material_textures as _build_material_textures
from .material_payload import (
    _infer_material_texture_roles as _infer_material_texture_roles,
)
from .material_payload import (
    _map_animated_uniforms_to_edmprops as _map_animated_uniforms_to_edmprops,
)
from .material_payload import (
    _material_animated_uniform_payload as _material_animated_uniform_payload,
)
from .material_payload import _material_prop_scalar as _material_prop_scalar
from .material_payload import _material_prop_value as _material_prop_value
from .material_payload import _material_texture_payload as _material_texture_payload
from .material_payload import _material_uniform_payload as _material_uniform_payload
from .material_payload import _preserve_material_payload as _preserve_material_payload
from .material_payload import (
    _preserve_object_material_args as _preserve_object_material_args,
)
from .material_uv_shift import _UV_SHIFT_SLOTS as _UV_SHIFT_SLOTS
from .material_uv_shift import (
    _keyframe_material_uv_location as _keyframe_material_uv_location,
)
from .material_uv_shift import _uv_shift_mapping_node as _uv_shift_mapping_node
from .materials_bridge import (
    _attach_official_material_bridge as _attach_official_material_bridge,
)


def _find_texture_file(name):
    """
    Searches for a texture file given a basename without extension.

    The current working directory will be searched, as will any
    subdirectories called "textures/", for any file starting with the
    designated name '{name}.'
    """
    files = glob.glob(name + ".*")
    if not files:
        matcher = re.compile(fnmatch.translate(name + ".*"), re.IGNORECASE)
        files = [x for x in glob.glob("*.*") if matcher.match(x)]
        if not files:
            files = glob.glob("textures/" + name + ".*")
            if not files:
                matcher = re.compile(
                    fnmatch.translate("textures/" + name + ".*"), re.IGNORECASE
                )
                files = [x for x in glob.glob("textures/*.*") if matcher.match(x)]
                if not files:
                    print("Warning: Could not find texture named {}".format(name))
                    return None
    # print("Found {} as: {}".format(name, files))
    if len(files) > 1:
        print(
            "Warning: Multiple matches for texture '{}'; using {}".format(
                name, files[0]
            )
        )
        files = [files[0]]
    textureFilename = files[0]
    return os.path.abspath(textureFilename)


def _ensure_placeholder_texture_image(name):
    """Create a tiny generated image so the official exporter can still recover
    the EDM texture name from the image datablock when the source file is absent.
    """
    image_name = str(name or "__EMPTY__")
    existing = bpy.data.images.get(image_name)
    if existing is not None:
        return existing
    image = bpy.data.images.new(name=image_name, width=1, height=1, alpha=True)
    try:
        image.generated_color = (1.0, 1.0, 1.0, 1.0)
    except Exception:
        _logger.debug("Ignoring optional operation failure", exc_info=True)
    return image


def _wire_uv_transform(nodes, links, tex_def, tex_image):
    """Add a TexCoord and Mapping chain for a non-identity UV matrix."""
    matrix = getattr(tex_def, "matrix", None)
    if matrix is None:
        return
    try:
        is_id = all(
            abs(float(matrix[r][c]) - (1.0 if r == c else 0.0)) < 1e-5
            for r in range(4)
            for c in range(4)
        )
        if is_id:
            return
    except Exception:
        return
    try:
        loc, rot, scale = matrix.decompose()
    except Exception as e:
        print(
            "Warning: Could not decompose UV matrix for "
            f"'{getattr(tex_def, 'name', '')}': {e}"
        )
        return
    x = tex_image.location[0]
    y = tex_image.location[1]
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (x - 200, y)
    mapping.vector_type = "POINT"
    try:
        mapping.inputs["Location"].default_value = (loc.x, loc.y, loc.z)
        mapping.inputs["Rotation"].default_value = rot.to_euler("XYZ")
        mapping.inputs["Scale"].default_value = (scale.x, scale.y, scale.z)
    except Exception as e:
        print(f"Warning: Could not set mapping values: {e}")
    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (x - 400, y)
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], tex_image.inputs["Vector"])


def create_material(material):
    """Create a blender node-based PBR material from an EDM one."""
    mat, nodes, links, principled_bsdf = _create_material_node_tree(material)
    texture_nodes = _build_material_textures(
        material,
        nodes,
        links,
        _find_texture_file,
        _ensure_placeholder_texture_image,
        _wire_uv_transform,
    )
    _connect_principled_textures(nodes, links, principled_bsdf, texture_nodes)
    _set_principled_uniforms(material, links, principled_bsdf, texture_nodes)
    _finish_material(material, mat, nodes, principled_bsdf, texture_nodes)
    return mat


def _connect_principled_textures(nodes, links, principled_bsdf, texture_nodes):
    if 0 in texture_nodes:  # Diffuse
        tex_image = texture_nodes[0]
        tex_image.image.colorspace_settings.name = "sRGB"
        links.new(tex_image.outputs["Color"], principled_bsdf.inputs["Base Color"])

    if 1 in texture_nodes:  # Normal
        tex_image = texture_nodes[1]
        tex_image.image.colorspace_settings.name = "Non-Color"
        normal_map_node = nodes.new("ShaderNodeNormalMap")
        normal_map_node.location = (-200, -300)
        links.new(tex_image.outputs["Color"], normal_map_node.inputs["Color"])
        links.new(normal_map_node.outputs["Normal"], principled_bsdf.inputs["Normal"])

    if 2 in texture_nodes:  # Specular
        tex_image = texture_nodes[2]
        tex_image.image.colorspace_settings.name = "Non-Color"
        links.new(
            tex_image.outputs["Color"], principled_bsdf.inputs["Specular IOR Level"]
        )


def _set_principled_uniforms(material, links, principled_bsdf, texture_nodes):
    _metallic_materials = {"chrome_material", "aluminium_material"}
    if (material.material_name or "").lower() in _metallic_materials:
        principled_bsdf.inputs["Metallic"].default_value = 1.0
        # reflectionBlurring: 0.0 = mirror-sharp, 1.0 = fully diffuse.
        # chrome uses ~0.01 (near-perfect mirror); aluminium uses ~0.8 (brushed).
        reflection_blurring = material.uniforms.get("reflectionBlurring", None)
        if reflection_blurring is not None:
            try:
                principled_bsdf.inputs["Roughness"].default_value = max(
                    0.0, min(1.0, float(reflection_blurring))
                )
            except Exception:
                _logger.debug("Ignoring optional operation failure", exc_info=True)
    else:
        principled_bsdf.inputs["Metallic"].default_value = 0.0

    # Roughness from specPower (non-metallic materials only)
    specPower = material.uniforms.get("specPower", None)
    if (
        specPower is not None
        and (material.material_name or "").lower() not in _metallic_materials
    ):
        # This assumes specPower is in the common legacy range of 0-1024.
        # Higher specPower means a smaller highlight and lower roughness.
        # Clamp before sqrt: specPower above 1024 would produce a complex number.
        roughness = (
            max(0.0, 1.0 - (specPower / 1024.0)) ** 0.5
        )  # Using sqrt for a more perceptually linear mapping
        principled_bsdf.inputs["Roughness"].default_value = max(
            0.0, min(1.0, roughness)
        )
    elif (material.material_name or "").lower() not in _metallic_materials:
        principled_bsdf.inputs["Roughness"].default_value = 0.5

    # Specular from specFactor
    specFactor = material.uniforms.get("specFactor", None)
    if specFactor is not None:
        principled_bsdf.inputs["Specular IOR Level"].default_value = specFactor

    # Emissive from selfIlluminationValue/selfIlluminationColor (self-illum materials).
    # The official exporter stores these as animated_uniforms in emissive blocks.
    _self_illum_names = {
        "self_illum_material",
        "transparent_self_illum_material",
        "additive_self_illum_material",
        "additive_self_illum_color_material",
        "additive_self_illum_tex_material",
    }
    if (material.material_name or "").lower() in _self_illum_names:
        # Connect diffuse texture to Emission socket so the material visibly glows.
        if 0 in texture_nodes:
            try:
                links.new(
                    texture_nodes[0].outputs["Color"],
                    principled_bsdf.inputs["Emission Color"],
                )
            except Exception:
                _logger.debug("Ignoring optional operation failure", exc_info=True)
        # selfIlluminationValue drives emission strength.
        siv = material.uniforms.get("selfIlluminationValue", None)
        if siv is None:
            # Also check animated_uniforms, stored as a key list when animation is
            # not preserved.
            anim_siv = material.animated_uniforms.get("selfIlluminationValue", None)
            if anim_siv and hasattr(anim_siv, "__iter__"):
                try:
                    siv = float(next(iter(anim_siv)).value)
                except Exception:
                    _logger.debug("Ignoring optional operation failure", exc_info=True)
        try:
            principled_bsdf.inputs["Emission Strength"].default_value = (
                float(siv) if siv is not None else 1.0
            )
        except Exception:
            _logger.debug("Ignoring optional operation failure", exc_info=True)


def _finish_material(material, mat, nodes, principled_bsdf, texture_nodes):
    links = mat.node_tree.links
    if material.blending in (1, 2):  # Alpha blending or Alpha test
        # Connect diffuse texture alpha to Principled BSDF Alpha input
        if 0 in texture_nodes:
            links.new(
                texture_nodes[0].outputs["Alpha"], principled_bsdf.inputs["Alpha"]
            )
        mat.surface_render_method = "DITHERED"
    elif (
        material.blending == 3
    ):  # SUM_BLENDING (additive) — used by self-illumination materials
        # Additive blending: drive Alpha by texture alpha if present; use BLENDED mode.
        if 0 in texture_nodes:
            links.new(
                texture_nodes[0].outputs["Alpha"], principled_bsdf.inputs["Alpha"]
            )
        mat.surface_render_method = "BLENDED"

    # --- Handle Culling ---
    # EDM culling=0 → standard backface culling enabled; non-zero → two-sided.
    try:
        mat.use_backface_culling = getattr(material, "culling", 0) == 0
    except Exception:
        _logger.debug("Ignoring optional operation failure", exc_info=True)

    # Add official exporter-compatible EDM group node when available.
    official_attached = _attach_official_material_bridge(mat, material, texture_nodes)
    if not official_attached:
        print(
            f"Warning: Material bridge not attached for '{material.material_name}' "
            f"(material '{material.name}') — re-export may not recognise this material"
        )
    if official_attached:
        # Keep imported material node layout close to official reference scenes:
        # Output + official EDM group (+ texture nodes), without extra Principled.
        try:
            nodes.remove(principled_bsdf)
        except Exception as e:
            print(f"Warning in blender_importer/material_setup.py: {e}")

    # Set other material properties
    mat.edm_material = material.material_name
    mat.edm_blending = str(material.blending)
    _preserve_material_payload(mat, material)

    # Create node-tree animation keyframes so the exporter can read them back.
    _create_material_socket_animations(mat, material)
    _create_material_uv_animations(mat, material, texture_nodes)

    return
