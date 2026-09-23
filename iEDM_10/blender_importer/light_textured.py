import bpy
from mathutils import Vector

from .light_geometry import (
    _add_fake_spot_direction_child,
    _create_fake_light_mesh,
    _create_fake_omni_mesh,
    _default_fake_spot_uvs,
)
from .light_materials import (
    _create_fake_light_material,
    _find_fake_light_group_node,
    _find_or_create_emissive_texture_node,
)
from .light_real import _import_light_properties
from .materials_bridge import (
    _link_texture_to_group_input,
)
from .prelude import (
    _set_official_special_type,
)


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
            obj["_iedm_light_texture_uv_matrix"] = [
                float(v) for row in uv_transform for v in row
            ]
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
        _link_texture_to_group_input(
            material.node_tree.links, tex_node, group_node, "Emissive"
        )
    except Exception:
        pass


def _create_textured_light_surrogate(node):
    light_props = getattr(node, "lightProps", None) or {}
    has_spot_keys = any(k in light_props for k in ("Phi", "Theta", "phi", "theta"))
    name = node.name or "TexturedLight"

    if has_spot_keys:
        mesh = _create_fake_light_mesh(name, [Vector((0.0, 0.0, 0.0))])
        obj = bpy.data.objects.new(name, mesh)
        _set_official_special_type(obj, "FAKE_LIGHT")
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
        preview_light = bpy.data.lights.new(name=f"{name}_Preview", type="SPOT")
        try:
            _import_light_properties(node, obj, preview_light, "SPOT")
        finally:
            preview_ad = preview_light.animation_data
            preview_action = preview_ad.action if preview_ad else None
            bpy.data.lights.remove(preview_light)
            # The preview light's animation has no remaining user once the light is gone.
            if preview_action is not None and preview_action.users == 0:
                bpy.data.actions.remove(preview_action)
        bpy.context.collection.objects.link(obj)
        return obj

    mesh, location = _create_fake_omni_mesh(name, [Vector((0.0, 0.0, 0.0))])
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    _set_official_special_type(obj, "FAKE_LIGHT")
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
