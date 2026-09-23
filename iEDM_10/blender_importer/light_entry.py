import bpy

from .light_real import _import_light_properties
from .light_textured import _create_textured_light_surrogate


def _unlink_default_emission(light_data):
    """Disconnect the Emission -> Light Output link Blender 5.x adds to new lights.

    The official exporter reads the colour from a linked Emission node when one
    exists; its RGBA default crashes pyedm.PropertyFloat3 and would override the
    imported colour. Unlinked, the exporter falls back to light_data.color.
    """
    tree = getattr(light_data, "node_tree", None)
    if tree is None:
        return
    for link in list(tree.links):
        if (
            link.from_node.bl_idname == "ShaderNodeEmission"
            and link.to_node.bl_idname == "ShaderNodeOutputLight"
        ):
            tree.links.remove(link)


def create_lamp(node):
    """Creates a blender lamp from an edm renderable LampNode"""
    if getattr(node, "texture", None) is not None:
        surrogate = _create_textured_light_surrogate(node)
        if surrogate is not None:
            return surrogate

    light_props = getattr(node, "lightProps", None) or {}
    has_spot_keys = any(k in light_props for k in ("Phi", "Theta", "phi", "theta"))
    light_type = "SPOT" if has_spot_keys else "POINT"
    light_data = bpy.data.lights.new(name=node.name, type=light_type)
    _unlink_default_emission(light_data)
    obj = bpy.data.objects.new(name=node.name, object_data=light_data)
    _import_light_properties(node, obj, light_data, light_type)
    bpy.context.collection.objects.link(obj)
    return obj
