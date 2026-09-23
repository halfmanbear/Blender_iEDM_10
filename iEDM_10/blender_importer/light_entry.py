import bpy

from .light_real import _import_light_properties
from .light_textured import _create_textured_light_surrogate


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
    obj = bpy.data.objects.new(name=node.name, object_data=light_data)
    _import_light_properties(node, obj, light_data, light_type)
    bpy.context.collection.objects.link(obj)
    return obj
