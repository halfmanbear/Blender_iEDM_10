"""
rna

Contains extensions to the blender data model.
EDMPropsGroup and EDMObjectSettings are owned by the official io_scene_edm
exporter plugin — we defer to its registrations and only add what it doesn't
provide, guarded by hasattr checks to avoid double-registration.
"""

import bpy

_owns_fallback_edmprops = False

# Keep enum ordering/indexes aligned with official io_scene_edm so values stay
# stable if that addon is enabled later and overrides Object.EDMProps.
_FALLBACK_SPECIAL_TYPE_ITEMS = [
    ('UNKNOWN_TYPE', 'unknown_type', "Unspecified object type: geometry/animation empty", 0),
    ('USER_BOX', 'user_box', "Box covering only geometry. Is used for backlight", 1),
    ('BOUNDING_BOX', 'bounding_box', "Box covering geometry and all animations. Is used for camera cutoff", 2),
    ('COLLISION_LINE', 'collision_line', "Geometry edge used for collision", 3),
    ('COLLISION_SHELL', 'collision_shell', "Geometry mesh used for collision", 4),
    ('CONNECTOR', 'connector', "Type Connector", 5),
    ('FAKE_LIGHT', 'fake_light', "Type BANO", 6),
    ('LIGHT_BOX', 'light_box', "Box - Limiter for light source", 7),
    ('NUMBER_TYPE', 'number_type', "Dynamic digits for bort number", 8),
    ('SKIN_BOX', 'skin_box', "Box covering bones geometry", 9),
    ('WATER_MASK', 'water_mask', "Plane to cover water in boats", 10),
]


class IEDMFallbackEDMPropsGroup(bpy.types.PropertyGroup):
    """Fallback for official EDM object props when io_scene_edm is unavailable."""
    bl_idname = "iedm.EDMPropsFallback"

    SPECIAL_TYPE: bpy.props.EnumProperty(
        name="Obj.type",
        description="Choose EDM object semantic type",
        items=_FALLBACK_SPECIAL_TYPE_ITEMS,
        default='UNKNOWN_TYPE',
    )
    CONNECTOR_EXT: bpy.props.StringProperty(
        name="connector ext",
        default="",
    )


class EDMObjectSettings(bpy.types.PropertyGroup):
    is_collision_shell: bpy.props.BoolProperty(
        name="Is Collision Shell",
        default=False
    )


def _is_fallback_edmprops_bound():
    try:
        prop = bpy.types.Object.bl_rna.properties.get("EDMProps")
        if prop is None:
            return False
        fixed = getattr(prop, "fixed_type", None)
        if fixed is None:
            return False
        return fixed.identifier == IEDMFallbackEDMPropsGroup.bl_rna.identifier
    except Exception:
        return False


def register():
    global _owns_fallback_edmprops
    if not hasattr(bpy.types.Action, "argument"):
        bpy.types.Action.argument = bpy.props.IntProperty(name="Argument", default=-1, min=-1)
    if not hasattr(bpy.types.Scene, "active_edm_argument"):
        bpy.types.Scene.active_edm_argument = bpy.props.IntProperty(name="Active Argument", default=-1, min=-1)
    if not hasattr(bpy.types.Scene, "edm_version"):
        bpy.types.Scene.edm_version = bpy.props.IntProperty(
            name="EDM Version",
            default=10,
            description="EDM file format version detected on import")
    if not hasattr(bpy.types.Material, "edm_material"):
        bpy.types.Material.edm_material = bpy.props.StringProperty(name="EDM Material Name")
    if not hasattr(bpy.types.Material, "edm_blending"):
        bpy.types.Material.edm_blending = bpy.props.StringProperty(name="EDM Blending Mode")

    if not hasattr(bpy.types.Object, "EDMProps"):
        try:
            bpy.utils.register_class(IEDMFallbackEDMPropsGroup)
        except RuntimeError:
            pass
        bpy.types.Object.EDMProps = bpy.props.PointerProperty(type=IEDMFallbackEDMPropsGroup)
        _owns_fallback_edmprops = True

    if not hasattr(bpy.types.Object, "edm"):
        bpy.utils.register_class(EDMObjectSettings)
        bpy.types.Object.edm = bpy.props.PointerProperty(type=EDMObjectSettings)


def unregister():
    global _owns_fallback_edmprops
    if hasattr(bpy.types.Object, "edm"):
        del bpy.types.Object.edm
        try:
            bpy.utils.unregister_class(EDMObjectSettings)
        except RuntimeError:
            pass

    # Only remove fallback EDMProps if this addon still owns the binding.
    if _owns_fallback_edmprops and hasattr(bpy.types.Object, "EDMProps") and _is_fallback_edmprops_bound():
        del bpy.types.Object.EDMProps
    if _owns_fallback_edmprops:
        try:
            bpy.utils.unregister_class(IEDMFallbackEDMPropsGroup)
        except RuntimeError:
            pass
    _owns_fallback_edmprops = False

    if hasattr(bpy.types.Material, "edm_blending"):
        del bpy.types.Material.edm_blending
    if hasattr(bpy.types.Material, "edm_material"):
        del bpy.types.Material.edm_material
    if hasattr(bpy.types.Scene, "edm_version"):
        del bpy.types.Scene.edm_version
    if hasattr(bpy.types.Scene, "active_edm_argument"):
        del bpy.types.Scene.active_edm_argument
    if hasattr(bpy.types.Action, "argument"):
        del bpy.types.Action.argument
