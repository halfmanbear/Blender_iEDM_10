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
    TWO_SIDED: bpy.props.BoolProperty(
        name="two sided object",
        default=False,
    )
    SURFACE_MODE: bpy.props.BoolProperty(
        name="surface mode",
        default=False,
    )
    DAMAGE_ARG: bpy.props.IntProperty(
        name="Damage arg",
        default=-1,
        min=-1,
    )
    AO_ARG: bpy.props.IntProperty(
        name="LightMap arg",
        default=-1,
        min=-1,
    )
    COLOR_ARG: bpy.props.IntProperty(
        name="Base color arg",
        default=-1,
        min=-1,
    )
    EMISSIVE_ARG: bpy.props.IntProperty(
        name="Emissive value arg",
        default=-1,
        min=-1,
    )
    EMISSIVE_COLOR_ARG: bpy.props.IntProperty(
        name="Emissive color arg",
        default=-1,
        min=-1,
    )
    UV_LB: bpy.props.FloatVectorProperty(
        name="tex_coords_lb",
        size=2,
        default=(0.0, 0.0),
        subtype='COORDINATES',
    )
    UV_RT: bpy.props.FloatVectorProperty(
        name="tex_coords_rt",
        size=2,
        default=(1.0, 1.0),
        subtype='COORDINATES',
    )
    SIZE: bpy.props.FloatProperty(
        name="size",
        default=3.0,
        min=0.0,
    )
    UV_LB_BACK: bpy.props.FloatVectorProperty(
        name="tex_coords_lb_back",
        size=2,
        default=(0.0, 0.0),
    )
    UV_RT_BACK: bpy.props.FloatVectorProperty(
        name="tex_coords_rt_back",
        size=2,
        default=(1.0, 1.0),
    )
    ANIMATED_BRIGHTNESS: bpy.props.FloatProperty(
        name="object luminance",
        default=1.0,
        min=0.0,
        options={'ANIMATABLE'},
    )
    LIGHT_SOFTNESS: bpy.props.FloatProperty(
        name="light softness",
        default=0.0,
        min=0.0,
        options={'ANIMATABLE'},
    )
    NUMBER_UV_X_ARG: bpy.props.IntProperty(
        name="xArg",
        default=-1,
        min=-1,
    )
    NUMBER_UV_X_SCALE: bpy.props.FloatProperty(
        name="xScale",
        default=0.0,
        min=0.0,
        max=1.0,
    )
    NUMBER_UV_Y_ARG: bpy.props.IntProperty(
        name="yArg",
        default=-1,
        min=-1,
    )
    NUMBER_UV_Y_SCALE: bpy.props.FloatProperty(
        name="yScale",
        default=0.0,
        min=0.0,
        max=1.0,
    )
    LIGHT_COLOR_ARG: bpy.props.IntProperty(
        name="Light color arg",
        default=-1,
        min=-1,
    )
    LIGHT_POWER_ARG: bpy.props.IntProperty(
        name="Light power arg",
        default=-1,
        min=-1,
    )
    LIGHT_DISTANCE_ARG: bpy.props.IntProperty(
        name="Light distance arg",
        default=-1,
        min=-1,
    )
    LIGHT_SPECULAR_ARG: bpy.props.IntProperty(
        name="Light specular arg",
        default=-1,
        min=-1,
    )
    LIGHT_SPOT_SHAPE_ARG: bpy.props.IntProperty(
        name="Light spot shape arg",
        default=-1,
        min=-1,
    )
    LIGHT_VOLUME_RADIUS_FACTOR: bpy.props.FloatProperty(
        name="Volume radius factor",
        default=0.0,
        min=0.0,
        max=1.0,
    )
    LIGHT_VOLUME_DENSITY_FACTOR: bpy.props.FloatProperty(
        name="Volume density factor",
        default=0.0,
        min=0.0,
        max=1.0,
    )
    LIGHT_VOLUME_NEAR_DISTANCE: bpy.props.FloatProperty(
        name="Volume near distance",
        default=0.0,
        min=0.0,
    )
    LIGHT_VOLUME_TYPE: bpy.props.EnumProperty(
        name="Volume type",
        items=[
            ('LANDING', 'landing', "light type: landing", 0),
            ('NAV', 'nav', "light type: nav", 1),
            ('TAXI', 'taxi', "light type: taxi", 2),
            ('BANO', 'bano', "light type: bano", 3),
            ('NONE', 'none', "", 4),
        ],
        default="NONE",
    )
    OPACITY_VALUE_ARG: bpy.props.IntProperty(
        name="Opacity value arg",
        default=-1,
        min=-1,
    )


class EDMObjectSettings(bpy.types.PropertyGroup):
    is_collision_shell: bpy.props.BoolProperty(
        name="Is Collision Shell",
        default=False
    )
    billboard_type: bpy.props.EnumProperty(
        name="Billboard Type",
        items=[
            ('direction', 'direction', "Directional billboard", 0),
            ('point', 'point', "Point billboard", 1),
        ],
        default='point',
    )
    billboard_axis: bpy.props.EnumProperty(
        name="Billboard Axis",
        items=[
            ('all', 'all', "All axes", 0),
            ('x', 'x', "X axis", 1),
            ('y', 'y', "Y axis", 2),
            ('z', 'z', "Z axis", 3),
            ('along_x', 'along_x', "Along X", 4),
            ('along_y', 'along_y', "Along Y", 5),
            ('along_z', 'along_z', "Along Z", 6),
        ],
        default='all',
    )
    is_connector: bpy.props.BoolProperty(
        name="Is Connector",
        default=False
    )
    is_collision_line: bpy.props.BoolProperty(
        name="Is Collision Line",
        default=False
    )
    is_renderable: bpy.props.BoolProperty(
        name="Is Renderable",
        default=False
    )
    damage_argument: bpy.props.IntProperty(
        name="Damage Argument",
        default=-1,
        min=-1,
    )
    is_lod_root: bpy.props.BoolProperty(
        name="Is LOD Root",
        default=False
    )
    lod_min_distance: bpy.props.FloatProperty(
        name="LOD Min Distance",
        default=0.0,
        min=0.0,
    )
    lod_max_distance: bpy.props.FloatProperty(
        name="LOD Max Distance",
        default=0.0,
        min=0.0,
    )
    nouse_lod_distance: bpy.props.BoolProperty(
        name="No Use LOD Distance",
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
