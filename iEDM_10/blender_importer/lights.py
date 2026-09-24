"""Compatibility imports for the focused implementation modules."""

import json as json
import math as math
import os as os

import bpy as bpy
from mathutils import Matrix as Matrix
from mathutils import Vector as Vector

from ..edm_format.mathtypes import vector_to_blender as vector_to_blender

# Public names formerly imported by this module remain available.
from ..utils import action_fcurves as action_fcurves
from ..utils import chdir as chdir
from .light_animation import (
    _apply_animated_fake_omni_brightness as _apply_animated_fake_omni_brightness,
)
from .light_animation import (
    _apply_fake_light_animation_payload as _apply_fake_light_animation_payload,
)
from .light_animation import (
    _has_animated_fake_omni_payload as _has_animated_fake_omni_payload,
)
from .light_billboard import _billboard_plane_axes as _billboard_plane_axes
from .light_billboard import (
    _create_billboard_surrogate_mesh as _create_billboard_surrogate_mesh,
)
from .light_billboard import (
    _create_official_render_material as _create_official_render_material,
)
from .light_billboard import create_billboard as create_billboard
from .light_entry import create_lamp as create_lamp
from .light_fake import create_fake_als_lights as create_fake_als_lights
from .light_fake import create_fake_omni_lights as create_fake_omni_lights
from .light_fake import create_fake_spot_lights as create_fake_spot_lights
from .light_geometry import (
    _add_fake_spot_direction_child as _add_fake_spot_direction_child,
)
from .light_geometry import _axis_box_layout as _axis_box_layout
from .light_geometry import _classify_fake_spot_mode as _classify_fake_spot_mode
from .light_geometry import _create_axis_box_mesh as _create_axis_box_mesh
from .light_geometry import _create_fake_light_mesh as _create_fake_light_mesh
from .light_geometry import _create_fake_omni_mesh as _create_fake_omni_mesh
from .light_geometry import _default_fake_spot_uvs as _default_fake_spot_uvs
from .light_geometry import _fake_light_world_from_edm as _fake_light_world_from_edm
from .light_materials import (
    _apply_fake_light_material_payload as _apply_fake_light_material_payload,
)
from .light_materials import _create_fake_light_material as _create_fake_light_material
from .light_materials import _find_fake_light_group_node as _find_fake_light_group_node
from .light_materials import (
    _find_or_create_emissive_texture_node as _find_or_create_emissive_texture_node,
)
from .light_materials import _material_for_fake_light as _material_for_fake_light
from .light_materials import (
    _material_matches_fake_light_kind as _material_matches_fake_light_kind,
)
from .light_materials import _safe_float as _safe_float
from .light_materials import _safe_vec as _safe_vec
from .light_materials import _set_material_group_input as _set_material_group_input
from .light_real import _import_light_properties as _import_light_properties
from .light_textured import (
    _apply_textured_light_material as _apply_textured_light_material,
)
from .light_textured import (
    _create_textured_light_surrogate as _create_textured_light_surrogate,
)
from .light_textured import (
    _store_textured_light_metadata as _store_textured_light_metadata,
)
from .light_values import _add_edmprop_keyframes as _add_edmprop_keyframes
from .light_values import _add_light_keyframes as _add_light_keyframes
from .light_values import (
    _edm_light_brightness_to_blender_energy as _edm_light_brightness_to_blender_energy,
)
from .light_values import _extract_light_property as _extract_light_property
from .light_values import _get_prop_any as _get_prop_any
from .light_values import _new_fcurve as _new_fcurve
from .light_values import _push_object_action_to_nla as _push_object_action_to_nla
from .light_values import _to_float as _to_float
from .light_values import _to_vec3 as _to_vec3
