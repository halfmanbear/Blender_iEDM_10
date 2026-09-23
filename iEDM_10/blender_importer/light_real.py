import math

import bpy

from ..utils import action_fcurves
from .light_values import (
    _add_edmprop_keyframes,
    _add_light_keyframes,
    _edm_light_brightness_to_blender_energy,
    _extract_light_property,
    _get_prop_any,
    _push_object_action_to_nla,
    _to_float,
    _to_vec3,
)
from .prelude import (
    _set_edmprop,
)


def _import_light_properties(node, obj, light_data, light_type):
    props = getattr(node, "lightProps", None) or {}
    if not props:
        return

    color_arg, color_keys, color_static = _extract_light_property(
        _get_prop_any(props, "Color", "color")
    )
    bright_arg, bright_keys, bright_static = _extract_light_property(
        _get_prop_any(props, "Brightness", "brightness")
    )
    dist_arg, dist_keys, dist_static = _extract_light_property(
        _get_prop_any(props, "Distance", "distance")
    )
    phi_arg, phi_keys, phi_static = _extract_light_property(
        _get_prop_any(props, "Phi", "phi")
    )
    theta_arg, theta_keys, theta_static = _extract_light_property(
        _get_prop_any(props, "Theta", "theta")
    )
    spec_arg, spec_keys, spec_static = _extract_light_property(
        _get_prop_any(props, "specularAmount", "specular_amount", "specular")
    )
    soft_arg, soft_keys, soft_static = _extract_light_property(
        _get_prop_any(props, "softness", "Softness")
    )
    vol_radius_arg, vol_radius_keys, vol_radius_static = _extract_light_property(
        _get_prop_any(props, "VolumeRadiusFactor", "radiusFactor")
    )
    vol_density_arg, vol_density_keys, vol_density_static = _extract_light_property(
        _get_prop_any(props, "VolumeDensityFactor", "densityFactor")
    )
    vol_near_arg, vol_near_keys, vol_near_static = _extract_light_property(
        _get_prop_any(props, "VolumeNearDistance", "nearDistance")
    )
    vol_type_arg, vol_type_keys, vol_type_static = _extract_light_property(
        _get_prop_any(props, "VolumeType", "volumeType")
    )

    if color_static is not None:
        light_data.color = _to_vec3(color_static, light_data.color)
    if bright_static is not None:
        light_data.energy = _edm_light_brightness_to_blender_energy(
            bright_static, light_type
        )
    if dist_static is not None:
        light_data.use_custom_distance = True
        light_data.cutoff_distance = max(0.0, _to_float(dist_static, 0.0))
    if spec_static is not None and hasattr(light_data, "specular_factor"):
        light_data.specular_factor = max(
            0.0, _to_float(spec_static, light_data.specular_factor)
        )

    if light_type == "SPOT":
        if phi_static is not None:
            light_data.spot_size = min(
                math.radians(170.0),
                max(0.0, _to_float(phi_static, light_data.spot_size)),
            )
        if theta_static is not None:
            light_data.spot_blend = min(
                1.0, max(0.0, _to_float(theta_static, light_data.spot_blend))
            )

    # Map EDM animated-property arguments into official exporter EDMProps fields.
    if color_arg >= 0:
        _set_edmprop(obj, "LIGHT_COLOR_ARG", int(color_arg))
    if bright_arg >= 0:
        _set_edmprop(obj, "LIGHT_POWER_ARG", int(bright_arg))
    if dist_arg >= 0:
        _set_edmprop(obj, "LIGHT_DISTANCE_ARG", int(dist_arg))
    if spec_arg >= 0:
        _set_edmprop(obj, "LIGHT_SPECULAR_ARG", int(spec_arg))
    if soft_static is not None:
        _set_edmprop(obj, "LIGHT_SOFTNESS", max(0.0, _to_float(soft_static, 0.0)))
    if vol_radius_static is not None:
        _set_edmprop(
            obj,
            "LIGHT_VOLUME_RADIUS_FACTOR",
            min(1.0, max(0.0, _to_float(vol_radius_static, 0.0))),
        )
    if vol_density_static is not None:
        _set_edmprop(
            obj,
            "LIGHT_VOLUME_DENSITY_FACTOR",
            min(1.0, max(0.0, _to_float(vol_density_static, 0.0))),
        )
    if vol_near_static is not None:
        _set_edmprop(
            obj, "LIGHT_VOLUME_NEAR_DISTANCE", max(0.0, _to_float(vol_near_static, 0.0))
        )
    if vol_type_static is not None:
        try:
            vol_type_int = int(round(_to_float(vol_type_static, 4)))
            volume_types = {0: "LANDING", 1: "NAV", 2: "TAXI", 3: "BANO"}
            _set_edmprop(
                obj, "LIGHT_VOLUME_TYPE", volume_types.get(vol_type_int, "NONE")
            )
        except Exception as e:
            print(f"Warning in blender_importer/lights.py: {e}")
    if light_type == "SPOT":
        spot_arg = phi_arg if phi_arg >= 0 else theta_arg
        if spot_arg >= 0:
            _set_edmprop(obj, "LIGHT_SPOT_SHAPE_ARG", int(spot_arg))

    # Recreate light-data animation curves for exporter parity.
    has_anim = any(
        keys
        for keys in (
            color_keys,
            bright_keys,
            dist_keys,
            phi_keys,
            theta_keys,
            spec_keys,
        )
    )
    if has_anim:
        action = bpy.data.actions.new("Light_{}".format(obj.name))
        action_fcurves(action, id_type="LIGHT")
        if hasattr(action, "argument"):
            first_arg = next(
                (
                    a
                    for a in (
                        color_arg,
                        bright_arg,
                        dist_arg,
                        phi_arg,
                        theta_arg,
                        spec_arg,
                        soft_arg,
                    )
                    if a >= 0
                ),
                -1,
            )
            if first_arg >= 0:
                action.argument = int(first_arg)

        if color_keys:
            for idx in range(3):
                _add_light_keyframes(
                    action,
                    "color",
                    color_keys,
                    lambda v, c=idx: _to_vec3(v)[c],
                    array_index=idx,
                )
        if bright_keys:
            _add_light_keyframes(
                action,
                "energy",
                bright_keys,
                lambda v: _edm_light_brightness_to_blender_energy(v, light_type),
            )
        if dist_keys:
            light_data.use_custom_distance = True
            _add_light_keyframes(
                action,
                "cutoff_distance",
                dist_keys,
                lambda v: max(0.0, _to_float(v, 0.0)),
            )
        if spec_keys and hasattr(light_data, "specular_factor"):
            _add_light_keyframes(
                action,
                "specular_factor",
                spec_keys,
                lambda v: max(0.0, _to_float(v, 0.0)),
            )
        if light_type == "SPOT":
            if phi_keys:
                _add_light_keyframes(
                    action,
                    "spot_size",
                    phi_keys,
                    lambda v: min(math.radians(170.0), max(0.0, _to_float(v, 0.0))),
                )
            if theta_keys:
                _add_light_keyframes(
                    action,
                    "spot_blend",
                    theta_keys,
                    lambda v: min(1.0, max(0.0, _to_float(v, 0.0))),
                )

        anim_data = light_data.animation_data_create()
        anim_data.action = action

    edmprop_anim = any(
        keys for keys in (soft_keys, vol_radius_keys, vol_density_keys, vol_near_keys)
    )
    if edmprop_anim and hasattr(obj, "EDMProps"):
        prop_action = bpy.data.actions.new("LightProps_{}".format(obj.name))
        if hasattr(prop_action, "argument"):
            first_arg = next(
                (
                    a
                    for a in (
                        soft_arg,
                        vol_radius_arg,
                        vol_density_arg,
                        vol_near_arg,
                        vol_type_arg,
                    )
                    if a >= 0
                ),
                -1,
            )
            if first_arg >= 0:
                prop_action.argument = int(first_arg)

        if soft_keys:
            _add_edmprop_keyframes(
                prop_action,
                "EDMProps.LIGHT_SOFTNESS",
                soft_keys,
                lambda v: max(0.0, _to_float(v, 0.0)),
            )
        if vol_radius_keys:
            _add_edmprop_keyframes(
                prop_action,
                "EDMProps.LIGHT_VOLUME_RADIUS_FACTOR",
                vol_radius_keys,
                lambda v: min(1.0, max(0.0, _to_float(v, 0.0))),
            )
        if vol_density_keys:
            _add_edmprop_keyframes(
                prop_action,
                "EDMProps.LIGHT_VOLUME_DENSITY_FACTOR",
                vol_density_keys,
                lambda v: min(1.0, max(0.0, _to_float(v, 0.0))),
            )
        if vol_near_keys:
            _add_edmprop_keyframes(
                prop_action,
                "EDMProps.LIGHT_VOLUME_NEAR_DISTANCE",
                vol_near_keys,
                lambda v: max(0.0, _to_float(v, 0.0)),
            )

        # Keep object-level EDMProps animation separate from later transform /
        # visibility action assignment on the same object.
        if not _push_object_action_to_nla(obj, prop_action):
            obj_anim_data = obj.animation_data_create()
            obj_anim_data.action = prop_action
