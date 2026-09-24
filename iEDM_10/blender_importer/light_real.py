import math
from types import SimpleNamespace

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

    _apply_static_light_data(
        light_data,
        light_type,
        color_static,
        bright_static,
        dist_static,
        spec_static,
        phi_static,
        theta_static,
    )

    _apply_static_light_properties(
        obj,
        light_type,
        (color_arg, bright_arg, dist_arg, spec_arg, phi_arg, theta_arg),
        (
            soft_static,
            vol_radius_static,
            vol_density_static,
            vol_near_static,
            vol_type_static,
        ),
    )

    # Recreate light-data animation curves for exporter parity.
    _create_light_data_animation(
        obj,
        light_data,
        light_type,
        (
            color_arg,
            color_keys,
            bright_arg,
            bright_keys,
            dist_arg,
            dist_keys,
            phi_arg,
            phi_keys,
            theta_arg,
            theta_keys,
            spec_arg,
            spec_keys,
            soft_arg,
        ),
    )

    _create_light_property_animation(
        obj,
        (
            soft_arg,
            soft_keys,
            vol_radius_arg,
            vol_radius_keys,
            vol_density_arg,
            vol_density_keys,
            vol_near_arg,
            vol_near_keys,
            vol_type_arg,
        ),
    )


def _theta_to_spot_blend(theta, spot_size):
    """EDM theta is the inner cone angle; pyedm writes
    theta = 2 * atan(tan(phi / 2) * (1 - spot_blend)), so invert that."""
    outer = math.tan(min(spot_size, math.radians(170.0)) / 2.0)
    if outer <= 1e-9:
        return 0.0
    return min(1.0, max(0.0, 1.0 - math.tan(max(0.0, theta) / 2.0) / outer))


def _clamp_at_zero_crossings(keys):
    """Blender energy cannot go below 0 and the exporter's power curve cannot
    take it (C-130J strobes key -1.6 -> 0 -> 4, tail lights -0.7 -> 1.3).
    Clamping a key alone changes the fade, so a key is added where the curve
    crosses 0: exact if DCS treats negative brightness as off."""
    points = [(float(k.frame), _to_float(k.value, 0.0)) for k in keys]
    out = []
    for index, (frame, value) in enumerate(points):
        if index:
            f0, v0 = points[index - 1]
            if v0 * value < 0.0:
                out.append((f0 + (frame - f0) * v0 / (v0 - value), 0.0))
        out.append((frame, max(0.0, value)))
    return [SimpleNamespace(frame=f, value=v) for f, v in out]


def _apply_static_light_data(
    light_data, light_type, color, brightness, distance, specular, phi, theta
):
    if color is not None:
        light_data.color = _to_vec3(color, light_data.color)
    if brightness is not None:
        light_data.energy = _edm_light_brightness_to_blender_energy(
            brightness, light_type
        )
    if distance is not None:
        light_data.use_custom_distance = True
        light_data.cutoff_distance = max(0.0, _to_float(distance, 0.0))
    if specular is not None and hasattr(light_data, "specular_factor"):
        light_data.specular_factor = max(
            0.0, _to_float(specular, light_data.specular_factor)
        )
    if light_type == "SPOT":
        if phi is not None:
            light_data.spot_size = min(
                math.radians(170.0), max(0.0, _to_float(phi, light_data.spot_size))
            )
        if theta is not None:
            light_data.spot_blend = _theta_to_spot_blend(
                _to_float(theta, 0.0), light_data.spot_size
            )


def _apply_static_light_properties(obj, light_type, args, values):
    color_arg, bright_arg, dist_arg, spec_arg, phi_arg, theta_arg = args
    soft, vol_radius, vol_density, vol_near, vol_type = values
    fields = (
        (color_arg, "LIGHT_COLOR_ARG"),
        (bright_arg, "LIGHT_POWER_ARG"),
        (dist_arg, "LIGHT_DISTANCE_ARG"),
        (spec_arg, "LIGHT_SPECULAR_ARG"),
    )
    for argument, field in fields:
        if argument >= 0:
            _set_edmprop(obj, field, int(argument))
    for value, field, convert in (
        (soft, "LIGHT_SOFTNESS", lambda v: max(0.0, _to_float(v, 0.0))),
        (
            vol_radius,
            "LIGHT_VOLUME_RADIUS_FACTOR",
            lambda v: min(1.0, max(0.0, _to_float(v, 0.0))),
        ),
        (
            vol_density,
            "LIGHT_VOLUME_DENSITY_FACTOR",
            lambda v: min(1.0, max(0.0, _to_float(v, 0.0))),
        ),
        (vol_near, "LIGHT_VOLUME_NEAR_DISTANCE", lambda v: max(0.0, _to_float(v, 0.0))),
    ):
        if value is not None:
            _set_edmprop(obj, field, convert(value))
    if vol_type is not None:
        try:
            volume_types = {0: "LANDING", 1: "NAV", 2: "TAXI", 3: "BANO"}
            value = int(round(_to_float(vol_type, 4)))
            _set_edmprop(obj, "LIGHT_VOLUME_TYPE", volume_types.get(value, "NONE"))
        except Exception as e:
            print(f"Warning in blender_importer/lights.py: {e}")
    if light_type == "SPOT":
        spot_arg = phi_arg if phi_arg >= 0 else theta_arg
        if spot_arg >= 0:
            _set_edmprop(obj, "LIGHT_SPOT_SHAPE_ARG", int(spot_arg))


def _create_light_data_animation(obj, light_data, light_type, values):
    (
        color_arg,
        color_keys,
        bright_arg,
        bright_keys,
        dist_arg,
        dist_keys,
        phi_arg,
        phi_keys,
        theta_arg,
        theta_keys,
        spec_arg,
        spec_keys,
        soft_arg,
    ) = values
    if not any((color_keys, bright_keys, dist_keys, phi_keys, theta_keys, spec_keys)):
        return
    action = bpy.data.actions.new("Light_{}".format(obj.name))
    action_fcurves(action, id_type="LIGHT")
    if hasattr(action, "argument"):
        action.argument = next(
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
            _clamp_at_zero_crossings(bright_keys),
            lambda v: _edm_light_brightness_to_blender_energy(
                v, light_type, animated=True
            ),
        )
    if dist_keys:
        light_data.use_custom_distance = True
        _add_light_keyframes(
            action, "cutoff_distance", dist_keys, lambda v: max(0.0, _to_float(v, 0.0))
        )
    if spec_keys and hasattr(light_data, "specular_factor"):
        _add_light_keyframes(
            action, "specular_factor", spec_keys, lambda v: max(0.0, _to_float(v, 0.0))
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
                lambda v: _theta_to_spot_blend(_to_float(v, 0.0), light_data.spot_size),
            )
    light_data.animation_data_create().action = action


def _create_light_property_animation(obj, values):
    (
        soft_arg,
        soft_keys,
        radius_arg,
        radius_keys,
        density_arg,
        density_keys,
        near_arg,
        near_keys,
        type_arg,
    ) = values
    if not any((soft_keys, radius_keys, density_keys, near_keys)) or not hasattr(
        obj, "EDMProps"
    ):
        return
    action = bpy.data.actions.new("LightProps_{}".format(obj.name))
    if hasattr(action, "argument"):
        action.argument = next(
            (
                a
                for a in (soft_arg, radius_arg, density_arg, near_arg, type_arg)
                if a >= 0
            ),
            -1,
        )
    for keys, path, convert in (
        (soft_keys, "EDMProps.LIGHT_SOFTNESS", lambda v: max(0.0, _to_float(v, 0.0))),
        (
            radius_keys,
            "EDMProps.LIGHT_VOLUME_RADIUS_FACTOR",
            lambda v: min(1.0, max(0.0, _to_float(v, 0.0))),
        ),
        (
            density_keys,
            "EDMProps.LIGHT_VOLUME_DENSITY_FACTOR",
            lambda v: min(1.0, max(0.0, _to_float(v, 0.0))),
        ),
        (
            near_keys,
            "EDMProps.LIGHT_VOLUME_NEAR_DISTANCE",
            lambda v: max(0.0, _to_float(v, 0.0)),
        ),
    ):
        if keys:
            _add_edmprop_keyframes(action, path, keys, convert)
    if not _push_object_action_to_nla(obj, action):
        obj.animation_data_create().action = action
