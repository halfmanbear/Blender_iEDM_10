import logging
import math

from ..utils import action_fcurves
from .prelude import (
    _BLENDER_LAMP_ENERGY_COEFFICIENT,
    _BLENDER_LAMP_WEAK_COEFFICIENT,
    _PBR_WATTS_TO_LUMENS,
    _anim_frame_to_scene_frame,
)

logger = logging.getLogger(__name__)


def _extract_light_property(prop_value):
    """Return (argument, keyframes, static_value) for a light property payload."""
    argument = -1
    keys = None
    static_value = None

    if hasattr(prop_value, "keys") and hasattr(prop_value, "argument"):
        try:
            argument = int(prop_value.argument)
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning("Invalid light argument; using -1: %s", exc)
            argument = -1
        keys = list(getattr(prop_value, "keys", []) or [])
        if keys:
            static_value = getattr(keys[0], "value", None)
    elif isinstance(prop_value, list):
        keys = prop_value
        if keys:
            static_value = getattr(keys[0], "value", None)
    else:
        static_value = prop_value

    return argument, keys, static_value


def _get_prop_any(props, *names):
    """Fetch first present key from a light/property dict."""
    if not props:
        return None
    for name in names:
        if name in props:
            return props.get(name)
    return None


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        # Optional light properties may be absent or use legacy string values.
        logger.warning("Invalid light scalar %r; using %r: %s", value, default, exc)
        return float(default)


def _to_vec3(value, default=(1.0, 1.0, 1.0)):
    if value is None:
        return default
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, OverflowError, IndexError, KeyError) as exc:
        logger.warning("Invalid light vector %r; using %r: %s", value, default, exc)
        return default


def _edm_light_brightness_to_blender_energy(
    brightness_value, light_type, animated=False
):
    """
    Invert the official exporter conversion path from Blender light energy to EDM
    Brightness property, so importing then exporting preserves values.
    Animated keys go through light_power_to_energy, which skips the 1/(4*pi)
    the static path applies to non-sun lights.
    """
    brightness = max(0.0, _to_float(brightness_value, 0.0))
    if brightness <= 0.0:
        return 0.0

    base = max(0.0, brightness) ** (1.0 / _BLENDER_LAMP_WEAK_COEFFICIENT)
    denom = _PBR_WATTS_TO_LUMENS * _BLENDER_LAMP_ENERGY_COEFFICIENT
    if denom <= 0.0:
        return 0.0
    energy = base / denom
    if light_type != "SUN" and not animated:
        energy *= 4.0 * math.pi
    return max(0.0, energy)


def _new_fcurve(action, data_path, array_index=None):
    idx = 0 if array_index is None else int(array_index)
    existing = action_fcurves(action).find(data_path, index=idx)
    if existing is not None:
        return existing
    if array_index is None:
        return action_fcurves(action).new(data_path=data_path)
    return action_fcurves(action).new(data_path=data_path, index=int(array_index))


def _add_light_keyframes(action, data_path, keys, value_fn, array_index=None):
    curve = _new_fcurve(action, data_path, array_index)
    for framedata in keys:
        frame = _anim_frame_to_scene_frame(getattr(framedata, "frame", 0.0))
        value = value_fn(getattr(framedata, "value", 0.0))
        key = curve.keyframe_points.insert(frame, float(value), options={"FAST"})
        key.interpolation = "LINEAR"
    return curve


def _add_edmprop_keyframes(action, data_path, keys, value_fn):
    curve = _new_fcurve(action, data_path)
    for framedata in keys:
        frame = _anim_frame_to_scene_frame(getattr(framedata, "frame", 0.0))
        value = value_fn(getattr(framedata, "value", 0.0))
        key = curve.keyframe_points.insert(frame, float(value), options={"FAST"})
        key.interpolation = "LINEAR"
    return curve


def _push_object_action_to_nla(obj, action):
    if obj is None or action is None:
        return False
    anim_data = obj.animation_data_create()
    try:
        track = anim_data.nla_tracks.new()
        track.name = action.name
        # Start the strip at the action's first key so keys keep their scene frames.
        start = float(action.frame_range[0])
        strip = track.strips.new(action.name, int(start), action)
        if getattr(strip, "action_slot", False) is None and action.slots:
            strip.action_slot = action.slots[0]
        if abs(strip.frame_start - start) > 1e-6 and hasattr(strip, "frame_start_ui"):
            strip.frame_start_ui = start
        strip.extrapolation = "HOLD"
        strip.blend_type = "REPLACE"
        return True
    except Exception as e:
        print(f"Warning in blender_importer/lights.py: {e}")
        return False
