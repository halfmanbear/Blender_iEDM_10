import bpy

from ..utils import action_fcurves
from .light_materials import _safe_float
from .prelude import (
    _anim_frame_to_scene_frame,
)


def _apply_fake_light_animation_payload(obj, edm_material):
    if obj is None or not hasattr(obj, "EDMProps") or edm_material is None:
        return
    anim_uniforms = getattr(edm_material, "animated_uniforms", None) or {}
    luminance_prop = anim_uniforms.get("luminance")
    if luminance_prop is None or not hasattr(luminance_prop, "keys"):
        return

    keys = list(getattr(luminance_prop, "keys", []) or [])
    if not keys:
        return
    arg = getattr(luminance_prop, "argument", None)
    base_luminance = _safe_float(getattr(keys[0], "value", 1.0), 1.0)
    if abs(base_luminance) <= 1.0e-8:
        base_luminance = 1.0
    try:
        obj.EDMProps.ANIMATED_BRIGHTNESS = 1.0
    except Exception:
        pass

    try:
        action_name = "FakeLight_{}".format(obj.name)
        if arg is not None and int(arg) >= 0:
            action_name = "{}_{}".format(int(arg), obj.name)
        action = bpy.data.actions.new(action_name)
        if hasattr(action, "argument") and arg is not None and int(arg) >= 0:
            action.argument = int(arg)
        anim_data = obj.animation_data_create()
        anim_data.action = action
        for framedata in keys:
            frame = _anim_frame_to_scene_frame(getattr(framedata, "frame", 0.0))
            value = _safe_float(getattr(framedata, "value", 0.0), 0.0) / base_luminance
            obj.EDMProps.ANIMATED_BRIGHTNESS = value
            obj.keyframe_insert(data_path="EDMProps.ANIMATED_BRIGHTNESS", frame=frame)
        curve = action_fcurves(action).find("EDMProps.ANIMATED_BRIGHTNESS")
        if curve is not None:
            for key in curve.keyframe_points:
                key.interpolation = "LINEAR"
    except Exception as e:
        print(f"Warning in blender_importer/lights.py: {e}")


def _has_animated_fake_omni_payload(node):
    return (
        getattr(node, "anim_arg_handle", None) is not None
        and getattr(node, "anim_arg_handle", 0xFFFFFFFF) != 0xFFFFFFFF
        and getattr(node, "anim_data_count", 0) > 0
        and getattr(node, "anim_data_raw", b"")
    )


def _apply_animated_fake_omni_brightness(ob, node, light_count, verts_per_light=1):
    """Reconstruct per-light brightness animation from AnimatedFakeOmniLightsNode payload.

    The binary stores lightCount * 128 float32 brightness values — each light's
    curve presampled at 128 evenly-spaced arg positions. The official exporter
    expects: one vertex per light, a vertex group whose weight encodes each
    light's timing delay, and an action on EDMProps.ANIMATED_BRIGHTNESS named
    "{arg_handle}_{object_name}" with the master brightness curve as keyframes.
    """
    import struct as _struct

    arg_handle = getattr(node, "anim_arg_handle", None)
    data_count = getattr(node, "anim_data_count", 0)
    anim_data_raw = getattr(node, "anim_data_raw", b"")

    if not anim_data_raw or data_count == 0 or light_count <= 0:
        return

    available = len(anim_data_raw) // 4
    if data_count > available:
        print(
            "Warning: animated fake light '{}': anim_data_count={} but payload holds {} floats; truncating".format(
                getattr(node, "name", ""), data_count, available
            )
        )
        data_count = available
        if data_count < light_count:
            return

    # Format spec: anim_sample_rate must be 128. Use the explicit field when
    # available and consistent; fall back to data_count // light_count otherwise.
    explicit_rate = getattr(node, "anim_sample_rate", None)
    n_samples = data_count // light_count
    if explicit_rate is not None and explicit_rate > 0:
        if explicit_rate != n_samples:
            print(
                "Warning: animated fake light '{}': anim_sample_rate={} but data_count/light_count={}; "
                "using explicit rate".format(
                    getattr(node, "name", ""), explicit_rate, n_samples
                )
            )
        n_samples = int(explicit_rate)
    if n_samples == 0:
        return
    if light_count * n_samples > data_count:
        print(
            "Warning: animated fake light '{}': {} lights x {} samples exceeds {} floats; skipping brightness animation".format(
                getattr(node, "name", ""), light_count, n_samples, data_count
            )
        )
        return

    all_floats = _struct.unpack_from("<{}f".format(data_count), anim_data_raw)
    lights_samples = [
        all_floats[i * n_samples : (i + 1) * n_samples] for i in range(light_count)
    ]
    master = lights_samples[0]

    # Recover per-light delay by finding circular shift vs master curve.
    # Shift of k samples → delay = k/n_samples * 2.0 in EDM time units,
    # stored as vertex group weight (exporter reads weight directly as delay).
    delays = [0.0]
    for j in range(1, light_count):
        samp = lights_samples[j]
        best_shift, best_score = 0, float("inf")
        for shift in range(n_samples):
            score = sum(
                (master[i] - samp[(i + shift) % n_samples]) ** 2
                for i in range(n_samples)
            )
            if score < best_score:
                best_score = score
                best_shift = shift
        delays.append(min(best_shift / n_samples * 2.0, 1.0))

    # Assign delay weights — one vertex group; every vertex of a light (4 per
    # quad in surface mode) carries that light's delay.
    delay_group = ob.vertex_groups.new(name="brightness_delay")
    vertex_count = len(ob.data.vertices)
    for li, delay in enumerate(delays):
        indices = [
            vi
            for vi in range(li * verts_per_light, (li + 1) * verts_per_light)
            if vi < vertex_count
        ]
        if indices:
            delay_group.add(indices, float(delay), "REPLACE")

    # Build brightness action from master curve.
    # 128 samples span Blender frames [0, 200]; frame_i = i * 200 / n_samples.
    try:
        ob.EDMProps.ANIMATED_BRIGHTNESS = 1.0
    except Exception:
        pass

    action_name = "{}_{}".format(int(arg_handle), ob.name)
    action = bpy.data.actions.new(action_name)
    try:
        if hasattr(action, "argument"):
            action.argument = int(arg_handle)
    except Exception:
        pass

    anim_data = ob.animation_data_create()
    anim_data.action = action

    frame_step = 200.0 / n_samples
    for i, brightness in enumerate(master):
        frame = i * frame_step
        try:
            ob.EDMProps.ANIMATED_BRIGHTNESS = float(brightness)
            ob.keyframe_insert(data_path="EDMProps.ANIMATED_BRIGHTNESS", frame=frame)
        except Exception:
            pass

    curve = action_fcurves(action).find("EDMProps.ANIMATED_BRIGHTNESS")
    if curve is not None:
        for kp in curve.keyframe_points:
            kp.interpolation = "LINEAR"
        try:
            curve.extrapolation = "CONSTANT"
        except Exception:
            pass
