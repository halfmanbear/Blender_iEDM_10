"""Convert EDM visibility intervals to Blender scene frames and keys."""

from .import_context import FRAME_SCALE


def _anim_frame_to_scene_frame(frame_value):
    """Map EDM normalized animation frame [-1..1] to scene frame [0..FRAME_SCALE]."""
    try:
        val = (float(frame_value) + 1.0) * FRAME_SCALE / 2.0
        # Preserve subframes; integer rounding shifts damage thresholds.
        return min(2147483647, max(-2147483648, val))
    except (OverflowError, ValueError):
        return 0


def _visibility_scene_ranges(ranges):
    """Union EDM visibility windows without quantizing their argument boundaries."""
    merged = []
    for start, end in sorted(ranges):
        first = _anim_frame_to_scene_frame(start)
        last = _anim_frame_to_scene_frame(end)
        if last <= first:
            continue
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
        else:
            merged.append((first, last))
    return merged


def _visibility_scene_keys(ranges):
    intervals = _visibility_scene_ranges(ranges)
    keys = {min(0.0, intervals[0][0] - 1.0) if intervals else 0.0: 0.0}
    for start, end in intervals:
        keys[start] = 1.0
        keys[end] = 0.0
    return sorted(keys.items())
