"""Keep hold keys alive through the official exporter's key reduction.

pyedm (the exporter's native writer) drops a key when the next key has the
same value and an earlier, non-adjacent key had that value too. A hold that
returns to an earlier pose (A ... B, A, A, ... ) therefore loses its first
key and the curve slides from B straight to the last A, e.g. a crew head
bone that lost its arg 0 key came back turned by ~13 degrees at rest.

Components within ~1e-4 count as equal, and the tolerance grows with the
value (a 1e-4 change at 100 m is still dropped). The first key of each such
hold is nudged clear of it: rotations by 1e-3 rad (0.06 degrees), locations
and scales by 1e-3 or 1e-5 of the value, whichever is larger.
"""

from collections import defaultdict

import bpy
from mathutils import Euler, Quaternion

from ..utils import action_fcurves

_TOL = 1e-4
_REL_TOL = 1e-5
_NUDGE_ANGLE = 1e-3
_NUDGE = 1e-3
_NUDGE_QUAT = Quaternion((0.0, 0.0, 1.0), _NUDGE_ANGLE)
_PATHS = ("location", "rotation_quaternion", "rotation_euler", "scale")


def _same(a, b):
    tol = max(_TOL, _REL_TOL * max(abs(x) for x in (*a, *b)))
    return max(abs(x - y) for x, y in zip(a, b, strict=True)) < tol


def _channel_groups(action):
    groups = defaultdict(dict)
    for fcurve in action_fcurves(action):
        if fcurve.data_path.rsplit(".", 1)[-1] in _PATHS:
            groups[fcurve.data_path][fcurve.array_index] = fcurve
    return groups


def _hold_starts(values):
    """Indices pyedm would drop: equal to the next and to an earlier key."""
    starts = []
    for i in range(1, len(values) - 1):
        if not _same(values[i], values[i + 1]) or _same(values[i - 1], values[i]):
            continue
        if any(_same(values[j], values[i]) for j in range(i - 1)):
            starts.append(i)
    return starts


def _nudged(path, value):
    if path.endswith("rotation_quaternion") and len(value) == 4:
        return list(Quaternion(value) @ _NUDGE_QUAT)
    if path.endswith("rotation_euler") and len(value) == 3:
        euler = Euler(value)
        euler.z += _NUDGE_ANGLE
        return list(euler)
    nudged = list(value)
    i = max(range(len(nudged)), key=lambda k: abs(nudged[k]))
    nudged[i] += max(_NUDGE, _REL_TOL * abs(nudged[i]))
    return nudged


def _set_key(fcurve, frame, value):
    for key in fcurve.keyframe_points:
        if abs(key.co[0] - frame) < 1e-6:
            key.co[1] = value
            key.handle_left[1] = value
            key.handle_right[1] = value
            return
    fcurve.keyframe_points.insert(frame, value, options={"FAST"})


def _protect_channel(path, fcurves):
    frames = sorted({k.co[0] for fc in fcurves for k in fc.keyframe_points})
    values = [[fc.evaluate(f) for fc in fcurves] for f in frames]
    starts = _hold_starts(values)
    for i in starts:
        for fcurve, value in zip(fcurves, _nudged(path, values[i]), strict=True):
            _set_key(fcurve, frames[i], value)
    if starts:
        for fcurve in fcurves:
            fcurve.update()
    return len(starts)


def protect_hold_keys():
    """Nudge every transform hold start that pyedm would otherwise drop."""
    count = 0
    for action in bpy.data.actions:
        for path, comps in _channel_groups(action).items():
            fcurves = [comps[i] for i in sorted(comps)]
            count += _protect_channel(path, fcurves)
    return count
