"""Keep hold keys alive through the official exporter's key reduction.

pyedm (the exporter's native writer) drops a key that lies on the
interpolation (slerp for rotations) between any earlier key and the next key,
not only its neighbours. A hold that returns to an earlier pose
(A ... B, A, A, ... ) therefore loses its first key and the curve slides from
B straight to the last A, e.g. a crew head bone that lost its arg 0 key came
back turned by ~13 degrees at rest; sampled sweeps lose keys the same way.

Components within ~1e-4 count as equal, and the tolerance grows with the
value (a 1e-4 change at 100 m is still dropped). The first key of each such
hold is nudged clear of it: rotations by 4.5e-4 rad (0.026 degrees), locations
and scales by 1e-3 or 1e-5 of the value, whichever is larger. Larger rotation
nudges can visibly displace vertices far from a bone's pivot.
"""

import functools
from collections import defaultdict

import bpy
from mathutils import Euler, Quaternion

from ..utils import action_fcurves

_TOL = 1e-4
_REL_TOL = 1e-5
# For unit quaternions the largest component change is at least ~angle / 4.
# Keep it above pyedm's ~1e-4 component tolerance, even for balanced quaternions.
_NUDGE_ANGLE = 4.5e-4
_NUDGE_LOCATION = 1e-3
_NUDGE_SCALE = 1e-3
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


def _interpolate(path, frames, values, a, b, frame):
    f = (frame - frames[a]) / (frames[b] - frames[a])
    if path.endswith("rotation_quaternion") and len(values[a]) == 4:
        return list(Quaternion(values[a]).slerp(Quaternion(values[b]), f))
    return [x + (y - x) * f for x, y in zip(values[a], values[b], strict=True)]


def _hold_starts(path, frames, values):
    """Indices pyedm would drop although they matter.

    pyedm drops a key lying on the interpolation between ANY earlier key and
    the next key (a hold returning to an earlier value is one case). Only keys
    off their neighbours' interpolation change the curve when dropped: F4U-1D
    gear strut pivot arg 5 lost 0.3 (on slerp(0.0, 0.4), 0.45 degrees off
    slerp(0.2, 0.4)).
    """
    starts = []
    for i in range(1, len(values) - 1):
        on = functools.partial(
            _interpolate, path, frames, values, b=i + 1, frame=frames[i]
        )
        if _same(on(a=i - 1), values[i]):
            continue
        if any(_same(on(a=j), values[i]) for j in range(i - 1)):
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
    minimum = _NUDGE_SCALE if path.endswith("scale") else _NUDGE_LOCATION
    nudged[i] += max(minimum, _REL_TOL * abs(nudged[i]))
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
    starts = _hold_starts(path, frames, values)
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
