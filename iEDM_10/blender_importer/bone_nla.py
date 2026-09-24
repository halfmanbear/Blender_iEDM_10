"""Bake the source-driven bone pose into per-argument armature NLA strips.

The official exporter reads bone animation only from armature NLA strips whose
names start with the argument number, keyed on pose-bone location, quaternion
and scale in pose-basis space. Bones are driven by constraints to the helper
graph in bone_controls, which the exporter cannot read, so each argument is
isolated on that graph, sampled, and written as pose-basis keys. The
constraints stay in place, so Blender playback remains exact.
"""

import logging
import re

import bpy
from mathutils import Vector

from ..utils import action_fcurves

_logger = logging.getLogger(__name__)

_ARG_PREFIX = re.compile(r"^(\d+)_")
_NEUTRAL_FRAME = 100.0
_TOLERANCE = 1e-5
_SAMPLE_STEP = 10.0
_REFINE_TOLERANCE = 1e-4
_MIN_REFINE_STEP = 1.25
_PATHS = ("location", "rotation_quaternion", "scale")


def _control_actions(collection):
    """Group helper-graph (object, action) pairs by argument number."""
    by_arg = {}
    for obj in collection.objects:
        action = obj.animation_data.action if obj.animation_data else None
        if action is None:
            continue
        match = _ARG_PREFIX.match(action.name)
        if match is None:
            continue
        by_arg.setdefault(int(match.group(1)), []).append((obj, action))
    return by_arg


def _pose_helpers(controls, frame):
    """Write helper-empty channels for ``frame`` without changing scene time.

    Moving the scene frame would re-evaluate every imported object and leave
    objects hidden at the import frame with stale matrices, which the exporter
    then reads.
    """
    for obj, action in controls:
        for curve in action_fcurves(action):
            values = getattr(obj, curve.data_path)
            values[curve.array_index] = curve.evaluate(frame)
    bpy.context.view_layer.update()


def _sample(rig, controls, frames):
    """Return {bone: [(frame, loc, quat, scale)]} for the evaluated pose."""
    samples = {pb.name: [] for pb in rig.pose.bones}
    for frame in frames:
        _pose_helpers(controls, frame)
        for pb in rig.pose.bones:
            basis = rig.convert_space(
                pose_bone=pb, matrix=pb.matrix, from_space="POSE", to_space="LOCAL"
            )
            loc, quat, scale = basis.decompose()
            samples[pb.name].append((frame, loc, quat, scale))
    return samples


def _sample_arg(rig, arg, by_arg):
    """Sample every bone while only ``arg`` moves; others hold arg 0."""
    keys = {
        point.co[0]
        for _obj, action in by_arg[arg]
        for curve in action_fcurves(action)
        for point in curve.keyframe_points
    }
    # Chained rotations about offset pivots do not interpolate linearly in
    # pose-basis space, so sample between the source keys as well.
    low, high = min(keys), max(keys)
    count = int((high - low) // _SAMPLE_STEP)
    frames = sorted(keys | {low + i * _SAMPLE_STEP for i in range(1, count + 1)})
    try:
        return _refine(rig, by_arg[arg], frames, _sample(rig, by_arg[arg], frames))
    finally:
        _pose_helpers(by_arg[arg], _NEUTRAL_FRAME)


def _misses(a, b, middle):
    """True when interpolating ``a``..``b`` (lerp, and slerp as DCS does for
    rotations) misses the sampled ``middle``."""
    _f, la, qa, sa = a
    _f, lb, qb, sb = b
    _f, lm, qm, sm = middle
    if (la.lerp(lb, 0.5) - lm).length > _REFINE_TOLERANCE:
        return True
    if qa.slerp(qb, 0.5).rotation_difference(qm).angle > _REFINE_TOLERANCE:
        return True
    return _differs(sa.lerp(sb, 0.5), sm, _REFINE_TOLERANCE)


def _refine(rig, controls, frames, samples):
    """Bisect sample intervals whose midpoint the exported keys would miss.

    A location swinging about an offset pivot follows an arc, so the 10-frame
    grid cuts chords (fa-18c wing fold: 44% voxel hits between samples).
    """
    rows = {bone: dict((s[0], s) for s in values) for bone, values in samples.items()}
    pending = [
        (f0, f1) for f0, f1 in zip(frames, frames[1:]) if f1 - f0 > _MIN_REFINE_STEP
    ]
    while pending:
        middles = [(f0 + f1) / 2.0 for f0, f1 in pending]
        sampled = _sample(rig, controls, middles)
        split = []
        for index, (f0, f1) in enumerate(pending):
            middle = middles[index]
            if not any(
                _misses(rows[bone][f0], rows[bone][f1], values[index])
                for bone, values in sampled.items()
            ):
                continue
            for bone, values in sampled.items():
                rows[bone][middle] = values[index]
            if middle - f0 > _MIN_REFINE_STEP:
                split += [(f0, middle), (middle, f1)]
        pending = split
    return {bone: [by_frame[f] for f in sorted(by_frame)] for bone, by_frame in rows.items()}


def _differs(a, b, tolerance=_TOLERANCE):
    return any(abs(x - y) > tolerance for x, y in zip(a, b, strict=True))


def _relative(sample, neutral, first):
    """Express a sample for a secondary argument relative to the neutral pose.

    The exporter composes arguments like EDM: locations add, rotations and
    scales multiply. The first argument keeps absolute values; later ones
    carry only their change from the all-zero pose.
    """
    frame, loc, quat, scale = sample
    if first:
        return frame, loc, quat, scale
    _nframe, nloc, nquat, nscale = neutral
    delta_scale = Vector(
        s / n if abs(n) > _TOLERANCE else 1.0
        for s, n in zip(scale, nscale, strict=True)
    )
    return frame, loc - nloc, nquat.inverted() @ quat, delta_scale


def _write_curves(action, bone, samples, paths):
    fcurves = action_fcurves(action)
    group = bone
    previous = None
    rows = []
    for frame, loc, quat, scale in samples:
        if previous is not None and previous.dot(quat) < 0.0:
            quat = -quat
        previous = quat
        rows.append(
            (frame, {"location": loc, "rotation_quaternion": quat, "scale": scale})
        )
    for path in paths:
        data_path = f'pose.bones["{bone}"].{path}'
        for index in range(len(rows[0][1][path])):
            if hasattr(action, "fcurves"):
                curve = fcurves.new(data_path, index=index, action_group=group)
            else:
                curve = fcurves.new(data_path, index=index, group_name=group)
            curve.keyframe_points.add(len(rows))
            for point, (frame, values) in zip(curve.keyframe_points, rows, strict=True):
                point.co = (frame, values[path][index])
                point.interpolation = "LINEAR"
            curve.extrapolation = "CONSTANT"
            curve.update()


def _animated_paths(samples, neutral):
    """Return the channels of a bone that change while its argument moves."""
    paths = []
    for index, path in enumerate(_PATHS, start=1):
        if any(_differs(sample[index], neutral[index]) for sample in samples):
            paths.append(path)
    return paths


def _is_identity(sample):
    _frame, loc, quat, scale = sample
    return (
        loc.length < _TOLERANCE
        and abs(abs(quat.w) - 1.0) < _TOLERANCE
        and not _differs(scale, (1.0, 1.0, 1.0))
    )


def _plan_actions(rig, by_arg):
    """Return {arg: {bone: (samples, paths)}} ready to be written."""
    everything = [pair for pairs in by_arg.values() for pair in pairs]
    neutral = {
        name: rows[0]
        for name, rows in _sample(rig, everything, [_NEUTRAL_FRAME]).items()
    }
    plan = {}
    owner = {}
    for arg in sorted(by_arg):
        for bone, samples in _sample_arg(rig, arg, by_arg).items():
            paths = _animated_paths(samples, neutral[bone])
            if not paths:
                continue
            first = bone not in owner
            if first:
                owner[bone] = arg
                paths = list(_PATHS)
            rows = [_relative(s, neutral[bone], first) for s in samples]
            plan.setdefault(arg, {})[bone] = (rows, paths)
    if plan:
        # Every bone is exported through the NLA path once any strip exists, so
        # static bones need their constant basis stored in some strip.
        static_arg = min(plan)
        for bone, sample in neutral.items():
            if bone in owner or _is_identity(sample):
                continue
            rows = [(frame, *sample[1:]) for frame in (0.0, 200.0)]
            plan[static_arg][bone] = (rows, list(_PATHS))
    return plan


def _push_strip(rig, action):
    track = rig.animation_data.nla_tracks.new()
    track.name = action.name
    start = float(action.frame_range[0])
    strip = track.strips.new(action.name, int(start), action)
    if getattr(strip, "action_slot", False) is None and action.slots:
        strip.action_slot = action.slots[0]
    if abs(strip.frame_start - start) > 1e-6 and hasattr(strip, "frame_start_ui"):
        strip.frame_start_ui = start
    strip.name = action.name
    strip.extrapolation = "NOTHING"


def bake_bone_nla(rig, collection):
    """Write per-argument pose-basis NLA strips for the exporter."""
    by_arg = _control_actions(collection)
    if not by_arg:
        return 0
    plan = _plan_actions(rig, by_arg)
    rig.animation_data_create()
    rig.animation_data.use_nla = True
    for arg in sorted(plan):
        action = bpy.data.actions.new(f"{arg}_{rig.name}_bones")
        if hasattr(action, "argument"):
            action.argument = arg
        for bone, (rows, paths) in sorted(plan[arg].items()):
            _write_curves(action, bone, rows, paths)
        _push_strip(rig, action)
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
    _logger.debug("Baked bone animation for %d argument(s)", len(plan))
    return len(plan)
