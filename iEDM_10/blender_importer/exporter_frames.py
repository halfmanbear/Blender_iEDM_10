"""Pre-invert the frame rotations the official exporter adds on export.

The exporter always inserts a "Connector Transform" Rx(+90) below each
connector empty and a "Fake Light Transform" Ry(+90) below each light. The
imported object therefore carries the EDM frame times the inverse rotation,
so a re-export reproduces the source frame exactly.
"""

import math

import bpy
from mathutils import Matrix

from ..utils import action_fcurves

_CONNECTOR_UNDO = Matrix.Rotation(math.radians(-90.0), 4, "X")
_LIGHT_UNDO = Matrix.Rotation(math.radians(-90.0), 4, "Y")
_TRANSFORM_PATHS = {
    "location",
    "rotation_quaternion",
    "rotation_euler",
    "rotation_axis_angle",
    "scale",
}


def _is_connector(obj):
    props = getattr(obj, "EDMProps", None)
    return obj.type == "EMPTY" and getattr(props, "SPECIAL_TYPE", "") == "CONNECTOR"


def _has_transform_keys(obj):
    action = obj.animation_data.action if obj.animation_data else None
    if action is None:
        return False
    return any(c.data_path in _TRANSFORM_PATHS for c in action_fcurves(action))


def _rotate_local(obj, rotation):
    """Right-multiply the object's frame, keeping children where they are.

    Children are compensated through their parent inverse so their own
    (possibly animated) basis stays untouched.
    """
    obj.matrix_basis = obj.matrix_basis @ rotation
    inverse = rotation.inverted()
    for child in obj.children:
        child.matrix_parent_inverse = inverse @ child.matrix_parent_inverse


def undo_exporter_frame_rotations():
    """Apply the inverse exporter rotation to every connector and light."""
    counts = {"connector": 0, "light": 0, "skipped": 0}
    for obj in bpy.data.objects:
        if _is_connector(obj):
            kind, rotation = "connector", _CONNECTOR_UNDO
        elif obj.type == "LIGHT":
            kind, rotation = "light", _LIGHT_UNDO
        else:
            continue
        if _has_transform_keys(obj):
            # Own transform keys would need re-keying; leave them untouched.
            counts["skipped"] += 1
            continue
        _rotate_local(obj, rotation)
        counts[kind] += 1
    return counts
