"""Evaluate objects that are viewport-hidden at the import frame.

The depsgraph skips objects whose ``hide_viewport`` is set, which is how
visibility animation hides objects at the import frame. Objects created or
re-parented while hidden (the export visibility helper chains) are never
evaluated, so their world matrix stays at identity, and the depsgraph copies
that stale matrix back over any value written to the original. The official
exporter bakes it into the exported vertices: a crew figure hidden at
arg 0 by a visibility argument came back ~3 m out of place.

Un-hiding everything for one update gives every object a correct evaluated
matrix; once hidden again the depsgraph leaves it alone.
"""

import bpy

from ..utils import action_fcurves


def _hide_viewport_controls(obj):
    anim = obj.animation_data
    if anim is None:
        return []
    controls = [d for d in anim.drivers if d.data_path == "hide_viewport"]
    action = anim.action
    if action is not None:
        controls += [
            c for c in action_fcurves(action) if c.data_path == "hide_viewport"
        ]
    return controls


def evaluate_hidden_objects():
    """Evaluate every viewport-hidden object once, then restore the hiding."""
    hidden = [obj for obj in bpy.data.objects if obj.hide_viewport]
    if not hidden:
        return 0
    muted = []
    for obj in hidden:
        for control in _hide_viewport_controls(obj):
            if not control.mute:
                control.mute = True
                muted.append(control)
        obj.hide_viewport = False
    bpy.context.view_layer.update()
    for control in muted:
        control.mute = False
    for obj in hidden:
        obj.hide_viewport = True
    bpy.context.view_layer.update()
    return len(hidden)
