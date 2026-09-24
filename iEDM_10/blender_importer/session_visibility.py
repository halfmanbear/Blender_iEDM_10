"""Propagate visibility intervals and hide drivers to render objects."""

import bpy as bpy

from ..utils import action_fcurves
from .import_context import FRAME_SCALE
from .node_identity import _assign_action
from .visibility_timeline import _visibility_scene_keys, _visibility_scene_ranges


def _visibility_frame_intervals(vis_data):
    """Intersect controls and union each control's preview-timeline ranges."""
    intervals = [(0, FRAME_SCALE + 1)]
    for _arg, ranges in vis_data:
        control = _visibility_scene_ranges(ranges)
        intervals = [
            (max(a, c), min(b, d))
            for a, b in intervals
            for c, d in control
            if max(a, c) < min(b, d)
        ]
    return intervals


def _propagate_visibility_hide_to_render_nodes(graph):
    """Preview inherited visibility using actions that obey exporter argument muting.

    Each distinct argument/range set has a custom-property action. The official
    exporter ignores that property, while its argument tools mute the action just
    like the authored VISIBLE actions. Mesh hide drivers read these evaluated
    properties instead of bypassing action muting with direct frame expressions.
    """
    controllers = {}

    for node in graph.nodes:
        obj = node.blender
        if obj is None or obj.type != "MESH" or node.render is None:
            continue
        controls = []
        ancestor = node
        seen = set()
        while ancestor is not None:
            transforms = [ancestor.transform] + list(
                getattr(ancestor, "_collapsed_transforms", None) or []
            )
            for transform in transforms:
                if transform is not None and id(transform) not in seen:
                    seen.add(id(transform))
                    for arg, ranges in getattr(transform, "visData", None) or []:
                        control = _visibility_controller_for(arg, ranges, controllers)
                        if control not in controls:
                            controls.append(control)
            ancestor = ancestor.parent
        if not controls:
            continue
        # Keep expressions below Blender's driver length limit for deep hierarchies.
        while len(controls) > 24:
            combined = []
            for offset in range(0, len(controls), 24):
                helper = bpy.data.objects.new("IEDM_VisibilityIntersection", None)
                bpy.context.collection.objects.link(helper)
                helper.empty_display_size = 0.01
                helper["_iedm_visible"] = 1.0
                _add_visibility_hide_driver(
                    helper,
                    '["_iedm_visible"]',
                    controls[offset : offset + 24],
                    invert=False,
                )

                combined.append(helper)
            controls = combined
        for path in ("hide_viewport", "hide_render"):
            _add_visibility_hide_driver(obj, path, controls)


def _visibility_controller_for(arg, ranges, controllers):
    """Create or reuse the action-backed visibility controller for an argument."""
    signature = (arg, tuple(tuple(pair) for pair in ranges))
    if signature in controllers:
        return controllers[signature]
    helper = bpy.data.objects.new("IEDM_Visibility_{}".format(arg), None)
    bpy.context.collection.objects.link(helper)
    helper.empty_display_size = 0.01
    helper["_iedm_visibility_preview_control"] = True
    intervals = sorted(_visibility_frame_intervals([(arg, ranges)]))
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    helper["_iedm_visible"] = float(any(a <= 100 < b for a, b in merged))
    action = bpy.data.actions.new("{}_IEDM_VisibilityPreview".format(arg))
    if hasattr(action, "argument"):
        action.argument = arg
    curve = action_fcurves(action).new(data_path='["_iedm_visible"]')
    for frame, value in _visibility_scene_keys(ranges):
        curve.keyframe_points.add(1)
        key = curve.keyframe_points[-1]
        key.co = (frame, value)
        key.interpolation = "CONSTANT"
    curve.update()
    _assign_action(helper, action)
    controllers[signature] = helper
    return helper


def _add_visibility_hide_driver(obj, path, controls, invert=True):
    """Drive an object's hide state from one or more visibility controllers."""
    driver = obj.driver_add(path).driver
    driver.type = "SCRIPTED"
    for variable in list(driver.variables):
        driver.variables.remove(variable)
    for index, control in enumerate(controls):
        variable = driver.variables.new()
        variable.name = "v{}".format(index)
        variable.type = "SINGLE_PROP"
        variable.targets[0].id = control
        variable.targets[0].data_path = '["_iedm_visible"]'
    expression = " and ".join("v{}".format(i) for i in range(len(controls)))
    driver.expression = "not (" + expression + ")" if invert else expression
