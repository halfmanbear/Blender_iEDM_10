"""Retarget EDM bone actions to Blender pose-bone curves and NLA tracks."""

import bpy as bpy

from ...edm_format.mathtypes import Matrix, Quaternion
from ...edm_format.types import AnimatingNode, ArgVisibilityNode
from ...utils import action_fcurves, new_grouped_fcurve
from ..anim_actions import get_actions_for_node
from ..import_context import _import_ctx, _log
from ..import_logging import _log_bone_debug_event
from ..node_identity import _is_bone_transform


def _copy_fcurve_to_action(src_curve, dst_action, dst_path, action_group):
    dst_curve = new_grouped_fcurve(
        dst_action,
        data_path=dst_path,
        index=src_curve.array_index,
        action_group=action_group,
    )
    dst_curve.extrapolation = src_curve.extrapolation
    for key in src_curve.keyframe_points:
        new_key = dst_curve.keyframe_points.insert(
            key.co[0], key.co[1], options={"FAST"}
        )
        new_key.interpolation = key.interpolation
        try:
            new_key.handle_left_type = key.handle_left_type
            new_key.handle_right_type = key.handle_right_type
            new_key.handle_left = key.handle_left
            new_key.handle_right = key.handle_right
        except Exception as e:
            print(f"Warning in blender_importer/nodes/armature.py: {e}")
    dst_curve.update()


def _action_has_fcurve(action, data_path, index=None):
    if action is None:
        return False
    for fcu in action_fcurves(action):
        if fcu.data_path != data_path:
            continue
        if index is None or fcu.array_index == index:
            return True
    return False


def _copy_bone_rotation_curves(src_action, dst_action, bone_name, rest_quat_inv):
    """Copy rotation_quaternion curves from world to bone-local space.

    Object-level actions bake leftRot (the bone's world-space rest rotation) into
    every keyframe so that rotation_quaternion=leftRot at the rest frame. Pose bone
    rotation_quaternion is relative to the edit-bone orientation, so the rest frame
    must be identity. This removes leftRot by premultiplying each keyframe quaternion
    by inv(rest_rot), giving rotation_quaternion=identity at rest.
    """
    src_curves = {
        fcu.array_index: fcu
        for fcu in action_fcurves(src_action)
        if fcu.data_path == "rotation_quaternion"
    }
    if not src_curves:
        return
    dst_path = 'pose.bones["{}"].rotation_quaternion'.format(bone_name)
    if any(
        action_fcurves(dst_action).find(dst_path, index=i) is not None for i in range(4)
    ):
        return

    dst_curves = []
    for i in range(4):
        dc = new_grouped_fcurve(
            dst_action, data_path=dst_path, index=i, action_group=bone_name
        )
        if i in src_curves:
            dc.extrapolation = src_curves[i].extrapolation
        dst_curves.append(dc)

    all_frames = sorted(
        {kp.co[0] for fcu in src_curves.values() for kp in fcu.keyframe_points}
    )
    for frame in all_frames:
        comps = []
        for i in range(4):
            fcu = src_curves.get(i)
            if fcu is None:
                comps.append(1.0 if i == 0 else 0.0)
                continue
            val = next(
                (
                    kp.co[1]
                    for kp in fcu.keyframe_points
                    if abs(kp.co[0] - frame) < 0.001
                ),
                None,
            )
            comps.append(float(val) if val is not None else (1.0 if i == 0 else 0.0))
        local_q = rest_quat_inv @ Quaternion((comps[0], comps[1], comps[2], comps[3]))
        for i, component in enumerate([local_q.w, local_q.x, local_q.y, local_q.z]):
            new_kp = dst_curves[i].keyframe_points.insert(
                frame, component, options={"FAST"}
            )
            src_fcu = src_curves.get(i)
            if src_fcu:
                src_kp = next(
                    (
                        k
                        for k in src_fcu.keyframe_points
                        if abs(k.co[0] - frame) < 0.001
                    ),
                    None,
                )
                if src_kp:
                    new_kp.interpolation = src_kp.interpolation
                    try:
                        new_kp.handle_left_type = src_kp.handle_left_type
                        new_kp.handle_right_type = src_kp.handle_right_type
                    except Exception as exc:
                        _log.debug("Optional operation failed: {}".format(exc), level=2)
    for dc in dst_curves:
        dc.update()


def _merge_visibility_action_into_transform_action(
    transform_action, vis_action, obj_name
):
    """Clone a transform action and append VISIBLE fcurves from a visibility action."""
    if transform_action is None or vis_action is None:
        return transform_action
    if _action_has_fcurve(transform_action, "VISIBLE"):
        return transform_action

    vis_curves = [
        fcu for fcu in action_fcurves(vis_action) if fcu.data_path == "VISIBLE"
    ]
    if not vis_curves:
        return transform_action

    # Official exporter extracts the EDM argument from the action name prefix.
    # Preserve the visibility action's name prefix (e.g. "10_*") so VISIBLE
    # wrappers are emitted for merged visibility+transform actions.
    merged_name = (
        getattr(vis_action, "name", "") or ""
    ).strip() or transform_action.name
    merged = bpy.data.actions.new(merged_name)
    for src_curve in action_fcurves(transform_action):
        group_name = (
            src_curve.group.name if getattr(src_curve, "group", None) else "Transform"
        )
        _copy_fcurve_to_action(src_curve, merged, src_curve.data_path, group_name)
    for src_curve in vis_curves:
        if (
            action_fcurves(merged).find("VISIBLE", index=src_curve.array_index)
            is not None
        ):
            continue
        group_name = (
            src_curve.group.name if getattr(src_curve, "group", None) else "Visibility"
        )
        _copy_fcurve_to_action(src_curve, merged, "VISIBLE", group_name)
    return merged


def _transfer_bone_actions_to_armature(graph, arm_obj, node_to_bone_name):
    """Retarget per-node ArgAnimatedBone actions to armature pose-bone actions."""
    action_map = {}
    source_graph_nodes = set()
    source_transforms = set()
    bone_nodes = sorted(
        node_to_bone_name.keys(),
        key=lambda n: getattr(n.transform, "_graph_idx", 1 << 30),
    )
    _bone_rest_mats = (_import_ctx.bone_import_ctx or {}).get(
        "bone_rest_matrix_by_name", {}
    )
    for node in bone_nodes:
        bone_name = node_to_bone_name[node]
        rest_mat = _bone_rest_mats.get(bone_name)
        rest_quat_inv = (
            Matrix(rest_mat).to_quaternion().inverted()
            if rest_mat is not None
            else Quaternion()
        )
        # EDM bone animation is typically encoded on wrapper nodes above the Bone.
        seen_sources = set()
        current = node
        while (
            current is not None
            and current.parent is not None
            and current.render is None
            and current.transform is not None
        ):
            tfnode = current.transform
            if isinstance(tfnode, AnimatingNode) and not isinstance(
                tfnode, ArgVisibilityNode
            ):
                if id(tfnode) not in seen_sources:
                    seen_sources.add(id(tfnode))
                    src_actions = get_actions_for_node(tfnode)
                    if src_actions:
                        source_graph_nodes.add(current)
                        source_transforms.add(tfnode)
                    for src_action in src_actions:
                        _log_bone_debug_event(
                            "retarget-source",
                            {
                                "graph_node": getattr(tfnode, "name", "")
                                or type(tfnode).__name__,
                                "graph_node_type": type(tfnode).__name__,
                                "bone_name": bone_name,
                                "action_name": src_action.name,
                                "fcurves": [
                                    fcu.data_path for fcu in action_fcurves(src_action)
                                ],
                            },
                            getattr(tfnode, "name", "") or type(tfnode).__name__,
                            bone_name,
                            src_action.name,
                        )
                        dst_action = action_map.get(src_action.name)
                        if dst_action is None:
                            dst_action = bpy.data.actions.new(src_action.name)
                            action_map[src_action.name] = dst_action
                        # rotation_quaternion curves carry leftRot baked into the rest
                        # value.
                        # Pose bone rotation is relative to edit-bone orientation, so
                        # rest frame must be identity. Use the dedicated
                        # helper to remove the rest rotation from each keyframe.
                        _copy_bone_rotation_curves(
                            src_action, dst_action, bone_name, rest_quat_inv
                        )
                        for src_curve in action_fcurves(src_action):
                            if src_curve.data_path == "rotation_quaternion":
                                continue
                            dst_path = 'pose.bones["{}"].{}'.format(
                                bone_name, src_curve.data_path
                            )
                            # Skip existing FCurves to prevent a crash on re-import.
                            if (
                                action_fcurves(dst_action).find(
                                    dst_path, index=src_curve.array_index
                                )
                                is not None
                            ):
                                continue
                            _copy_fcurve_to_action(
                                src_curve, dst_action, dst_path, bone_name
                            )
            parent = current.parent
            if parent is None or _is_bone_transform(parent.transform):
                break
            current = parent

    if not action_map:
        return source_graph_nodes, source_transforms

    _log_bone_debug_event(
        "retarget-summary",
        {
            "armature": getattr(arm_obj, "name", None),
            "actions": sorted(action_map.keys()),
            "source_graph_nodes": sorted(
                (
                    getattr(getattr(n, "transform", None), "name", "")
                    or type(getattr(n, "transform", None)).__name__
                )
                for n in source_graph_nodes
            ),
        },
        getattr(arm_obj, "name", None),
    )

    _attach_retargeted_bone_actions(arm_obj, action_map)
    return source_graph_nodes, source_transforms


def _attach_retargeted_bone_actions(arm_obj, action_map):
    """Attach copied bone actions to the armature using matching NLA tracks."""
    arm_obj.animation_data_create()
    ad = arm_obj.animation_data
    ad.use_nla = True
    ad.action = None
    for track in list(ad.nla_tracks):
        ad.nla_tracks.remove(track)

    for action_name in sorted(action_map.keys()):
        action = action_map[action_name]
        track = ad.nla_tracks.new()
        track.name = action.name
        # Start the strip at the action's first key so keys keep their scene frames.
        start = float(action.frame_range[0])
        strip = track.strips.new(action.name, int(start), action)
        if getattr(strip, "action_slot", False) is None and action.slots:
            strip.action_slot = action.slots[0]
        if abs(strip.frame_start - start) > 1e-6 and hasattr(strip, "frame_start_ui"):
            strip.frame_start_ui = start
        strip.name = action.name
        strip.extrapolation = "NOTHING"
