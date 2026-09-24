"""Copy action curves and build oriented-scale and argument animations."""

import bpy as bpy

from ..edm_format.mathtypes import Matrix, Quaternion, Vector
from ..utils import action_fcurves
from .action_values import _scale_orientation_quaternion
from .animation import (
    _finalize_authored_transform_action,
    add_position_fcurves,
    add_rotation_fcurves,
    add_scale_fcurves,
)
from .graph_diagnostics import _anim_quaternion_to_blender
from .graph_pipeline import _get_action_argument
from .import_context import _log
from .import_logging import _log_bone_debug_event, _matrix_trs_summary
from .node_identity import _strip_anim_prefix
from .visibility_timeline import _anim_frame_to_scene_frame


def _copy_fcurve_points_local(src_curve, dst_curve):
    for kp in src_curve.keyframe_points:
        try:
            frame = float(kp.co[0])
            value = float(kp.co[1])
        except Exception as exc:
            _log.debug("Skipping invalid animation keyframe: {}".format(exc), level=2)
            continue
        new_kp = dst_curve.keyframe_points.insert(frame, value, options={"FAST"})
        try:
            new_kp.interpolation = kp.interpolation
            new_kp.handle_left_type = kp.handle_left_type
            new_kp.handle_right_type = kp.handle_right_type
            new_kp.easing = kp.easing
        except Exception as exc:
            _log.debug(
                "Could not copy animation keyframe metadata: {}".format(exc), level=2
            )


def _clone_action_filtered(
    action, name_suffix="", exclude_paths=None, include_paths=None
):
    if action is None:
        return None
    exclude_paths = set(exclude_paths or [])
    include_paths = set(include_paths or [])
    cloned = bpy.data.actions.new("{}{}".format(action.name, name_suffix))
    arg = _get_action_argument(action)
    if arg is not None and hasattr(cloned, "argument"):
        cloned.argument = int(arg)
    copied = 0
    for src_curve in action_fcurves(action):
        if include_paths and src_curve.data_path not in include_paths:
            continue
        if src_curve.data_path in exclude_paths:
            continue
        dst_curve = action_fcurves(cloned).new(
            data_path=src_curve.data_path, index=src_curve.array_index
        )
        _copy_fcurve_points_local(src_curve, dst_curve)
        copied += 1
    if copied == 0:
        try:
            bpy.data.actions.remove(cloned)
        except Exception as exc:
            _log.debug("Could not remove empty cloned action: {}".format(exc), level=2)
        return None
    return cloned


def _create_scale_orientation_rotation_action(
    name, keys4, frame_mapper=None, invert=False
):
    if not keys4:
        return None
    action = bpy.data.actions.new(name)
    curves = []
    for idx in range(4):
        curves.append(
            action_fcurves(action).new(data_path="rotation_quaternion", index=idx)
        )
    frame_mapper = frame_mapper or _anim_frame_to_scene_frame
    previous_quat = None
    for framedata in keys4:
        quat = _scale_orientation_quaternion(framedata.value)
        if invert:
            quat = quat.conjugated()
        if previous_quat is not None and previous_quat.dot(quat) < 0.0:
            quat = -quat
        previous_quat = quat.copy()
        frame = frame_mapper(framedata.frame)
        for curve, component in zip(curves, quat, strict=False):
            curve.keyframe_points.add(1)
            curve.keyframe_points[-1].co = (frame, float(component))
            curve.keyframe_points[-1].interpolation = "LINEAR"
    for curve in curves:
        try:
            curve.update()
        except Exception as exc:
            _log.debug("Could not update animation curve: {}".format(exc), level=2)
    return action


def _build_arganimation_action(
    node,
    arg,
    basis_local,
    frame_mapper=None,
    include_scale=True,
    action_name=None,
    rotation_basis_local=None,
    position_prefix_lifted=False,
):
    """Build a single action for one ArgAnimationNode argument on a chosen basis.

    Used both for the normal import path and for oriented-scale wrapper reconstruction,
    where the animated transform must be re-derived against the wrapper's actual local
    basis instead of cloning curves from the original node. With
    position_prefix_lifted, `base.matrix @ T(base.position)` lives on a parent helper,
    so position keys are already in the parent frame.
    """
    posData = [x[1] for x in node.posData if x[0] == arg]
    rotData = [x[1] for x in node.rotData if x[0] == arg]
    scaleData = [x[1] for x in node.scaleData if x[0] == arg] if include_scale else []

    action = bpy.data.actions.new(
        action_name or "{}_{}".format(arg, _strip_anim_prefix(node.name or "anim"))
    )
    if hasattr(action, "argument"):
        action.argument = arg

    rot_arg_order = [
        entry_arg for entry_arg, keys in (getattr(node, "rotData", None) or []) if keys
    ]
    rot_arg_index = {entry_arg: idx for idx, entry_arg in enumerate(rot_arg_order)}
    if arg in rot_arg_index and len(rot_arg_order) > 1:
        action["_iedm_chain_sort"] = int(rot_arg_index[arg])
        action["_iedm_rotation_accum_args"] = int(len(rot_arg_order))

    static_loc, static_rot, _static_scale = basis_local.decompose()
    if rotation_basis_local is not None:
        try:
            _rot_loc, static_rot, _rot_scale = rotation_basis_local.decompose()
        except Exception as exc:
            _log.debug("Could not decompose rotation basis: {}".format(exc), level=2)
    leftRot = static_rot
    rightRot = Quaternion((1, 0, 0, 0))
    if rotation_basis_local is None and basis_local.to_3x3().determinant() < 0:
        # decompose() folds a single-axis mirror F into the rotation (R @ F);
        # animated keys belong between R and F, not after R @ F.
        base_scale = Vector(node.base.scale)
        flip = Matrix.Diagonal([-1.0 if s >= 0 else 1.0 for s in base_scale])
        if flip.determinant() > 0:
            flip_quat = flip.to_quaternion()
            authored = static_rot @ flip_quat.inverted()
            rebuilt = authored.to_matrix() @ Matrix.Diagonal(base_scale)
            target = basis_local.to_3x3()
            if (
                max(
                    abs(rebuilt[r][c] - target[r][c])
                    for r in range(3)
                    for c in range(3)
                )
                < 1e-4
            ):
                leftRot = authored
                rightRot = flip_quat
    leftPos = (
        # Position deltas precede the default rotation in the EDM transform.
        Matrix.Translation(static_loc) @ Matrix(node.base.matrix).to_3x3().to_4x4()
        if posData and not position_prefix_lifted
        else Matrix.Identity(4)
    )
    rightPos = Matrix.Identity(4)
    base_scale_vec = Vector(
        (node.base.scale[0], node.base.scale[1], node.base.scale[2])
    )
    # Scale keys replace the object's whole scale, so they must carry a uniform
    # scale folded into base.matrix too (F4U-1D Bano glow nodes: 1.032).
    matrix_scale = Matrix(node.base.matrix).to_3x3().to_scale()
    if max(matrix_scale) - min(matrix_scale) < 1e-4 * max(matrix_scale):
        base_scale_vec *= matrix_scale[0]

    def key_quat_to_blender(q):
        return _anim_quaternion_to_blender(q)

    for pos in posData:
        add_position_fcurves(
            action, pos, leftPos, rightPos, node=node, frame_mapper=frame_mapper
        )
    for rot in rotData:
        add_rotation_fcurves(
            action,
            rot,
            leftRot,
            rightRot,
            quat_to_blender=key_quat_to_blender,
            frame_mapper=frame_mapper,
            use_euler=False,
        )
    for sca_pair in scaleData:
        keys3 = sca_pair[1] if isinstance(sca_pair, tuple) and len(sca_pair) > 1 else []
        add_scale_fcurves(
            action, keys3, frame_mapper=frame_mapper, base_scale=base_scale_vec
        )

    _log_bone_debug_event(
        "anim-action",
        {
            "node_name": getattr(node, "name", "") or type(node).__name__,
            "node_type": type(node).__name__,
            "argument": int(arg) if isinstance(arg, int) else arg,
            "action_name": action.name,
            "has_pos": bool(posData),
            "has_rot": bool(rotData),
            "has_scale": bool(scaleData),
            "left_rotation": [round(float(v), 6) for v in leftRot],
            "right_rotation": [round(float(v), 6) for v in rightRot],
            "left_position": _matrix_trs_summary(leftPos),
            "right_position": _matrix_trs_summary(rightPos),
        },
        getattr(node, "name", "") or type(node).__name__,
        action.name,
    )
    _finalize_authored_transform_action(action)
    return action
