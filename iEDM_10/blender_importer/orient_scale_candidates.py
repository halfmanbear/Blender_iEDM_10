"""Select and describe oriented-scale control rewrite candidates."""

from ..edm_format.mathtypes import Quaternion, Vector
from .anim_actions import _has_nonidentity_scale_orientation_keys
from .animation import _quat_is_identity
from .graph_pipeline import _get_action_argument


def _single_scale_key_set(source_tf):
    scale_sets = []
    for entry in list(getattr(source_tf, "scaleData", None) or []):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        arg, pair = entry
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        keys4, keys3 = list(pair[0] or []), list(pair[1] or [])
        if keys4 or keys3:
            scale_sets.append((arg, keys4, keys3))
    if len(scale_sets) > 1:
        return None
    return scale_sets[0] if scale_sets else (None, [], [])


def _action_arguments_match(active_action, scale_arg):
    action_arg = _get_action_argument(active_action)
    if active_action is None or scale_arg is None or action_arg is None:
        return True, action_arg
    return int(scale_arg) == int(action_arg), action_arg


def _oriented_scale_rewrite_inputs(
    node,
    bone_anim_source_nodes,
    bone_anim_source_transforms,
    source_for_node,
):
    obj = getattr(node, "blender", None)
    if obj is None or getattr(obj, "type", "") == "ARMATURE":
        return None
    source_tf = source_for_node(node)
    if source_tf is None or node in bone_anim_source_nodes:
        return None
    if source_tf in bone_anim_source_transforms:
        return None
    if getattr(source_tf, "_iedm_oriented_scale_wrapped", False):
        return None
    if getattr(source_tf, "bmat_inv", None) is not None:
        return None
    animation_data = getattr(obj, "animation_data", None)
    active_action = getattr(animation_data, "action", None)
    if animation_data is not None and len(
        list(getattr(animation_data, "nla_tracks", []) or [])
    ) > 0:
        return None

    scale_data = _single_scale_key_set(source_tf)
    if scale_data is None:
        return None
    scale_arg, keys4, keys3 = scale_data
    args_match, action_arg = _action_arguments_match(active_action, scale_arg)
    if not args_match:
        return None

    has_anim_scale = bool(keys3)
    has_anim_orient = _has_nonidentity_scale_orientation_keys(keys4)
    q2_raw = getattr(getattr(source_tf, "base", None), "quat_2", None)
    if not hasattr(q2_raw, "to_matrix"):
        q2_raw = Quaternion(q2_raw or (1.0, 0.0, 0.0, 0.0))
    base_scale = Vector(
        getattr(getattr(source_tf, "base", None), "scale", (1.0, 1.0, 1.0))[:3]
    )
    has_base_scale = any(abs(float(base_scale[i]) - 1.0) > 1e-6 for i in range(3))
    has_base_orient = not _quat_is_identity(q2_raw) and has_base_scale
    if not ((has_anim_scale and has_anim_orient) or has_base_orient):
        return None
    return (
        obj,
        source_tf,
        active_action,
        action_arg,
        scale_arg,
        keys4,
        keys3,
        has_anim_scale,
        has_anim_orient,
        q2_raw,
        base_scale,
        has_base_scale,
    )
