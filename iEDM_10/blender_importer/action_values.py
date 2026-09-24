"""Interpret animation keys, scale orientation, and action execution order."""

from ..edm_format.mathtypes import MatrixScale, Quaternion, Vector
from ..edm_format.types import ArgAnimationNode, ArgVisibilityNode
from .animation import _quat_is_identity
from .graph_pipeline import _get_action_argument
from .import_context import _log
from .visibility_graph import _is_authored_argvis_control_pair, _is_child_of_file_root


def _plain_root_unit_interval_rot_sets(node):
    if type(node).__name__ != "ArgRotationNode":
        return None
    if _is_authored_argvis_control_pair(node):
        return None
    if not _is_child_of_file_root(node):
        return None
    children = getattr(node, "children", None) or []
    vis = children[0] if children else None
    if vis is not None and isinstance(vis, ArgVisibilityNode):
        return None
    rot_sets = [keys for _arg, keys in (getattr(node, "rotData", None) or []) if keys]
    if (
        not rot_sets
        or getattr(node, "posData", None)
        or getattr(node, "scaleData", None)
    ):
        return None
    frames = [float(getattr(k, "frame", 0.0)) for keys in rot_sets for k in keys]
    if not frames:
        return None
    if min(frames) < -1e-6 or max(frames) > 1.0 + 1e-6:
        return None
    if min(frames) < 0.0 - 1e-6:
        return None
    return rot_sets


def _frame_value_components(value):
    if hasattr(value, "to_matrix"):
        return tuple(float(v) for v in value)
    try:
        return tuple(float(v) for v in value)
    except Exception:
        try:
            return (float(value),)
        except Exception:
            return ()


def _frame_values_close(a, b, eps=1e-5):
    av = _frame_value_components(a)
    bv = _frame_value_components(b)
    if len(av) != len(bv):
        return False
    if not av:
        return False
    if all(abs(x - y) <= eps for x, y in zip(av, bv, strict=False)):
        return True
    if len(av) == 4:
        return all(abs(x + y) <= eps for x, y in zip(av, bv, strict=False))
    return False


def _is_plain_root_unit_interval_argrot(node):
    return _plain_root_unit_interval_rot_sets(node) is not None


def _scale_orientation_quaternion(value):
    if hasattr(value, "to_matrix"):
        return value
    comps = tuple(float(v) for v in value[:4])
    # Scale orientation keys are stored as (x, y, z, w).
    return Quaternion((comps[3], comps[0], comps[1], comps[2]))


def _compose_oriented_scale_matrix(scale_vec, orientation_quat):
    scale_mat = MatrixScale(Vector((scale_vec[0], scale_vec[1], scale_vec[2])))
    orient = (
        orientation_quat
        if hasattr(orientation_quat, "to_matrix")
        else Quaternion(orientation_quat)
    )
    if _quat_is_identity(orient):
        return scale_mat
    orient_mat = orient.to_matrix().to_4x4()
    return orient_mat @ scale_mat @ orient_mat.inverted()


def _action_chain_sort_value(action, default=-1):
    if action is None:
        return default
    try:
        return int(action.get("_iedm_chain_sort", default))
    except Exception:
        return default


def _sorted_transform_actions_for_execution(actions):
    transform_actions = list(actions or [])
    if len(transform_actions) <= 1:
        return transform_actions
    return sorted(
        transform_actions,
        key=lambda action: (
            _action_chain_sort_value(action, -1),
            _get_action_argument(action) or -1,
        ),
        reverse=True,
    )


def _needs_multi_arg_rotation_helper_split(node):
    tf = getattr(node, "transform", None)
    if tf is None or not isinstance(tf, ArgAnimationNode):
        return False
    # A non-armature object can only hold one active Blender action. Any node driven
    # by more than one distinct control argument - whether they all rotate, or split
    # across pos/rot/scale like a translate-then-rotate actuator - would otherwise be
    # pushed onto NLA tracks, which the EDM exporter only reads off ARMATURE objects.
    args = set()
    for attr in ("posData", "rotData", "scaleData"):
        for arg, keys in getattr(tf, attr, None) or []:
            if keys:
                args.add(arg)
    return len(args) > 1


def _has_nonidentity_scale_orientation_keys(keys4):
    for key in list(keys4 or []):
        try:
            quat = _scale_orientation_quaternion(key.value)
        except Exception as exc:
            _log.debug(
                "Skipping invalid scale-orientation key: {}".format(exc), level=2
            )
            continue
        if not _quat_is_identity(quat):
            return True
    return False
