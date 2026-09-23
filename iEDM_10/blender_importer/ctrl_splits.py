# Fragment: multi-arg control splitting and control/mesh-pair renaming.
import bpy

from ..edm_format.mathtypes import Matrix
from ..edm_format.types import ArgAnimationNode
from ..utils import action_fcurves
from .anim_actions import (
    _action_chain_sort_value,
    _action_has_visibility_curve,
    _build_arganimation_action,
    _build_nonarmature_action_plan,
    _clear_object_animation_tracks,
    _clone_action_filtered,
    _collect_merged_transform_actions_for_graph_node,
    _needs_multi_arg_rotation_helper_split,
    _sorted_transform_actions_for_execution,
    _visibility_source_for_graph_node,
    get_actions_for_node,
)
from .graph_pipeline import _get_action_argument
from .prelude import (
    _SUFFIX_RE,
    _assign_action,
    _import_ctx,
    _strip_anim_prefix,
)


def _action_paths(action):
    return {fc.data_path for fc in action_fcurves(action)}


def _relative_keys_from_source(action, node):
    """Re-key an inner split carrier; the owner holds the static part."""
    arg = _get_action_argument(action)
    sources = [
        t
        for t in getattr(node, "_collapsed_transforms", None) or [node.transform]
        if isinstance(t, ArgAnimationNode)
        and any(a == arg and k for a, k in list(t.posData) + list(t.rotData))
    ]
    if not sources:
        return
    rebuilt = _build_arganimation_action(
        sources[0], arg, Matrix.Identity(4), include_scale=False
    )
    for fc in action_fcurves(action):
        if fc.data_path not in {"location", "rotation_quaternion"}:
            continue
        src = action_fcurves(rebuilt).find(fc.data_path, index=fc.array_index)
        if src is None or len(src.keyframe_points) != len(fc.keyframe_points):
            continue
        for dst, point in zip(fc.keyframe_points, src.keyframe_points, strict=False):
            dst.co.y = dst.handle_left.y = dst.handle_right.y = point.co.y
        fc.update()
    bpy.data.actions.remove(rebuilt)


def _split_multi_arg_rotation_controls(graph):
    ctx = _import_ctx.bone_import_ctx or {}
    scene_collection = bpy.context.collection

    for node in getattr(graph, "nodes", []) or []:
        ob = getattr(node, "blender", None)
        if ob is None or getattr(ob, "type", "") == "ARMATURE":
            continue
        if not _needs_multi_arg_rotation_helper_split(node):
            continue

        vis_source = _visibility_source_for_graph_node(node)
        vis_actions = get_actions_for_node(vis_source) if vis_source else []
        transform_actions = _sorted_transform_actions_for_execution(
            _collect_merged_transform_actions_for_graph_node(node, ctx)
        )
        if len(transform_actions) <= 1:
            continue

        planned_actions = _build_nonarmature_action_plan(transform_actions, vis_actions)
        if len(planned_actions) <= 1:
            continue

        planned_actions = _divide_position_rotation_actions(planned_actions)
        static_loc, static_rot, static_scale = ob.matrix_basis.decompose()
        seen_paths = set()
        for action in planned_actions:
            paths = _action_paths(action)
            if paths & seen_paths & {"location", "rotation_quaternion"}:
                _relative_keys_from_source(action, node)
            seen_paths |= paths
        later_paths = set()
        for action in planned_actions[1:]:
            later_paths |= _action_paths(action)
        inner_static = None
        if "location" in later_paths:
            # Position deltas are neither rotated nor scaled.
            # Keep static R @ S innermost.
            ob.matrix_basis = Matrix.Translation(static_loc)
            inner_static = Matrix.LocRotScale(
                None,
                None if "rotation_quaternion" in later_paths else static_rot,
                static_scale,
            )
        elif "rotation_quaternion" in later_paths - _action_paths(planned_actions[0]):
            ob.matrix_basis = Matrix.LocRotScale(static_loc, None, static_scale)

        _build_rotation_helper_chain(
            ob, planned_actions, inner_static, scene_collection
        )


def _divide_position_rotation_actions(actions):
    """Separate arguments that animate position and rotation into distinct links."""
    divided = []
    for action in actions:
        paths = _action_paths(action)
        sort_value = _action_chain_sort_value(action, -1)
        if sort_value > 0 and {"location", "rotation_quaternion"} <= paths:
            position = _clone_action_filtered(
                action, "_pos", include_paths={"location"}
            )
            rotation = _clone_action_filtered(
                action, "_rot", exclude_paths={"location"}
            )
            if position is not None and rotation is not None:
                position["_iedm_chain_sort"] = -1
                rotation["_iedm_chain_sort"] = sort_value
                divided.extend((position, rotation))
                continue
        divided.append(action)
    # EDM evaluates position before rotations, then rotations in authored order.
    divided.sort(
        key=lambda action: (
            "location" not in _action_paths(action),
            _action_chain_sort_value(action, -1),
        )
    )
    return divided


def _build_rotation_helper_chain(obj, actions, inner_static, scene_collection):
    """Assign split actions to identity helpers and preserve direct children."""
    children = list(obj.children)
    _clear_object_animation_tracks(obj)
    _assign_action(obj, actions[0])
    obj["_iedm_multi_arg_rotation_split"] = True
    parent = obj
    helpers = []
    for action in actions[1:]:
        helper = bpy.data.objects.new(obj.name, None)
        helper.empty_display_size = 0.1
        scene_collection.objects.link(helper)
        helper.parent = parent
        helper.matrix_parent_inverse = Matrix.Identity(4)
        helper.matrix_basis = Matrix.Identity(4)
        if "rotation_quaternion" in _action_paths(action):
            helper.rotation_mode = "QUATERNION"
        _assign_action(helper, action)
        helper["_iedm_identity_passthrough"] = True
        helper["_iedm_narrow_identity_passthrough"] = True
        if _action_has_visibility_curve(action):
            helper["_iedm_vis_passthrough"] = True
        helpers.append(helper)
        parent = helper
    if inner_static is not None and not _matrix_is_identity(inner_static):
        helper = bpy.data.objects.new(obj.name, None)
        helper.empty_display_size = 0.1
        scene_collection.objects.link(helper)
        helper.parent = parent
        helper.matrix_basis = inner_static
        helpers.append(helper)
    if helpers:
        for child in children:
            if child not in helpers and child.parent == obj:
                child.parent = helpers[-1]


def _matrix_is_identity(matrix):
    return all(
        abs(matrix[row][column] - (1.0 if row == column else 0.0)) <= 1e-6
        for row in range(4)
        for column in range(4)
    )


def _rename_control_wrapper_mesh_pairs(graph):
    """Keep visible meshes on the semantic base name instead of `.001` helpers.

    Official plain-root visibility scenes frequently contain a control empty with the
    same semantic name as its sole mesh child. Blender auto-suffixes the child mesh
    (`tf_0466.001`), which is noisy. Rename the control wrapper to its authored control
    prefix so the visible mesh can reclaim the base name.
    """
    prefix_by_cls = {
        "ArgVisibilityNode": "v_",
        "ArgRotationNode": "ar_",
        "ArgPositionNode": "al_",
        "ArgScaleNode": "as_",
    }

    for node in getattr(graph, "nodes", []) or []:
        tf = getattr(node, "transform", None)
        ob = getattr(node, "blender", None)
        if tf is None or ob is None:
            continue
        if getattr(node, "render", None) is not None:
            continue
        if getattr(ob, "type", "") != "EMPTY":
            continue

        prefix = prefix_by_cls.get(type(tf).__name__, "")
        if not prefix:
            continue

        children = [
            child
            for child in (getattr(node, "children", []) or [])
            if getattr(child, "blender", None) is not None
        ]
        if len(children) != 1:
            continue

        child = children[0]
        child_render = getattr(child, "render", None)
        child_ob = getattr(child, "blender", None)
        if child_render is None or child_ob is None:
            continue
        if getattr(child, "transform", None) is not None:
            continue
        if getattr(child_ob, "type", "") != "MESH":
            continue

        desired_mesh_name = str(getattr(child_render, "name", "") or "")
        desired_mesh_name = _strip_anim_prefix(desired_mesh_name)
        if desired_mesh_name.startswith("Empty_"):
            desired_mesh_name = desired_mesh_name[len("Empty_") :]
        if not desired_mesh_name:
            desired_mesh_name = _SUFFIX_RE.sub("", getattr(child_ob, "name", "") or "")
        if not desired_mesh_name or desired_mesh_name.lower() == "root":
            continue

        current_parent_name = str(getattr(ob, "name", "") or "")
        current_parent_base = _SUFFIX_RE.sub(
            "", _strip_anim_prefix(current_parent_name)
        )
        current_mesh_name = str(getattr(child_ob, "name", "") or "")
        desired_parent_name = prefix + desired_mesh_name

        # Only rewrite when the current names are a Blender duplicate split of the same
        # semantic base (`tf_0466` + `tf_0466.001`) or when the child carries the
        # semantic base but the parent still steals it.
        if (
            current_parent_base != desired_mesh_name
            and current_mesh_name != desired_mesh_name
        ):
            continue

        try:
            ob.name = desired_parent_name
            child_ob.name = desired_mesh_name
        except Exception as e:
            print(f"Warning in _rename_control_wrapper_mesh_pairs: {e}")
