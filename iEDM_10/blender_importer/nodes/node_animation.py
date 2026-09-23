import logging

# Fragment: core node processing — creates Blender objects from the EDM graph.
from ...edm_format.mathtypes import (
    Matrix,
    Vector,
)
from ...edm_format.types import (
    AnimatingNode,
    ArgVisibilityNode,
)
from ..anim_actions import get_actions_for_node
from ..graph_pipeline import (
    _get_action_argument,
    _merge_actions_by_argument,
    _push_action_to_nla,
)
from ..node_transform import apply_node_transform
from ..prelude import (
    _assign_action,
)
from .armature import (
    _merge_visibility_action_into_transform_action,
)

_logger = logging.getLogger(__name__)


def _hookup_node_animations(node, ctx, vis_actions, used_shared_parent):
    """Connect animation actions and apply the rest transform."""
    anim_transforms = _node_animation_transforms(node)
    should_apply_transform = bool(
        node.transform
        and (
            not isinstance(node.transform, ArgVisibilityNode)
            or any(
                isinstance(tf, AnimatingNode)
                for tf in getattr(node, "_collapsed_transforms", [])
            )
        )
    )
    if should_apply_transform:
        if anim_transforms and not _skip_node_object_animation(
            node, ctx, anim_transforms
        ):
            actions = _collect_node_animation_actions(node, anim_transforms)
            actions = _merge_node_visibility_actions(node, actions, vis_actions)
            if actions:
                _assign_node_actions(node.blender, actions)
        _apply_node_rest_transform(node, anim_transforms, used_shared_parent)
        if node.blender.type == "EMPTY":
            if (node.blender.scale - Vector((1, 1, 1))).length < 0.01:
                node.blender.empty_display_size = 0.01
        elif vis_actions:
            _assign_node_actions(node.blender, vis_actions)
    elif vis_actions:
        _assign_node_actions(node.blender, vis_actions)
    _compensate_bone_tail_export(node.blender)


def _node_animation_transforms(node):
    transforms = getattr(node, "_collapsed_transforms", None)
    if transforms is None:
        transforms = [node.transform] if node.transform else []
    return [tf for tf in transforms if isinstance(tf, AnimatingNode)]


def _skip_node_object_animation(node, context, anim_transforms):
    return node in context.get("bone_anim_source_nodes", set()) or any(
        tf in context.get("bone_anim_source_transforms", set())
        for tf in anim_transforms
    )


def _collect_node_animation_actions(node, anim_transforms):
    if isinstance(node.transform, ArgVisibilityNode):
        actions = [
            action for tf in anim_transforms for action in get_actions_for_node(tf)
        ]
    else:
        actions = get_actions_for_node(node.transform)
        for transform in anim_transforms:
            if transform is not node.transform:
                actions.extend(get_actions_for_node(transform))
    return _merge_actions_by_argument(actions)


def _merge_node_visibility_actions(node, actions, vis_actions):
    if not vis_actions:
        return actions
    visibility_by_arg = {}
    visibility_without_arg = []
    for action in vis_actions:
        argument = _get_action_argument(action)
        if argument is None:
            visibility_without_arg.append(action)
        elif argument not in visibility_by_arg:
            visibility_by_arg[argument] = action
    merged = []
    for action in actions:
        visibility = visibility_by_arg.pop(_get_action_argument(action), None)
        if visibility is not None:
            action = _merge_visibility_action_into_transform_action(
                action, visibility, node.blender.name
            )
        merged.append(action)
    return merged + list(visibility_by_arg.values()) + visibility_without_arg


def _assign_node_actions(obj, actions):
    obj.animation_data_create()
    if len(actions) == 1:
        _assign_action(obj, actions[0])
        return
    pushed = sum(_push_action_to_nla(obj, action) for action in actions)
    if pushed:
        obj.animation_data.action = None
    else:
        _assign_action(obj, actions[0])


def _apply_node_rest_transform(node, anim_transforms, used_shared_parent):
    transform = node
    if isinstance(node.transform, ArgVisibilityNode) and anim_transforms:
        transform = next(
            (tf for tf in anim_transforms if not isinstance(tf, ArgVisibilityNode)),
            anim_transforms[0],
        )
    apply_node_transform(transform, node.blender, used_shared_parent=used_shared_parent)
    if transform is not node and transform is not node.transform:
        try:
            transform._blender_obj = node.blender
        except Exception:
            _logger.debug("Ignoring optional operation failure", exc_info=True)


def _compensate_bone_tail_export(obj):
    try:
        tail_length = float(obj.get("_iedm_compensate_bone_tail_export", 0.0))
    except Exception:
        tail_length = 0.0
    if abs(tail_length) <= 1e-8:
        return
    try:
        obj.matrix_basis = (
            Matrix.Translation((0.0, -tail_length, 0.0)) @ obj.matrix_basis
        )
    except Exception as exc:
        print(f"Warning in blender_importer/nodes/core.py: {exc}")
