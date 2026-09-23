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
    _all_collapsed = getattr(node, "_collapsed_transforms", [])
    _has_collapsed_anim = any(isinstance(tf, AnimatingNode) for tf in _all_collapsed)
    if node.transform and (
        not isinstance(node.transform, ArgVisibilityNode) or _has_collapsed_anim
    ):
        all_transforms = getattr(
            node, "_collapsed_transforms", [node.transform] if node.transform else []
        )
        anim_transforms = [tf for tf in all_transforms if isinstance(tf, AnimatingNode)]
        bone_anim_source_nodes = ctx.get("bone_anim_source_nodes", set())
        bone_anim_source_transforms = ctx.get("bone_anim_source_transforms", set())
        skip_object_anim = node in bone_anim_source_nodes or any(
            tf in bone_anim_source_transforms for tf in anim_transforms
        )

        if anim_transforms and not skip_object_anim:
            if isinstance(node.transform, ArgVisibilityNode):
                actions = []
                for extra_tf in anim_transforms:
                    actions.extend(get_actions_for_node(extra_tf))
            else:
                actions = get_actions_for_node(node.transform)
                for extra_tf in anim_transforms:
                    if extra_tf is not node.transform:
                        actions.extend(get_actions_for_node(extra_tf))
            actions = _merge_actions_by_argument(actions)

            if actions:
                if vis_actions:
                    vis_by_arg = {}
                    vis_no_arg = []
                    for vis_action in vis_actions:
                        vis_arg = _get_action_argument(vis_action)
                        if vis_arg is None:
                            vis_no_arg.append(vis_action)
                        elif vis_arg not in vis_by_arg:
                            vis_by_arg[vis_arg] = vis_action

                    merged_actions = []
                    for action in actions:
                        vis_action = vis_by_arg.pop(_get_action_argument(action), None)
                        if vis_action is not None:
                            action = _merge_visibility_action_into_transform_action(
                                action, vis_action, node.blender.name
                            )
                        merged_actions.append(action)
                    actions = merged_actions + list(vis_by_arg.values()) + vis_no_arg

                node.blender.animation_data_create()
                if len(actions) == 1:
                    _assign_action(node.blender, actions[0])
                else:
                    nla_pushed = 0
                    for action in actions:
                        if _push_action_to_nla(node.blender, action):
                            nla_pushed += 1
                    if nla_pushed > 0:
                        node.blender.animation_data.action = None
                    else:
                        _assign_action(node.blender, actions[0])

        # Apply rest transform: for authored-pair collapse use the AnimatingNode's
        # zero_transform_local_matrix rather than the ArgVisibilityNode (identity).
        if isinstance(node.transform, ArgVisibilityNode) and anim_transforms:
            _tf_to_apply = next(
                (tf for tf in anim_transforms if not isinstance(tf, ArgVisibilityNode)),
                anim_transforms[0],
            )
            apply_node_transform(
                _tf_to_apply, node.blender, used_shared_parent=used_shared_parent
            )
            if _tf_to_apply is not node.transform:
                try:
                    _tf_to_apply._blender_obj = node.blender
                except Exception:
                    _logger.debug("Ignoring optional operation failure", exc_info=True)
        else:
            apply_node_transform(
                node, node.blender, used_shared_parent=used_shared_parent
            )

        if node.blender.type == "EMPTY":
            distFromScale = node.blender.scale - Vector((1, 1, 1))
            if distFromScale.length < 0.01:
                node.blender.empty_display_size = 0.01
        elif vis_actions:
            node.blender.animation_data_create()
            if len(vis_actions) == 1:
                _assign_action(node.blender, vis_actions[0])
            else:
                nla_pushed = 0
                for action in vis_actions:
                    if _push_action_to_nla(node.blender, action):
                        nla_pushed += 1
                if nla_pushed > 0:
                    node.blender.animation_data.action = None
                else:
                    _assign_action(node.blender, vis_actions[0])

    elif vis_actions:
        node.blender.animation_data_create()
        if len(vis_actions) == 1:
            _assign_action(node.blender, vis_actions[0])
        else:
            nla_pushed = 0
            for action in vis_actions:
                if _push_action_to_nla(node.blender, action):
                    nla_pushed += 1
            if nla_pushed > 0:
                node.blender.animation_data.action = None
            else:
                _assign_action(node.blender, vis_actions[0])

    try:
        tail_len = float(node.blender.get("_iedm_compensate_bone_tail_export", 0.0))
    except Exception:
        tail_len = 0.0
    if abs(tail_len) > 1e-8:
        try:
            # The official exporter inserts an implicit "End Of <bone>" transform
            # for direct bone children.  Pre-apply the inverse so imported authored
            # controls round-trip at their EDM positions instead of at the bone tail.
            node.blender.matrix_basis = (
                Matrix.Translation((0.0, -tail_len, 0.0)) @ node.blender.matrix_basis
            )
        except Exception as e:
            print(f"Warning in blender_importer/nodes/core.py: {e}")
