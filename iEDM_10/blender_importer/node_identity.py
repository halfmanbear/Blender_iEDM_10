"""Identify connectors and skeleton nodes and normalize display names."""

import re as re

from ..edm_format.types import AnimatingNode, ArgVisibilityNode, TransformNode


def _ob_local_is_identity(o, eps=1e-6):
    """Return True if o.matrix_local is (approximately) the identity matrix."""
    try:
        m = o.matrix_local
        return all(
            abs(float(m[r][c]) - (1.0 if r == c else 0.0)) <= eps
            for r in range(4)
            for c in range(4)
        )
    except Exception:
        return False


def _assign_action(owner, action):
    """Assign an action and bind its slot; Blender 4.4+ can leave it unbound."""
    ad = owner.animation_data or owner.animation_data_create()
    ad.action = action
    if (
        action is not None
        and getattr(ad, "action_slot", False) is None
        and action.slots
    ):
        ad.action_slot = action.slots[0]
    return ad


def _is_connector_object(obj):
    if obj is None:
        return False
    if hasattr(obj, "edm") and getattr(obj.edm, "is_connector", False):
        return True
    if (
        hasattr(obj, "EDMProps")
        and getattr(obj.EDMProps, "SPECIAL_TYPE", "") == "CONNECTOR"
    ):
        return True
    return False


def _is_connector_transform(tfnode, blender_obj=None):
    # A TransformNode is a connector wrapper if it's named "Connector Transform"
    # (the standard io_scene_edm export name) OR if its blender_obj is a connector
    # empty created by create_connector() (some EDMs use the connector's own
    # name, e.g. "BANO_0", "GUN_POINT", "Pylon1", as the wrapper TransformNode
    # name). Apply the -90° X correction only when legacy exporter behavior is
    # positively identified.
    if not isinstance(tfnode, TransformNode):
        return False
    if not _is_connector_object(blender_obj):
        return False
    try:
        return bool(blender_obj.get("_iedm_connector_apply_xfix", False))
    except Exception:
        return False


def _strip_anim_prefix(name):
    """Strip known EDM animation/control prefixes from node names."""
    if not name:
        return name
    for prefix in ("ar_", "al_", "as_", "tl_", "tr_", "s_", "v_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _is_generic_render_name(name):
    if not name:
        return True
    if re.match(r"^_[0-9]+$", name):
        return True
    if re.match(r"^Object(\.[0-9]+)?$", name):
        return True
    return False


def _is_anim_node_name(name):
    """Check if a name has animation/control prefixes that indicate a skeleton node."""
    if not name:
        return False
    return any(
        name.startswith(p) for p in ("ar_", "al_", "as_", "tl_", "tr_", "s_", "v_")
    )


def _is_bone_transform(tfnode):
    return tfnode is not None and "Bone" in type(tfnode).__name__


def _is_skeleton_controller(node):
    """Return whether the node itself carries animation or light structure."""
    if node.transform:
        name = getattr(node.transform, "name", "") or ""
        if isinstance(node.transform, AnimatingNode):
            return True
        if _is_anim_node_name(name) or "Light Transform" in name:
            return True
    return bool(node.render and type(node.render).__name__ == "LightNode")


def is_skeleton_node(node):
    """Determine if a graph node should be preserved as a separate Empty/Armature.

    Skeleton nodes are animation controllers, bone ancestors, light transforms,
    or nodes with renderable descendants that need structural hierarchy.
    """
    if node is None:
        return False
    if getattr(node, "_is_graph_root", False):
        return False

    # AnimatingNodes, LightNodes and nodes with anim prefixes are always skeletal
    if _is_skeleton_controller(node):
        return True

    # Ancestors of Bone nodes should be preserved. _mark_skeleton_nodes caches the
    # subtree flags in one pass; fall back to walking the subtree otherwise.
    def _has_bone(n):
        cached = getattr(n, "_subtree_has_bone", None)
        if cached is not None:
            return cached
        if n.transform and "Bone" in type(n.transform).__name__:
            return True
        return any(_has_bone(c) for c in n.children)

    def _has_render(n):
        cached = getattr(n, "_subtree_has_render", None)
        if cached is not None:
            return cached
        if n.render is not None:
            return True
        return any(_has_render(c) for c in n.children)

    # Check ancestors up to graph root
    res = False
    curr = node
    while curr and not getattr(curr, "_is_graph_root", False):
        if curr.transform and "Bone" in type(curr.transform).__name__:
            res = True
            break
        curr = curr.parent

    if not res:
        res = _has_bone(node) or _has_render(node)

    return res


def _transform_display_name(tfnode):
    raw_name = getattr(tfnode, "name", "") or ""
    if raw_name:
        name = raw_name
    else:
        idx = getattr(tfnode, "_graph_idx", None)
        if idx is not None:
            name = "tf_{:04d}".format(idx)
        else:
            name = type(tfnode).__name__
    if isinstance(tfnode, ArgVisibilityNode):
        return name
    name = _strip_anim_prefix(name)
    # Strip "Armature : " prefix (exporter adds it to bone names)
    if " : " in name:
        name = name.split(" : ")[-1]
    return name
