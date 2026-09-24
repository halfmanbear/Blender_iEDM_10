"""Classify authored visibility pairs and root-wrapper relationships."""

import re as re

from ..edm_format.types import AnimatingNode, ArgVisibilityNode


def _is_authored_argvis_control_pair(node):
    """Check for an authored ArgVisibility -> Arg* pair with the same name.

    Official exports often encode control empties as:
      ArgVisibilityNode("Dummy604") -> ArgRotationNode("Dummy604")
      ArgVisibilityNode("Dummy001") -> ArgAnimationNode("Dummy001")
    These are structurally different from a plain top-level arg node even when
    _is_child_of_file_root() returns True through a pure ArgVisibility chain.
    Treating them like direct root children applies the v10 basis fix one layer
    too early.
    """
    try:
        parent = getattr(node, "parent", None)
        if parent is None:
            return False
        parent_tf = getattr(parent, "transform", None)
        if not (
            isinstance(parent, ArgVisibilityNode)
            or isinstance(parent_tf, ArgVisibilityNode)
        ):
            return False
        name = getattr(node, "name", "") or ""
        parent_name = (
            getattr(parent, "name", "") or getattr(parent_tf, "name", "") or ""
        )
        if not (name and parent_name):
            return False
        if name == parent_name:
            return True
        return _canonical_control_name(name) == _canonical_control_name(parent_name)
    except Exception:
        return False


def _canonical_control_name(name):
    """Normalize authored control names so DAM_* and *_DAM wrappers still pair.

    Some 3ds Max exports write the visibility wrapper as `DAM_Rudder_L` and the
    driven control as `Rudder_L_DAM` or `Flap_L_DAM`. Those are the same
    authored control pair even though the strings are not byte-identical.
    """
    import re

    raw = str(name or "").lower()
    raw = re.sub(r"\.\d{3}$", "", raw)
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", raw) if tok]
    norm = []
    for tok in tokens:
        if len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        norm.append(tok)
    return tuple(sorted(norm))


def _is_nested_authored_argvis_control_pair(node):
    """Check for same-name ArgVisibility -> Arg* under another animated parent.

    Example pattern:
    ArgRotationNode("A") -> ArgVisibilityNode("B")
    -> ArgRotationNode("B")

    In these nested pairs, the inner Arg* node's non-zero base.position is the
    static offset of the middle wrapper object, while the deepest child object
    should primarily own the animated rotation.
    """
    try:
        if not _is_authored_argvis_control_pair(node):
            return False
        parent = getattr(node, "parent", None)
        grandparent = getattr(parent, "parent", None) if parent is not None else None
        grandparent_tf = (
            getattr(grandparent, "transform", grandparent)
            if grandparent is not None
            else None
        )
        return isinstance(grandparent_tf, AnimatingNode) and not isinstance(
            grandparent_tf, ArgVisibilityNode
        )
    except Exception:
        return False


def _is_root_visibility_chain_authored_pair(node):
    """Check same-name ArgVis -> Arg* pairs under only ArgVisibility ancestors.

    These pairs are not the same as the nested animated-parent case above. They
    sit under a pure visibility wrapper chain from the file root, so they still
    need the top-level q1 basis handling used for root-authored controls.
    """
    try:
        if not _is_authored_argvis_control_pair(node):
            return False
        if _is_nested_authored_argvis_control_pair(node):
            return False
        if not _is_child_of_file_root(node):
            return False
        grandparent = getattr(getattr(node, "parent", None), "parent", None)
        grandparent_tf = (
            getattr(grandparent, "transform", grandparent)
            if grandparent is not None
            else None
        )
        return isinstance(grandparent_tf, ArgVisibilityNode)
    except Exception:
        return False


def _is_top_level_visibility_authored_pair(node):
    """Immediate same-name ArgVis -> Arg* pair under a root visibility wrapper.

    This is the narrow legacy 3ds Max shape that still behaves like a top-level
    authored control even though one visibility wrapper sits between it and the
    file root:

    Root -> ArgVisibility("Dummy648") -> ArgVisibility("Dummy603")
    -> ArgRotation("Dummy603")

    Deeper visibility-only chains such as:

    Root -> ArgVisibility("Dummy638") -> ArgVisibility("Dummy637")
    -> ArgVisibility("Dummy636") -> ArgRotation("Dummy597")

    must not be treated as top-level controls.
    """
    try:
        if not _is_authored_argvis_control_pair(node):
            return False
        parent = getattr(node, "parent", None)
        grandparent = getattr(parent, "parent", None) if parent is not None else None
        if grandparent is None:
            return False
        grandparent_tf = getattr(grandparent, "transform", grandparent)
        if not isinstance(grandparent_tf, ArgVisibilityNode):
            return False
        great_grandparent = getattr(grandparent, "parent", None)
        return (
            great_grandparent is not None
            and getattr(great_grandparent, "parent", None) is None
        )
    except Exception:
        return False


def _has_only_visibility_ancestors(node):
    try:
        parent = getattr(node, "parent", None)
        if parent is None:
            return False
        current = parent
        seen_any = False
        while current is not None:
            current_tf = getattr(current, "transform", current)
            if not isinstance(current_tf, ArgVisibilityNode):
                return False
            seen_any = True
            current = getattr(current, "parent", None)
        return seen_any
    except Exception:
        return False


def _visibility_node_has_direct_anim_child(vis_node):
    try:
        for ch in getattr(vis_node, "children", None) or []:
            ch_tf = getattr(ch, "transform", None)
            if isinstance(ch_tf, AnimatingNode) and not isinstance(
                ch_tf, ArgVisibilityNode
            ):
                return True
        return False
    except Exception:
        return False


def _nearest_visibility_ancestor_with_direct_anim_child(node):
    """Return nearest ArgVisibility ancestor with a direct Arg* child."""
    try:
        current = getattr(node, "parent", None)
        while current is not None:
            current_tf = getattr(current, "transform", current)
            if not isinstance(current_tf, ArgVisibilityNode):
                return None
            if _visibility_node_has_direct_anim_child(current):
                return current
            current = getattr(current, "parent", None)
        return None
    except Exception:
        return None


def _is_static_root_visibility_wrapper(node):
    """Find a static ArgVisibility wrapper with an outer Arg* sibling."""
    try:
        if node is None:
            return False
        tf = getattr(node, "transform", None)
        if not isinstance(tf, ArgVisibilityNode):
            return False
        if _visibility_node_has_direct_anim_child(node):
            return False
        ancestor = _nearest_visibility_ancestor_with_direct_anim_child(node)
        return ancestor is not None and getattr(node, "parent", None) is ancestor
    except Exception:
        return False


def _is_root_visibility_pair_child(node):
    try:
        tf = getattr(node, "transform", None)
        if (
            tf is None
            or not isinstance(tf, AnimatingNode)
            or isinstance(tf, ArgVisibilityNode)
        ):
            return False
        if not _is_authored_argvis_control_pair(node):
            return False
        if not _has_only_visibility_ancestors(node):
            return False
        return _nearest_visibility_ancestor_with_direct_anim_child(node) is not None
    except Exception:
        return False


def _is_child_of_file_root(node):
    parent = getattr(node, "parent", None)
    if parent is None:
        # In raw EDM graph, a top-level node has parent=None.
        # Exclude the graph root, which has a generic name or children without a base.
        if hasattr(node, "transform") and not hasattr(node, "base"):
            return False
        return True
    if getattr(parent, "parent", None) is None:
        return True
    try:
        from ..edm_format.types import ArgVisibilityNode

        # Walk up through any number of consecutive ArgVisibilityNode ancestors.
        # A node is "child of file root" if every ancestor between it and the
        # root are ArgVisibilityNode wrappers. The tree root is the Node whose
        # parent is None; a direct child of root has parent.parent == None.
        current = parent
        while current is not None:
            current_tf = getattr(current, "transform", None)
            is_argvis = isinstance(current, ArgVisibilityNode) or isinstance(
                current_tf, ArgVisibilityNode
            )
            if not is_argvis:
                break
            current_parent = getattr(current, "parent", None)
            if current_parent is None:
                return True  # current itself is root (unlikely for ArgVis)
            if getattr(current_parent, "parent", None) is None:
                return True  # current_parent is the tree root node
            current = current_parent
    except Exception:  # noqa: S110 - logging must not recurse when its own writer fails
        pass
    return False
