from ...edm_format.mathtypes import Matrix
from ...edm_format.types import (
    AnimatingNode,
    ArgVisibilityNode,
    TransformNode,
)
from .mesh import _is_identity_matrix_approx


def _compact_visibility_identity_intermediate(node):
    """Hoist fake-light helper transforms toward the semantic node under a v_* wrapper.

    Handles both patterns:
      v_* -> identity helper -> fake-light
      v_* -> semantic(identity) -> transformed helper -> fake-light
    """
    if node is None or not getattr(node, "blender", None):
        return
    _renderless_helper_mode = getattr(node, "render", None) is None and isinstance(
        getattr(node, "transform", None), TransformNode
    )
    if getattr(node, "render", None) is None and not _renderless_helper_mode:
        return

    fake_obj = node.blender

    def _trace(msg):
        return

    if _renderless_helper_mode:
        helper_obj = node.blender
        fake_obj = None
        _trace("renderless-helper mode")
    else:
        helper_obj = getattr(fake_obj, "parent", None)
        if helper_obj is None or getattr(helper_obj, "type", None) != "EMPTY":
            _trace(
                "skip helper_obj parent missing/non-empty type={}".format(
                    getattr(helper_obj, "type", None) if helper_obj else None
                )
            )
            return
    _trace(
        "objs helper={} helper_parent={}".format(
            getattr(helper_obj, "name", None),
            getattr(getattr(helper_obj, "parent", None), "name", None),
        )
    )
    # Graph chain from helper transform node (either current renderless node, or parent of render node).
    helper_graph = node if _renderless_helper_mode else getattr(node, "parent", None)
    if helper_graph is None:
        _trace("skip helper_graph missing")
        return
    if not isinstance(
        getattr(helper_graph, "transform", None), (TransformNode, AnimatingNode)
    ):
        _trace(
            "skip helper_graph type={}".format(
                type(getattr(helper_graph, "transform", None)).__name__
            )
        )
        return
    _trace(
        "graph helper_tf={} parent_tf={}".format(
            type(getattr(helper_graph, "transform", None)).__name__,
            type(
                getattr(getattr(helper_graph, "parent", None), "transform", None)
            ).__name__
            if getattr(helper_graph, "parent", None)
            else None,
        )
    )

    def _has_object_animation(obj):
        ad = getattr(obj, "animation_data", None)
        if ad is None:
            return False
        if getattr(ad, "action", None) is not None:
            return True
        try:
            return len(getattr(ad, "nla_tracks", [])) > 0
        except Exception:
            return False

    def _get_local(obj):
        try:
            return obj.matrix_basis.copy()
        except Exception:
            try:
                return obj.matrix_local.copy()
            except Exception:
                return None

    def _set_local(obj, mat):
        try:
            obj.matrix_basis = mat
            return True
        except Exception:
            try:
                loc, rot, scale = mat.decompose()
                obj.location = loc
                obj.rotation_mode = "XYZ"
                obj.rotation_euler = rot.to_euler("XYZ")
                obj.scale = scale
                return True
            except Exception:
                return False

    def _collapse_redundant_helper_empty(
        helper_obj, semantic_obj, preferred_child=None
    ):
        """Remove a static identity helper EMPTY after its transform was hoisted."""
        # Only collapse in the render-node path. If we delete the current node's
        # object during renderless-helper processing, later debug/output paths still
        # in process_node can touch a dead reference.
        if _renderless_helper_mode:
            return False
        if helper_obj is None or semantic_obj is None:
            return False
        if (
            getattr(helper_obj, "type", None) != "EMPTY"
            or getattr(semantic_obj, "type", None) != "EMPTY"
        ):
            return False
        if _has_object_animation(helper_obj):
            return False
        helper_name = str(getattr(helper_obj, "name", "") or "")
        helper_local = _get_local(helper_obj)
        if helper_local is None or not _is_identity_matrix_approx(helper_local):
            return False
        children = list(getattr(helper_obj, "children", []) or [])
        if len(children) != 1:
            _trace(
                "skip collapse helper '{}' child_count={}".format(
                    helper_name, len(children)
                )
            )
            return False
        child = children[0]
        if preferred_child is not None and child is not preferred_child:
            return False
        # This helper is still owned by a translation-graph node and its source
        # transform. Retain it as an identity passthrough: deleting it here leaves
        # dangling RNA references for later graph passes (notably on C130J lights).
        helper_obj["_iedm_identity_passthrough"] = True
        return False

    # Case 1: v_* -> identity helper -> fake-light (legacy helper compaction)
    helper_parent_graph = getattr(helper_graph, "parent", None)
    if isinstance(getattr(helper_parent_graph, "transform", None), ArgVisibilityNode):
        vis_name = str(
            getattr(getattr(helper_parent_graph, "transform", None), "name", "") or ""
        )
        if not vis_name.startswith(
            ("Cylinder", "Fspot", "Omni", "Omni_l", "Box", "ChamferBox", "Object")
        ):
            _trace("skip case1 vis_name={!r}".format(vis_name))
            return
        semantic_obj = getattr(helper_obj, "parent", None)
        helper_local = _get_local(helper_obj)
        if (
            semantic_obj is not None
            and getattr(semantic_obj, "type", None) == "EMPTY"
            and not _has_object_animation(helper_obj)
            and helper_local is not None
            and not _is_identity_matrix_approx(helper_local)
            and _is_identity_matrix_approx(semantic_obj.matrix_basis)
            # Hoisting moves every sibling. Blender children do not exist yet at
            # this point, so the graph decides whether the helper is alone.
            and len(getattr(helper_parent_graph, "children", []) or []) == 1
        ):
            if _set_local(semantic_obj, helper_local):
                _trace("case1 moved helper_local to semantic")
                _set_local(helper_obj, Matrix.Identity(4))
                _collapse_redundant_helper_empty(
                    helper_obj, semantic_obj, preferred_child=fake_obj
                )
                return
        if not _has_object_animation(helper_obj) and _is_identity_matrix_approx(
            helper_obj.matrix_basis
        ):
            fake_local = _get_local(fake_obj) if fake_obj is not None else None
            if fake_local is not None and not _is_identity_matrix_approx(fake_local):
                if _set_local(helper_obj, fake_local):
                    _trace("case1 moved fake_local to helper")
                    _set_local(fake_obj, Matrix.Identity(4))
            else:
                _trace("case1 fake_local none/identity")
        else:
            _trace("case1 helper animated or non-identity helper_basis")
        return

    # Case 2: v_* -> semantic(identity) -> transformed helper -> fake-light
    semantic_graph = helper_parent_graph
    if semantic_graph is None or not isinstance(
        getattr(semantic_graph, "transform", None), (TransformNode, AnimatingNode)
    ):
        _trace(
            "skip case2 semantic_graph tf={}".format(
                type(getattr(semantic_graph, "transform", None)).__name__
                if semantic_graph
                else None
            )
        )
        return
    vis_graph = getattr(semantic_graph, "parent", None)
    if vis_graph is None or not isinstance(
        getattr(vis_graph, "transform", None), ArgVisibilityNode
    ):
        _trace(
            "skip case2 vis_graph tf={}".format(
                type(getattr(vis_graph, "transform", None)).__name__
                if vis_graph
                else None
            )
        )
        return
    vis_name = str(getattr(getattr(vis_graph, "transform", None), "name", "") or "")
    if not vis_name.startswith(
        ("Cylinder", "Fspot", "Omni", "Omni_l", "Box", "ChamferBox", "Object")
    ):
        _trace("skip case2 vis_name={!r}".format(vis_name))
        return

    semantic_obj = getattr(helper_obj, "parent", None)
    if semantic_obj is None or getattr(semantic_obj, "type", None) != "EMPTY":
        _trace(
            "skip case2 semantic_obj missing/non-empty type={}".format(
                getattr(semantic_obj, "type", None) if semantic_obj else None
            )
        )
        return
    if _has_object_animation(helper_obj):
        _trace("skip case2 helper animated")
        return

    helper_local = _get_local(helper_obj)
    if helper_local is None or _is_identity_matrix_approx(helper_local):
        _trace("skip case2 helper_local none/identity")
        return
    if not _is_identity_matrix_approx(semantic_obj.matrix_basis):
        _trace("skip case2 semantic basis non-identity")
        return
    # Hoisting moves every sibling. Blender children do not exist yet at this
    # point, so the graph decides whether the helper is alone.
    if len(getattr(semantic_graph, "children", []) or []) != 1:
        _trace("skip case2 semantic has siblings")
        return

    if _set_local(semantic_obj, helper_local):
        _trace("case2 moved helper_local to semantic")
        _set_local(helper_obj, Matrix.Identity(4))
        _collapse_redundant_helper_empty(
            helper_obj, semantic_obj, preferred_child=fake_obj
        )
