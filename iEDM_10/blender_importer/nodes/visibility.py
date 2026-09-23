from ...edm_format.mathtypes import Matrix
from ...edm_format.types import (
    AnimatingNode,
    ArgVisibilityNode,
    TransformNode,
)
from .mesh import _is_identity_matrix_approx


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
    helper_obj, semantic_obj, preferred_child, renderless_mode
):
    """Mark a static identity helper as a passthrough when it can be collapsed."""
    if renderless_mode or helper_obj is None or semantic_obj is None:
        return False
    if (
        getattr(helper_obj, "type", None) != "EMPTY"
        or getattr(semantic_obj, "type", None) != "EMPTY"
    ):
        return False
    if _has_object_animation(helper_obj):
        return False
    helper_local = _get_local(helper_obj)
    if helper_local is None or not _is_identity_matrix_approx(helper_local):
        return False
    children = list(getattr(helper_obj, "children", []) or [])
    if len(children) != 1 or (
        preferred_child is not None and children[0] is not preferred_child
    ):
        return False
    helper_obj["_iedm_identity_passthrough"] = True
    return False


def _compact_visibility_identity_intermediate(node):
    """Hoist fake-light helper transforms toward the semantic node under a v_* wrapper.

    Handles both patterns:
      v_* -> identity helper -> fake-light
      v_* -> semantic(identity) -> transformed helper -> fake-light
    """
    if node is None or not getattr(node, "blender", None):
        return
    renderless = getattr(node, "render", None) is None and isinstance(
        getattr(node, "transform", None), TransformNode
    )
    if getattr(node, "render", None) is None and not renderless:
        return
    fake_obj = None if renderless else node.blender
    helper_obj = node.blender if renderless else getattr(fake_obj, "parent", None)
    if not renderless and (
        helper_obj is None or getattr(helper_obj, "type", None) != "EMPTY"
    ):
        return
    helper_graph = node if renderless else getattr(node, "parent", None)
    if helper_graph is None or not isinstance(
        getattr(helper_graph, "transform", None), (TransformNode, AnimatingNode)
    ):
        return
    parent_graph = getattr(helper_graph, "parent", None)
    if isinstance(getattr(parent_graph, "transform", None), ArgVisibilityNode):
        _compact_legacy_visibility_helper(
            helper_obj, fake_obj, renderless, helper_graph, parent_graph
        )
    else:
        _compact_nested_visibility_helper(
            helper_obj, fake_obj, renderless, parent_graph
        )


def _compact_legacy_visibility_helper(
    helper_obj, fake_obj, renderless, helper_graph, parent_graph
):
    """Handle v_* -> identity helper -> fake-light chains."""
    visibility_name = str(getattr(parent_graph.transform, "name", "") or "")
    if not _is_fake_light_visibility_name(visibility_name):
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
        and len(getattr(parent_graph, "children", []) or []) == 1
        and _set_local(semantic_obj, helper_local)
    ):
        _set_local(helper_obj, Matrix.Identity(4))
        _collapse_redundant_helper_empty(helper_obj, semantic_obj, fake_obj, renderless)
        return
    if _has_object_animation(helper_obj) or not _is_identity_matrix_approx(
        helper_obj.matrix_basis
    ):
        return
    fake_local = _get_local(fake_obj) if fake_obj is not None else None
    if fake_local is not None and not _is_identity_matrix_approx(fake_local):
        if _set_local(helper_obj, fake_local):
            _set_local(fake_obj, Matrix.Identity(4))


def _compact_nested_visibility_helper(helper_obj, fake_obj, renderless, semantic_graph):
    """Handle v_* -> semantic -> transformed helper -> fake-light chains."""
    if semantic_graph is None or not isinstance(
        getattr(semantic_graph, "transform", None), (TransformNode, AnimatingNode)
    ):
        return
    vis_graph = getattr(semantic_graph, "parent", None)
    if vis_graph is None or not isinstance(
        getattr(vis_graph, "transform", None), ArgVisibilityNode
    ):
        return
    if not _is_fake_light_visibility_name(getattr(vis_graph.transform, "name", "")):
        return
    semantic_obj = getattr(helper_obj, "parent", None)
    if semantic_obj is None or getattr(semantic_obj, "type", None) != "EMPTY":
        return
    if _has_object_animation(helper_obj):
        return
    helper_local = _get_local(helper_obj)
    if helper_local is None or _is_identity_matrix_approx(helper_local):
        return
    if not _is_identity_matrix_approx(semantic_obj.matrix_basis):
        return
    if len(getattr(semantic_graph, "children", []) or []) != 1:
        return
    if _set_local(semantic_obj, helper_local):
        _set_local(helper_obj, Matrix.Identity(4))
        _collapse_redundant_helper_empty(helper_obj, semantic_obj, fake_obj, renderless)


def _is_fake_light_visibility_name(name):
    return str(name or "").startswith(
        ("Cylinder", "Fspot", "Omni", "Omni_l", "Box", "ChamferBox", "Object")
    )
