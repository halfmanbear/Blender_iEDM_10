# Fragment: core node processing — creates Blender objects from the EDM graph.


from ...edm_format.types import (
    AnimatingNode,
    ArgVisibilityNode,
    Connector,
    SkinNode,
)
from ..node_transform import apply_node_transform
from ..prelude import (
    _ROOT_BASIS_FIX,
    _import_ctx,
    _import_profile_flag,
    _ob_local_is_identity,
)
from .node_helpers import (
    _is_narrow_safe_identity_helper_name,
    _wrap_skin_object_with_skin_box,
)


def _apply_render_positioning(node):
    """Apply local matrix for render-only nodes; tag identity passthrough helpers.

    Returns _used_shared_parent_fallback (bool) for use in the animation hookup.
    """
    _used_shared_parent_fallback = False

    _is_render_only_positioning = (
        node.render
        and node.blender
        and node.blender.type == "MESH"
        and (not node.transform or isinstance(node.transform, ArgVisibilityNode))
    )

    if (
        isinstance(getattr(node, "render", None), Connector)
        and getattr(node, "transform", None) is None
        and getattr(node, "blender", None) is not None
    ):
        _apply_connector_position(node)

    if _is_render_only_positioning:
        shared_parent = getattr(node.render, "shared_parent", None)
        parent_transform = getattr(getattr(node, "parent", None), "transform", None)
        parent_is_real_transform = parent_transform is not None and not isinstance(
            parent_transform, ArgVisibilityNode
        )
        has_parent_obj = bool(node.parent and node.parent.blender)
        (
            getattr(shared_parent, "_blender_obj", None)
            if shared_parent is not None
            else None
        )

        if (
            shared_parent is not None
            and not has_parent_obj
            and not parent_is_real_transform
        ):
            apply_node_transform(shared_parent, node.blender, used_shared_parent=True)
            _used_shared_parent_fallback = True

        local_bl = getattr(node, "_local_bl", None)
        if local_bl is not None and not _used_shared_parent_fallback:
            _apply_render_local_matrix(node, local_bl)

        if (
            _import_ctx.mesh_origin_mode == "APPROX"
            and node.parent
            and node.parent.blender
            and not isinstance(
                node.parent.transform, (ArgVisibilityNode, AnimatingNode)
            )
            and not isinstance(getattr(node, "render", None), SkinNode)
        ):
            if node.blender.location.length < 1e-9:
                # Later identity/basis rewrites discard an early compensating offset.
                # Defer editing-origin changes until the authored graph has settled.
                node._recenter_render_origin = True

        if isinstance(getattr(node, "render", None), SkinNode):
            _wrap_skin_object_with_skin_box(node.blender, node.render)

    _tag_render_identity_passthrough(node)
    _tag_combined_fake_light_helper(node)

    return _used_shared_parent_fallback


def _apply_connector_position(node):
    try:
        apply_node_transform(node, node.blender, used_shared_parent=False)
    except Exception as exc:
        print(f"Warning in blender_importer/nodes/core.py: {exc}")


def _apply_render_local_matrix(node, local_matrix):
    try:
        has_render_local = hasattr(node.render, "matrix") or hasattr(node.render, "pos")
        if has_render_local:
            apply_node_transform(node, node.blender, used_shared_parent=False)
            return
        matrix_to_apply = (
            local_matrix.copy() if hasattr(local_matrix, "copy") else local_matrix
        )
        needs_root_fix = _needs_render_root_basis_fix(node)
        has_skin_override = isinstance(
            getattr(node, "render", None), SkinNode
        ) and bool(node.blender.get("_iedm_skin_parent_override"))
        if not needs_root_fix and isinstance(getattr(node, "render", None), SkinNode):
            context = getattr(_import_ctx, "bone_import_ctx", None) or {}
            needs_root_fix = (
                bool(context.get("arm_carries_basis_fix")) and not has_skin_override
            )
        if needs_root_fix and not has_skin_override:
            matrix_to_apply = _ROOT_BASIS_FIX @ matrix_to_apply
        node.blender.matrix_basis = matrix_to_apply
    except Exception as exc:
        print(f"Warning in blender_importer/nodes/core.py: {exc}")


def _needs_render_root_basis_fix(node):
    parent = getattr(node, "parent", None)
    if not (
        _import_profile_flag("plain_root_render_local_basis_fix")
        and _import_ctx.edm_version >= 10
        and node.transform is None
        and not getattr(_import_ctx, "use_scene_root_basis_object", True)
        and str(getattr(node.render, "name", "") or "")
    ):
        return False
    if getattr(parent, "_is_graph_root", False):
        return True
    is_visibility_root = isinstance(
        getattr(parent, "transform", None), ArgVisibilityNode
    ) and getattr(getattr(parent, "parent", None), "_is_graph_root", False)
    has_animated_child = any(
        isinstance(getattr(child, "transform", None), AnimatingNode)
        and not isinstance(getattr(child, "transform", None), ArgVisibilityNode)
        for child in list(getattr(parent, "children", []) or [])
    )
    return is_visibility_root and has_animated_child


def _tag_render_identity_passthrough(node):
    try:
        if not (node.blender and node.render is not None and node.transform is None):
            return
        obj = node.blender
        name = getattr(obj, "name", "") or ""
        has_dup_suffix = len(name) > 4 and name[-4] == "." and name[-3:].isdigit()
        parent_is_visibility = isinstance(
            getattr(node.parent, "transform", None), ArgVisibilityNode
        )
        fake_light = type(node.render).__name__ in {
            "FakeOmniLightsNode",
            "FakeSpotLightsNode",
        }
        if not (
            getattr(obj, "parent", None) is not None
            and _ob_local_is_identity(obj)
            and (has_dup_suffix or (parent_is_visibility and fake_light))
        ):
            return
        obj["_iedm_identity_passthrough"] = True
        if (has_dup_suffix and _is_narrow_safe_identity_helper_name(name)) or (
            parent_is_visibility and fake_light
        ):
            obj["_iedm_narrow_identity_passthrough"] = True
    except Exception as exc:
        print(f"Warning in blender_importer/nodes/core.py: {exc}")


def _tag_combined_fake_light_helper(node):
    try:
        if not (
            node.blender and node.render is not None and node.transform is not None
        ):
            return
        parent_is_visibility = isinstance(
            getattr(node.parent, "transform", None), ArgVisibilityNode
        )
        fake_light = type(node.render).__name__ in {
            "FakeOmniLightsNode",
            "FakeSpotLightsNode",
        }
        if (
            parent_is_visibility
            and fake_light
            and not isinstance(node.transform, AnimatingNode)
            and _ob_local_is_identity(node.blender)
        ):
            node.blender["_iedm_identity_passthrough"] = True
            node.blender["_iedm_narrow_identity_passthrough"] = True
    except Exception as exc:
        print(f"Warning in blender_importer/nodes/core.py: {exc}")
