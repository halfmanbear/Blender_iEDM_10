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
        try:
            apply_node_transform(node, node.blender, used_shared_parent=False)
        except Exception as e:
            print(f"Warning in blender_importer/nodes/core.py: {e}")

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
            try:
                has_render_local = hasattr(node.render, "matrix") or hasattr(
                    node.render, "pos"
                )
                if has_render_local:
                    apply_node_transform(node, node.blender, used_shared_parent=False)
                else:
                    matrix_to_apply = (
                        local_bl.copy() if hasattr(local_bl, "copy") else local_bl
                    )
                    _prrbf_base = (
                        _import_profile_flag("plain_root_render_local_basis_fix")
                        and _import_ctx.edm_version >= 10
                        and node.transform is None
                        and not getattr(
                            _import_ctx, "use_scene_root_basis_object", True
                        )
                        and str(getattr(node.render, "name", "") or "")
                    )
                    _parent_node = getattr(node, "parent", None)
                    _needs_render_basis_fix = _prrbf_base and (
                        getattr(_parent_node, "_is_graph_root", False)
                        or (
                            isinstance(
                                getattr(_parent_node, "transform", None),
                                ArgVisibilityNode,
                            )
                            and getattr(
                                getattr(_parent_node, "parent", None),
                                "_is_graph_root",
                                False,
                            )
                            and any(
                                isinstance(
                                    getattr(sib, "transform", None), AnimatingNode
                                )
                                and not isinstance(
                                    getattr(sib, "transform", None), ArgVisibilityNode
                                )
                                for sib in list(
                                    getattr(_parent_node, "children", []) or []
                                )
                            )
                        )
                    )
                    if (
                        not _needs_render_basis_fix
                        and isinstance(getattr(node, "render", None), SkinNode)
                        and not bool(node.blender.get("_iedm_skin_parent_override"))
                    ):
                        _bone_ctx = getattr(_import_ctx, "bone_import_ctx", None) or {}
                        if bool(_bone_ctx.get("arm_carries_basis_fix")):
                            _needs_render_basis_fix = True
                    if _needs_render_basis_fix:
                        if not (
                            isinstance(getattr(node, "render", None), SkinNode)
                            and bool(node.blender.get("_iedm_skin_parent_override"))
                        ):
                            matrix_to_apply = _ROOT_BASIS_FIX @ matrix_to_apply
                    node.blender.matrix_basis = matrix_to_apply
            except Exception as e:
                print(f"Warning in blender_importer/nodes/core.py: {e}")

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

    # Tag identity passthrough for render-only nodes
    try:
        if node.blender and node.render is not None and node.transform is None:
            ob = node.blender
            ob_name = getattr(ob, "name", "") or ""
            has_parent = getattr(ob, "parent", None) is not None
            has_blender_dup_suffix = (
                len(ob_name) > 4 and ob_name[-4] == "." and ob_name[-3:].isdigit()
            )
            parent_is_vis = isinstance(
                getattr(node.parent, "transform", None), ArgVisibilityNode
            )
            render_cls_name = type(node.render).__name__
            is_fake_light_render = render_cls_name in {
                "FakeOmniLightsNode",
                "FakeSpotLightsNode",
            }
            if (
                has_parent
                and _ob_local_is_identity(ob)
                and (has_blender_dup_suffix or (parent_is_vis and is_fake_light_render))
            ):
                ob["_iedm_identity_passthrough"] = True
                if has_blender_dup_suffix and _is_narrow_safe_identity_helper_name(
                    ob_name
                ):
                    ob["_iedm_narrow_identity_passthrough"] = True
                if parent_is_vis and is_fake_light_render:
                    ob["_iedm_narrow_identity_passthrough"] = True
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")

    # Combined transform+render fake-light helpers under visibility wrapper
    try:
        if node.blender and node.render is not None and node.transform is not None:
            ob = node.blender
            parent_is_vis = isinstance(
                getattr(node.parent, "transform", None), ArgVisibilityNode
            )
            is_anim_tf = isinstance(node.transform, AnimatingNode)
            render_cls_name = type(node.render).__name__
            is_fake_light_render = render_cls_name in {
                "FakeOmniLightsNode",
                "FakeSpotLightsNode",
            }
            if (
                parent_is_vis
                and is_fake_light_render
                and (not is_anim_tf)
                and _ob_local_is_identity(ob)
            ):
                ob["_iedm_identity_passthrough"] = True
                ob["_iedm_narrow_identity_passthrough"] = True
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")

    return _used_shared_parent_fallback
