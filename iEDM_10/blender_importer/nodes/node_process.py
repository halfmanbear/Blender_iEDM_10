# Fragment: core node processing — creates Blender objects from the EDM graph.


from ...edm_format.types import (
    ArgVisibilityNode,
    LodNode,
    SkinNode,
)
from ..anim_actions import get_actions_for_node
from ..graph_pipeline import (
    _debug_dump_node_transform,
)
from ..prelude import (
    _debug_log_event,
    _import_ctx,
)
from .armature import (
    _bind_skin_object,
)
from .node_animation import _hookup_node_animations
from .node_create import _create_node_object
from .node_helpers import _idprop_diag_value
from .node_parent import _parent_node_object
from .node_position import _apply_render_positioning
from .node_properties import _stamp_node_properties
from .visibility import _compact_visibility_identity_intermediate


def _dump_node_diagnostics(node):
    """Dump all numeric/vector EDM node attributes to Blender IDprops."""
    if not node.blender:
        return

    def _dump_diag(target, prefix="EDM_RAW_"):
        for attr in dir(target):
            if attr.startswith("_") or attr in ("blender", "children", "parent"):
                continue
            try:
                val = getattr(target, attr)
            except Exception:
                continue
            idprop_val = _idprop_diag_value(val)
            if idprop_val is None:
                continue
            try:
                node.blender[prefix + attr] = idprop_val
            except Exception:
                pass

    if node.transform:
        _dump_diag(node.transform, "EDM_TF_")
        if hasattr(node.transform, "base"):
            _dump_diag(node.transform.base, "EDM_BASE_")
    if node.render:
        _dump_diag(node.render, "EDM_RN_")

    if hasattr(node, "_local_bl"):
        node.blender["IEDM_LOCAL_BL_MAT"] = [
            float(v) for row in node._local_bl for v in row
        ]


def process_node(node):
    """Processes a single node of the transform graph."""
    if node.parent is None:
        return

    node._is_primary = False

    # ArgVisibilityNode wrappers always keep their own object: the official
    # exporter reconstructs the visibility node from that object's action.
    if node.render is None and isinstance(node.transform, ArgVisibilityNode):
        tf = getattr(node, "transform", None)
        child_count = len(getattr(node, "children", []) or [])
        _debug_log_event(
            "[iEDM][VISDBG] keep-own-object idx={} name={!r} children={}".format(
                getattr(tf, "_graph_idx", None),
                getattr(tf, "name", "") if tf is not None else "",
                child_count,
            )
        )

    # EDM bone-control chains map to a single armature object
    ctx = _import_ctx.bone_import_ctx or {}
    arm_obj = ctx.get("armature")
    bone_chain_nodes = ctx.get("bone_chain_nodes", set())
    if arm_obj is not None and node.render is None and node in bone_chain_nodes:
        node.blender = arm_obj
        return

    is_skel = getattr(node, "_is_skeleton", False)

    _create_node_object(node, ctx, is_skel)

    if node.blender and isinstance(node.render, SkinNode):
        _bind_skin_object(node.blender, node.render)

    if not node.blender:
        node.blender = node.parent.blender if node.parent else None
        return

    node._is_primary = True

    _parent_node_object(node, ctx)
    _stamp_node_properties(node, ctx)

    _used_shared_parent_fallback = _apply_render_positioning(node)

    # Visibility source: own transform or collapsed parent wrapper
    _vis_source = None
    if isinstance(node.transform, ArgVisibilityNode):
        _vis_source = node.transform
    elif (
        node.parent
        and isinstance(node.parent.transform, ArgVisibilityNode)
        and node.parent.blender == node.blender
    ):
        _vis_source = node.parent.transform

    vis_actions = get_actions_for_node(_vis_source) if _vis_source else []

    # Material assignment
    material_target = node.render_blender if node.render_blender else node.blender
    if (
        hasattr(node.render, "material")
        and node.render.material
        and hasattr(node.render.material, "blender_material")
        and node.render.material.blender_material
        and material_target
        and hasattr(material_target, "data")
        and material_target.data
    ):
        render_cls_name = type(node.render).__name__
        if render_cls_name not in {
            "FakeOmniLightsNode",
            "FakeSpotLightsNode",
            "FakeALSNode",
        }:
            material_target.data.materials.append(node.render.material.blender_material)

    _hookup_node_animations(node, ctx, vis_actions, _used_shared_parent_fallback)

    _compact_visibility_identity_intermediate(node)

    _dump_node_diagnostics(node)

    _debug_dump_node_transform(node)

    if isinstance(node.transform, LodNode):
        node._lod_post_children = True


def _process_lod_post_children(node):
    """Apply LodNode properties after children have been processed."""
    if not getattr(node, "_lod_post_children", False):
        return
    assert node.blender.type == "EMPTY"
    node.blender.edm.is_lod_root = True
    try:
        node.blender["IEDM_LOD_LEVELS"] = [
            [float(start), float(end)]
            for start, end in getattr(node.transform, "level", [])
        ]
    except Exception:
        pass
    for (start, end), child in zip(node.transform.level, node.children):
        child.blender.edm.lod_min_distance = start
        child.blender.edm.lod_max_distance = end
        child.blender.edm.nouse_lod_distance = end > 1e6


def _apply_shadeless(mat):
    """Make a material shadeless by routing Base Color to Emission Color."""
    if not mat.use_nodes:
        return
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = None
    for node in nodes:
        if node.type == "BSDF_PRINCIPLED":
            principled = node
            break
    if not principled:
        return
    base_color_input = principled.inputs["Base Color"]
    if base_color_input.links:
        source_socket = base_color_input.links[0].from_socket
        links.new(source_socket, principled.inputs["Emission Color"])
        principled.inputs["Emission Strength"].default_value = 1.0
