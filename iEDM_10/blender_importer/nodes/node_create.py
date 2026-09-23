# Fragment: core node processing â€” creates Blender objects from the EDM graph.


import bpy

from ...edm_format.mathtypes import (
    Matrix,
)
from ...edm_format.types import (
    AnimatingNode,
    ArgAnimationNode,
    ArgVisibilityNode,
    BillboardNode,
    Connector,
    FakeALSNode,
    FakeOmniLightsNode,
    FakeSpotLightsNode,
    LightNode,
    NumberNode,
    RenderNode,
    SegmentsNode,
    ShellNode,
    SkinNode,
    TransformNode,
)
from ..graph_pipeline import (
    _SEMANTIC_NAME_MAP,
    _is_neg90_x_basis_matrix,
)
from ..lights import (
    create_billboard,
    create_fake_als_lights,
    create_fake_omni_lights,
    create_fake_spot_lights,
    create_lamp,
)
from ..object_create import (
    create_connector,
    create_object,
    create_segments,
)
from ..prelude import (
    _ROOT_BASIS_FIX,
    _SUFFIX_RE,
    _import_ctx,
    _is_connector_object,
    _is_generic_render_name,
    _strip_anim_prefix,
    _transform_display_name,
)
from .node_helpers import _preferred_control_wrapper_name


def _is_generic_shell_name(name):
    if _is_generic_render_name(name):
        return True
    if not name:
        return True
    return name.startswith("Node") and name[4:].isdigit()


def _nearest_shell_alias(node):
    cur = getattr(node, "parent", None)
    while cur is not None:
        tf = getattr(cur, "transform", None)
        if tf is None:
            cur = getattr(cur, "parent", None)
            continue
        candidate = _transform_display_name(tf)
        if isinstance(tf, ArgVisibilityNode) and candidate.startswith("v_"):
            candidate = candidate[2:]
        candidate = _strip_anim_prefix(candidate)
        if candidate.startswith("Empty_"):
            candidate = candidate[len("Empty_") :]
        if (
            candidate
            and candidate.lower() != "root"
            and not _is_generic_render_name(candidate)
        ):
            return candidate
        cur = getattr(cur, "parent", None)
    ridx = getattr(node.render, "_graph_render_idx", None)
    if ridx is not None:
        return "shell_{:04d}".format(ridx)
    return "shell"


def _nearest_collision_alias(node, suffix):
    base = _nearest_shell_alias(node)
    if base.endswith("_shell") or base.endswith("_segments"):
        return base
    return "{}_{}".format(base, suffix)


def _name_render_node(node):
    render_name = getattr(node.render, "name", "") or ""
    name_unknown = getattr(node.render, "name_unknown", False)
    if isinstance(node.render, NumberNode) and render_name.endswith("_Number"):
        name_unknown = True
    is_connector_render = isinstance(node.render, Connector)
    is_shell_render = isinstance(node.render, ShellNode)
    is_segments_render = isinstance(node.render, SegmentsNode)

    final_name = render_name
    parent_is_arg_wrapper = isinstance(
        getattr(getattr(node, "parent", None), "transform", None),
        (ArgVisibilityNode, ArgAnimationNode),
    )

    if _needs_render_name_resolution(
        final_name, name_unknown, is_shell_render, is_segments_render
    ):
        final_name = _resolve_render_name(
            node,
            final_name,
            name_unknown,
            is_shell_render,
            is_segments_render,
            is_connector_render,
        )

    final_name = _apply_semantic_render_name(
        node, final_name, name_unknown, parent_is_arg_wrapper
    )

    if final_name:
        node.render.name = final_name


def _needs_render_name_resolution(name, unknown, shell, segments):
    return (
        (shell and _is_generic_shell_name(name))
        or (segments and _is_generic_shell_name(name))
        or _is_generic_render_name(name)
        or unknown
    )


def _resolve_render_name(node, name, unknown, shell, segments, connector):
    if shell and _is_generic_shell_name(name):
        name = "{}_shell".format(_nearest_shell_alias(node))
    elif segments and _is_generic_shell_name(name):
        name = _nearest_collision_alias(node, "segments")
    if connector:
        alias = _connector_parent_alias(node)
        return "Connector_{}".format(alias) if alias else "Connector"
    if (_is_generic_render_name(name) or unknown) and isinstance(
        node.transform, ArgVisibilityNode
    ):
        alias = _visibility_render_alias(node.transform)
        if alias:
            name = alias
    if (
        (_is_generic_render_name(name) or unknown)
        and node.parent
        and node.parent.transform
    ):
        alias = _parent_render_alias(node.parent.transform)
        if alias:
            name = alias
    return name


def _connector_parent_alias(node):
    if not node.parent or not node.parent.transform:
        return ""
    alias = _transform_display_name(node.parent.transform)
    if alias.startswith("v_"):
        alias = alias[2:]
    alias = _strip_anim_prefix(alias)
    if alias.startswith("Empty_"):
        alias = alias[len("Empty_") :]
    return "" if alias.lower() == "root" else alias


def _visibility_render_alias(transform):
    alias = getattr(transform, "name", "") or ""
    if alias.startswith("v_"):
        alias = alias[2:]
    if alias.startswith("Empty_"):
        alias = alias[len("Empty_") :]
    return (
        alias
        if alias and alias.lower() != "root" and not _is_generic_render_name(alias)
        else ""
    )


def _parent_render_alias(transform):
    alias = _transform_display_name(transform)
    if isinstance(transform, ArgVisibilityNode) and alias.startswith("v_"):
        alias = alias[2:]
    if alias.startswith("Empty_"):
        alias = alias[len("Empty_") :]
    return (
        alias
        if alias and alias.lower() != "root" and not _is_generic_render_name(alias)
        else ""
    )


def _apply_semantic_render_name(node, name, unknown, parent_is_arg_wrapper):
    if not parent_is_arg_wrapper and (_is_generic_render_name(name) or unknown):
        material = getattr(node.render, "material", None)
        material_name = getattr(material, "name", "") or ""
        if material_name:
            name = _SEMANTIC_NAME_MAP.get(material_name, material_name)
    if not parent_is_arg_wrapper:
        name = _SEMANTIC_NAME_MAP.get(name, name)
        if _is_generic_render_name(name):
            render_index = getattr(node.render, "_graph_render_idx", None)
            if render_index is not None:
                name = "rn_{:04d}".format(render_index)
    return name


def _prepare_connector_metadata(node):
    _set_connector_render_name(node)
    try:
        node.blender["_iedm_connector_apply_xfix"] = False
        _record_connector_parent_transform(node)
        _record_connector_own_transform(node)
        _record_connector_fallback_transform(node)
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")


def _set_connector_render_name(node):
    try:
        _connector_edm_name = str(getattr(node.render, "name", "") or "")
        if _connector_edm_name and node.blender.name != _connector_edm_name:
            _conflict = bpy.data.objects.get(_connector_edm_name)
            if _conflict is not None and _conflict is not node.blender:
                if not _is_connector_object(_conflict):
                    _conflict.name = _connector_edm_name + "_tf"
                    node.blender.name = _connector_edm_name
            else:
                node.blender.name = _connector_edm_name
        if _connector_edm_name:
            node.blender["_iedm_connector_name"] = _connector_edm_name
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")


def _record_connector_parent_transform(node):
    """Preserve static parent-transform details used by connector export."""
    parent_tf = getattr(getattr(node, "parent", None), "transform", None)
    if not (
        node.transform is None
        and isinstance(parent_tf, TransformNode)
        and not isinstance(parent_tf, AnimatingNode)
    ):
        return
    name = getattr(parent_tf, "name", "") or ""
    matrix = _connector_transform_matrix(node, parent_tf, name)
    if name == "Connector Transform":
        try:
            node.blender["_iedm_connector_apply_xfix"] = (
                matrix is None or _is_neg90_x_basis_matrix(matrix)
            )
        except Exception:
            node.blender["_iedm_connector_apply_xfix"] = True
    elif name:
        node.blender["_iedm_connector_tf_name"] = name


def _record_connector_own_transform(node):
    """Preserve a connector node's own static transform metadata."""
    transform = node.transform
    if transform is None or isinstance(transform, AnimatingNode):
        return
    name = getattr(transform, "name", "") or ""
    matrix = _connector_transform_matrix(node, transform, name)
    if name == "Connector Transform":
        try:
            node.blender["_iedm_connector_apply_xfix"] = (
                matrix is None or _is_neg90_x_basis_matrix(matrix)
            )
        except Exception:
            node.blender["_iedm_connector_apply_xfix"] = True
    elif name:
        node.blender["_iedm_connector_tf_name"] = name


def _connector_transform_matrix(node, transform, name):
    if not hasattr(transform, "matrix"):
        return None
    try:
        matrix = Matrix(transform.matrix)
        flat = [float(value) for row in matrix for value in row]
        parent_obj = getattr(getattr(node, "parent", None), "blender", None)
        if transform is getattr(getattr(node, "parent", None), "transform", None):
            if parent_obj is not None:
                parent_obj["_iedm_raw_transform_matrix"] = flat
                if name:
                    parent_obj["_iedm_raw_transform_name"] = name
        node.blender["_iedm_connector_tf_matrix"] = flat
        return matrix
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
        return None


def _record_connector_fallback_transform(node):
    if "_iedm_connector_tf_matrix" not in node.blender:
        local_matrix = getattr(node, "_local_bl", None)
        if local_matrix is not None:
            try:
                matrix = Matrix(local_matrix)
                node.blender["_iedm_connector_tf_matrix"] = [
                    float(value) for row in matrix for value in row
                ]
            except Exception as e:
                print(f"Warning in blender_importer/nodes/core.py: {e}")
    if "_iedm_connector_tf_name" not in node.blender:
        name = str(getattr(getattr(node, "render", None), "name", "") or "")
        if name:
            node.blender["_iedm_connector_tf_name"] = _SUFFIX_RE.sub("", name)


def _create_node_object(node, ctx, is_skel):
    """Create the Blender object for a graph node and store it in node.blender."""
    node.render_blender = None
    if node.render:
        _name_render_node(node)
        if isinstance(node.render, Connector):
            node.blender = create_connector(node.render)
            _prepare_connector_metadata(node)
        elif isinstance(node.render, (RenderNode, ShellNode, NumberNode, SkinNode)):
            node.blender = create_object(node.render)
        elif isinstance(node.render, SegmentsNode):
            node.blender = create_segments(node.render)
            if (
                node.blender is not None
                and getattr(_import_ctx, "collision_geometry_basis_fix", False)
                and getattr(node.blender, "data", None) is not None
            ):
                try:
                    node.blender.data.transform(_ROOT_BASIS_FIX)
                    node.blender.data.update()
                except Exception as e:
                    print(f"Warning in blender_importer/nodes/core.py: {e}")
        elif isinstance(node.render, LightNode):
            node.blender = create_lamp(node.render)
        elif (
            isinstance(node.render, BillboardNode)
            or type(node.render).__name__ == "BillboardNode"
        ):
            node.blender = create_billboard(node.render)
        elif (
            isinstance(node.render, FakeOmniLightsNode)
            or type(node.render).__name__ == "FakeOmniLightsNode"
        ):
            node.blender = create_fake_omni_lights(node.render)
        elif (
            isinstance(node.render, FakeSpotLightsNode)
            or type(node.render).__name__ == "FakeSpotLightsNode"
        ):
            node.blender = create_fake_spot_lights(node.render)
        elif (
            isinstance(node.render, FakeALSNode)
            or type(node.render).__name__ == "FakeALSNode"
        ):
            node.blender = create_fake_als_lights(node.render)
        else:
            print("Warning: No case yet for object node {}".format(node.render))

    elif is_skel:
        empty_name = (
            _transform_display_name(node.transform) or type(node.transform).__name__
        )
        _node_children_skel = getattr(node, "children", []) or []
        if (
            len(_node_children_skel) == 1
            and isinstance(getattr(_node_children_skel[0], "render", None), Connector)
            and getattr(_node_children_skel[0], "transform", None) is None
        ):
            empty_name = empty_name + "_tf"
        ob = bpy.data.objects.new(empty_name, None)
        ob.empty_display_size = 0.1
        bpy.context.collection.objects.link(ob)
        node.blender = ob
    else:
        tf_name = (
            _transform_display_name(node.transform) or type(node.transform).__name__
        )
        tf_name = _preferred_control_wrapper_name(node, tf_name)
        _node_children_ns = getattr(node, "children", []) or []
        if (
            len(_node_children_ns) == 1
            and isinstance(getattr(_node_children_ns[0], "render", None), Connector)
            and getattr(_node_children_ns[0], "transform", None) is None
        ):
            tf_name = tf_name + "_tf"
        ob = bpy.data.objects.new(tf_name, None)
        ob.empty_display_size = 0.1
        bpy.context.collection.objects.link(ob)
        node.blender = ob
