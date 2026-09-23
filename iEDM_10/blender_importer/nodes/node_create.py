# Fragment: core node processing — creates Blender objects from the EDM graph.


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

    if (
        (is_shell_render and _is_generic_shell_name(final_name))
        or (is_segments_render and _is_generic_shell_name(final_name))
        or _is_generic_render_name(final_name)
        or name_unknown
    ):
        if is_shell_render and _is_generic_shell_name(final_name):
            final_name = "{}_shell".format(_nearest_shell_alias(node))
        elif is_segments_render and _is_generic_shell_name(final_name):
            final_name = _nearest_collision_alias(node, "segments")

        if is_connector_render:
            candidate = ""
            if node.parent and node.parent.transform:
                candidate = _transform_display_name(node.parent.transform)
                if candidate.startswith("v_"):
                    candidate = candidate[2:]
                candidate = _strip_anim_prefix(candidate)
                if candidate.startswith("Empty_"):
                    candidate = candidate[len("Empty_") :]
            if candidate and candidate.lower() != "root":
                final_name = "Connector_{}".format(candidate)
            else:
                final_name = "Connector"

        if (_is_generic_render_name(final_name) or name_unknown) and isinstance(
            node.transform, ArgVisibilityNode
        ):
            candidate = getattr(node.transform, "name", "") or ""
            if candidate.startswith("v_"):
                candidate = candidate[2:]
            if candidate.startswith("Empty_"):
                candidate = candidate[len("Empty_") :]
            if (
                candidate
                and candidate.lower() != "root"
                and not _is_generic_render_name(candidate)
            ):
                final_name = candidate

        if (
            (_is_generic_render_name(final_name) or name_unknown)
            and node.parent
            and node.parent.transform
        ):
            candidate = _transform_display_name(node.parent.transform)
            if isinstance(
                node.parent.transform, ArgVisibilityNode
            ) and candidate.startswith("v_"):
                candidate = candidate[2:]
            if candidate.startswith("Empty_"):
                candidate = candidate[len("Empty_") :]
            if (
                candidate
                and candidate.lower() != "root"
                and not _is_generic_render_name(candidate)
            ):
                final_name = candidate

    if (
        _is_generic_render_name(final_name) or name_unknown
    ) and not parent_is_arg_wrapper:
        mat_name = ""
        if hasattr(node.render, "material") and node.render.material:
            mat_name = getattr(node.render.material, "name", "") or ""
        if mat_name:
            final_name = _SEMANTIC_NAME_MAP.get(mat_name, mat_name)

    if not parent_is_arg_wrapper:
        final_name = _SEMANTIC_NAME_MAP.get(final_name, final_name)

    if _is_generic_render_name(final_name) and not parent_is_arg_wrapper:
        ridx = getattr(node.render, "_graph_render_idx", None)
        if ridx is not None:
            final_name = "rn_{:04d}".format(ridx)

    if final_name:
        node.render.name = final_name


def _prepare_connector_metadata(node):
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
    try:
        node.blender["_iedm_connector_apply_xfix"] = False
        parent_tf = getattr(getattr(node, "parent", None), "transform", None)
        if (
            node.transform is None
            and isinstance(parent_tf, TransformNode)
            and not isinstance(parent_tf, AnimatingNode)
        ):
            _ptf_name = getattr(parent_tf, "name", "") or ""
            if hasattr(parent_tf, "matrix"):
                try:
                    _pm = Matrix(parent_tf.matrix)
                    pbl = getattr(node.parent, "blender", None)
                    if pbl is not None:
                        pbl["_iedm_raw_transform_matrix"] = [
                            float(v) for row in _pm for v in row
                        ]
                        if _ptf_name:
                            pbl["_iedm_raw_transform_name"] = _ptf_name
                    node.blender["_iedm_connector_tf_matrix"] = [
                        float(v) for row in _pm for v in row
                    ]
                except Exception as e:
                    print(f"Warning in blender_importer/nodes/core.py: {e}")
            if _ptf_name == "Connector Transform":
                try:
                    node.blender["_iedm_connector_apply_xfix"] = (
                        _is_neg90_x_basis_matrix(_pm)
                    )
                except Exception:
                    node.blender["_iedm_connector_apply_xfix"] = True
            if _ptf_name and _ptf_name != "Connector Transform":
                node.blender["_iedm_connector_tf_name"] = _ptf_name

        if node.transform is not None and not isinstance(node.transform, AnimatingNode):
            _ctf_name = getattr(node.transform, "name", "") or ""
            if hasattr(node.transform, "matrix"):
                try:
                    _m = Matrix(node.transform.matrix)
                    node.blender["_iedm_connector_tf_matrix"] = [
                        float(v) for row in _m for v in row
                    ]
                except Exception as e:
                    print(f"Warning in blender_importer/nodes/core.py: {e}")
            if _ctf_name == "Connector Transform":
                try:
                    node.blender["_iedm_connector_apply_xfix"] = (
                        _is_neg90_x_basis_matrix(_m)
                    )
                except Exception:
                    node.blender["_iedm_connector_apply_xfix"] = True
            if _ctf_name and _ctf_name != "Connector Transform":
                node.blender["_iedm_connector_tf_name"] = _ctf_name
        if (
            "_iedm_connector_tf_matrix" not in node.blender
            and getattr(node, "_local_bl", None) is not None
        ):
            try:
                _ml = Matrix(node._local_bl)
                node.blender["_iedm_connector_tf_matrix"] = [
                    float(v) for row in _ml for v in row
                ]
            except Exception as e:
                print(f"Warning in blender_importer/nodes/core.py: {e}")
        if "_iedm_connector_tf_name" not in node.blender:
            _fallback_name = str(
                getattr(getattr(node, "render", None), "name", "") or ""
            )
            if _fallback_name:
                _fallback_name = _SUFFIX_RE.sub("", _fallback_name)
                node.blender["_iedm_connector_tf_name"] = _fallback_name
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")


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
