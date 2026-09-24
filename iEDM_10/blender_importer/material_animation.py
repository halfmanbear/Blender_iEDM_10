"""Map animated uniforms to shader sockets and UV keyframes."""

import logging as logging

import bpy as bpy

from ..utils import action_fcurves
from .material_uv_shift import (
    _UV_SHIFT_SLOTS,
    _keyframe_material_uv_location,
    _uv_shift_mapping_node,
)

_ANIMATED_UNIFORM_TO_EDMPROPS = {
    "emissiveValue": "EMISSIVE_ARG",
    "colorShift": "COLOR_ARG",
    "diffuseShift": "COLOR_ARG",
    "selfIlluminationValue": "EMISSIVE_ARG",
    "selfIlluminationColor": "EMISSIVE_COLOR_ARG",
    "emissiveShift": "EMISSIVE_COLOR_ARG",
    "aoValue": "AO_ARG",
    "lightMapValue": "AO_ARG",
    "lightMapShift": "AO_ARG",
    "opacityValue": "OPACITY_VALUE_ARG",
}


_logger = logging.getLogger(__name__)


_ANIM_UNIFORM_TO_SOCKET = {
    "emissiveValue": "Emissive Value",
    "selfIlluminationValue": "Emissive Value",
    "selfIlluminationColor": "Emissive",
    "colorShift": "Base Color",
    "aoValue": "AO Value*",
    "lightMapValue": "AO Value*",
    "opacityValue": "Opacity Value",
}


_UV_ANIM_KEYWORDS = ("uv", "UV", "scroll", "Scroll", "move", "Move", "shift", "Shift")


_HEURISTIC_ANIM_UNIFORM_TARGETS = {
    "emissive_value": {"socket": "Emissive Value", "edmprop": "EMISSIVE_ARG"},
    "emissive_color": {"socket": "Emissive", "edmprop": "EMISSIVE_COLOR_ARG"},
    "color": {"socket": "Base Color", "edmprop": "COLOR_ARG"},
    "ao": {"socket": "AO Value*", "edmprop": "AO_ARG"},
    "opacity": {"socket": "Opacity Value", "edmprop": "OPACITY_VALUE_ARG"},
    "uv": {"socket": None, "edmprop": None},
}


def _normalized_uniform_name(name):
    raw = str(name or "").strip().lower()
    if not raw:
        return ""
    return "".join(ch for ch in raw if ("a" <= ch <= "z") or ("0" <= ch <= "9"))


def _heuristic_uniform_category(name):
    norm = _normalized_uniform_name(name)
    if not norm:
        return None

    if any(
        token in norm
        for token in (
            "uv",
            "texcoord",
            "texcoords",
            "scroll",
            "offset",
            "pan",
            "translate",
        )
    ):
        return "uv"
    if any(token in norm for token in ("opacity", "alpha", "transparen")):
        return "opacity"
    if any(token in norm for token in ("lightmap", "ambientocclusion", "ao")):
        return "ao"
    if any(
        token in norm
        for token in ("emissive", "selfillum", "selfillumination", "glow", "incand")
    ):
        if any(token in norm for token in ("color", "colour", "tint", "shift")):
            return "emissive_color"
        return "emissive_value"
    if any(
        token in norm
        for token in (
            "diffuse",
            "albedo",
            "basecolor",
            "basecolour",
            "color",
            "colour",
            "tint",
        )
    ):
        return "color"
    return None


def _resolve_animated_uniform_target(uniform_name):
    socket_name = _ANIM_UNIFORM_TO_SOCKET.get(uniform_name)
    edmprops_field = _ANIMATED_UNIFORM_TO_EDMPROPS.get(uniform_name)
    if socket_name is not None or edmprops_field is not None:
        return {
            "mode": "exact",
            "socket": socket_name,
            "edmprop": edmprops_field,
            "category": None,
        }

    category = _heuristic_uniform_category(uniform_name)
    if category is None:
        return None
    target = _HEURISTIC_ANIM_UNIFORM_TARGETS.get(category)
    if target is None:
        return None
    return {
        "mode": "heuristic",
        "socket": target.get("socket"),
        "edmprop": target.get("edmprop"),
        "category": category,
    }


def _looks_like_uv_anim_uniform(uniform_name):
    if uniform_name in _UV_SHIFT_SLOTS:
        return True
    target = _resolve_animated_uniform_target(uniform_name)
    if target is not None:
        return target.get("category") == "uv"
    return any(kw in str(uniform_name or "") for kw in _UV_ANIM_KEYWORDS)


def _mat_ensure_node_tree_action(mat):
    if mat.node_tree.animation_data is None:
        mat.node_tree.animation_data_create()
    if mat.node_tree.animation_data.action is None:
        mat.node_tree.animation_data.action = bpy.data.actions.new(
            mat.name + "_mat_anim"
        )
    return mat.node_tree.animation_data.action


def _mat_set_linear_on_path(action, anim_path):
    for fc in action_fcurves(action, id_type="NODETREE"):
        if anim_path in fc.data_path:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


def _create_material_socket_animations(mat, material):
    """Create node-tree animation keyframes for animated material uniforms.

    The exporter checks bpy_material.node_tree.animation_data for animation on
    specific socket paths (via path_from_id). Without these keyframes the *_ARG
    int indices are set but the exporter treats every property as static.
    """
    if not (mat and mat.use_nodes and mat.node_tree):
        return
    anim_uniforms = getattr(material, "animated_uniforms", None)
    if not anim_uniforms:
        return

    try:
        from ..edm_format.typereader import AnimatedProperty
    except ImportError:
        return

    # The bridge node is usually a custom official node (type "CUSTOM"), not a
    # plain ShaderNodeGroup; match anything that carries a group node tree.
    group_node = next(
        (n for n in mat.node_tree.nodes if getattr(n, "node_tree", None) is not None),
        None,
    )
    if group_node is None:
        return

    for uniform_name, prop in anim_uniforms.items():
        target = _resolve_animated_uniform_target(uniform_name)
        socket_name = target.get("socket") if target else None
        if socket_name is None:
            continue
        if not isinstance(prop, AnimatedProperty):
            continue
        keys = list(getattr(prop, "keys", []) or [])
        if not keys:
            continue

        socket = next(
            (inp for inp in group_node.inputs if inp.name == socket_name), None
        )
        if socket is None:
            continue

        anim_path = socket.path_from_id("default_value")
        action = _mat_ensure_node_tree_action(mat)

        for framedata in keys:
            try:
                _keyframe_material_socket(mat, socket, anim_path, framedata)
            except Exception as e:
                print(
                    f"Warning in material_setup._create_material_socket_animations: {e}"
                )

        _mat_set_linear_on_path(action, anim_path)


def _create_material_uv_animations(mat, material, texture_nodes):
    """Label Mapping nodes with arg numbers and animate their Location inputs.

    The exporter finds UV-scroll animation in
    ``extract_arg_number(mapping_node.label)``.
    and checking bpy_material.node_tree.animation_data on the Location input path.
    Without the label and keyframes, UV scroll is exported as a static offset.
    """
    if not (mat and mat.use_nodes and mat.node_tree):
        return
    anim_uniforms = getattr(material, "animated_uniforms", None)
    if not anim_uniforms:
        return

    try:
        from ..edm_format.typereader import AnimatedProperty
    except ImportError:
        return

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for uniform_name, prop in anim_uniforms.items():
        target = _resolve_animated_uniform_target(uniform_name)
        if target and target.get("socket") is not None:
            continue  # handled by _create_material_socket_animations
        if not _looks_like_uv_anim_uniform(uniform_name):
            continue
        if not isinstance(prop, AnimatedProperty):
            continue
        keys = list(getattr(prop, "keys", []) or [])
        if not keys:
            continue
        arg = getattr(prop, "argument", None)
        if arg is None or int(arg) < 0:
            continue
        tex_node = texture_nodes.get(_UV_SHIFT_SLOTS.get(uniform_name, 0))
        if tex_node is None:
            continue

        mapping_node = _uv_shift_mapping_node(nodes, links, tex_node)
        # Label encodes the arg number: exporter calls extract_arg_number(node.label).
        mapping_node.label = str(int(arg))

        loc_input = mapping_node.inputs.get("Location") or mapping_node.inputs[1]
        anim_path = loc_input.path_from_id("default_value")
        action = _mat_ensure_node_tree_action(mat)

        for framedata in keys:
            try:
                _keyframe_material_uv_location(mat, loc_input, anim_path, framedata)
            except Exception as e:
                print(f"Warning in material_setup._create_material_uv_animations: {e}")

        _mat_set_linear_on_path(action, anim_path)


def _keyframe_material_socket(mat, socket, anim_path, framedata):
    """Apply one animated material socket value and insert its keyframe."""
    value = getattr(framedata, "value", None)
    if value is None:
        return
    blender_frame = (float(getattr(framedata, "frame", 0.0)) + 1.0) * 100.0
    if socket.type == "VALUE":
        socket.default_value = float(value)
    elif socket.type in ("RGBA", "VECTOR"):
        value = tuple(float(component) for component in value)
        if socket.type == "RGBA" and len(value) == 3:
            value += (1.0,)
        socket.default_value = value
    else:
        return
    mat.node_tree.keyframe_insert(data_path=anim_path, frame=blender_frame)
