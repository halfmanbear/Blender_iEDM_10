# Fragment: core node processing — creates Blender objects from the EDM graph.
import json
import logging
import math

import bpy

from ...edm_format.mathtypes import (
    Matrix,
    Vector,
    vector_to_blender,
)
from ...edm_format.types import (
    ArgPositionNode,
    ArgRotationNode,
    ArgScaleNode,
    ArgVisibilityNode,
)
from ..prelude import (
    _ROOT_BASIS_FIX,
    _import_ctx,
    _import_profile_flag,
    _is_generic_render_name,
    _set_official_special_type,
    _strip_anim_prefix,
)

_logger = logging.getLogger(__name__)


def _is_narrow_safe_identity_helper_name(name):
    if not name:
        return False
    if name in {"Connector Transform", "Fake Light Transform"}:
        return True
    if name.startswith(("Connector_", "Light_Dir", "Omni_", "Fspot", "Point", "Dummy")):
        return True
    return False


def _idprop_sequence_value(value):
    def _convert(item, depth=0):
        if depth > 2:
            return None
        if isinstance(item, (int, float, bool, str)):
            return item
        if isinstance(item, (bytes, bytearray)):
            return bytes(item).hex()
        item_to_list = getattr(item, "tolist", None)
        if callable(item_to_list):
            try:
                return _convert(item_to_list(), depth)
            except Exception:
                return None
        if isinstance(item, (list, tuple)):
            if len(item) > 16:
                return None
            converted = [_convert(sub, depth + 1) for sub in item]
            if any(sub is None for sub in converted):
                return None
            return converted
        return None

    if not isinstance(value, (list, tuple)):
        return None
    if len(value) > 16:
        return None
    converted = _convert(value, 0)
    if converted is None:
        return None
    if all(isinstance(item, (int, float, bool)) for item in converted):
        return converted
    return __import__("json").dumps(converted, separators=(",", ":"))


def _idprop_diag_value(value):
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return _idprop_sequence_value(value)


def _control_wrapper_prefix(tfnode):
    if isinstance(tfnode, ArgVisibilityNode):
        return "v_"
    if isinstance(tfnode, ArgRotationNode):
        return "ar_"
    if isinstance(tfnode, ArgPositionNode):
        return "al_"
    if isinstance(tfnode, ArgScaleNode):
        return "as_"
    return ""


def _preferred_control_wrapper_name(node, base_name):
    """Disambiguate control empties from their sole visible child mesh.

    Plain-root visibility/control assets often import as:
      tf_0466   (ArgRotation EMPTY)
        -> tf_0466.001 (RenderNode MESH)

    because Blender auto-suffixes the child mesh after the control empty claims
    the semantic base name. Naming the control object with its authored control
    prefix keeps the visible mesh on the base name.
    """
    tf = getattr(node, "transform", None)
    if tf is None:
        return base_name

    prefix = _control_wrapper_prefix(tf)
    if not prefix or base_name.startswith(prefix):
        return base_name

    children = list(getattr(node, "children", []) or [])
    if len(children) != 1:
        return base_name

    child = children[0]
    if (
        getattr(child, "transform", None) is not None
        or getattr(child, "render", None) is None
    ):
        return base_name

    child_name = str(getattr(child.render, "name", "") or "")
    if child_name.startswith("v_") and isinstance(tf, ArgVisibilityNode):
        child_name = child_name[2:]
    child_name = _strip_anim_prefix(child_name)
    if child_name.startswith("Empty_"):
        child_name = child_name[len("Empty_") :]

    base_plain = _strip_anim_prefix(base_name)
    if base_plain.startswith("Empty_"):
        base_plain = base_plain[len("Empty_") :]

    if not child_name or _is_generic_render_name(child_name):
        child_name = base_plain

    if not child_name or child_name.lower() == "root":
        return base_name

    if (
        child_name == base_plain
        or _is_generic_render_name(base_plain)
        or base_plain.startswith("tf_")
    ):
        return prefix + child_name

    return base_name


def _skin_bbox_local_matrix(skin_node):
    bbox = list(getattr(skin_node, "bbox", None) or [])
    if len(bbox) != 6:
        return None

    try:
        min_vec = vector_to_blender(bbox[0:3])
        max_vec = vector_to_blender(bbox[3:6])
        if any(math.isinf(v) or math.isnan(v) for v in min_vec) or any(
            math.isinf(v) or math.isnan(v) for v in max_vec
        ):
            return None

        if getattr(_import_ctx, "edm_version", 0) >= 10 and _import_profile_flag(
            "skin_mesh_geometry_root_basis_fix"
        ):
            corners = [
                (_ROOT_BASIS_FIX @ Vector((x, y, z, 1.0))).to_3d()
                for x in (min_vec.x, max_vec.x)
                for y in (min_vec.y, max_vec.y)
                for z in (min_vec.z, max_vec.z)
            ]
            min_vec = Vector(
                (
                    min(c.x for c in corners),
                    min(c.y for c in corners),
                    min(c.z for c in corners),
                )
            )
            max_vec = Vector(
                (
                    max(c.x for c in corners),
                    max(c.y for c in corners),
                    max(c.z for c in corners),
                )
            )

        dims = max_vec - min_vec
        if dims.length <= 1.0e-9:
            return None

        center = (min_vec + max_vec) * 0.5
        scale = Vector((abs(dims.x) * 0.5, abs(dims.y) * 0.5, abs(dims.z) * 0.5))
        return Matrix.Translation(center) @ Matrix.Diagonal(
            (scale.x, scale.y, scale.z, 1.0)
        )
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
        return None


def _wrap_skin_object_with_skin_box(mesh_obj, skin_node):
    if mesh_obj is None or getattr(mesh_obj, "type", "") != "MESH":
        return None
    if bool(mesh_obj.get("_iedm_skin_box_wrapped")):
        return None

    bbox_local = _skin_bbox_local_matrix(skin_node)
    if bbox_local is None:
        return None

    try:
        old_parent = getattr(mesh_obj, "parent", None)
        old_local = mesh_obj.matrix_basis.copy()
        old_world = mesh_obj.matrix_world.copy()

        skin_box = bpy.data.objects.new(f"{mesh_obj.name}_Skin_Box", None)
        skin_box.empty_display_type = "CUBE"
        skin_box.empty_display_size = 1.0
        _set_official_special_type(skin_box, "SKIN_BOX")
        skin_box["_iedm_translation_status"] = "exact"
        skin_box["_iedm_translation_source"] = "SkinNode.bbox"
        try:
            skin_box["_iedm_raw_skin_box"] = json.dumps(
                [float(v) for v in (getattr(skin_node, "bbox", None) or [])],
                separators=(",", ":"),
            )
        except Exception:
            _logger.debug("Ignoring optional operation failure", exc_info=True)

        target_collections = list(getattr(mesh_obj, "users_collection", []) or [])
        if not target_collections:
            target_collections = [bpy.context.collection]
        for col in target_collections:
            try:
                col.objects.link(skin_box)
            except RuntimeError:
                pass

        skin_box.parent = old_parent
        skin_box.matrix_parent_inverse = Matrix.Identity(4)
        skin_box.matrix_basis = old_local @ bbox_local

        mesh_obj.parent = skin_box
        mesh_obj.matrix_parent_inverse = Matrix.Identity(4)
        mesh_obj.matrix_basis = bbox_local.inverted_safe()
        mesh_obj["_iedm_skin_box_wrapped"] = True

        try:
            mesh_obj.matrix_world = old_world
        except Exception:
            _logger.debug("Ignoring optional operation failure", exc_info=True)

        return skin_box
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
        return None
