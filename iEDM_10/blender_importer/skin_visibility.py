"""Restore skin visibility transforms and inverse-scale rest offsets."""

import math as math

import bpy as bpy
from mathutils import Matrix

from .graph_postprocess import _reparent_preserve_world
from .import_context import _import_ctx, _log
from .node_identity import _ob_local_is_identity


def _flat_to_matrix(value):
    try:
        if hasattr(value, "to_list"):
            value = value.to_list()
        elif not isinstance(value, (list, tuple)):
            value = list(value)
    except Exception:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 16:
        return None
    try:
        return Matrix(
            (
                tuple(float(v) for v in value[0:4]),
                tuple(float(v) for v in value[4:8]),
                tuple(float(v) for v in value[8:12]),
                tuple(float(v) for v in value[12:16]),
            )
        )
    except Exception:
        return None


def _matrix_close(a, b, eps=1e-5):
    try:
        return all(
            abs(float(a[r][c]) - float(b[r][c])) <= eps
            for r in range(4)
            for c in range(4)
        )
    except Exception:
        return False


def _restore_skin_visibility_transform_basis():
    """Restore same-name transform helper basis for root visibility skin branches."""
    if _import_ctx.edm_version < 10:
        return
    for vis_ob in list(getattr(bpy.data, "objects", []) or []):
        if getattr(vis_ob, "type", "") != "EMPTY":
            continue
        if not bool(vis_ob.get("_iedm_vis_passthrough")):
            continue
        _restore_skin_visibility_object(vis_ob)


def _restore_skin_visibility_object(vis_ob):
    """Move skin meshes back under their matching transform helper."""
    vis_name = str(getattr(vis_ob, "name", "") or "")
    if not vis_name:
        return
    transform_children, vis_skin_children = _matching_visibility_children(
        vis_ob, vis_name
    )
    if len(transform_children) != 1:
        return
    helper = transform_children[0]
    helper_skin_children = _matching_skin_children(helper, vis_name)
    if not vis_skin_children and not helper_skin_children:
        return
    authored_local = _flat_to_matrix(helper.get("IEDM_LOCAL_BL_MAT"))
    if (
        authored_local is None
        or _matrix_close(authored_local, Matrix.Identity(4))
        or not _ob_local_is_identity(helper)
        or not _matrix_close(vis_ob.matrix_basis, authored_local)
    ):
        return
    try:
        vis_ob.matrix_basis = Matrix.Identity(4)
        helper.matrix_basis = authored_local
    except Exception as exc:
        _log.warn("_restore_skin_visibility_transform_basis", exc=exc)
        return
    _reparent_visibility_skin_children(vis_skin_children, helper)
    _tag_visibility_skin_children(helper_skin_children, helper)


def _matching_visibility_children(vis_ob, vis_name):
    transform_children = []
    skin_children = []
    for child in list(getattr(vis_ob, "children", []) or []):
        if getattr(child, "type", "") == "EMPTY":
            if (
                str(child.get("_iedm_dbg_tf_cls", "") or "") == "TransformNode"
                and str(child.get("_iedm_dbg_tf_name", "") or "") == vis_name
            ):
                transform_children.append(child)
        elif getattr(child, "type", "") == "MESH" and _is_matching_skin_child(
            child, vis_name
        ):
            skin_children.append(child)
    return transform_children, skin_children


def _matching_skin_children(parent, vis_name):
    return [
        child
        for child in list(getattr(parent, "children", []) or [])
        if getattr(child, "type", "") == "MESH"
        and _is_matching_skin_child(child, vis_name)
    ]


def _is_matching_skin_child(child, vis_name):
    return (
        bool(child.get("_iedm_skin_parent_override"))
        and str(child.get("_iedm_src_render_cls", "") or "") == "SkinNode"
        and str(child.get("_iedm_src_render_name", "") or "") == vis_name
    )


def _reparent_visibility_skin_children(children, helper):
    for child in children:
        try:
            _reparent_preserve_world(child, helper)
            child["_iedm_dbg_skin_helper_name"] = helper.name
        except Exception as exc:
            _log.warn("_restore_skin_visibility_transform_basis reparent", exc=exc)


def _tag_visibility_skin_children(children, helper):
    for child in children:
        try:
            child["_iedm_dbg_skin_helper_name"] = helper.name
        except Exception as exc:
            _log.warn("_restore_skin_visibility_transform_basis tag", exc=exc)


def _has_inverse_scale_child(ob):
    for child in list(getattr(ob, "children", []) or []):
        try:
            loc, rot, scale = child.matrix_basis.decompose()
        except Exception:
            continue
        max_scale = max(abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z)))
        min_scale = min(abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z)))
        if max_scale <= 0.0 or max_scale >= 0.1:
            continue
        if min_scale / max_scale < 0.95:
            continue
        if (
            abs(float(loc.x)) > 1e-4
            or abs(float(loc.y)) > 1e-4
            or abs(float(loc.z)) > 1e-4
        ):
            continue
        try:
            if abs(rot.angle) > math.radians(1.0):
                continue
        except Exception as exc:
            _log.debug("Optional operation failed: {}".format(exc), level=2)
        if any(
            getattr(desc, "type", "") == "MESH"
            for desc in getattr(child, "children_recursive", []) or []
        ):
            return True
    return False


def _is_identityish_basis(ob):
    try:
        loc, rot, scale = ob.matrix_basis.decompose()
        if (
            abs(float(loc.x)) > 1e-4
            or abs(float(loc.y)) > 1e-4
            or abs(float(loc.z)) > 1e-4
        ):
            return False
        if (
            abs(float(scale.x) - 1.0) > 1e-4
            or abs(float(scale.y) - 1.0) > 1e-4
            or abs(float(scale.z) - 1.0) > 1e-4
        ):
            return False
        return abs(rot.angle) <= math.radians(1.0)
    except Exception:
        return False


def _same_world_rotation(a, b):
    try:
        _la, ra, _sa = a.matrix_world.decompose()
        _lb, rb, _sb = b.matrix_world.decompose()
        return abs(ra.rotation_difference(rb).angle) <= math.radians(1.0)
    except Exception:
        return False


def _fix_inverse_scaled_visibility_rest_offset():
    """Repair inverse-scaled visibility wrapper rest offsets structurally.

    Some legacy 3ds Max v10 assets encode render geometry under an identity
    ArgVisibilityNode wrapper, then under a tiny uniform-scale TransformNode. The
    visibility wrapper must stay under its animated parent so animation still
    drives the mesh, but its rest offset may need to resolve against the parent
    pivot's parent transform. Detect that pattern by structure only; do not key
    off object names or a specific EDM file.
    """

    changed = False
    for ob in list(getattr(bpy.data, "objects", []) or []):
        if getattr(ob, "type", "") != "EMPTY":
            continue
        if str(ob.get("_iedm_dbg_tf_cls", "") or "") != "ArgVisibilityNode":
            continue
        if not _is_identityish_basis(ob):
            continue
        if not _has_inverse_scale_child(ob):
            continue

        parent = getattr(ob, "parent", None)
        candidate = getattr(parent, "parent", None) if parent is not None else None
        if parent is None or candidate is None:
            continue
        if (
            getattr(parent, "type", "") != "EMPTY"
            or getattr(candidate, "type", "") != "EMPTY"
        ):
            continue
        if not _same_world_rotation(parent, candidate):
            continue

        try:
            parent_loc = parent.matrix_world.to_translation()
            candidate_loc = candidate.matrix_world.to_translation()
            delta = candidate_loc - parent_loc
        except Exception:
            continue

        if delta.length < 0.05 or delta.length > 1.0:
            continue

        try:
            old_world = ob.matrix_world.copy()
            old_loc, old_rot, old_scale = old_world.decompose()
            _ = old_loc
            target_loc = candidate.matrix_world.to_translation().copy()
            target_world = Matrix.LocRotScale(target_loc, old_rot, old_scale)
            target_local = parent.matrix_world.inverted_safe() @ target_world
            ob.matrix_parent_inverse = Matrix.Identity(4)
            ob.matrix_basis = target_local
            ob["_iedm_inverse_scaled_visibility_rest_offset_fix"] = True
            ob["_iedm_inverse_scaled_visibility_parent"] = getattr(parent, "name", "")
            ob["_iedm_inverse_scaled_visibility_candidate"] = getattr(
                candidate, "name", ""
            )
            changed = True
        except Exception as e:
            _log.warn("_fix_inverse_scaled_visibility_rest_offset", exc=e)

    if changed:
        try:
            bpy.context.view_layer.update()
        except Exception as exc:
            _log.debug("Optional operation failed: {}".format(exc), level=2)
