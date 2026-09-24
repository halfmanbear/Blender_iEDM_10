"""Compute oriented-scale bases and create parent wrapper objects."""

import bpy as bpy

from ..edm_format.mathtypes import Matrix, MatrixScale, Quaternion, Vector
from ..edm_format.types import ArgAnimationNode
from .action_values import _compose_oriented_scale_matrix
from .animation import _quat_is_identity
from .import_context import _ROOT_BASIS_FIX, _import_ctx, _log
from .visibility_graph import (
    _is_child_of_file_root,
    _is_top_level_visibility_authored_pair,
)


def _render_local_matrix_for_graph_node(node):
    render = getattr(node, "render", None)
    if render is None:
        return Matrix.Identity(4)
    try:
        if hasattr(render, "pos"):
            return Matrix.Translation(Vector(render.pos[:3]))
        if hasattr(render, "matrix"):
            return Matrix(render.matrix)
    except Exception as e:
        _log.warn(
            "render local matrix read for '{}': {}".format(
                getattr(render, "name", type(render).__name__), e
            ),
            exc=e,
        )
    return Matrix.Identity(4)


def _scale_orientation_source_for_graph_node(node):
    if node is None:
        return None
    transforms = list(getattr(node, "_collapsed_transforms", None) or [])
    if not transforms:
        tf = getattr(node, "transform", None)
        if tf is not None:
            transforms = [tf]
    for tf in transforms:
        if not isinstance(tf, ArgAnimationNode):
            continue
        scale_sets = list(getattr(tf, "scaleData", None) or [])
        has_scale_keys = any(
            (entry[1][0] or entry[1][1])
            for entry in scale_sets
            if isinstance(entry, (list, tuple)) and len(entry) == 2
        )
        q2 = getattr(getattr(tf, "base", None), "quat_2", None)
        if has_scale_keys or (q2 is not None and not _quat_is_identity(q2)):
            return tf
    return None


def _object_collection_targets(ob):
    cols = list(getattr(ob, "users_collection", None) or [])
    if cols:
        return cols
    return [bpy.context.collection]


def _insert_parent_wrapper_object(
    child, name, local_matrix=None, reset_child_local=False
):
    parent = getattr(child, "parent", None)
    parent_type = child.parent_type
    parent_bone = child.parent_bone
    # Bone attachments carry their EDM bone-frame correction in the parent inverse.
    parent_inverse = child.matrix_parent_inverse.copy()
    helper = bpy.data.objects.new(name, None)
    helper.empty_display_size = 0.1
    for collection in _object_collection_targets(child):
        try:
            collection.objects.link(helper)
        except RuntimeError:
            pass
    helper.parent = parent
    helper.parent_type = parent_type
    helper.parent_bone = parent_bone
    helper.matrix_parent_inverse = parent_inverse
    helper.matrix_basis = (
        local_matrix.copy()
        if hasattr(local_matrix, "copy")
        else (local_matrix or Matrix.Identity(4))
    )
    child.parent = helper
    child.parent_type = "OBJECT"
    child.parent_bone = ""
    child.matrix_parent_inverse = Matrix.Identity(4)
    if reset_child_local:
        child.matrix_basis = Matrix.Identity(4)
    return helper


def _promote_oriented_scale_top_name(node, top_wrapper, leaf):
    """Keep the authored EDM control name on the object that carries its action.

    For control-only empties, the top oriented-scale wrapper represents the real
    EDM node transform. The identity leaf exists only to realise Blender's lack of
    per-object oriented scale, so it should not keep the authored node name.
    """
    if node is None or top_wrapper is None or leaf is None:
        return
    if getattr(leaf, "type", "") != "EMPTY":
        return
    if getattr(node, "render", None) is not None:
        return

    original_name = getattr(leaf, "name", "") or ""
    if not original_name:
        return

    try:
        leaf.name = "{}_iedm_leaf".format(original_name)
    except Exception:
        return

    try:
        top_wrapper.name = original_name
    except Exception:
        return

    try:
        top_wrapper["_iedm_oriented_scale_node_name"] = original_name
        leaf["_iedm_oriented_scale_leaf_of"] = original_name
    except Exception as exc:
        _log.debug("Optional operation failed: {}".format(exc), level=2)


def _reconstruct_edm_local_matrix(node):
    """Return (local_edm, rot_scale_edm, base_mat, pos_vec) for an ArgAnimationNode.

    local_edm  = base_mat @ T(pos) @ q1m @ oriented_scale  (DLL formula verbatim)
    rot_scale_edm = base_mat @ q1m @ oriented_scale  (no translation)
    pos_vec       = Vector(base.position)              (Z-up passthrough)

    The DLL reads default_position and keyframe Vec3d values with the same readVec3d
    call — same coordinate space. Since vector_to_blender / _anim_vector_to_blender
    are both passthroughs for v10, base.position is already in Blender/Z-up space
    and must NOT receive _ROOT_BASIS_FIX.
    """
    base_mat = Matrix(node.base.matrix)
    pos_vec = Vector(
        (node.base.position[0], node.base.position[1], node.base.position[2])
    )
    base_scale_vec = Vector(
        (node.base.scale[0], node.base.scale[1], node.base.scale[2])
    )
    q1 = (
        node.base.quat_1
        if hasattr(node.base.quat_1, "to_matrix")
        else Quaternion(node.base.quat_1)
    )
    q2 = (
        node.base.quat_2
        if hasattr(node.base.quat_2, "to_matrix")
        else Quaternion(node.base.quat_2)
    )

    q1m = q1.to_matrix().to_4x4()
    oriented_scale = _compose_oriented_scale_matrix(base_scale_vec, q2)
    rot_scale_edm = base_mat @ q1m @ oriented_scale
    local_edm = base_mat @ Matrix.Translation(pos_vec) @ q1m @ oriented_scale

    return local_edm, rot_scale_edm, base_mat, pos_vec


def _arganimation_prerotation_basis_local(node):
    """Return the constant affine prefix before `quat_1` is applied."""
    base_mat = Matrix(node.base.matrix)
    pos_vec = Vector(
        (node.base.position[0], node.base.position[1], node.base.position[2])
    )

    acts_like_top_level_control = _is_child_of_file_root(
        node
    ) or _is_top_level_visibility_authored_pair(node)
    apply_root_basis = bool(
        _import_ctx.edm_version >= 10
        and acts_like_top_level_control
        and not getattr(_import_ctx, "use_scene_root_basis_object", True)
    )
    if apply_root_basis:
        return Matrix.Translation(pos_vec) @ (_ROOT_BASIS_FIX @ base_mat)
    return base_mat @ Matrix.Translation(pos_vec)


def _arganimation_rotation_basis_local(node):
    """Return the full static basis up to and including `quat_1`.

    Legacy single-object basis used when Blender can store the full affine+rotation
    top transform without drift.
    """
    return _arganimation_prerotation_basis_local(
        node
    ) @ _arganimation_key_rotation_basis_local(node)


def _arganimation_key_rotation_basis_local(node):
    """Return the exact static `quat_1` rotation basis.

    For oriented-scale wrapper rebuilds the animated control object should keep only
    the authored rotation basis. Any affine prefix from `base.matrix` and
    `base.position` can be lifted into a constant parent helper when Blender cannot
    round-trip the combined affine+rotation matrix exactly.
    """
    q1 = (
        node.base.quat_1
        if hasattr(node.base.quat_1, "to_matrix")
        else Quaternion(node.base.quat_1)
    )
    return q1.to_matrix().to_4x4()


def _matrix_transform_roundtrip_error(local_mat):
    try:
        loc, rot, scale = local_mat.decompose()
        rebuilt = (
            Matrix.Translation(loc) @ rot.to_matrix().to_4x4() @ MatrixScale(scale)
        )
        return max(
            abs(float(local_mat[r][c]) - float(rebuilt[r][c]))
            for r in range(4)
            for c in range(4)
        )
    except Exception:
        return float("inf")


def _needs_prerotation_affine_split(
    top_local, prerotation_local, rotation_local, eps=1e-4
):
    """Detect top wrappers Blender cannot store exactly as one object transform.

    Some ArgRotation/ArgAnimation controls carry a representable affine prefix in
    `base.matrix @ T(base.position)` and a representable authored `quat_1` rotation,
    but their product becomes a sheared affine transform that drifts when Blender
    decomposes it. Split those into:

      prefix_helper @ animated_rotation_object
    """
    top_err = _matrix_transform_roundtrip_error(top_local)
    if top_err <= eps:
        return False
    pre_err = _matrix_transform_roundtrip_error(prerotation_local)
    rot_err = _matrix_transform_roundtrip_error(rotation_local)
    return pre_err <= eps and rot_err <= eps
