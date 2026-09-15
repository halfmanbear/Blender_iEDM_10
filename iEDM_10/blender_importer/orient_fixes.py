# Bone-transform prefix correction for static render descendants.

import bpy
from mathutils import Matrix

from .prelude import _ROOT_BASIS_FIX, _import_ctx, _log

# Rz(-90°): rotates the DCS nose-forward axis to Blender +X for BT-prefix EDMs.
_Rz_neg90 = Matrix(((0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
# Full DCS-to-Blender world transform applied to BT-prefix content.
_BT_CONTENT_TO_BLENDER = _Rz_neg90 @ _ROOT_BASIS_FIX


def _fix_bonetransform_bone_child_render_world_positions():
    """Post-pass: correct render meshes below bone empties in the no-armature case.

    When armature creation is skipped for Bonetransform-prefix EDMs (no SkinNodes),
    ArgAnimatedBone/Bone nodes become empties at their bone rest world positions.
    RenderNode meshes and static TransformNode wrappers authored below those bones
    store their placement in world space, identical to the convention used when the
    parent is an armature at identity.  Placing that world-space matrix_basis under
    a non-identity bone empty multiplies in the bone's world transform, displacing
    the part.

    Some assets put wrappers between the bone and mesh:
      Mesh -> ArgVisibilityNode -> TransformNode -> Bone
    so fixing only meshes whose direct parent is a bone misses most animated parts.
    This pass finds a static, non-animated correction target in the chain below the
    nearest bone and sets matrix_world = matrix_basis so Blender recomputes the
    correct local offset while preserving bone/visibility animation hierarchy.
    """
    if getattr(_import_ctx, "bonetransform_prefix_matrix", None) is None:
        return
    if not getattr(_import_ctx, "file_has_bones", False):
        return

    bpy.context.view_layer.update()

    def _has_anim_data(ob):
        ad = getattr(ob, "animation_data", None)
        return ad is not None and (
            getattr(ad, "action", None) is not None
            or len(getattr(ad, "nla_tracks", []) or []) > 0
        )

    def _debug_tf_cls(ob):
        return str(ob.get("_iedm_dbg_tf_cls", "") or "")

    def _is_bone_empty(ob):
        return _debug_tf_cls(ob) in {"ArgAnimatedBone", "Bone"}

    def _is_prefix_empty(ob):
        try:
            return bool(ob.get("_iedm_bt_prefix", False))
        except Exception:
            return False

    def _matrix_is_identity(mat, eps=1e-6):
        try:
            ident = type(mat).Identity(4)
            for r in range(4):
                for c in range(4):
                    if abs(float(mat[r][c]) - float(ident[r][c])) > eps:
                        return False
            return True
        except Exception:
            return False

    def _has_small_uniform_scale(ob, max_scale=0.1):
        try:
            _, _, scale = ob.matrix_basis.decompose()
            vals = [abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z))]
            return max(vals) <= max_scale and min(vals) > 1e-8
        except Exception:
            return False

    def _find_bone_descendant_correction_target(mesh_obj):
        """Return the object whose basis is authored as world space, or None."""
        chain = []
        cur = mesh_obj
        bone = None
        while cur is not None:
            if _is_bone_empty(cur):
                bone = cur
                break
            if _is_prefix_empty(cur):
                break
            chain.append(cur)
            cur = getattr(cur, "parent", None)

        if bone is None:
            return None

        # Prefer the highest static TransformNode wrapper below the bone.  Visibility
        # wrappers are often animated and should remain untouched; their child mesh
        # or transform wrapper will inherit the corrected world placement.
        for ob in reversed(chain):
            if getattr(ob, "type", "") != "EMPTY":
                continue
            if _debug_tf_cls(ob) != "TransformNode":
                continue
            if _has_anim_data(ob):
                continue
            if _matrix_is_identity(ob.matrix_basis):
                continue
            # The misplaced static render wrappers in this class of EDMs carry the
            # authored model-space scale, typically 0.01.  Full-size bone descendants
            # can be intentionally composed through their bone hierarchy and must not
            # be moved into raw world space by this no-armature repair pass.
            if not _has_small_uniform_scale(ob):
                continue
            return ob

        # Fallback for render nodes parented directly to bones.
        if (
            not _has_anim_data(mesh_obj)
            and not _matrix_is_identity(mesh_obj.matrix_basis)
            and _has_small_uniform_scale(mesh_obj)
        ):
            return mesh_obj

        return None

    fixed = 0
    skipped_animated = 0
    seen_targets = set()
    for ob in list(getattr(bpy.data, "objects", []) or []):
        if getattr(ob, "type", "") != "MESH":
            continue

        target = _find_bone_descendant_correction_target(ob)
        if target is None:
            continue
        if _has_anim_data(target):
            skipped_animated += 1
            continue
        target_key = target.as_pointer()
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)

        try:
            # matrix_basis is in DCS/EDM world space; apply Rz(-90°)@RBF to convert to Blender world.
            _basis_copy = target.matrix_basis.copy()
            _basis_loc = _basis_copy.to_translation()
            print(
                "[iEDM][BTFIX] target={} mesh={} basis_loc=({:.4f},{:.4f},{:.4f}) parent={}".format(
                    target.name,
                    ob.name,
                    _basis_loc.x,
                    _basis_loc.y,
                    _basis_loc.z,
                    getattr(getattr(target, "parent", None), "name", None),
                )
            )
            intended_world = _BT_CONTENT_TO_BLENDER @ _basis_copy
            target.matrix_world = intended_world
            bpy.context.view_layer.update()
            fixed += 1
        except Exception as e:
            _log.warn("_fix_bonetransform_bone_child_render_world_positions", exc=e)

    _log.debug(
        "_fix_bonetransform_bone_child_render_world_positions: fixed {} skipped_animated {}".format(
            fixed,
            skipped_animated,
        ),
        level=1,
    )
