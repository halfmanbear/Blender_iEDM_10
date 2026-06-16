# Fragment: post-pass world orientation fixes for meshes and empties.

import math

import bpy
from mathutils import Matrix

from .prelude import _ROOT_BASIS_FIX, _import_ctx, _import_profile_flag, _log

# Rz(-90°): rotates the DCS nose-forward axis to Blender +X for BT-prefix EDMs.
_Rz_neg90 = Matrix(((0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
# Full DCS-to-Blender world transform applied to BT-prefix content.
_BT_CONTENT_TO_BLENDER = _Rz_neg90 @ _ROOT_BASIS_FIX


def _apply_collision_mesh_orientation_fix():
  """Post-pass: fix collision meshes whose root-basis inheritance was broken.

  When the root basis object carries _ROOT_BASIS_FIX (90° X), all mesh descendants
  should inherit that rotation. Meshes at identity world X (0°) indicate the parent
  chain broke the rotation inheritance. Apply _ROOT_BASIS_FIX directly to these meshes.
  """
  if _import_ctx.edm_version < 10:
    return
  if getattr(_import_ctx, "collision_geometry_basis_fix", False):
    return
  if not _import_profile_flag("embedded_collision_scene_root_basis_object"):
    return
  if not getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  bpy.context.view_layer.update()

  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "type", "") != "MESH":
      continue
    ad = getattr(ob, "animation_data", None)
    if ad is not None and (getattr(ad, "action", None) is not None or len(getattr(ad, "nla_tracks", []) or []) > 0):
      continue
    try:
      world_mat = ob.matrix_world.copy()
      _, world_rot, _ = world_mat.decompose()
      world_x_deg = abs(math.degrees(world_rot.to_euler("XYZ").x))
      if world_x_deg < 5.0 or world_x_deg > 175.0:
        ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
    except Exception as e:
      _log.warn("_apply_collision_mesh_orientation_fix", exc=e)


def _apply_scene_root_mesh_orientation_fix():
  """Post-pass: fix scene-root meshes at wrong world orientation.

  When the root basis object carries _ROOT_BASIS_FIX (90° X), all mesh descendants
  should inherit that rotation. Meshes at identity world X (0°) need _ROOT_BASIS_FIX
  applied. For animated meshes with only visibility fcurves, the fix may be applied
  to the ArgVisibilityNode parent instead.
  """
  _log.debug("_apply_scene_root_mesh_orientation_fix: starting", level=1)
  if _import_ctx.edm_version < 10:
    _log.debug("_apply_scene_root_mesh_orientation_fix: skipped (version < 10)", level=1)
    return
  if getattr(_import_ctx, "bonetransform_prefix_matrix", None) is not None:
    _log.debug("_apply_scene_root_mesh_orientation_fix: skipped (bonetransform prefix)", level=1)
    return
  if not _import_profile_flag("scene_root_world_orientation_postfix"):
    _log.debug("_apply_scene_root_mesh_orientation_fix: skipped (capability disabled)", level=1)
    return
  use_root_basis = getattr(_import_ctx, "use_scene_root_basis_object", True)
  _log.debug("_apply_scene_root_mesh_orientation_fix: use_scene_root_basis_object={}".format(use_root_basis), level=1)
  if not use_root_basis:
    _log.debug("_apply_scene_root_mesh_orientation_fix: skipped (no root basis)", level=1)
    return

  bpy.context.view_layer.update()

  def _mat_to_euler_x_deg(mat):
    try:
      _, rot, _ = mat.decompose()
      return abs(math.degrees(rot.to_euler("XYZ").x))
    except:
      return 0.0

  def _has_rotation_fcurves(ob):
    ad = getattr(ob, "animation_data", None)
    if ad is None:
      return False
    action = getattr(ad, "action", None)
    if action is None:
      return False
    for fc in action.fcurves:
      if fc.data_path in ("rotation_euler", "rotation_quaternion"):
        return True
    return False

  def _has_only_visibility_fcurves(ob):
    ad = getattr(ob, "animation_data", None)
    if ad is None:
      return False
    action = getattr(ad, "action", None)
    if action is None:
      return False
    for fc in action.fcurves:
      if fc.data_path not in ("VISIBLE", "hide_viewport"):
        return False
    return len(action.fcurves) > 0

  def _get_root_name(obj):
    root = obj
    while getattr(root, 'parent', None) is not None:
      root = root.parent
    return getattr(root, 'name', '')

  def _is_argvis_empty(ob):
    return str(ob.get("_iedm_dbg_tf_cls", "")) == "ArgVisibilityNode"

  fixed_count = 0
  skipped_rotation_fcurves = 0
  still_broken = 0
  parent_fixed = 0
  visibility_only_count = 0
  cleared_anim_count = 0
  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "type", "") != "MESH":
      continue

    try:
      world_mat = ob.matrix_world.copy()
      world_x_deg = _mat_to_euler_x_deg(world_mat)

      if world_x_deg >= 5.0:
        continue

      has_rot_fcurves = _has_rotation_fcurves(ob)

      if has_rot_fcurves:
        parent = getattr(ob, 'parent', None)
        root_name = _get_root_name(ob)

        if parent is not None and _is_argvis_empty(parent) and root_name == "_EDMFileRoot":
          if not _has_rotation_fcurves(parent):
            parent.matrix_basis = _ROOT_BASIS_FIX @ parent.matrix_basis
            bpy.context.view_layer.update()
            world_x_deg2 = _mat_to_euler_x_deg(ob.matrix_world)
            if world_x_deg2 > 5.0:
              parent_fixed += 1
              fixed_count += 1
              _log.debug("mesh_fix (rot_fcurves parent): {} world_x={:.2f} -> {:.2f}deg".format(ob.name, world_x_deg, world_x_deg2), level=1)
              continue
            else:
              parent.matrix_basis = _ROOT_BASIS_FIX.inverted() @ parent.matrix_basis

        skipped_rotation_fcurves += 1
        continue

      parent = getattr(ob, 'parent', None)
      root_name = _get_root_name(ob)
      has_vis = _has_only_visibility_fcurves(ob)

      if has_vis:
        visibility_only_count += 1

      if has_vis and parent is not None and _is_argvis_empty(parent) and root_name == "_EDMFileRoot":
        if not _has_rotation_fcurves(parent):
          parent.matrix_basis = _ROOT_BASIS_FIX @ parent.matrix_basis
          bpy.context.view_layer.update()
          world_x_deg2 = _mat_to_euler_x_deg(ob.matrix_world)
          if world_x_deg2 > 5.0:
            parent_fixed += 1
            fixed_count += 1
            continue
          else:
            parent.matrix_basis = _ROOT_BASIS_FIX.inverted() @ parent.matrix_basis

      if has_vis and not _has_rotation_fcurves(ob):
        ad = getattr(ob, "animation_data", None)
        if ad is not None:
          action = getattr(ad, "action", None)
          nla_tracks = list(getattr(ad, "nla_tracks", []) or [])
          if action is not None:
            ob.animation_data_clear()
            bpy.context.view_layer.update()
            new_world = parent.matrix_world @ _ROOT_BASIS_FIX if parent else _ROOT_BASIS_FIX
            ob.matrix_world = new_world
            bpy.context.view_layer.update()
            world_x_deg2 = _mat_to_euler_x_deg(ob.matrix_world)
            if world_x_deg2 > 5.0:
              cleared_anim_count += 1
              fixed_count += 1
              _log.debug("mesh_fix (cleared_anim): {} world_x={:.2f} -> {:.2f}deg".format(ob.name, world_x_deg, world_x_deg2), level=1)
            else:
              parent_world_x = _mat_to_euler_x_deg(parent.matrix_world) if parent else 0
              _log.debug("mesh_fix: {} FAILED world_x={:.2f} -> after_fix={:.2f}deg (parent_world={:.2f})".format(ob.name, world_x_deg, world_x_deg2, parent_world_x), level=1)
              still_broken += 1
              ob.animation_data_create()
              ad = ob.animation_data
              ad.action = action
              for track in nla_tracks:
                new_track = ad.nla_tracks.new()
                for strip in track.strips:
                  new_track.strips.new(strip.name, strip.frame_start, strip.action)
            continue

      if not has_vis:
        ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
        bpy.context.view_layer.update()
        world_x_deg2 = _mat_to_euler_x_deg(ob.matrix_world)
        if world_x_deg2 > 5.0:
          fixed_count += 1
          _log.debug("mesh_fix (direct): {} world_x={:.2f} -> {:.2f}deg".format(ob.name, world_x_deg, world_x_deg2), level=1)
        else:
          still_broken += 1
          _log.debug("mesh_fix (direct): {} FAILED world_x={:.2f} -> {:.2f}deg".format(ob.name, world_x_deg, world_x_deg2), level=1)
      else:
        still_broken += 1
    except Exception as e:
      _log.warn("_apply_scene_root_mesh_orientation_fix", exc=e)

  _log.debug(
    "_apply_scene_root_mesh_orientation_fix: fixed {} ({} via parent, {} cleared anim), skipped {} rotation, {} still broken, {} visibility-only".format(
      fixed_count,
      parent_fixed,
      cleared_anim_count,
      skipped_rotation_fcurves,
      still_broken,
      visibility_only_count,
    ),
    level=1,
  )


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
      print("[iEDM][BTFIX] target={} mesh={} basis_loc=({:.4f},{:.4f},{:.4f}) parent={}".format(
        target.name, ob.name,
        _basis_loc.x, _basis_loc.y, _basis_loc.z,
        getattr(getattr(target, "parent", None), "name", None),
      ))
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


def _apply_scene_root_empty_orientation_fix():
  """Post-pass: fix scene-root empty objects at wrong world orientation.

  Empty objects at identity world X rotation (not driven by rotation fcurves)
  need _ROOT_BASIS_FIX applied.
  """
  if _import_ctx.edm_version < 10:
    return
  if getattr(_import_ctx, "bonetransform_prefix_matrix", None) is not None:
    return
  if not _import_profile_flag("scene_root_world_orientation_postfix"):
    return
  if not getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  bpy.context.view_layer.update()

  def _mat_to_euler_x_deg(mat):
    try:
      _, rot, _ = mat.decompose()
      return abs(math.degrees(rot.to_euler("XYZ").x))
    except:
      return 0.0

  def _has_rotation_fcurves(ob):
    ad = getattr(ob, "animation_data", None)
    if ad is None:
      return False
    action = getattr(ad, "action", None)
    if action is None:
      return False
    for fc in action.fcurves:
      if fc.data_path in ("rotation_euler", "rotation_quaternion"):
        return True
    return False

  fixed_count = 0
  skipped_rotation_fcurves = 0
  still_broken = 0
  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "type", "") != "EMPTY":
      continue
    if ob.get("_iedm_bt_prefix"):
      continue

    try:
      world_mat = ob.matrix_world.copy()
      world_x_deg = _mat_to_euler_x_deg(world_mat)

      if world_x_deg < 5.0:
        if _has_rotation_fcurves(ob):
          skipped_rotation_fcurves += 1
          continue

        saved_basis = ob.matrix_basis.copy()
        ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
        bpy.context.view_layer.update()
        world_x_deg2 = _mat_to_euler_x_deg(ob.matrix_world)

        if world_x_deg2 > 5.0:
          _log.debug("scene_root_empty_fix: {} world_x={:.2f}deg -> after_fix={:.2f}deg".format(ob.name, world_x_deg, world_x_deg2), level=1)
          fixed_count += 1
        else:
          ob.matrix_basis = saved_basis
          still_broken += 1
    except Exception as e:
      _log.warn("_apply_scene_root_empty_orientation_fix", exc=e)

  _log.debug(
    "_apply_scene_root_empty_orientation_fix: fixed {}, skipped {} rotation, {} still broken".format(
      fixed_count,
      skipped_rotation_fcurves,
      still_broken,
    ),
    level=1,
  )
