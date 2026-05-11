import bpy
from mathutils import Matrix

from ..edm_format.types import ArgScaleNode, ArgVisibilityNode
from .prelude import (
  _ROOT_BASIS_FIX,
  _import_ctx,
  _import_profile_flag,
  _is_root_visibility_pair_child,
  _is_static_root_visibility_wrapper,
  _log,
)


def _reparent_preserve_world(child, new_parent):
  if child is None:
    return
  world = child.matrix_world.copy()
  child.parent = new_parent
  child.matrix_parent_inverse = Matrix.Identity(4)
  child.matrix_world = world


def _fix_owner_encoded_render_offsets(graph):
  """Fix owner-encoded render chunks whose shared_parent differs from graph parent."""
  def _accum_world_basis(obj):
    mats = []
    o = obj
    while o is not None:
      mats.append(o.matrix_basis.copy())
      o = o.parent
    result = Matrix.Identity(4)
    for m in reversed(mats):
      result = result @ m
    return result

  for node in graph.nodes:
    if node.render is None or node.blender is None:
      continue
    if getattr(node.blender, "type", "") != "MESH":
      continue
    if node.transform is not None and not isinstance(node.transform, ArgVisibilityNode):
      continue
    if getattr(node.render, "split_owner_encoded", False):
      continue
    shared_parent = getattr(node.render, "shared_parent", None)
    if shared_parent is None:
      continue
    parent_transform = getattr(getattr(node, "parent", None), "transform", None)
    if shared_parent is parent_transform:
      continue
    sp_obj = getattr(shared_parent, "_blender_obj", None)
    parent_obj = getattr(getattr(node, "parent", None), "blender", None)
    if sp_obj is None or parent_obj is None:
      continue
    try:
      relative = _accum_world_basis(parent_obj).inverted() @ _accum_world_basis(sp_obj)
      current_local = node.blender.matrix_basis.copy()
      node.blender.matrix_basis = relative @ current_local
      node.blender.pop("_iedm_identity_passthrough", None)
      node.blender.pop("_iedm_narrow_identity_passthrough", None)
    except Exception as e:
      _log.warn("_fix_owner_encoded_render_offsets", exc=e)


def _zero_render_child_mesh_locals_under_transform(graph):
  """Leave render-only mesh children at identity under explicit transform wrappers."""
  for node in getattr(graph, "nodes", []) or []:
    render = getattr(node, "render", None)
    ob = getattr(node, "blender", None)
    if render is None or ob is None or getattr(ob, "type", "") != "MESH":
      continue
    if getattr(node, "transform", None) is not None:
      continue
    parent = getattr(node, "parent", None)
    parent_tf = getattr(parent, "transform", None) if parent is not None else None
    if parent_tf is None or isinstance(parent_tf, ArgScaleNode):
      continue
    shared_parent = getattr(render, "shared_parent", None)
    if shared_parent is not None and shared_parent is not parent_tf:
      continue
    try:
      _loc, _rot, _scale = ob.matrix_basis.decompose()
      if abs(_rot.angle) > 1e-3:
        continue
      ob.matrix_basis = Matrix.Identity(4)
    except Exception as e:
      _log.warn("_zero_render_child_mesh_locals_under_transform", exc=e)


def _basis_is_identity(obj, eps=1e-6):
  try:
    m = obj.matrix_basis
    return all(
      abs(float(m[r][c]) - (1.0 if r == c else 0.0)) <= eps
      for r in range(4) for c in range(4)
    )
  except Exception:
    return False


def _apply_plain_root_visibility_object_basis_fix():
  if not _import_profile_flag("plain_root_visibility_object_basis_fix"):
    return
  if _import_ctx.edm_version < 10:
    return
  if getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "parent", None) is not None:
      continue
    if getattr(ob, "type", "") != "EMPTY":
      continue
    if not bool(ob.get("_iedm_vis_passthrough")):
      continue
    if not _basis_is_identity(ob):
      continue
    stack = list(getattr(ob, "children", []) or [])
    while stack:
      child = stack.pop()
      child_type = getattr(child, "type", "")
      if child_type == "EMPTY":
        if _basis_is_identity(child):
          stack.extend(list(getattr(child, "children", []) or []))
        continue
      if child_type != "MESH":
        continue
      if str(child.get("_iedm_dbg_r_cls", "") or "") != "RenderNode":
        continue
      if not _basis_is_identity(child):
        continue
      try:
        child.matrix_basis = _ROOT_BASIS_FIX @ child.matrix_basis
      except Exception as e:
        _log.warn("_apply_plain_root_visibility_object_basis_fix", exc=e)


def _apply_plain_root_visibility_mesh_basis_fix():
  if not _import_profile_flag("plain_root_visibility_mesh_basis_fix"):
    return
  if _import_ctx.edm_version < 10:
    return
  if getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  for ob in list(getattr(bpy.data, "objects", []) or []):
    if getattr(ob, "parent", None) is not None:
      continue
    if getattr(ob, "type", "") != "MESH":
      continue
    if str(ob.get("_iedm_dbg_r_cls", "") or "") != "RenderNode":
      continue
    if not ob.get("_iedm_src_vis_chain"):
      continue
    if not _basis_is_identity(ob):
      continue
    try:
      ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
    except Exception as e:
      _log.warn("_apply_plain_root_visibility_mesh_basis_fix", exc=e)


def _apply_root_visibility_pair_wrapper_basis_fix(graph):
  seen = set()
  for node in getattr(graph, "nodes", []) or []:
    if not _is_root_visibility_pair_child(node):
      continue
    parent = getattr(node, "parent", None)
    ob = getattr(parent, "blender", None)
    if ob is None:
      continue
    key = ob.name_full if hasattr(ob, "name_full") else id(ob)
    if key in seen:
      continue
    seen.add(key)
    try:
      ob.matrix_basis = _ROOT_BASIS_FIX.inverted() @ ob.matrix_basis
    except Exception as e:
      _log.warn("_apply_root_visibility_pair_wrapper_basis_fix", exc=e)


def _apply_static_root_visibility_wrapper_basis_fix(graph):
  # When mesh children receive the basis fix directly, do not also rotate their
  # EMPTY visibility wrapper parents.
  if _import_profile_flag("plain_root_visibility_object_basis_fix"):
    return
  if getattr(_import_ctx, "use_scene_root_basis_object", True):
    return

  seen_objects = set()
  for node in getattr(graph, "nodes", []) or []:
    if not _is_static_root_visibility_wrapper(node):
      continue
    ob = getattr(node, "blender", None)
    if ob is None:
      continue
    ob_key = ob.name_full if hasattr(ob, "name_full") else id(ob)
    if ob_key in seen_objects:
      continue
    seen_objects.add(ob_key)
    try:
      ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
    except Exception as e:
      _log.warn("_apply_static_root_visibility_wrapper_basis_fix", exc=e)
