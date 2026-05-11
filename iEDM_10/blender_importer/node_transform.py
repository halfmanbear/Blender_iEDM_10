# Fragment: apply_node_transform — assigns the local matrix to a Blender object.

import math

from mathutils import Matrix, Vector

from ..edm_format.types import AnimatingNode, ArgAnimationNode, Connector, TransformNode
from .animation import _is_pos90_x_basis_matrix, _normalize_euler_xyz
from .graph_pipeline import _debug_filter_terms, _debug_fmt_rot_deg, _debug_fmt_vec3, _is_neg90_x_basis_matrix
from .prelude import (
  _ROOT_BASIS_FIX,
  _import_ctx,
  _import_profile_flag,
  _is_child_of_file_root,
  _is_connector_object,
  _is_connector_transform,
  _log,
)


def _transform_uses_quaternion_rotation(tfnode, obj=None):
  try:
    if obj is not None and _is_connector_object(obj):
      return False
  except Exception:
    pass
  try:
    action = getattr(getattr(obj, "animation_data", None), "action", None)
    if action is not None:
      if any(fc.data_path == "rotation_euler" for fc in action.fcurves):
        return False
      if any(fc.data_path == "rotation_quaternion" for fc in action.fcurves):
        return True
  except Exception:
    pass
  try:
    if isinstance(tfnode, ArgAnimationNode) and getattr(tfnode, "rotData", None):
      return True
  except Exception:
    pass
  return False


def apply_node_transform(node, obj, used_shared_parent=False):
  """Assigns the transform to a given node. node can be a TranslationNode or raw EDM node."""
  tnode = node if hasattr(node, "transform") else None
  tfnode = tnode.transform if tnode else node

  render = tnode.render if tnode else getattr(node, "render", None)
  if tfnode is None and render is None:
    return

  wants_quaternion_rotation = _transform_uses_quaternion_rotation(tfnode, obj) if tfnode is not None else False
  obj.rotation_mode = "QUATERNION" if wants_quaternion_rotation else "XYZ"

  def _debug_set_trace(source_label, local_mat):
    if not _import_ctx.transform_debug.get("enabled"):
      return
    node_name = getattr(tfnode, "name", "") or type(tfnode).__name__
    filter_terms = _debug_filter_terms()
    if filter_terms:
      if not any(term in node_name.lower() or term in obj.name.lower() for term in filter_terms):
        return
    loc, rot, scale = local_mat.decompose()
    print("[iEDM][TFSET] {} src={} obj={} loc={} rot_deg={} scale={}".format(
      node_name, source_label, obj.name, _debug_fmt_vec3(loc), _debug_fmt_rot_deg(rot), _debug_fmt_vec3(scale)
    ))

  def _debug_dump_parent_chain(local_mat):
    if not _import_ctx.transform_debug.get("enabled"):
      return
    filter_terms = _debug_filter_terms()
    if not filter_terms:
      return
    node_name = getattr(tfnode, "name", "") or type(tfnode).__name__
    if not any(term in node_name.lower() or term in obj.name.lower() for term in filter_terms):
      return
    dumped = _import_ctx.transform_debug.setdefault("chain_dumped", set())
    dump_key = "{}::{}".format(node_name, obj.name)
    if dump_key in dumped:
      return
    dumped.add(dump_key)

    print("[iEDM][CHAIN] begin node={} obj={} graph_parent={} blender_parent={}".format(
      node_name,
      obj.name,
      getattr(getattr(getattr(tnode, "parent", None), "transform", None), "name", "<ROOT>") if tnode is not None else "<ROOT>",
      getattr(getattr(obj, "parent", None), "name", None),
    ))
    if tnode is not None:
      cur = tnode
      graph_parts = []
      while cur is not None:
        tf_cur = getattr(cur, "transform", None)
        rn_cur = getattr(cur, "render", None)
        if tf_cur is not None:
          label = "{}<{}>".format(getattr(tf_cur, "name", "") or type(tf_cur).__name__, type(tf_cur).__name__)
        elif rn_cur is not None:
          label = "{}<{}>".format(getattr(rn_cur, "name", "") or type(rn_cur).__name__, type(rn_cur).__name__)
        else:
          label = "<ROOT>"
        graph_parts.append(label)
        cur = getattr(cur, "parent", None)
      print("[iEDM][CHAIN] graph_path={}".format(" / ".join(reversed(graph_parts))))
    try:
      loc, rot, scale = local_mat.decompose()
      print("[iEDM][CHAIN] assigned_local loc={} rot_deg={} scale={}".format(
        _debug_fmt_vec3(loc), _debug_fmt_rot_deg(rot), _debug_fmt_vec3(scale)
      ))
    except Exception:
      pass

    current = obj
    level = 0
    while current is not None:
      try:
        basis_loc, basis_rot, basis_scale = current.matrix_basis.decompose()
        world_loc, world_rot, world_scale = current.matrix_world.decompose()
        print("[iEDM][CHAIN] level={} obj={} type={} parent={} basis_loc={} basis_rot_deg={} basis_scale={} world_loc={} world_rot_deg={} world_scale={}".format(
          level,
          current.name,
          getattr(current, "type", ""),
          getattr(getattr(current, "parent", None), "name", None),
          _debug_fmt_vec3(basis_loc),
          _debug_fmt_rot_deg(basis_rot),
          _debug_fmt_vec3(basis_scale),
          _debug_fmt_vec3(world_loc),
          _debug_fmt_rot_deg(world_rot),
          _debug_fmt_vec3(world_scale),
        ))
      except Exception as e:
        print("[iEDM][CHAIN] level={} obj={} error={}".format(level, getattr(current, "name", "<unknown>"), e))
      current = getattr(current, "parent", None)
      level += 1
    print("[iEDM][CHAIN] end node={} obj={}".format(node_name, obj.name))

  def _set_local_matrix(local_mat):
    try:
      obj.matrix_basis = local_mat
      return
    except Exception as e:
      _log.warn("matrix_basis assign for '{}': {}".format(
        getattr(obj, "name", "<unknown>"), e), exc=e)
    loc, rot, scale = local_mat.decompose()
    obj.location = loc
    if wants_quaternion_rotation:
      obj.rotation_mode = "QUATERNION"
      obj.rotation_quaternion = rot
    else:
      obj.rotation_mode = "XYZ"
      obj.rotation_euler = _normalize_euler_xyz(rot.to_euler('XYZ'))
    obj.scale = scale

  def _compose_with_parent_if_enabled(local_mat):
    if _import_ctx.legacy_v10_parent_compose and obj.parent is not None and _import_ctx.edm_version >= 10:
      try:
        return obj.parent.matrix_basis @ local_mat
      except Exception:
        return local_mat
    return local_mat

  def _is_plain_root_connector_wrapper(graph_node, blender_obj):
    if not _import_profile_flag("plain_root_connector_basis_fix"):
      return False
    if _import_ctx.edm_version < 10:
      return False
    if getattr(_import_ctx, "use_scene_root_basis_object", True):
      return False
    if blender_obj is None or getattr(blender_obj, "parent", None) is not None:
      return False
    if getattr(blender_obj, "type", "") != "EMPTY":
      return False
    try:
      children = list(getattr(graph_node, "children", []) or [])
    except Exception:
      return False
    return any(
      isinstance(getattr(ch, "render", None), Connector)
      or _is_connector_object(getattr(ch, "blender", None))
      for ch in children
    )

  def _is_plain_root_connector_child(graph_node, blender_obj):
    if not _import_profile_flag("plain_root_connector_child_basis_fix"):
      return False
    if _import_ctx.edm_version < 10:
      return False
    if getattr(_import_ctx, "use_scene_root_basis_object", True):
      return False
    if graph_node is None or blender_obj is None:
      return False
    if getattr(blender_obj, "parent", None) is None:
      return False
    if not _is_connector_object(blender_obj):
      return False
    return _is_child_of_file_root(graph_node)

  # 1. Base Transform from Node.transform
  final_local = Matrix(node._local_bl) if (tfnode is None and hasattr(node, "_local_bl") and getattr(node, "_local_bl", None) is not None) else Matrix.Identity(4)
  if tfnode is None and _is_connector_object(obj) and obj.parent is None:
    try:
      flat = obj.get("_iedm_connector_tf_matrix", None)
      if flat is not None and len(flat) == 16:
        connector_local = Matrix((
          tuple(float(v) for v in flat[0:4]),
          tuple(float(v) for v in flat[4:8]),
          tuple(float(v) for v in flat[8:12]),
          tuple(float(v) for v in flat[12:16]),
        ))
        loc = connector_local.to_translation()
        rot_mat = connector_local.to_3x3().to_4x4()
        if _is_pos90_x_basis_matrix(rot_mat):
          rot_mat = Matrix.Identity(4)
        elif _is_neg90_x_basis_matrix(rot_mat):
          rot_mat = Matrix.Identity(4)
        if (
          _import_profile_flag("plain_root_connector_basis_fix")
          and obj.parent is None
          and _import_ctx.edm_version >= 10
          and not getattr(_import_ctx, "use_scene_root_basis_object", True)
        ):
          loc = (_ROOT_BASIS_FIX @ Matrix.Translation(loc)).to_translation()
        final_local = Matrix.Translation(loc) @ rot_mat
    except Exception as e:
      _log.warn("connector tf-matrix parse for '{}': {}".format(
        getattr(obj, "name", "<unknown>"), e), exc=e)
  if isinstance(tfnode, TransformNode):
    is_connector_obj = _is_connector_transform(tfnode, obj)
    is_connector_wrapper = _is_plain_root_connector_wrapper(tnode, obj)
    is_connector_child = _is_plain_root_connector_child(tnode, obj)
    raw_local_mat = Matrix(tfnode.matrix)
    local_mat = Matrix(raw_local_mat)
    if is_connector_obj or is_connector_wrapper or is_connector_child:
      if (
        _import_profile_flag("plain_root_connector_basis_fix")
        and obj.parent is None
        and _import_ctx.edm_version >= 10
        and not getattr(_import_ctx, "use_scene_root_basis_object", True)
      ):
        local_mat = _ROOT_BASIS_FIX @ local_mat
      elif is_connector_child:
        local_mat = _ROOT_BASIS_FIX @ local_mat
      elif (
        is_connector_wrapper
        and obj.parent is not None
        and _import_ctx.edm_version >= 10
        and _is_neg90_x_basis_matrix(raw_local_mat.to_3x3().to_4x4())
      ):
        try:
          parent_rot = obj.parent.matrix_world.to_3x3().to_4x4()
        except Exception:
          parent_rot = Matrix.Identity(4)
        if _is_pos90_x_basis_matrix(parent_rot):
          local_mat = Matrix.Translation(raw_local_mat.to_translation())
      if is_connector_obj:
        local_mat = local_mat @ Matrix.Rotation(math.radians(-90.0), 4, "X")
      elif not is_connector_wrapper and not _is_neg90_x_basis_matrix(raw_local_mat.to_3x3().to_4x4()):
        local_mat = local_mat @ Matrix.Rotation(math.radians(-90.0), 4, "X")
    final_local = _compose_with_parent_if_enabled(local_mat)
  elif isinstance(tfnode, AnimatingNode):
    if hasattr(tfnode, "zero_transform_local_matrix"):
      final_local = _compose_with_parent_if_enabled(tfnode.zero_transform_local_matrix)
    elif hasattr(tfnode, "zero_transform_matrix"):
      final_local = _compose_with_parent_if_enabled(tfnode.zero_transform_matrix)
    elif hasattr(tfnode, "zero_transform"):
      loc, rot, scale = tfnode.zero_transform
      final_local = _compose_with_parent_if_enabled(Matrix.LocRotScale(loc, rot, scale))

    wrapper_offset = getattr(tfnode, "wrapper_rest_translation_matrix", None)
    if wrapper_offset is not None and tnode is not None:
      parent_graph = getattr(tnode, "parent", None)
      parent_obj = getattr(parent_graph, "blender", None) if parent_graph is not None else None
      if parent_obj is not None and not getattr(tfnode, "_wrapper_rest_translation_applied", False):
        try:
          parent_obj.matrix_basis = parent_obj.matrix_basis @ wrapper_offset
          tfnode._wrapper_rest_translation_applied = True
        except Exception as e:
          _log.warn("wrapper_rest_translation apply for '{}': {}".format(
            getattr(parent_obj, "name", "<unknown>"), e), exc=e)

  # 2. Inherit RenderNode local offset if present
  if render and tfnode is not None and not _is_connector_object(obj):
    m_rn = Matrix.Identity(4)
    if hasattr(render, "pos"):
      m_rn = Matrix.Translation(Vector(render.pos[:3]))
    elif hasattr(render, "matrix"):
      m_rn = Matrix(render.matrix)

    if not m_rn.is_identity:
      final_local = final_local @ m_rn

  _set_local_matrix(final_local)
  _debug_set_trace(type(tfnode).__name__, final_local)
  _debug_dump_parent_chain(final_local)
