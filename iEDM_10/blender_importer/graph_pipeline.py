# Fragment: graph debug utilities, action helpers, and collection assignment.
# Graph construction lives in graph_build.py; bbox utilities in bbox_utils.py.

import math

import bpy
from mathutils import Matrix, Quaternion, Vector

from ..edm_format.types import AnimatingNode, ArgVisibilityNode, TransformNode
from .prelude import _import_ctx, _is_connector_transform, _log, _transform_display_name


# ---------------------------------------------------------------------------
# Transform debug utilities
# ---------------------------------------------------------------------------

def _reset_transform_debug(options):
  options = options or {}
  enabled = bool(options.get("debug_transforms", False))
  raw_limit = options.get("debug_transform_limit", 200)
  try:
    limit = max(1, int(raw_limit))
  except Exception:
    limit = 200
  name_filter = options.get("debug_transform_filter", None)
  if isinstance(name_filter, str) and not name_filter.strip():
    name_filter = None
  _import_ctx.transform_debug = {
    "enabled": enabled,
    "filter": name_filter,
    "limit": limit,
    "emitted": 0,
    "log_path": None,
    "chain_dumped": set(),
  }
  if enabled:
    filter_str = " filter='{}'".format(name_filter) if name_filter else ""
    print("Info: Transform debug enabled{} limit={}".format(filter_str, limit))


def _reset_mesh_origin_mode(options):
  options = options or {}
  mode = str(options.get("mesh_origin_mode", "APPROX")).upper()
  if mode not in {"APPROX", "RAW"}:
    mode = "APPROX"
  _import_ctx.mesh_origin_mode = mode


def _debug_node_label(node):
  if getattr(node, "transform", None):
    tf = node.transform
    name = _transform_display_name(tf) or type(tf).__name__
    return "tf:{}<{}>".format(name, type(tf).__name__)
  if getattr(node, "render", None):
    rn = node.render
    name = getattr(rn, "name", "") or type(rn).__name__
    return "rn:{}<{}>".format(name, type(rn).__name__)
  if getattr(node, "blender", None):
    return "bl:{}<{}>".format(node.blender.name, node.blender.type)
  return "<ROOT>"


def _debug_node_path(node):
  parts = []
  current = node
  while current is not None:
    parts.append(_debug_node_label(current))
    current = getattr(current, "parent", None)
  return " / ".join(reversed(parts))


def _debug_fmt_vec3(vec):
  return "({:+.6f}, {:+.6f}, {:+.6f})".format(vec.x, vec.y, vec.z)


def _debug_fmt_rot_deg(rot):
  euler = rot.to_euler("XYZ")
  return "({:+.3f}, {:+.3f}, {:+.3f})".format(
    math.degrees(euler.x), math.degrees(euler.y), math.degrees(euler.z)
  )


def _debug_fmt_trs(mat):
  loc, rot, scale = mat.decompose()
  return _debug_fmt_vec3(loc), _debug_fmt_rot_deg(rot), _debug_fmt_vec3(scale)


def _debug_filter_terms():
  dbg = getattr(_import_ctx, "transform_debug", {}) or {}
  needle = dbg.get("filter")
  if not needle:
    return []
  if isinstance(needle, str):
    return [term.strip().lower() for term in needle.split(",") if term.strip()]
  return [str(needle).strip().lower()]


def _anim_vector_to_blender(v):
  return Vector(v)


def _anim_quaternion_to_blender(q):
  return q if hasattr(q, "to_matrix") else Quaternion(q)


def _is_neg90_x_basis_matrix(mat, eps=1e-3):
  try:
    target = (
      (1.0, 0.0, 0.0),
      (0.0, 0.0, 1.0),
      (0.0, -1.0, 0.0),
    )
    for r in range(3):
      for c in range(3):
        if abs(float(mat[r][c]) - target[r][c]) > eps:
          return False
    return True
  except Exception:
    return False


def _anim_scale_components(value):
  if len(value) < 3:
    return (1.0, 1.0, 1.0)
  return (value[0], value[1], value[2])


def _expected_local_matrix(tfnode, blender_obj=None):
  if tfnode is None:
    return None
  if isinstance(tfnode, TransformNode):
    is_connector = _is_connector_transform(tfnode, blender_obj)
    local_mat = Matrix(tfnode.matrix)
    if is_connector:
      local_mat = local_mat @ Matrix.Rotation(math.radians(-90.0), 4, "X")
    if (
      blender_obj is not None
      and blender_obj.parent is not None
      and _import_ctx.edm_version >= 10
    ):
      local_mat = blender_obj.parent.matrix_basis @ local_mat
    return local_mat
  if isinstance(tfnode, AnimatingNode):
    local_mat = None
    if hasattr(tfnode, "zero_transform_local_matrix"):
      local_mat = tfnode.zero_transform_local_matrix
    elif hasattr(tfnode, "zero_transform_matrix"):
      local_mat = tfnode.zero_transform_matrix
    elif hasattr(tfnode, "zero_transform"):
      loc, rot, scale = tfnode.zero_transform
      local_mat = Matrix.LocRotScale(loc, rot, scale)
    if local_mat is not None and blender_obj is not None and blender_obj.parent is not None and _import_ctx.edm_version >= 10:
      local_mat = blender_obj.parent.matrix_basis @ local_mat
    return local_mat
  return None


def _debug_filter_match(node, path):
  name_filter = _import_ctx.transform_debug.get("filter")
  if not name_filter:
    return True
  needle = name_filter.lower()
  candidates = [path]
  if getattr(node, "blender", None):
    candidates.append(node.blender.name)
  if getattr(node, "transform", None):
    candidates.append(getattr(node.transform, "name", ""))
  if getattr(node, "render", None):
    candidates.append(getattr(node.render, "name", ""))
  return any(needle in (item or "").lower() for item in candidates)


def _debug_dump_node_transform(node):
  if not _import_ctx.transform_debug.get("enabled"):
    return
  if _import_ctx.transform_debug["emitted"] >= _import_ctx.transform_debug["limit"]:
    return
  if not getattr(node, "blender", None):
    return

  path = _debug_node_path(node)
  if not _debug_filter_match(node, path):
    return

  obj = node.blender
  bpy.context.view_layer.update()

  edm_local = _expected_local_matrix(getattr(node, "transform", None), obj)
  blender_local = obj.matrix_local.copy()
  blender_basis = obj.matrix_basis.copy()
  blender_world = obj.matrix_world.copy()
  parent_name = obj.parent.name if obj.parent else "<none>"

  print("[iEDM][TFDBG] {}".format(path))
  print("  object={} type={} parent={}".format(obj.name, obj.type, parent_name))

  if edm_local is not None:
    loc, rot, scale = _debug_fmt_trs(edm_local)
    print("  edm_local  loc={} rot_deg={} scale={}".format(loc, rot, scale))
  else:
    print("  edm_local  <none>")

  loc, rot, scale = _debug_fmt_trs(blender_local)
  print("  bl_localM  loc={} rot_deg={} scale={}".format(loc, rot, scale))
  loc, rot, scale = _debug_fmt_trs(blender_basis)
  print("  bl_basis   loc={} rot_deg={} scale={}".format(loc, rot, scale))
  loc, rot, scale = _debug_fmt_trs(blender_world)
  print("  bl_world   loc={} rot_deg={} scale={}".format(loc, rot, scale))

  if edm_local is not None:
    ed_loc, ed_rot, ed_scale = edm_local.decompose()
    bl_loc, bl_rot, bl_scale = blender_basis.decompose()
    loc_error = (bl_loc - ed_loc).length
    scale_error = (bl_scale - ed_scale).length
    rot_error = math.degrees(ed_rot.rotation_difference(bl_rot).angle)
    print("  local_err  loc={:.6g} rot_deg={:.6g} scale={:.6g}".format(
      loc_error, rot_error, scale_error
    ))

  _import_ctx.transform_debug["emitted"] += 1


# ---------------------------------------------------------------------------
# Semantic name mapping (populated by import capabilities)
# ---------------------------------------------------------------------------

_SEMANTIC_NAME_MAP = {}


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _get_action_argument(action):
  if action is None:
    return None
  try:
    if hasattr(action, "argument"):
      arg = int(getattr(action, "argument"))
      if arg >= 0:
        return arg
  except Exception as e:
    _log.warn("action.argument read: {}".format(e), exc=e)
  try:
    prefix = str(getattr(action, "name", "")).split("_", 1)[0]
    if prefix and (prefix.isdigit() or (prefix.startswith("-") and prefix[1:].isdigit())):
      return int(prefix)
  except Exception as e:
    _log.warn("action name prefix parse: {}".format(e), exc=e)
  return None


def _copy_fcurve_points(src_curve, dst_curve):
  for kp in src_curve.keyframe_points:
    try:
      frame = float(kp.co[0])
      value = float(kp.co[1])
    except Exception:
      continue
    new_kp = dst_curve.keyframe_points.insert(frame, value, options={'FAST'})
    try:
      new_kp.interpolation = kp.interpolation
      new_kp.handle_left_type = kp.handle_left_type
      new_kp.handle_right_type = kp.handle_right_type
      new_kp.easing = kp.easing
    except Exception as e:
      _log.debug("fcurve interpolation copy: {}".format(e), level=2)


def _merge_actions_by_argument(actions):
  """Merge multiple source actions into one action per EDM argument."""
  if not actions or len(actions) <= 1:
    return actions

  grouped = {}
  order = []
  for action in actions:
    arg = _get_action_argument(action)
    key = ("arg", arg) if arg is not None else ("name", getattr(action, "name", ""))
    if key not in grouped:
      grouped[key] = []
      order.append(key)
    grouped[key].append(action)

  if all(len(grouped[k]) == 1 for k in order):
    return actions

  merged_actions = []
  for key in order:
    src_actions = grouped[key]
    if len(src_actions) == 1:
      merged_actions.append(src_actions[0])
      continue

    base = src_actions[0]
    merged = bpy.data.actions.new(base.name)
    arg = key[1] if key[0] == "arg" else None
    if arg is not None and hasattr(merged, "argument"):
      merged.argument = int(arg)

    for src in src_actions:
      for src_curve in src.fcurves:
        dst_curve = merged.fcurves.find(src_curve.data_path, index=src_curve.array_index)
        if dst_curve is None:
          dst_curve = merged.fcurves.new(data_path=src_curve.data_path, index=src_curve.array_index)
        _copy_fcurve_points(src_curve, dst_curve)

    merged_actions.append(merged)

  return merged_actions


def _push_action_to_nla(ob, action):
  """Push action onto an NLA track when supported."""
  if ob is None:
    return False
  if not ob.animation_data:
    ob.animation_data_create()
  try:
    track = ob.animation_data.nla_tracks.new()
    track.name = action.name
    strip = track.strips.new(action.name, 0, action)
    strip.extrapolation = 'HOLD'
    if "Visib" in action.name:
      strip.blend_type = 'COMBINED'
    else:
      strip.blend_type = 'REPLACE'
    return True
  except Exception as e:
    _log.warn("NLA push for '{}' on '{}': {}".format(
      getattr(action, "name", "<action>"),
      getattr(ob, "name", "<unknown>"), e), exc=e)
    return False


# ---------------------------------------------------------------------------
# Collection assignment
# ---------------------------------------------------------------------------

def _assign_collections(graph):
  """Create named import collections and assign Blender objects to them.

  Rules:
  - ShellNode / SegmentsNode meshes and their ancestor empties -> "Collision"
  - Animated/hierarchical render meshes -> "Vehicle"
  - Identity-style root render helpers -> "Texture_Animation"
  - Other nodes stay in Scene Collection
  - Empties and connectors inherit the category of their nearby branch
  """
  scene = bpy.context.scene

  def _get_or_create_child_col(parent, name):
    col = bpy.data.collections.get(name)
    if col is None:
      col = bpy.data.collections.new(name)
    if parent is not None and col.name not in parent.children.keys():
      parent.children.link(col)
    return col

  col_vehicle = _get_or_create_child_col(scene.collection, "Vehicle")
  col_collision = _get_or_create_child_col(col_vehicle, "Collision")
  col_tex_anim = _get_or_create_child_col(scene.collection, "Texture_Animation")

  obj_category = {}

  for n in graph.nodes:
    if not getattr(n, "_is_primary", False) or not n.blender:
      continue
    if n.render is None:
      obj_category[n.blender] = None
      continue
    rtype = type(n.render).__name__
    if rtype in ("ShellNode", "SegmentsNode"):
      obj_category[n.blender] = "collision"
    elif rtype == "Connector":
      obj_category[n.blender] = "vehicle"
    elif rtype == "RenderNode":
      if n.blender.parent is None:
        render_local = Matrix.Identity(4)
        try:
          if getattr(n, "_local_bl", None) is not None:
            render_local = Matrix(n._local_bl)
        except Exception:
          render_local = Matrix.Identity(4)
        try:
          if hasattr(n.render, "matrix"):
            render_local = render_local @ Matrix(n.render.matrix)
          elif hasattr(n.render, "pos"):
            render_local = render_local @ Matrix.Translation(Vector(n.render.pos[:3]))
        except Exception:
          pass
        render_name = str(getattr(n.render, "name", "") or "")
        obj_category[n.blender] = "texture_anim" if (not render_name and render_local.is_identity) else None
      else:
        obj_category[n.blender] = None if isinstance(getattr(n, "transform", None), ArgVisibilityNode) else "vehicle"
    else:
      obj_category[n.blender] = None

  # Resolve empties based on children categories
  changed = True
  while changed:
    changed = False
    for n in graph.nodes:
      if not getattr(n, "_is_primary", False) or not n.blender:
        continue
      if obj_category.get(n.blender) is not None:
        continue
      child_cats = set()
      for child_obj in n.blender.children:
        cat = obj_category.get(child_obj)
        if cat:
          child_cats.add(cat)
      if "collision" in child_cats and "vehicle" not in child_cats:
        obj_category[n.blender] = "collision"
        changed = True
      elif "vehicle" in child_cats or "texture_anim" in child_cats:
        obj_category[n.blender] = "vehicle"
        changed = True

  # Helper empties inserted after graph construction follow their categorized parent
  changed = True
  while changed:
    changed = False
    for obj, cat in list(obj_category.items()):
      if cat is None:
        continue
      for child in list(obj.children):
        if obj_category.get(child) is not None:
          continue
        if child.get("_iedm_identity_passthrough") or child.get("_iedm_vis_passthrough") or getattr(child, "type", "") == "EMPTY":
          obj_category[child] = cat
          changed = True

  _col_map = {
    "collision": col_collision,
    "vehicle": col_vehicle,
    "texture_anim": col_tex_anim,
  }

  # Move _EDMFileRoot to Vehicle when all categorized children are vehicle content
  if graph.root.blender:
    root_name = getattr(graph.root.blender, 'name', '')
    if root_name == '_EDMFileRoot' and graph.root.blender.parent is None:
      child_cats = set()
      for child_obj in graph.root.blender.children:
        cat = obj_category.get(child_obj)
        if cat:
          child_cats.add(cat)
      if "vehicle" in child_cats and "collision" not in child_cats:
        obj_category[graph.root.blender] = "vehicle"

  for obj, cat in obj_category.items():
    target = _col_map.get(cat)
    if target is None:
      continue
    for cur_col in list(obj.users_collection):
      try:
        cur_col.objects.unlink(obj)
      except Exception as e:
        _log.warn("collection unlink '{}': {}".format(
          getattr(obj, "name", "<unknown>"), e), exc=e)
    try:
      target.objects.link(obj)
    except RuntimeError:
      pass

  def _exclude_layer_collection(layer_collection, name):
    if layer_collection.collection.name == name:
      layer_collection.exclude = True
      return True
    for child in layer_collection.children:
      if _exclude_layer_collection(child, name):
        return True
    return False

  _exclude_layer_collection(bpy.context.view_layer.layer_collection, "Texture_Animation")
