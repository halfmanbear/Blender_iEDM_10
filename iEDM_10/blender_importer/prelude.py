"""
reader

Contains the central EDM->Blender conversion code
"""

import bpy
import bmesh

from ..utils import chdir, print_edm_graph
from ..edm_format import EDMFile
from ..edm_format.mathtypes import *
from ..edm_format.types import *

from ..translation import TranslationGraph, TranslationNode

import re
import glob
import fnmatch
import os
import itertools
import math
import json
from .classifier import (
  CLASS_3DSMAX_BANO,
  CLASS_3DSMAX_DEF,
  CLASS_BLENDER_DEF,
  CLASS_COLLISION_MESH,
  CLASS_BLOB_COLLISION,
  CLASS_BLOB_RENDER,
  CLASS_BLOB_SKELETAL,
  CLASS_UNKNOWN,
  CLASS_INVALID,
)
from .import_capabilities import DEFAULT_CAPABILITIES
from .import_profiles import DEFAULT_PROFILE, get_legacy_import_profile

_SUFFIX_RE = re.compile(r"\.\d{3}$")  # matches ".000" to ".999" at end


FRAME_SCALE = 200

# Official EDM exports wrap scene space in a root basis fix matrix.
_ROOT_BASIS_FIX = Matrix(((1, 0, 0, 0),
                          (0, 0, -1, 0),
                          (0, 1, 0, 0),
                          (0, 0, 0, 1)))


def _default_transform_debug_state():
  return {"enabled": False, "filter": None, "limit": 200, "emitted": 0, "log_path": None}


def _default_render_split_debug_state():
  return {"enabled": False}


def _default_visibility_debug_state():
  return {"enabled": False, "target": None, "log_path": None, "event_limit": 200, "events": 0}


import threading
class ImportContext(threading.local):
  """Mutable importer runtime state shared across reader modules."""

  def __init__(self):
    self.edm_version = 10
    self.edm_exporter_class = CLASS_UNKNOWN
    self.edm_exporter_detail = ""
    self.import_profile = DEFAULT_PROFILE
    self.import_capabilities = DEFAULT_CAPABILITIES
    self.transform_debug = _default_transform_debug_state()
    self.render_split_debug = _default_render_split_debug_state()
    self.visibility_debug = _default_visibility_debug_state()
    self.mesh_origin_mode = "APPROX"
    self.source_dir = None
    self.texture_search_cache = {}
    self.official_material_bridge = None
    self.bone_import_ctx = None
    self.legacy_v10_parent_compose = False
    self.file_has_bones = False
    self.use_scene_root_basis_object = True
    self.collision_geometry_basis_fix = False


def _import_family_is(*families):
  current = getattr(_import_ctx, "edm_exporter_class", CLASS_UNKNOWN)
  return current in set(families)


def _import_family_is_3dsmax_bano():
  return _import_family_is(CLASS_3DSMAX_BANO)


def _import_capability_name():
  caps = getattr(_import_ctx, "import_capabilities", None)
  if caps is not None:
    name = getattr(caps, "name", None)
    if name:
      return name
  return DEFAULT_CAPABILITIES.name


def _import_capability_detail():
  caps = getattr(_import_ctx, "import_capabilities", None)
  if caps is not None:
    detail = getattr(caps, "detail", "")
    if detail:
      return detail
  return ""


def _import_profile_name():
  profile = getattr(_import_ctx, "import_profile", DEFAULT_PROFILE)
  return getattr(profile, "name", DEFAULT_PROFILE.name)


def _import_profile_detail():
  profile = getattr(_import_ctx, "import_profile", DEFAULT_PROFILE)
  detail = str(getattr(profile, "description", "") or "")
  if detail:
    return detail
  return str(getattr(_import_ctx, "edm_exporter_detail", "") or "")


def _import_profile_flag(flag_name, default=False):
  caps = getattr(_import_ctx, "import_capabilities", None)
  if caps is not None:
    flags = getattr(caps, "flags", None) or {}
    if flag_name in flags:
      return bool(flags[flag_name])
  profile = getattr(_import_ctx, "import_profile", DEFAULT_PROFILE)
  return bool(getattr(profile, flag_name, default))


_import_ctx = ImportContext()


def _append_debug_log_line(line):
  dbg = getattr(_import_ctx, "visibility_debug", {}) or {}
  if not dbg.get("enabled"):
    return
  log_path = dbg.get("log_path")
  if not log_path:
    return
  try:
    with open(log_path, "a", encoding="utf-8") as handle:
      handle.write(str(line))
      handle.write("\n")
  except Exception:
    pass


def _debug_log_event(line):
  dbg = getattr(_import_ctx, "visibility_debug", {}) or {}
  if not dbg.get("enabled"):
    return
  limit = int(dbg.get("event_limit", 200) or 200)
  emitted = int(dbg.get("events", 0) or 0)
  if emitted >= limit:
    return
  dbg["events"] = emitted + 1
  _append_debug_log_line(line)


def _append_transform_log_line(line):
  dbg = getattr(_import_ctx, "transform_debug", {}) or {}
  log_path = dbg.get("log_path")
  if not log_path:
    return
  try:
    with open(log_path, "a", encoding="utf-8") as handle:
      handle.write(str(line))
      handle.write("\n")
  except Exception:
    pass


def _transform_debug_matches(*names):
  dbg = getattr(_import_ctx, "transform_debug", {}) or {}
  if not dbg.get("enabled"):
    return False
  needle = dbg.get("filter")
  if not needle:
    return True
  try:
    needle_lc = str(needle).lower()
  except Exception:
    return False
  for name in names:
    if name is None:
      continue
    try:
      if needle_lc in str(name).lower():
        return True
    except Exception:
      continue
  return False


def _log_bone_debug_event(stage, payload, *names):
  if not _transform_debug_matches(*names):
    return
  try:
    line = "[iEDM][BONEDBG] {} {}".format(stage, json.dumps(payload, separators=(",", ":"), sort_keys=True))
  except Exception:
    return
  print(line)
  _append_transform_log_line(line)


def _matrix_flat_rows(mat):
  try:
    m = Matrix(mat)
    return [round(float(v), 9) for row in m for v in row]
  except Exception:
    return None


def _matrix_trs_summary(mat):
  try:
    loc, rot, scale = Matrix(mat).decompose()
    return {
      "loc": [round(float(loc.x), 6), round(float(loc.y), 6), round(float(loc.z), 6)],
      "rot_quat": [round(float(rot.w), 6), round(float(rot.x), 6), round(float(rot.y), 6), round(float(rot.z), 6)],
      "scale": [round(float(scale.x), 6), round(float(scale.y), 6), round(float(scale.z), 6)],
    }
  except Exception:
    return None


def _anim_rest_debug_enabled():
  dbg = getattr(_import_ctx, "transform_debug", {}) or {}
  return bool(dbg.get("log_path") and dbg.get("target") == "su-25t.edm")


def _log_anim_rest_event(stage, payload):
  if not _anim_rest_debug_enabled():
    return
  try:
    _append_transform_log_line("[iEDM][ANIMREST] {} {}".format(stage, json.dumps(payload, separators=(",", ":"), sort_keys=True)))
  except Exception:
    pass


def _node_visibility_chain_args(node):
  """Return visibility arg numbers from nearest parent to root for a graph node."""
  args = []
  curr = getattr(node, "parent", None)
  while curr is not None:
    tf = getattr(curr, "transform", None)
    if isinstance(tf, ArgVisibilityNode):
      try:
        vis_data = getattr(tf, "visData", []) or []
        if vis_data:
          arg = int(vis_data[0][0])
          args.append(arg)
      except Exception:
        pass
    curr = getattr(curr, "parent", None)
  return args

# Keep light conversion constants in sync with official exporter behavior.
_PBR_WATTS_TO_LUMENS = 683.0
_BLENDER_LAMP_ENERGY_COEFFICIENT = 0.0000305
_BLENDER_LAMP_WEAK_COEFFICIENT = 1.0 / 2.9


def _anim_frame_to_scene_frame(frame_value):
  """Map EDM normalized animation frame [-1..1] to scene frame [0..FRAME_SCALE]."""
  try:
    val = int(round((float(frame_value) + 1.0) * FRAME_SCALE / 2.0))
    # Clamp to 32-bit int for safety
    return min(2147483647, max(-2147483648, val))
  except (OverflowError, ValueError):
    return 0


def _is_authored_argvis_control_pair(node):
  """Return True for an authored ArgVisibility -> Arg* control pair with the same name.

  Official exports often encode control empties as:
    ArgVisibilityNode("Dummy604") -> ArgRotationNode("Dummy604")
    ArgVisibilityNode("Dummy001") -> ArgAnimationNode("Dummy001")
  These are structurally different from a plain top-level arg node even when
  _is_child_of_file_root() returns True through a pure ArgVisibility chain.
  Treating them like direct root children applies the v10 basis fix one layer
  too early.
  """
  try:
    parent = getattr(node, "parent", None)
    if parent is None:
      return False
    parent_tf = getattr(parent, "transform", None)
    if not (isinstance(parent, ArgVisibilityNode) or isinstance(parent_tf, ArgVisibilityNode)):
      return False
    name = getattr(node, "name", "") or ""
    parent_name = getattr(parent, "name", "") or getattr(parent_tf, "name", "") or ""
    if not (name and parent_name):
      return False
    if name == parent_name:
      return True
    return _canonical_control_name(name) == _canonical_control_name(parent_name)
  except Exception:
    return False


def _canonical_control_name(name):
  """Normalize authored control names so DAM_* and *_DAM wrappers still pair.

  Some 3ds Max exports write the visibility wrapper as `DAM_Rudder_L` and the
  driven control as `Rudder_L_DAM` or `Flap_L_DAM`. Those are the same
  authored control pair even though the strings are not byte-identical.
  """
  import re

  raw = str(name or "").lower()
  raw = re.sub(r"\.\d{3}$", "", raw)
  tokens = [tok for tok in re.split(r"[^a-z0-9]+", raw) if tok]
  norm = []
  for tok in tokens:
    if len(tok) > 3 and tok.endswith("s"):
      tok = tok[:-1]
    norm.append(tok)
  return tuple(sorted(norm))


def _is_nested_authored_argvis_control_pair(node):
  """Return True for a same-name ArgVisibility -> Arg* pair under another animated parent.

  Example pattern in su-27_lod3.edm:
    ArgRotationNode("Dummy670") -> ArgVisibilityNode("Dummy598") -> ArgRotationNode("Dummy598")

  In these nested pairs, the inner Arg* node's non-zero base.position is the
  static offset of the middle wrapper object, while the deepest child object
  should primarily own the animated rotation.
  """
  try:
    if not _is_authored_argvis_control_pair(node):
      return False
    parent = getattr(node, "parent", None)
    grandparent = getattr(parent, "parent", None) if parent is not None else None
    grandparent_tf = getattr(grandparent, "transform", grandparent) if grandparent is not None else None
    return isinstance(grandparent_tf, AnimatingNode) and not isinstance(grandparent_tf, ArgVisibilityNode)
  except Exception:
    return False


def _is_root_visibility_chain_authored_pair(node):
  """Return True for same-name ArgVis -> Arg* pairs under only ArgVisibility ancestors.

  These pairs are not the same as the nested animated-parent case above. They
  sit under a pure visibility wrapper chain from the file root, so they still
  need the top-level q1 basis handling used for root-authored controls.
  """
  try:
    if not _is_authored_argvis_control_pair(node):
      return False
    if _is_nested_authored_argvis_control_pair(node):
      return False
    if not _is_child_of_file_root(node):
      return False
    grandparent = getattr(getattr(node, "parent", None), "parent", None)
    grandparent_tf = getattr(grandparent, "transform", grandparent) if grandparent is not None else None
    return isinstance(grandparent_tf, ArgVisibilityNode)
  except Exception:
    return False


def _is_top_level_visibility_authored_pair(node):
  """Immediate same-name ArgVis -> Arg* pair under a root visibility wrapper.

  This is the narrow legacy 3ds Max shape that still behaves like a top-level
  authored control even though one visibility wrapper sits between it and the
  file root:

    Root -> ArgVisibility("Dummy648") -> ArgVisibility("Dummy603") -> ArgRotation("Dummy603")

  Deeper visibility-only chains such as:

    Root -> ArgVisibility("Dummy638") -> ArgVisibility("Dummy637") -> ArgVisibility("Dummy636") -> ArgRotation("Dummy597")

  must not be treated as top-level controls.
  """
  try:
    if not _is_authored_argvis_control_pair(node):
      return False
    parent = getattr(node, "parent", None)
    grandparent = getattr(parent, "parent", None) if parent is not None else None
    if grandparent is None:
      return False
    grandparent_tf = getattr(grandparent, "transform", grandparent)
    if not isinstance(grandparent_tf, ArgVisibilityNode):
      return False
    great_grandparent = getattr(grandparent, "parent", None)
    return great_grandparent is not None and getattr(great_grandparent, "parent", None) is None
  except Exception:
    return False


def _has_only_visibility_ancestors(node):
  try:
    parent = getattr(node, "parent", None)
    if parent is None:
      return False
    current = parent
    seen_any = False
    while current is not None:
      current_tf = getattr(current, "transform", current)
      if not isinstance(current_tf, ArgVisibilityNode):
        return False
      seen_any = True
      current = getattr(current, "parent", None)
    return seen_any
  except Exception:
    return False


def _visibility_node_has_direct_anim_child(vis_node):
  try:
    for ch in (getattr(vis_node, "children", None) or []):
      ch_tf = getattr(ch, "transform", None)
      if isinstance(ch_tf, AnimatingNode) and not isinstance(ch_tf, ArgVisibilityNode):
        return True
    return False
  except Exception:
    return False


def _nearest_visibility_ancestor_with_direct_anim_child(node):
  """Return nearest ArgVisibility ancestor whose direct children include Arg* control."""
  try:
    current = getattr(node, "parent", None)
    while current is not None:
      current_tf = getattr(current, "transform", current)
      if not isinstance(current_tf, ArgVisibilityNode):
        return None
      if _visibility_node_has_direct_anim_child(current):
        return current
      current = getattr(current, "parent", None)
    return None
  except Exception:
    return None


def _is_static_root_visibility_wrapper(node):
  """Static ArgVisibility wrapper inside a root visibility family with outer Arg* sibling."""
  try:
    if node is None:
      return False
    tf = getattr(node, "transform", None)
    if not isinstance(tf, ArgVisibilityNode):
      return False
    if _visibility_node_has_direct_anim_child(node):
      return False
    ancestor = _nearest_visibility_ancestor_with_direct_anim_child(node)
    return ancestor is not None and getattr(node, "parent", None) is ancestor
  except Exception:
    return False


def _is_root_visibility_pair_child(node):
  try:
    tf = getattr(node, "transform", None)
    if tf is None or not isinstance(tf, AnimatingNode) or isinstance(tf, ArgVisibilityNode):
      return False
    if not _is_authored_argvis_control_pair(node):
      return False
    if not _has_only_visibility_ancestors(node):
      return False
    return _nearest_visibility_ancestor_with_direct_anim_child(node) is not None
  except Exception:
    return False


def _is_mixed_root_visibility_direct_anim_child(node):
  """Return True for a direct animated child inside a mixed root visibility branch."""
  try:
    tf = getattr(node, "transform", None)
    if not isinstance(tf, AnimatingNode) or isinstance(tf, ArgVisibilityNode):
      return False
    if _is_authored_argvis_control_pair(tf):
      return False
    parent = getattr(node, "parent", None)
    parent_tf = getattr(parent, "transform", parent) if parent is not None else None
    if not isinstance(parent_tf, ArgVisibilityNode):
      return False
    if not _has_only_visibility_ancestors(node):
      return False
    children = list(getattr(parent, "children", None) or [])
    has_render_child = any(getattr(ch, "render", None) is not None for ch in children)
    has_anim_child = any(
      isinstance(getattr(ch, "transform", None), AnimatingNode)
      and not isinstance(getattr(ch, "transform", None), ArgVisibilityNode)
      for ch in children
    )
    return has_render_child and has_anim_child
  except Exception:
    return False


def _is_child_of_file_root(node):
  parent = getattr(node, "parent", None)
  if parent is None:
    # In raw EDM graph, a top-level node has parent=None.
    # Exclude the translation graph root itself (which has a generic name or children without a base).
    if hasattr(node, "transform") and not hasattr(node, "base"):
      return False
    return True
  if getattr(parent, "parent", None) is None:
    return True
  try:
    from ..edm_format.types import ArgVisibilityNode
    # Walk up through any number of consecutive ArgVisibilityNode ancestors.
    # A node is "child of file root" if every ancestor between it and the
    # root are ArgVisibilityNode wrappers. The tree root is the Node whose
    # parent is None; a direct child of root has parent.parent == None.
    current = parent
    while current is not None:
      current_tf = getattr(current, "transform", None)
      is_argvis = isinstance(current, ArgVisibilityNode) or isinstance(current_tf, ArgVisibilityNode)
      if not is_argvis:
        break
      current_parent = getattr(current, "parent", None)
      if current_parent is None:
        return True  # current itself is root (unlikely for ArgVis)
      if getattr(current_parent, "parent", None) is None:
        return True  # current_parent is the tree root node
      current = current_parent
  except Exception:
    pass
  return False


def _is_direct_child_of_file_root(node):
  """Return True only if node is a direct child of the file root (no ArgVisibility ancestor chain).
  
  Unlike _is_child_of_file_root which walks through ArgVisibilityNode ancestors,
  this only matches nodes whose immediate parent is the file root.
  """
  parent = getattr(node, "parent", None)
  if parent is None:
    if hasattr(node, "transform") and not hasattr(node, "base"):
      return False
    return True
  return getattr(parent, "parent", None) is None


def _ob_local_is_identity(o, eps=1e-6):
  """Return True if o.matrix_local is (approximately) the identity matrix."""
  try:
    m = o.matrix_local
    return all(
      abs(float(m[r][c]) - (1.0 if r == c else 0.0)) <= eps
      for r in range(4) for c in range(4)
    )
  except Exception:
    return False


def _is_connector_object(obj):
  if obj is None:
    return False
  if hasattr(obj, "edm") and getattr(obj.edm, "is_connector", False):
    return True
  if hasattr(obj, "EDMProps") and getattr(obj.EDMProps, "SPECIAL_TYPE", "") == "CONNECTOR":
    return True
  return False


def _is_connector_transform(tfnode, blender_obj=None):
  # A TransformNode is a connector wrapper if it's named "Connector Transform"
  # (the standard io_scene_edm export name) OR if its blender_obj is a connector
  # empty created by create_connector() (EDMs like Su-27 use the connector's own
  # name, e.g. "BANO_0", "GUN_POINT", "Pylon1", as the wrapper TransformNode
  # name). Apply the -90° X correction only when legacy exporter behavior is
  # positively identified.
  if not isinstance(tfnode, TransformNode):
    return False
  if not _is_connector_object(blender_obj):
    return False
  try:
    return bool(blender_obj.get("_iedm_connector_apply_xfix", False))
  except Exception:
    return False


def _strip_anim_prefix(name):
  """Strip known EDM animation/control prefixes from node names."""
  if not name:
    return name
  for prefix in ("ar_", "al_", "as_", "tl_", "tr_", "s_", "v_"):
    if name.startswith(prefix):
      return name[len(prefix):]
  return name


def _is_generic_render_name(name):
  if not name:
    return True
  if re.match(r"^_[0-9]+$", name):
    return True
  if re.match(r"^Object(\.[0-9]+)?$", name):
    return True
  return False


def _is_anim_node_name(name):
  """Check if a name has animation/control prefixes that indicate a skeleton node."""
  if not name:
    return False
  return any(name.startswith(p) for p in ("ar_", "al_", "as_", "tl_", "tr_", "s_", "v_"))


def _is_bone_transform(tfnode):
  return tfnode is not None and "Bone" in type(tfnode).__name__


def is_skeleton_node(node):
  """Determine if a graph node should be preserved as a separate Empty/Armature.

  Skeleton nodes are animation controllers, bone ancestors, light transforms,
  or nodes with renderable descendants that need structural hierarchy.
  """
  if node is None:
    return False
  if getattr(node, "_is_graph_root", False):
    return False

  # AnimatingNodes, LightNodes and nodes with anim prefixes are always skeletal
  if node.transform:
    name = getattr(node.transform, "name", "") or ""
    if isinstance(node.transform, AnimatingNode):
      return True
    if _is_anim_node_name(name):
      return True
    if "Light Transform" in name:
      return True

  if node.render and type(node.render).__name__ == "LightNode":
    return True

  # Ancestors of Bone nodes should be preserved
  def _has_bone(n):
    if n.transform and "Bone" in type(n.transform).__name__:
      return True
    for c in n.children:
      if _has_bone(c):
        return True
    return False

  def _has_render(n):
    if n.render is not None:
      return True
    for c in n.children:
      if _has_render(c):
        return True
    return False

  # Check ancestors up to graph root
  res = False
  curr = node
  while curr and not getattr(curr, "_is_graph_root", False):
    if curr.transform and "Bone" in type(curr.transform).__name__:
      res = True
      break
    curr = curr.parent

  if not res:
    res = _has_bone(node) or _has_render(node)

  return res


def _transform_display_name(tfnode):
  raw_name = getattr(tfnode, "name", "") or ""
  if raw_name:
    name = raw_name
  else:
    idx = getattr(tfnode, "_graph_idx", None)
    if idx is not None:
      name = "tf_{:04d}".format(idx)
    else:
      name = type(tfnode).__name__
  if isinstance(tfnode, ArgVisibilityNode):
    return name
  name = _strip_anim_prefix(name)
  # Strip "Armature : " prefix (exporter adds it to bone names)
  if " : " in name:
    name = name.split(" : ")[-1]
  return name


def _set_official_special_type(obj, special_type):
  """Set official EDM exporter object type when that addon is present."""
  if hasattr(obj, "EDMProps"):
    obj.EDMProps.SPECIAL_TYPE = special_type


def _ensure_official_material_bridge():
  """
  Resolve optional interop with the official EDM exporter material system.
  Returns a cached dict with:
    - available: bool
    - material_descs: dict[str, MatDesc]
    - names: dict[str, str]
    - error: str | None
  """
  if _import_ctx.official_material_bridge is not None:
    return _import_ctx.official_material_bridge

  bridge = {
    "available": False,
    "material_descs": {},
    "error": None,
    "names": {
      "default": "EDM_Default_Material",
      "deck": "EDM_Deck_Material",
      "fake_omni": "EDM_Fake_Omni_Material",
      "fake_spot": "EDM_Fake_Spot_Material",
      "glass": "EDM_Glass_Material",
      "mirror": "EDM_Mirror_Material",
    },
    "node_types": {
      "default": "EdmDefaultShaderNodeType",
      "deck": "EdmDeckShaderNodeType",
      "fake_omni": "EdmFakeOmniShaderNodeType",
      "fake_spot": "EdmFakeSpotShaderNodeType",
      "glass": "EdmGlassShaderNodeType",
      "mirror": "EdmMirrorShaderNodeType",
    },
  }

  try:
    # Available when official addon is loaded (io_scene_edm).
    from materials.materials import build_material_descriptions
    bridge["material_descs"] = build_material_descriptions() or {}
    bridge["available"] = True
  except Exception as exc:
    bridge["error"] = str(exc)

  _import_ctx.official_material_bridge = bridge
  return bridge
