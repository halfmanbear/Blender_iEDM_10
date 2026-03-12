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

_SUFFIX_RE = re.compile(r"\.\d{3}$")  # matches ".000" to ".999" at end


FRAME_SCALE = 200

# Official EDM exports wrap scene space in a root basis fix matrix.
_ROOT_BASIS_FIX = Matrix(((1, 0, 0, 0),
                          (0, 0, -1, 0),
                          (0, 1, 0, 0),
                          (0, 0, 0, 1)))


def _default_transform_debug_state():
  return {"enabled": False, "filter": None, "limit": 200, "emitted": 0}


def _default_render_split_debug_state():
  return {"enabled": False}


import threading
class ImportContext(threading.local):
  """Mutable importer runtime state shared across reader modules."""

  def __init__(self):
    self.edm_version = 10
    self.transform_debug = _default_transform_debug_state()
    self.render_split_debug = _default_render_split_debug_state()
    self.mesh_origin_mode = "APPROX"
    self.source_dir = None
    self.official_material_bridge = None
    self.bone_import_ctx = None
    self.legacy_v10_parent_compose = False
    self.file_has_bones = False


_import_ctx = ImportContext()

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


def _is_child_of_file_root(node):
  parent = getattr(node, "parent", None)
  if parent is None:
    return False
  if getattr(parent, "parent", None) is None:
    return True
  try:
    from ..edm_format.types import ArgVisibilityNode
    grandparent = getattr(parent, "parent", None)
    parent_tf = getattr(parent, "transform", None)
    if isinstance(parent_tf, ArgVisibilityNode) and grandparent is not None and getattr(grandparent, "parent", None) is None:
      return True
  except Exception:
    pass
  return False


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


