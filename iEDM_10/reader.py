"""
reader

Contains the central EDM->Blender conversion code
"""

import bpy
import bmesh

from .utils import chdir, print_edm_graph
from .edm import EDMFile
from .edm.mathtypes import *
from .edm.types import *

from .translation import TranslationGraph, TranslationNode

import re
import glob
import fnmatch
import os
import itertools
import math

_SUFFIX_RE = re.compile(r"\.\d{3}$")  # matches ".000" to ".999" at end

_PREFIXES = (
    "Cylinder", "Fspot", "Omni", "Object", "Line", "Flame",
    "BANO_", "light_bano_spot_", "Formation_Lights_",
    "f18c", "F18C_aoa", "forsag", "oreol", "Stv_",
    "lamp_off", "banno_", "RN_", "N_", "KIL_", "kil_",
    "pilot_", "Pilot_", "Cockpit_Glass", "Cockpit_part",
    "Texturte_Cockpit", "Texture_Cockpit",
    "Plane", "merge", "Merge",
    "RefuelingProbe_", "Refueling_Probe", "Line_bano_",
    "Box", "Cube", "f29_", "f_", "Pilon_", "Pylon",
    "Wing", "Flap", "Aileron", "Fin", "Stab", "Rudder",
    "Slat", "AirBrake", "Air_Brake", "chehly",
)

FRAME_SCALE = 200

# Global variable to track EDM version for coordinate system handling
_edm_version = 10
# Official EDM exports wrap scene space in a root basis fix matrix.
_ROOT_BASIS_FIX = Matrix(((1, 0, 0, 0),
                          (0, 0, -1, 0),
                          (0, 1, 0, 0),
                          (0, 0, 0, 1)))
_transform_debug = {"enabled": False, "filter": None, "limit": 200, "emitted": 0}
_mesh_origin_mode = "APPROX"
_official_material_bridge = None
_bone_import_ctx = None
_legacy_v10_parent_compose = False
_file_has_bones = False

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
  return parent is not None and getattr(parent, "parent", None) is None


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
  # name).  In both cases the importer must cancel the exporter's fixed +90° X
  # rotation by applying a matching -90° X correction to the Blender local matrix.
  if not isinstance(tfnode, TransformNode):
    return False
  if getattr(tfnode, "name", "") == "Connector Transform":
    return True
  return _is_connector_object(blender_obj)


def _is_narrow_safe_identity_helper_name(name) -> bool:
    """Whitelist helper wrappers that are safe to bypass."""
    s = str(name or "")
    s = _SUFFIX_RE.sub("", s)  # strip exporter ".###" suffix if present
    return s.startswith(_PREFIXES)


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


def _disable_official_export_optimizations():
  """Disable official exporter optimization toggles for round-trip parity.

  ALL optimizations are disabled.  Notably, OPTIMIZE_TRANFORMATIONS is kept
  OFF even though it could collapse identity chains: pyedm's C++ implementation
  is too aggressive and also removes ArgVisibilityNode wrappers (which control
  conditional visibility in ModelViewer2), causing a -67 regression on FA-18.
  Identity chain collapsing is handled instead by the Python-level
  _enable_official_identity_fallback_transform_prune patch, which only touches
  _mat/_bmat_inv/tl_/tr_/s_ TransformNode stacks and never removes
  ArgVisibilityNode / ArgAnimationNode / Bone nodes.
  MERGE_NODES is kept OFF because it can merge/rename named nodes.
  """
  class _NoOptDevProps:
    def __init__(self):
      self.EXPORT_CUR_ARG_ONLY = False
      self.OPTIMIZE_TRANFORMATIONS = False  # keep off: removes ArgVisibilityNode
      self.MERGE_PROPETIES_SET = False
      self.OPTIMIZE_TEXTURE_CHANNELS = False
      self.MERGE_NODES = False              # keep off: name-clobbering merges
      self.EXTREMLY_OPTIMIZE_MESHES = False
      self.OPTIMIZE_VERTEX_CACHE = False

  def _no_opt_props(_scene):
    return _NoOptDevProps()

  scene = bpy.context.scene
  dev_props = getattr(scene, "EDMDevModeProps", None)
  changed = False

  keys = (
    "EXPORT_CUR_ARG_ONLY",
    "OPTIMIZE_TRANFORMATIONS",
    "MERGE_PROPETIES_SET",
    "OPTIMIZE_TEXTURE_CHANNELS",
    "MERGE_NODES",
    "EXTREMLY_OPTIMIZE_MESHES",
    "OPTIMIZE_VERTEX_CACHE",
  )
  if dev_props is not None:
    for key in keys:
      if hasattr(dev_props, key) and getattr(dev_props, key):
        try:
          setattr(dev_props, key, False)
          changed = True
        except Exception:
          pass

  # Non-dev pyedm builds return a hardcoded dummy with all optimizations ON.
  # Monkeypatch the runtime getter so exporter defaults remain parity-safe.
  patched = False
  for mod_name in ("dev_mode", "io_scene_edm.dev_mode"):
    try:
      mod = __import__(mod_name, fromlist=["get_dev_mode_props"])
      if hasattr(mod, "get_dev_mode_props"):
        mod.get_dev_mode_props = _no_opt_props
        patched = True
    except Exception:
      pass
  for mod_name in ("collection_walker", "io_scene_edm.collection_walker"):
    try:
      mod = __import__(mod_name, fromlist=["get_dev_mode_props"])
      if hasattr(mod, "get_dev_mode_props"):
        mod.get_dev_mode_props = _no_opt_props
        patched = True
    except Exception:
      pass

  changed = changed or patched
  return changed


def _enable_official_visibility_name_alias():
  """Monkeypatch official exporter visibility naming to honor importer aliases.

  Official `io_scene_edm` emits visibility nodes as `v_ + obj.name`, but some
  source EDMs contain both an animation node and a visibility node with the same
  semantic name (e.g. `Aileron_L`). Blender object names must be unique, so the
  imported visibility wrapper object can be suffixed (`.002`), which leaks into
  the exported `v_*` name. This runtime patch lets the importer attach a custom
  `_iedm_vis_export_name` alias property used only for visibility wrapper names.
  """
  def _patch_module(mod):
    try:
      orig = getattr(mod, "extract_visibility_animation", None)
      if orig is None:
        return False
      if getattr(orig, "_iedm_vis_alias_patch", False):
        return False
      pyedm_mod = getattr(mod, "pyedm", None)
      anim_mod = getattr(mod, "anim", None)
      utils_mod = getattr(mod, "utils", None)
      if pyedm_mod is None or anim_mod is None or utils_mod is None:
        return False
    except Exception:
      return False

    def _extract_visibility_animation(parent, obj, allowed_args):
      if obj.animation_data and obj.animation_data.action:
        action = obj.animation_data.action
        arg = utils_mod.extract_arg_number(action.name)
        if arg >= 0:
          if allowed_args and arg not in allowed_args:
            return parent
          avis = anim_mod.extract_anim_float(action, 'VISIBLE')
          if avis:
            alias = None
            try:
              alias = obj.get("_iedm_vis_export_name", None)
            except Exception:
              alias = None
            # When the importer has stored the original EDM name as an alias
            # (which may or may not start with "v_"), use it directly as the
            # full export name.  This preserves Su-27-style plain names like
            # "Dummy001" that should NOT get a "v_" prefix, while FA-18/geometry
            # names stored as "v_Gas_Tanks" etc. are used verbatim too.
            if alias is not None:
              vis = pyedm_mod.VisibilityNode(str(alias))
            else:
              vis = pyedm_mod.VisibilityNode("v_" + obj.name)
            vis.setArguments([[arg, avis]])
            parent = parent.addChild(vis)
            return parent
      return parent

    _extract_visibility_animation._iedm_vis_alias_patch = True
    _extract_visibility_animation._iedm_vis_alias_orig = orig
    mod.extract_visibility_animation = _extract_visibility_animation
    return True

  patched = False
  patched_vis_fn = None
  for mod_name in ("visibility_animation", "io_scene_edm.visibility_animation"):
    try:
      mod = __import__(mod_name, fromlist=["extract_visibility_animation"])
      did_patch = _patch_module(mod)
      patched = did_patch or patched
      fn = getattr(mod, "extract_visibility_animation", None)
      if fn is not None and getattr(fn, "_iedm_vis_alias_patch", False):
        patched_vis_fn = fn
    except Exception:
      pass
  if patched_vis_fn is not None:
    for mod_name in ("collection_walker", "io_scene_edm.collection_walker"):
      try:
        mod = __import__(mod_name, fromlist=["extract_visibility_animation"])
        if hasattr(mod, "extract_visibility_animation"):
          mod.extract_visibility_animation = patched_vis_fn
          patched = True
      except Exception:
        pass
  return patched


def _enable_official_visibility_wrapper_transform_passthrough():
  """Skip exporter transform emission for identity visibility-only wrapper empties.

  Preserved renderless ArgVisibilityNode wrappers are needed for visibility arg
  parity, but the official exporter emits an extra identity TransformNode for the
  wrapper object before the child's animated stack (`v_* -> name -> name.001_mat`).
  For importer-marked wrapper empties, bypass `extract_transform_animation()` and
  return the existing parent node unchanged, so children attach directly under the
  visibility node.
  """
  def _patch_animation_module(mod):
    try:
      orig = getattr(mod, "extract_transform_animation", None)
      if orig is None:
        return False
      if getattr(orig, "_iedm_vis_passthrough_patch", False):
        return False
    except Exception:
      return False

    def _extract_transform_animation(parent, obj, allowed_args=None):
      try:
        _dbg_flag = False
        _dbg_ident_flag = False
        _dbg_narrow_ident_flag = False
        _dbg_name = ""
        if obj is not None:
          try:
            _dbg_name = str(getattr(obj, "name", "") or "")
          except Exception:
            _dbg_name = ""
          try:
            _dbg_flag = bool(obj.get("_iedm_vis_passthrough", False))
          except Exception:
            _dbg_flag = False
          try:
            _dbg_ident_flag = bool(obj.get("_iedm_identity_passthrough", False))
          except Exception:
            _dbg_ident_flag = False
          try:
            _dbg_narrow_ident_flag = bool(obj.get("_iedm_narrow_identity_passthrough", False))
          except Exception:
            _dbg_narrow_ident_flag = False
        if obj is not None and _dbg_flag:
          # Importer sets this flag only on preserved renderless visibility wrapper empties.
          return parent
        if obj is not None and _dbg_narrow_ident_flag:
          # Narrow safe class only (e.g. identity helper under ArgVisibilityNode
          # feeding a light/fake-light path). Keep generic identity passthrough
          # disabled to avoid ModelViewer2 blank-render regressions.
          return parent
        # Keep generic identity-helper passthrough disabled for now. It reduces
        # DOT wrapper noise substantially, but can also alter exporter control
        # hierarchy semantics enough to produce a blank render in ModelViewer2.
        # The visibility-wrapper passthrough above (_iedm_vis_passthrough) is
        # retained because it has a clearer, narrower effect.
      except Exception:
        pass
      return orig(parent, obj, allowed_args)

    _extract_transform_animation._iedm_vis_passthrough_patch = True
    _extract_transform_animation._iedm_vis_passthrough_orig = orig
    mod.extract_transform_animation = _extract_transform_animation
    return True

  patched = False
  for mod_name in ("animation", "io_scene_edm.animation"):
    try:
      mod = __import__(mod_name, fromlist=["extract_transform_animation"])
      patched = _patch_animation_module(mod) or patched
    except Exception:
      pass
  # `collection_walker` accesses `anim.extract_transform_animation`; rebinding the
  # module function is usually sufficient because `anim` is a module reference.
  return patched


def _enable_official_identity_fallback_transform_prune():
  """Prune no-op transform wrappers from official exporter animation stacks.

  `io_scene_edm.animation.extract_transform_anim()` emits fallback `tl_*`,
  `tr_*`, and `s_*` transforms even when they are identity. It may also emit
  identity `_mat` / `_bmat_inv` in some cases. These are semantically no-ops
  but inflate DOT structure heavily. This runtime patch preserves animated nodes
  and non-identity transforms while skipping identity wrappers.
  """
  def _patch_animation_module(mod):
    try:
      orig = getattr(mod, "extract_transform_anim", None)
      if orig is None:
        return False
      if getattr(orig, "_iedm_ident_prune_patch", False):
        return False
      Matrix_mod = getattr(mod, "Matrix", None)
      pyedm_mod = getattr(mod, "pyedm", None)
      Anim_Type_mod = getattr(mod, "Anim_Type", None)
      Allowed_mod = getattr(mod, "AllowedAnimationsEnum", None)
      if Matrix_mod is None or pyedm_mod is None or Anim_Type_mod is None or Allowed_mod is None:
        return False
    except Exception:
      return False

    def _is_identity_matrix(m, eps=1e-6):
      try:
        mm = Matrix_mod(m)
      except Exception:
        return False
      ident = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
      )
      try:
        for r in range(4):
          for c in range(4):
            if abs(float(mm[r][c]) - ident[r][c]) > eps:
              return False
        return True
      except Exception:
        return False

    def _extract_transform_anim(parent, obj, action, mat, bmat, name, anim_type,
                                allowed_anims=None, data_path_enum=None,
                                allowed_args=None, bone=None):
      # Preserve original defaults when callers omit these.
      if allowed_anims is None:
        allowed_anims = Allowed_mod.ALL
      if data_path_enum is None:
        data_path_enum = getattr(mod, "Data_Path_Enum")

      bmat_inv = bmat.inverted()
      bloc, brot, bsca = bmat.decompose()

      al = ar = asc = None

      # Check for original animation-node name override.  When the source EDM
      # used a plain name (e.g. "Dummy001") rather than the io_scene_edm
      # convention ("ar_Dummy001"), store the plain name as the desired output.
      # We temporarily swap mod.pyedm with a proxy so that pyedm.AnimationNode()
      # calls inside extract_transfrom_anim_fcurves use the plain name.
      _orig_anim_name = None
      try:
        if obj is not None and hasattr(obj, "get"):
          _raw = obj.get("_iedm_orig_anim_name", None)
          if (_raw
              and not str(_raw).startswith("al_")
              and not str(_raw).startswith("ar_")
              and not str(_raw).startswith("as_")):
            _orig_anim_name = str(_raw)
      except Exception:
        _orig_anim_name = None

      if anim_type is Anim_Type_mod.AT_FCURVES:
        if _orig_anim_name is not None:
          # Build a map: prefixed variant -> desired plain name.
          _name_map = {pfx + name: _orig_anim_name for pfx in ("al_", "ar_", "as_")}
          _real_pyedm = mod.pyedm
          class _AnimNameProxy:
            """Proxy for mod.pyedm that redirects AnimationNode names via _name_map."""
            def AnimationNode(self_p, n):  # noqa: N805
              return _real_pyedm.AnimationNode(_name_map.get(n, n))
            def __getattr__(self_p, a):  # noqa: N805
              return getattr(_real_pyedm, a)
          mod.pyedm = _AnimNameProxy()
          try:
            al, ar, asc = mod.extract_transfrom_anim_fcurves(
              action, bloc, brot, bsca, name, allowed_anims, data_path_enum, allowed_args
            )
          finally:
            mod.pyedm = _real_pyedm
        else:
          al, ar, asc = mod.extract_transfrom_anim_fcurves(
            action, bloc, brot, bsca, name, allowed_anims, data_path_enum, allowed_args
          )
      elif anim_type is Anim_Type_mod.AT_NLA:
        if getattr(obj, "type", None) == 'ARMATURE':
          al, ar, asc = mod.export_nla_animation(
            obj, bone, allowed_anims, bloc, brot, bsca, data_path_enum, allowed_args
          )

      a = parent

      # Keep parity-relevant wrappers, but skip exact identity transforms.
      mat_tf = Matrix_mod(mat)
      if not _is_identity_matrix(mat_tf):
        a = a.addChild(pyedm_mod.Transform(name + '_mat', mat_tf))

      bmat_inv_tf = Matrix_mod(bmat_inv)
      if not _is_identity_matrix(bmat_inv_tf):
        a = a.addChild(pyedm_mod.Transform(name + '_bmat_inv', bmat_inv_tf))

      if al:
        a = a.addChild(al)
      else:
        tl = Matrix_mod.LocRotScale(bloc, None, None)
        if not _is_identity_matrix(tl):
          a = a.addChild(pyedm_mod.Transform('tl_' + name, tl))

      if ar:
        a = a.addChild(ar)
      else:
        tr = Matrix_mod.LocRotScale(None, brot, None)
        if not _is_identity_matrix(tr):
          a = a.addChild(pyedm_mod.Transform('tr_' + name, tr))

      if asc:
        a = a.addChild(asc)
      else:
        s = Matrix_mod.LocRotScale(None, None, bsca)
        # Blender decomposition can introduce tiny scale noise (~1e-4..1e-3)
        # on otherwise identity scale fallbacks; prune those no-op wrappers too.
        if not _is_identity_matrix(s, eps=1e-3):
          a = a.addChild(pyedm_mod.Transform('s_' + name, s))

      return a

    _extract_transform_anim._iedm_ident_prune_patch = True
    _extract_transform_anim._iedm_ident_prune_orig = orig
    mod.extract_transform_anim = _extract_transform_anim
    return True

  patched = False
  patched_funcs = {}
  for mod_name in ("animation", "io_scene_edm.animation"):
    try:
      mod = __import__(mod_name, fromlist=["extract_transform_anim"])
      if _patch_animation_module(mod):
        patched = True
        patched_funcs[getattr(mod, "__name__", mod_name)] = getattr(mod, "extract_transform_anim", None)
    except Exception:
      pass

  # NOTE: Do not rebind export_armature.extract_transform_anim by default.
  #
  # Pruning identity fallback wrappers on the armature export path reduced DOT
  # noise substantially, but it can also remove bind/space-conversion wrappers
  # that appear semantically required for correct visual rendering in
  # ModelViewer2 (blank scene despite render nodes present). Keep the prune hook
  # limited to the generic animation module path for now.
  return patched


def _enable_official_connector_transform_naming():
  """Patch export_connectors.export_connector to name the wrapper TransformNode
  after the connector object instead of the hardcoded 'Connector Transform'.

  In original EDMs the wrapper TransformNode for a connector typically has the
  same name as the connector itself (e.g. TransformNode 'BANO_0' → Connector
  'BANO_0').  The official exporter always uses the literal 'Connector Transform'
  which breaks DOT name parity.  This patch intercepts pyedm.Transform() calls
  inside export_connector and substitutes the object's stored
  _iedm_connector_tf_name (set by the importer) or falls back to object.name.
  """
  def _patch(mod):
    orig = getattr(mod, "export_connector", None)
    if orig is None:
      return False
    if getattr(orig, "_iedm_connector_tf_name_patch", False):
      return False
    real_pyedm = getattr(mod, "pyedm", None)
    if real_pyedm is None:
      return False

    def _patched_export_connector(obj, transform_node, _orig=orig, _mod=mod, _real_pyedm=real_pyedm):
      # Determine the desired TransformNode name for this connector.
      tf_name = None
      try:
        tf_name = obj.get("_iedm_connector_tf_name", None)
      except Exception:
        tf_name = None
      if not tf_name:
        tf_name = "Connector Transform"

      class _ConnectorTfProxy:
        """Proxy for mod.pyedm that renames 'Connector Transform' → tf_name."""
        def Transform(self_p, name, *args, **kwargs):  # noqa: N805
          use = tf_name if name == "Connector Transform" else name
          return _real_pyedm.Transform(use, *args, **kwargs)
        def __getattr__(self_p, a):  # noqa: N805
          return getattr(_real_pyedm, a)

      _mod.pyedm = _ConnectorTfProxy()
      try:
        return _orig(obj, transform_node)
      finally:
        _mod.pyedm = _real_pyedm

    _patched_export_connector._iedm_connector_tf_name_patch = True
    mod.export_connector = _patched_export_connector
    return True

  patched = False
  for mod_name in ("export_connectors", "io_scene_edm.export_connectors"):
    try:
      mod = __import__(mod_name, fromlist=["export_connector"])
      if _patch(mod):
        patched = True
    except Exception:
      pass

  # Also rebind in collection_walker if it imported export_connector directly.
  patched_fn = None
  for mod_name in ("export_connectors", "io_scene_edm.export_connectors"):
    try:
      mod = __import__(mod_name, fromlist=["export_connector"])
      fn = getattr(mod, "export_connector", None)
      if fn is not None and getattr(fn, "_iedm_connector_tf_name_patch", False):
        patched_fn = fn
        break
    except Exception:
      pass
  if patched_fn is not None:
    for mod_name in ("collection_walker", "io_scene_edm.collection_walker"):
      try:
        mod = __import__(mod_name, fromlist=["export_connector"])
        if hasattr(mod, "export_connector"):
          mod.export_connector = patched_fn
      except Exception:
        pass

  return patched


def _ensure_official_material_bridge():
  """
  Resolve optional interop with the official EDM exporter material system.
  Returns a cached dict with:
    - available: bool
    - material_descs: dict[str, MatDesc]
    - names: dict[str, str]
    - error: str | None
  """
  global _official_material_bridge
  if _official_material_bridge is not None:
    return _official_material_bridge

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

  _official_material_bridge = bridge
  return bridge


def _map_edm_material_to_official_kind(edm_material_name):
  mat = (edm_material_name or "").lower()
  if mat in {"glass_material"}:
    return "glass"
  if mat in {"mirror_material"}:
    return "mirror"
  if mat in {"fake_omni_lights", "fake_omni_lights2", "bano_material", "fake_als_lights"}:
    return "fake_omni"
  if mat in {"fake_spot_lights"}:
    return "fake_spot"
  if mat in {"deck_material"}:
    return "deck"
  return "default"


def _create_official_group_node(nodes, bridge, official_kind, official_name):
  node_tree = _resolve_official_material_tree(bridge, official_name)
  custom_id = bridge.get("node_types", {}).get(official_kind)
  desc = bridge.get("material_descs", {}).get(official_name)

  if custom_id:
    try:
      custom_node = nodes.new(custom_id)
      # Custom official nodes need post_init to bind their EDM node tree.
      if desc is not None and hasattr(custom_node, "post_init"):
        custom_node.post_init(desc)
      elif node_tree is not None and hasattr(custom_node, "node_tree"):
        custom_node.node_tree = node_tree
      if getattr(custom_node, "node_tree", None) is not None:
        return custom_node
    except Exception:
      pass

  if node_tree is None:
    return None
  try:
    fallback_node = nodes.new("ShaderNodeGroup")
    fallback_node.node_tree = node_tree
    return fallback_node
  except Exception:
    return None


def _set_group_enum_property(group_node, prop_name, value):
  if not group_node or not prop_name:
    return
  if not hasattr(group_node, prop_name):
    return
  try:
    setattr(group_node, prop_name, value)
  except Exception:
    pass


def _map_edm_material_to_official_name(edm_material_name, bridge):
  names = bridge.get("names", {})
  kind = _map_edm_material_to_official_kind(edm_material_name)
  return names.get(kind, names.get("default", "EDM_Default_Material"))


def _official_transparency_enum(blending):
  try:
    value = int(blending)
  except Exception:
    value = 0
  return {
    0: "OPAQUE",
    1: "ALPHA_BLENDING",
    2: "ALPHA_TEST",
    3: "SUM_BLENDING",
  }.get(value, "OPAQUE")


def _official_shadow_enum(shadows):
  if shadows is None:
    return "SHADOW_CASTER_YES"
  if getattr(shadows, "cast_only", False):
    return "SHADOW_CASTER_ONLY"
  if getattr(shadows, "cast", False):
    return "SHADOW_CASTER_YES"
  return "SHADOW_CASTER_NO"


def _set_group_socket_default(group_node, socket_name, value):
  if not group_node:
    return
  for socket in group_node.inputs:
    if socket.name != socket_name:
      continue
    if not hasattr(socket, "default_value"):
      return
    try:
      socket.default_value = value
    except Exception:
      pass
    return


def _find_group_input_socket(group_node, socket_name):
  if not group_node:
    return None
  for socket in group_node.inputs:
    if socket.name == socket_name:
      return socket
  return None


def _link_texture_to_group_input(links, texture_node, group_node, input_name, output_name="Color"):
  if not texture_node or not group_node:
    return
  from_socket = texture_node.outputs.get(output_name)
  to_socket = _find_group_input_socket(group_node, input_name)
  if not from_socket or not to_socket:
    return
  for link in to_socket.links:
    if link.from_socket == from_socket:
      return
  try:
    links.new(from_socket, to_socket)
  except Exception:
    pass


def _ensure_uv_map_node(nodes):
  for node in nodes:
    if node.bl_idname == "ShaderNodeUVMap":
      return node
  try:
    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.name = "iEDM_UVMap"
    uv_node.label = "UV Map"
    uv_node.location = (-700, 200)
    return uv_node
  except Exception:
    return None


def _link_uv_to_texture_vector(links, uv_node, texture_node):
  if not uv_node or not texture_node:
    return
  from_socket = uv_node.outputs.get("UV")
  to_socket = texture_node.inputs.get("Vector")
  if not from_socket or not to_socket:
    return
  for link in to_socket.links:
    if link.from_socket == from_socket:
      return
  try:
    for link in list(to_socket.links):
      links.remove(link)
  except Exception:
    pass
  try:
    links.new(from_socket, to_socket)
  except Exception:
    pass


def _uv_map_name_from_channel(channel_index):
  if channel_index is None:
    return "UVMap"
  try:
    channel = int(channel_index)
  except Exception:
    channel = 0
  if channel <= 0:
    return "UVMap"
  return "UVMap.{:03d}".format(channel)


def _ensure_uv_map_node_for_channel(nodes, channel_index):
  uv_map_name = _uv_map_name_from_channel(channel_index)
  try:
    ch_idx = max(0, int(channel_index))
  except Exception:
    ch_idx = 0
  for node in nodes:
    if node.bl_idname != "ShaderNodeUVMap":
      continue
    if getattr(node, "uv_map", "") == uv_map_name:
      return node
  try:
    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = uv_map_name
    uv_node.name = "iEDM_UVMap_{}".format(uv_map_name.replace(".", "_"))
    uv_node.label = uv_map_name
    uv_node.location = (-700, 200 - 140 * ch_idx)
    return uv_node
  except Exception:
    return None


def _texture_uv_channel_map(edm_material):
  mapping = {}
  if edm_material is None:
    return mapping
  channels = getattr(edm_material, "texture_coordinates_channels", None) or []
  for tex_idx, uv_channel in enumerate(channels):
    try:
      uv_idx = int(uv_channel)
    except Exception:
      continue
    if uv_idx >= 0:
      mapping[int(tex_idx)] = uv_idx
  return mapping


def _find_active_material_output(nodes):
  first_output = None
  for node in nodes:
    if node.bl_idname != "ShaderNodeOutputMaterial":
      continue
    if first_output is None:
      first_output = node
    if getattr(node, "is_active_output", False):
      return node
  return first_output


def _link_group_surface_to_output(links, group_node, material_output):
  if not group_node or not material_output:
    return
  to_socket = material_output.inputs.get("Surface")
  if not to_socket:
    return

  from_socket = None
  for socket in group_node.outputs:
    if getattr(socket, "type", None) == "SHADER":
      from_socket = socket
      break
  if not from_socket:
    from_socket = group_node.outputs.get("Shader")
  if not from_socket:
    from_socket = group_node.outputs.get("BSDF")
  if not from_socket:
    return

  for link in list(to_socket.links):
    try:
      links.remove(link)
    except Exception:
      pass
  try:
    links.new(from_socket, to_socket)
  except Exception:
    pass


def _resolve_official_material_tree(bridge, official_material_name):
  descs = bridge.get("material_descs", {})
  desc = descs.get(official_material_name)
  if desc is not None:
    try:
      return desc.create()
    except Exception:
      pass

  node_tree = bpy.data.node_groups.get(official_material_name)
  if node_tree is not None:
    return node_tree

  try:
    return bpy.data.node_groups.new(official_material_name, "ShaderNodeTree")
  except Exception:
    return None


def _attach_official_material_bridge(mat, edm_material, texture_nodes):
  """
  Add an official-EDM-compatible material group node so meshes imported by iEDM
  can be exported by the official exporter addon.
  """
  bridge = _ensure_official_material_bridge()
  if not bridge.get("available"):
    return False

  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  names = bridge.get("names", {})
  official_kind = _map_edm_material_to_official_kind(edm_material.material_name)
  official_name = _map_edm_material_to_official_name(edm_material.material_name, bridge)

  # Reuse existing compatible node if one is already present.
  group_node = None
  for node in nodes:
    node_tree = getattr(node, "node_tree", None)
    if node_tree and node_tree.name == official_name:
      group_node = node
      break

  if group_node is None:
    group_node = _create_official_group_node(nodes, bridge, official_kind, official_name)
    if group_node is None:
      return False
    group_node.location = (320, -360)
    group_node.width = 320

  trans_mode = _official_transparency_enum(getattr(edm_material, "blending", 0))
  shadow_mode = _official_shadow_enum(getattr(edm_material, "shadows", None))

  # Set custom node enums (preferred path for official exporter parser).
  _set_group_enum_property(group_node, "transparency", trans_mode)
  _set_group_enum_property(group_node, "deck_transparency", trans_mode)
  _set_group_enum_property(group_node, "glass_transparency", trans_mode)
  _set_group_enum_property(group_node, "shadow_caster", shadow_mode)
  _set_group_enum_property(group_node, "glass_shadow_caster", shadow_mode)

  if official_name in {names["default"], names["deck"], names["glass"]}:
    _set_group_socket_default(group_node, "Transparency", trans_mode)
  if official_name in {names["default"], names["glass"]}:
    _set_group_socket_default(group_node, "Shadow Caster", shadow_mode)

  # Map common EDM texture channels to official group inputs.
  tex0 = texture_nodes.get(0)  # diffuse/base
  tex1 = texture_nodes.get(1)  # normal
  tex2 = texture_nodes.get(2)  # spec/roughmet-ish
  tex3 = texture_nodes.get(3)  # decal/number atlas in many default materials
  tex_uv_channels = _texture_uv_channel_map(edm_material)

  # Route explicit EDM UV-channel assignments to texture vector inputs so the
  # official exporter can recover texture_coord_channels deterministically.
  for tex_idx, tex_node in texture_nodes.items():
    uv_channel = tex_uv_channels.get(tex_idx, None)
    if uv_channel is None:
      continue
    if int(uv_channel) <= 0:
      continue
    uv_node = _ensure_uv_map_node_for_channel(nodes, uv_channel)
    _link_uv_to_texture_vector(links, uv_node, tex_node)

  if official_name == names["default"]:
    if 3 not in tex_uv_channels and tex3:
      uv_node = _ensure_uv_map_node(nodes)
      if uv_node:
        _link_uv_to_texture_vector(links, uv_node, tex3)
    _link_texture_to_group_input(links, tex0, group_node, "Base Color")
    _link_texture_to_group_input(links, tex3, group_node, "Decal Color")
    _link_texture_to_group_input(links, tex3, group_node, "Decal Alpha*", "Alpha")
    _link_texture_to_group_input(links, tex1, group_node, "Normal (Non-Color)")
    _link_texture_to_group_input(links, tex2, group_node, "RoughMet (Non-Color)")
  elif official_name == names["deck"]:
    _link_texture_to_group_input(links, tex0, group_node, "Tiled Base Color")
    _link_texture_to_group_input(links, tex0, group_node, "Base_Alpha", "Alpha")
    _link_texture_to_group_input(links, tex0, group_node, "Decal Base")
    _link_texture_to_group_input(links, tex0, group_node, "Decal Alpha", "Alpha")
    _link_texture_to_group_input(links, tex1, group_node, "Tiled Normal (Non-Color)")
    _link_texture_to_group_input(links, tex2, group_node, "Tiled RoughMet (Non-Color)")
  elif official_name == names["glass"]:
    _link_texture_to_group_input(links, tex0, group_node, "Diffuse Color (Dirt)")
    _link_texture_to_group_input(links, tex0, group_node, "Diffuse Alpha*", "Alpha")
    _link_texture_to_group_input(links, tex0, group_node, "Glass Color (Color Filter)")
    _link_texture_to_group_input(links, tex0, group_node, "Glass Alpha*", "Alpha")
    _link_texture_to_group_input(links, tex1, group_node, "Normal (Non-Color)")
    _link_texture_to_group_input(links, tex2, group_node, "RoughMet (Non-Color)")
  elif official_name == names["mirror"]:
    _link_texture_to_group_input(links, tex0, group_node, "Base Color")
    _link_texture_to_group_input(links, tex1, group_node, "Normal (Non-Color)")
  else:
    _link_texture_to_group_input(links, tex0, group_node, "Emissive")

  material_output = _find_active_material_output(nodes)
  _link_group_surface_to_output(links, group_node, material_output)
  return True


def _reset_transform_debug(options):
  options = options or {}
  global _transform_debug
  enabled = bool(options.get("debug_transforms", False))
  raw_limit = options.get("debug_transform_limit", 200)
  try:
    limit = max(1, int(raw_limit))
  except Exception:
    limit = 200
  name_filter = options.get("debug_transform_filter", None)
  if isinstance(name_filter, str) and not name_filter.strip():
    name_filter = None
  _transform_debug = {
    "enabled": enabled,
    "filter": name_filter,
    "limit": limit,
    "emitted": 0,
  }
  if enabled:
    filter_str = " filter='{}'".format(name_filter) if name_filter else ""
    print("Info: Transform debug enabled{} limit={}".format(filter_str, limit))


def _reset_mesh_origin_mode(options):
  global _mesh_origin_mode
  options = options or {}
  mode = str(options.get("mesh_origin_mode", "APPROX")).upper()
  if mode not in {"APPROX", "RAW"}:
    mode = "APPROX"
  _mesh_origin_mode = mode


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


def _anim_base_quaternion_q1_to_blender(node, q):
  """Convert only the ArgRotationNode base/rest quaternion when the node's base
  matrix carries the canonical EDM -90°X basis wrapper.

  Rotation key quaternions remain on the existing v10 path; this avoids the
  global basis regression seen when converting all v10 arg quaternions.
  """
  q_in = q if hasattr(q, "to_matrix") else Quaternion(q)
  if _edm_version >= 10 and isinstance(node, ArgRotationNode):
    try:
      if _is_neg90_x_basis_matrix(Matrix(node.base.matrix)):
        return quaternion_to_blender(q_in)
    except Exception:
      pass
  return _anim_quaternion_to_blender(q_in)


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
      # Remove the exporter's fixed 90-degree X rotation instead of stripping all rotation.
      local_mat = local_mat @ Matrix.Rotation(math.radians(-90.0), 4, "X")
    if (
      blender_obj is not None
      and blender_obj.parent is not None
      and _edm_version >= 10
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
    if local_mat is not None and blender_obj is not None and blender_obj.parent is not None and _edm_version >= 10:
      local_mat = blender_obj.parent.matrix_basis @ local_mat
    return local_mat
  return None


def _debug_filter_match(node, path):
  name_filter = _transform_debug.get("filter")
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
  if not _transform_debug.get("enabled"):
    return
  if _transform_debug["emitted"] >= _transform_debug["limit"]:
    return
  if not getattr(node, "blender", None):
    return

  path = _debug_node_path(node)
  if not _debug_filter_match(node, path):
    return

  obj = node.blender
  # Ensure matrix_local/world reflect latest assignments before comparison.
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

  _transform_debug["emitted"] += 1

def iterate_renderNodes(edmFile):
  """Iterates all renderNodes in an edmFile - whilst ignoring any nesting
  due to e.g. renderNode splitting"""
  for node in edmFile.renderNodes:
    if hasattr(node, "children") and node.children:
      for child in node.children:
        yield child
    else:
      yield node

def iterate_all_objects(edmFile):
  for node in itertools.chain(iterate_renderNodes(edmFile), edmFile.connectors, edmFile.shellNodes, edmFile.lightNodes):
    yield node

def build_graph(edmFile):
  "Build a translation graph object from an EDM file"
  graph = TranslationGraph()
  # The first node is ALWAYS the root node
  graph.root.transform = edmFile.nodes[0]
  nodeLookup = {edmFile.nodes[0]: graph.root}

  # Add stable indices to transform nodes for deterministic fallback naming.
  for i, tfnode in enumerate(edmFile.nodes):
    tfnode._graph_idx = i

  # Add an entry for every other transform node
  for tfnode in edmFile.nodes[1:]:
    newNode = TranslationNode()
    newNode.transform = tfnode
    nodeLookup[tfnode] = newNode
    graph.nodes.append(newNode)

  # Make the parent/child links
  for node in graph.nodes:
    if node.transform.parent:
      node.parent = nodeLookup[node.transform.parent]
      node.parent.children.append(node)

  # Verify we only have one root
  assert len([x for x in graph.nodes if not x.parent]) == 1, "More than one root node after reading EDM"

  # Connect every renderNode to it's place in the chain
  for i, node in enumerate(iterate_all_objects(edmFile)):
    node._graph_render_idx = i
    if not hasattr(node, "parent") or node.parent is None:
      if (isinstance(node, (FakeOmniLightsNode, FakeSpotLightsNode, FakeALSNode))
          or type(node).__name__ in ("FakeOmniLightsNode", "FakeSpotLightsNode", "FakeALSNode")):
        owner = graph.root
      else:
        print("Warning: Node {} has no parent attribute; skipping".format(node))
        continue
    elif node.parent not in nodeLookup:
      print("Warning: Node {} parent {} not in lookup; skipping".format(node, node.parent))
      continue
    else:
      owner = nodeLookup[node.parent]
    newNode = TranslationNode()
    newNode.render = node
    graph.attach_node(newNode, owner)

  # Keep transform and render nodes separate in general, but absorb simple
  # connector-helper chains to avoid creating implementation-detail empties.
  def _absorb_connector_helper(node):
    return # DISABLED: absorbing the Connector render payload into its parent TransformNode
           # prevents _collapse_chains from recognising that node as a "Connector Transform"
           # dummy child, leaving a spurious model:TransformNode wrapping model:Connector
           # objects in the Blender scene. Leave disabled so _collapse_chains can handle it.
    if node.type != "TRANSFORM":
      return
    if node.render or len(node.children) != 1:
      return
    child = node.children[0]
    if child.type != "RENDER" or child.children:
      return
    if not isinstance(child.render, Connector):
      return
    node.render = child.render
    graph.remove_node(child)

  graph.walk_tree(_absorb_connector_helper)

  # Collapse redundant TransformNode -> RenderNode chains to reduce graph size.
  # This merges the render component up into the transform node.
  def _collapse_transform_render_chains(node):
    if node.type != "TRANSFORM":
      return

    if node.render or len(node.children) != 1:
      return
    child = node.children[0]
    if child.type != "RENDER":
      return
    # Don't absorb Connector render nodes — absorbing them gives the parent
    # TransformNode a render payload, which prevents _collapse_chains from
    # treating it as a collapsible "Connector Transform" dummy child, leaving
    # a spurious model:TransformNode wrapping model:Connector in the scene.
    if isinstance(child.render, Connector):
      return
    # Only collapse static TransformNodes.
    # Collapsing ArgAnimationNode changes the basis frame for animations if the
    # child RenderNode had a rotation (e.g. from Y-up to Z-up), leading to
    # incorrect orientation or animation axes.
    if not isinstance(node.transform, TransformNode):
      return

    # Move render payload to parent
    node.render = child.render
    # Remove child
    graph.remove_node(child)

  graph.walk_tree(_collapse_transform_render_chains)

  # --- Pre-compute local/world matrices in Blender space for graph simplification ---
  graph.root._is_graph_root = True
  
  # For V10+, the graph root carries the global basis fix.
  m_root_edm = Matrix.Identity(4)
  root_tf = graph.root.transform
  if root_tf:
    if hasattr(root_tf, "matrix"):
      m_root_edm = Matrix(root_tf.matrix)
    elif hasattr(root_tf, "base") and hasattr(root_tf.base, "matrix"):
      m_root_edm = Matrix(root_tf.base.matrix)

  if _edm_version >= 10:
    graph.root._local_bl = _ROOT_BASIS_FIX @ m_root_edm
  else:
    graph.root._local_bl = Matrix.Identity(4)
    
  graph.root._world_bl = graph.root._local_bl.copy()

  for node in graph.nodes:
    if node == graph.root:
      continue
    m_edm = Matrix.Identity(4)
    tf = node.transform
    if tf:
      if hasattr(tf, "matrix"):
        m_edm = Matrix(tf.matrix)
      elif hasattr(tf, "base") and hasattr(tf.base, "matrix"):
        m_edm = Matrix(tf.base.matrix)
      elif isinstance(tf, Bone) and hasattr(tf, "data") and len(tf.data) >= 1:
        m_edm = Matrix(tf.data[0])
    rn = node.render
    if rn:
      m_rn = Matrix.Identity(4)
      if hasattr(rn, "pos"):
        m_rn = Matrix.Translation(Vector(rn.pos[:3]))
      elif hasattr(rn, "matrix"):
        m_rn = Matrix(rn.matrix)
      
      if not m_rn.is_identity:
        m_edm = m_edm @ m_rn

    # Convert to Blender space for graph decisions.
    # For V10, no per-node axis conversion; V8 uses a full axis swap.
    node._local_bl = m_edm

  # Calculate world matrices recursively
  _visited_world = set()
  def _calculate_world_bl(node):
    if id(node) in _visited_world:
      return
    _visited_world.add(id(node))
    if node.parent and hasattr(node.parent, "_world_bl"):
      node._world_bl = node.parent._world_bl @ node._local_bl
    elif not hasattr(node, "_world_bl"):
      node._world_bl = node._local_bl.copy()
    for child in node.children:
      _calculate_world_bl(child)

  _calculate_world_bl(graph.root)

  # --- Mark skeleton nodes before collapsing ---
  for n in graph.nodes:
    n._is_skeleton = is_skeleton_node(n) or (n.render is not None)

  # --- Surgical elimination of exporter artifacts ---
  nodes_to_remove = []
  for n in graph.nodes:
    if n == graph.root:
      continue
    if n.render is not None:
      continue
    name = _transform_display_name(n.transform)
    # Skip identity placeholder "root" nodes directly under graph root
    if name in {"root", ""} and n.parent == graph.root and n._local_bl.is_identity:
      nodes_to_remove.append(n)
      continue
    # Skip "Connector Transform" and "Fake Light Transform" wrapper artifacts
    if name in {"Connector Transform", "Fake Light Transform"} and n.parent:
      nodes_to_remove.append(n)
      continue

  for n in nodes_to_remove:
    p = n.parent
    if p is None:
      continue
    insert_at = p.children.index(n) if n in p.children else len(p.children)
    if n in p.children:
      p.children.pop(insert_at)
    for child in list(n.children):
      child.parent = p
      p.children.insert(insert_at, child)
      insert_at += 1
      child._local_bl = n._local_bl @ child._local_bl
    if n in graph.nodes:
      graph.nodes.remove(n)

  # Recalculate world matrices after elimination
  _visited_world.clear()
  _calculate_world_bl(graph.root)

  # --- Collapse redundant single-child transform chains ---
  def _collapse_chains(node):
    if not node.children or node.render:
      return
    if not hasattr(node, "_collapsed_transforms"):
      node._collapsed_transforms = [node.transform] if node.transform else []

    if len(node.children) == 1:
      child = node.children[0]
      if child.transform and not child.render:
        # Don't collapse skeleton nodes (preserves Empties for animation)
        if getattr(node, "_is_skeleton", False) or getattr(child, "_is_skeleton", False):
          return
        name_self = _transform_display_name(node.transform)
        name_child = _transform_display_name(child.transform)
        is_dummy_child = name_child in {"Connector Transform", "Fake Light Transform", ""}
        if (name_self and name_self == name_child) or is_dummy_child:
          node._local_bl = node._local_bl @ child._local_bl
          child_tfs = getattr(child, "_collapsed_transforms", [child.transform] if child.transform else [])
          node._collapsed_transforms.extend(child_tfs)
          for grandchild in list(child.children):
            grandchild.parent = node
            node.children.append(grandchild)
          node.children.remove(child)
          if child in graph.nodes:
            graph.nodes.remove(child)
          _collapse_chains(node)

  graph.walk_tree(_collapse_chains)

  # Recalculate world matrices after chain collapse
  _visited_world.clear()
  _calculate_world_bl(graph.root)

  # --- Collapse single generic-named render child into parent transform ---
  def _collapse_to_mesh(node):
    if node.transform is None or node.render is not None:
      return
    if len(node.children) != 1:
      return
    child = node.children[0]
    child_name = getattr(child.render, "name", "")
    if not child.render:
      return
    tf = node.transform

    # Existing behavior: collapse obvious implementation-detail render-only
    # children with generic names.
    # Do not collapse generic chunk renders under animated transform nodes
    # (e.g. v10 `_0`/`_1` mesh chunks under ArgRotationNode), because that
    # changes control-node identity and can move transforms/naming onto mesh
    # objects. Keep visibility nodes exempt so visibility wrappers continue
    # to collapse onto the controlled object.
    generic_child = _is_generic_render_name(child_name)
    is_anim_tf_for_generic_guard = isinstance(tf, AnimatingNode) and not isinstance(tf, ArgVisibilityNode)
    should_collapse = generic_child and not is_anim_tf_for_generic_guard

    # Also collapse the common v10 pattern where an animated transform node
    # owns a render-only child with the same semantic name. Keeping these
    # split creates an extra EMPTY helper in Blender and inflates exported
    # TransformNode chains (`v_* -> name -> name.001_mat -> ...`).
    if not should_collapse:
      tf_name = _transform_display_name(tf) or ""
      render_name = child_name or ""

      def _norm_sem_name(s):
        s = (s or "").strip()
        if s.startswith("v_"):
          s = s[2:]
        if s.startswith("Empty_"):
          s = s[len("Empty_"):]
        # Treat Blender duplicate suffixes as the same semantic node name
        # for importer-side helper/render split compaction (e.g. Foo/Foo.001).
        if len(s) > 4 and s[-4] == "." and s[-3:].isdigit():
          s = s[:-4]
        return s

      same_sem_name = _norm_sem_name(tf_name) and (_norm_sem_name(tf_name) == _norm_sem_name(render_name))
      is_anim_tf = isinstance(tf, AnimatingNode) and not isinstance(tf, (ArgVisibilityNode, Bone, ArgAnimatedBone))
      is_static_tf = (not isinstance(tf, AnimatingNode)) and not isinstance(tf, (ArgVisibilityNode, Bone, ArgAnimatedBone))
      render_shared = getattr(child.render, "shared_parent", None) is not None
      render_cls_name = type(child.render).__name__
      is_mesh_like_render = render_cls_name in {"RenderNode", "ShellNode", "NumberNode"}
      bone_related = _is_bone_transform(tf)
      if (
        is_anim_tf
        and same_sem_name
        and not render_shared
        and is_mesh_like_render
        and not bone_related
      ):
        should_collapse = True
      elif (
        is_static_tf
        and same_sem_name
        and not render_shared
        and is_mesh_like_render
        and not bone_related
      ):
        should_collapse = True

    if should_collapse:
      node.render = child.render
      node.children = child.children
      for c in node.children:
        c.parent = node
      if child in graph.nodes:
        graph.nodes.remove(child)
      _collapse_to_mesh(node)

  graph.walk_tree(_collapse_to_mesh)

  # Deterministic traversal ordering for round-trip parity:
  # preserve original transform stream ordering first, and only use render
  # stream ordering for render-only nodes that have no transform owner.
  def _child_sort_key(child):
    if child.transform is not None:
      tf_idx = getattr(child.transform, "_graph_idx", 1 << 30)
      tf_name = getattr(child.transform, "name", "") or ""
      return (0, tf_idx, tf_name)

    render_idx = 1 << 30
    if child.render is not None:
      render_idx = getattr(child.render, "_graph_render_idx", render_idx)
    render_name = getattr(child.render, "name", "") if child.render is not None else ""
    return (1, render_idx, render_name)

  def _sort_children(node):
    if node.children:
      node.children.sort(key=_child_sort_key)

  graph.walk_tree(_sort_children)

  return graph


_SEMANTIC_NAME_MAP = {}


def _push_action_to_nla(ob, action):
  """Push action onto an NLA track when supported.

  Official EDM exporter only supports NLA on armatures. Returning False lets
  callers fall back to a regular active action for non-armature objects.
  """
  if ob is None or getattr(ob, "type", "") != "ARMATURE":
    return False
  if not ob.animation_data:
    ob.animation_data_create()
  track = ob.animation_data.nla_tracks.new()
  track.name = action.name
  track.strips.new(action.name, 1, action)
  return True


def _assign_collections(graph):
  """Create named import collections and assign Blender objects to them.

  Rules:
  - ShellNode / SegmentsNode meshes and their ancestor empties -> "Collision"
  - RenderNode meshes that have a Blender parent (animated) -> "Vehicle"
  - RenderNode meshes without a Blender parent (root-level) -> "Texture_Animation"
  - Other nodes (SkinNode, NumberNode, LightNode, FakeLight) stay in Scene Collection
  - Empties follow the category of their children
  """
  scene = bpy.context.scene

  def _get_or_create_col(name):
    col = bpy.data.collections.get(name)
    if col is None:
      col = bpy.data.collections.new(name)
      scene.collection.children.link(col)
    return col

  col_vehicle = _get_or_create_col("Vehicle")
  col_collision = _get_or_create_col("Collision")
  col_tex_anim = _get_or_create_col("Texture_Animation")

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
    elif rtype == "RenderNode":
      if n.blender.parent is None:
        obj_category[n.blender] = "texture_anim"
      else:
        obj_category[n.blender] = "vehicle"
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

  _col_map = {
    "collision": col_collision,
    "vehicle": col_vehicle,
    "texture_anim": col_tex_anim,
  }

  for obj, cat in obj_category.items():
    target = _col_map.get(cat)
    if target is None:
      continue
    for cur_col in list(obj.users_collection):
      try:
        cur_col.objects.unlink(obj)
      except Exception:
        pass
    try:
      target.objects.link(obj)
    except RuntimeError:
      pass


def create_bounding_box_from_root(root):
  """Create an empty cube representing the EDM bounding box."""
  if not hasattr(root, "boundingBoxMin"):
    return None
  bmin = vector_to_blender(root.boundingBoxMin)
  bmax = vector_to_blender(root.boundingBoxMax)
  center = (bmin + bmax) * 0.5
  dims = Vector((abs(bmax.x - bmin.x), abs(bmax.y - bmin.y), abs(bmax.z - bmin.z)))
  ob = bpy.data.objects.new("Bounding_Box", None)
  ob.empty_display_type = "CUBE"
  ob.location = center
  ob.scale = dims * 0.5
  ob.empty_display_size = 1.0
  _set_official_special_type(ob, "BOUNDING_BOX")
  bpy.context.collection.objects.link(ob)
  return ob


def _has_special_box(special_type):
  """Return True if a scene object already represents a given special box type."""
  for obj in bpy.data.objects:
    if hasattr(obj, "EDMProps") and getattr(obj.EDMProps, "SPECIAL_TYPE", "") == special_type:
      return True
  return False


def _create_user_box_from_root(root):
  """Create an empty cube representing the EDM user box (unknownB)."""
  if not hasattr(root, "unknownB") or len(root.unknownB) < 2:
    return None
  min_vec_edm = Vector(root.unknownB[0])
  max_vec_edm = Vector(root.unknownB[1])
  if any(math.isinf(v) for v in min_vec_edm) or any(math.isinf(v) for v in max_vec_edm):
    return None
  if hasattr(root, "boundingBoxMin") and hasattr(root, "boundingBoxMax"):
    if Vector(root.boundingBoxMin) == min_vec_edm and Vector(root.boundingBoxMax) == max_vec_edm:
      return None
  min_vec = vector_to_blender(min_vec_edm)
  max_vec = vector_to_blender(max_vec_edm)
  center = (min_vec + max_vec) * 0.5
  dims = Vector((abs(max_vec.x - min_vec.x), abs(max_vec.y - min_vec.y), abs(max_vec.z - min_vec.z)))
  ob = bpy.data.objects.new("User_Box", None)
  ob.empty_display_type = "CUBE"
  ob.location = center
  ob.scale = dims * 0.5
  ob.empty_display_size = 1.0
  _set_official_special_type(ob, "USER_BOX")
  bpy.context.collection.objects.link(ob)
  return ob


def process_node(node):
  """Processes a single node of the transform graph"""
  _used_shared_parent_fallback = False
  # Root node has no processing
  if node.parent is None:
    return

  node._is_primary = False

  # Usually collapse visibility wrappers into the child object (tighter export
  # shape), but preserve them when the wrapped child carries transform animation.
  # In that case, keeping a separate wrapper avoids the exporter's one-action-per-
  # object limitation (visibility and transform args can differ).
  if node.render is None and isinstance(node.transform, ArgVisibilityNode):
    if not _visibility_wrapper_needs_own_object(node):
      node.blender = node.parent.blender
      return

  # EDM bone-control chains are represented by a single imported armature.
  ctx = _bone_import_ctx or {}
  arm_obj = ctx.get("armature")
  bone_chain_nodes = ctx.get("bone_chain_nodes", set())
  if arm_obj is not None and node.render is None and node in bone_chain_nodes:
    node.blender = arm_obj
    return

  is_skel = getattr(node, "_is_skeleton", False)

  # --- Create Blender object for this node ---
  node.render_blender = None
  if node.render:
    render_name = getattr(node.render, "name", "") or ""
    name_unknown = getattr(node.render, "name_unknown", False)
    if isinstance(node.render, NumberNode) and render_name.endswith("_Number"):
      name_unknown = True
    is_connector_render = isinstance(node.render, Connector)

    final_name = render_name

    if _is_generic_render_name(final_name) or name_unknown:
      # Avoid consuming semantic base names (e.g. Aileron_L / Flap_L) for generic
      # connector objects. Visibility wrappers later export as `v_ + obj.name`, so
      # if a connector grabs the base first Blender suffixes the wrapper (.001/.002).
      # Connector object names are not semantically important for parity here.
      if is_connector_render:
        candidate = ""
        if node.parent and node.parent.transform:
          candidate = _transform_display_name(node.parent.transform)
          if candidate.startswith("v_"):
            candidate = candidate[2:]
          candidate = _strip_anim_prefix(candidate)
          if candidate.startswith("Empty_"):
            candidate = candidate[len("Empty_"):]
        if candidate and candidate.lower() != "root":
          final_name = "Connector_{}".format(candidate)
        else:
          final_name = "Connector"

      if (_is_generic_render_name(final_name) or name_unknown) and isinstance(node.transform, ArgVisibilityNode):
        candidate = getattr(node.transform, "name", "") or ""
        if candidate.startswith("v_"):
          candidate = candidate[2:]
        if candidate.startswith("Empty_"):
          candidate = candidate[len("Empty_"):]
        if candidate and candidate.lower() != "root" and not _is_generic_render_name(candidate):
          final_name = candidate

      if (_is_generic_render_name(final_name) or name_unknown) and node.parent and node.parent.transform:
        candidate = _transform_display_name(node.parent.transform)
        if isinstance(node.parent.transform, ArgVisibilityNode) and candidate.startswith("v_"):
          candidate = candidate[2:]
        if candidate.startswith("Empty_"):
          candidate = candidate[len("Empty_"):]
        if candidate and candidate.lower() != "root" and not _is_generic_render_name(candidate):
          final_name = candidate

    if _is_generic_render_name(final_name) or name_unknown:
      mat_name = ""
      if hasattr(node.render, "material") and node.render.material:
        mat_name = getattr(node.render.material, "name", "") or ""
      if mat_name:
        final_name = _SEMANTIC_NAME_MAP.get(mat_name, mat_name)

    final_name = _SEMANTIC_NAME_MAP.get(final_name, final_name)

    if _is_generic_render_name(final_name):
      ridx = getattr(node.render, "_graph_render_idx", None)
      if ridx is not None:
        final_name = "rn_{:04d}".format(ridx)

    if final_name:
      node.render.name = final_name

    if isinstance(node.render, Connector):
      node.blender = create_connector(node.render)
      # Store the original wrapper TransformNode name so the export patch can
      # produce TransformNode "BANO_0" instead of "Connector Transform" etc.
      # In original EDMs the wrapper transform has the connector's own name.
      try:
        if (node.transform is not None
            and not isinstance(node.transform, AnimatingNode)):
          _ctf_name = getattr(node.transform, "name", "") or ""
          if _ctf_name and _ctf_name != "Connector Transform":
            node.blender["_iedm_connector_tf_name"] = _ctf_name
      except Exception:
        pass
    elif isinstance(node.render, (RenderNode, ShellNode, NumberNode, SkinNode)):
      node.blender = create_object(node.render)
    elif isinstance(node.render, SegmentsNode):
      node.blender = create_segments(node.render)
    elif isinstance(node.render, LightNode):
      node.blender = create_lamp(node.render)
    elif isinstance(node.render, FakeOmniLightsNode) or type(node.render).__name__ == "FakeOmniLightsNode":
      node.blender = create_fake_omni_lights(node.render)
    elif isinstance(node.render, FakeSpotLightsNode) or type(node.render).__name__ == "FakeSpotLightsNode":
      node.blender = create_fake_spot_lights(node.render)
    elif isinstance(node.render, FakeALSNode) or type(node.render).__name__ == "FakeALSNode":
      node.blender = create_fake_als_lights(node.render)
    else:
      print("Warning: No case yet for object node {}".format(node.render))

  elif is_skel:
    empty_name = _transform_display_name(node.transform) or type(node.transform).__name__
    ob = bpy.data.objects.new(empty_name, None)
    ob.empty_display_size = 0.1
    bpy.context.collection.objects.link(ob)
    node.blender = ob
  else:
    tf_name = _transform_display_name(node.transform) or type(node.transform).__name__
    ob = bpy.data.objects.new(tf_name, None)
    ob.empty_display_size = 0.1
    bpy.context.collection.objects.link(ob)
    node.blender = ob

  if node.blender and isinstance(node.render, SkinNode):
    _bind_skin_object(node.blender, node.render)

  if not node.blender:
    node.blender = node.parent.blender if node.parent else None
    return

  node._is_primary = True

  # --- (2) Parent in a “parity-safe” way: identity parent inverse ---
  if node.parent and node.parent.blender and node.parent.blender != node.blender:
    node.blender.parent = node.parent.blender
    node.blender.matrix_parent_inverse = Matrix.Identity(4)

  # Preserve mapping from EDM transform node -> created blender object
  if node.transform and node.blender:
    try:
      node.transform._blender_obj = node.blender
    except Exception:
      pass
    if isinstance(node.transform, ArgVisibilityNode):
      try:
        vis_alias = getattr(node.transform, "name", "") or ""
        # Store the FULL original EDM name (do not strip "v_").
        # The visibility export patch (_extract_visibility_animation) uses the
        # alias as-is, so "v_Gas_Tanks" → exports "v_Gas_Tanks" and plain
        # "Dummy001" → exports "Dummy001" (no extra "v_" prepended).
        # Only strip ar_/al_/as_ prefixes which are never part of the intended
        # visibility node name in a well-formed EDM.
        for _pfx in ("ar_", "al_", "as_"):
          if vis_alias.startswith(_pfx):
            vis_alias = vis_alias[len(_pfx):]
            break
        if vis_alias:
          node.blender["_iedm_vis_export_name"] = vis_alias
      except Exception:
        pass
      try:
        if node.render is None and getattr(node.blender, "type", None) == 'EMPTY':
          node.blender["_iedm_vis_passthrough"] = True
      except Exception:
        pass
    else:
      # Mark importer-created static helper empties that only contribute an
      # identity transform and exist due Blender duplicate-name splitting
      # (`Foo.001`, `Foo.002`, ...). The exporter otherwise emits no-op
      # TransformNodes for these helpers.
      try:
        ob = node.blender
        tf_name = getattr(node.transform, "name", "") or ""
        ob_name = getattr(ob, "name", "") or ""
        is_static_tf = not isinstance(node.transform, AnimatingNode)
        is_renderless_empty = (node.render is None and getattr(ob, "type", None) == 'EMPTY')
        has_one_child = len(getattr(node, "children", []) or []) == 1
        has_blender_dup_suffix = (len(ob_name) > 4 and ob_name[-4] == "." and ob_name[-3:].isdigit())
        not_bone_related = not isinstance(node.transform, (Bone, ArgAnimatedBone))
        if (
          is_static_tf
          and is_renderless_empty
          and has_one_child
          and has_blender_dup_suffix
          and not_bone_related
          and _ob_local_is_identity(ob)
        ):
          ob["_iedm_identity_passthrough"] = True
          if _is_narrow_safe_identity_helper_name(ob_name) or _is_narrow_safe_identity_helper_name(tf_name):
            ob["_iedm_narrow_identity_passthrough"] = True
        # Also allow a narrow fake-light helper case with no Blender suffix:
        # v_* -> Omni_r001 (identity EMPTY) -> LightNode
        elif (
          is_static_tf
          and is_renderless_empty
          and has_one_child
          and not_bone_related
          and isinstance(getattr(node.parent, "transform", None), ArgVisibilityNode)
          and _ob_local_is_identity(ob)
        ):
          try:
            child0 = (getattr(node, "children", []) or [None])[0]
            child_render_cls = type(getattr(child0, "render", None)).__name__ if child0 is not None else None
            child_tf_name = str(getattr(getattr(child0, "transform", None), "name", "") or "")
          except Exception:
            child_render_cls = None
            child_tf_name = ""
          if child_render_cls == "LightNode" or child_tf_name == "Fake Light Transform":
            ob["_iedm_identity_passthrough"] = True
            ob["_iedm_narrow_identity_passthrough"] = True
      except Exception:
        pass

  # Store the original EDM animation-node name so the export name-passthrough
  # patch can emit the plain name (e.g. "Dummy001", "pilot_SU_helmet_glass001")
  # instead of the prefixed name emitted by io_scene_edm ("ar_Dummy001").
  # Applies to all animated nodes — both pure empties AND animated mesh objects
  # — when the original EDM name has no al_/ar_/as_ prefix.  The proxy only
  # fires when the property is present, so prefixed names in the original are
  # never mis-renamed.
  if (node.transform is not None
      and isinstance(node.transform, AnimatingNode)
      and not isinstance(node.transform, (Bone, ArgAnimatedBone))):
    try:
      orig_anim_name = getattr(node.transform, "name", "") or ""
      if (orig_anim_name
          and not orig_anim_name.startswith("al_")
          and not orig_anim_name.startswith("ar_")
          and not orig_anim_name.startswith("as_")):
        node.blender["_iedm_orig_anim_name"] = orig_anim_name
    except Exception:
      pass

  # Temporary importer debug stamp: helps correlate Blender objects back to source
  # EDM transform/render classes while diagnosing cross-asset basis issues.
  try:
    if node.blender:
      node.blender["_iedm_dbg_tf_cls"] = type(node.transform).__name__ if node.transform is not None else ""
      node.blender["_iedm_dbg_r_cls"] = type(node.render).__name__ if node.render is not None else ""
      if node.transform is not None:
        node.blender["_iedm_dbg_tf_name"] = str(getattr(node.transform, "name", "") or "")
      if node.render is not None:
        node.blender["_iedm_dbg_r_name"] = str(getattr(node.render, "name", "") or "")
      try:
        src_m = None
        tf = node.transform
        if tf is not None:
          if hasattr(tf, "matrix"):
            src_m = Matrix(tf.matrix)
          elif hasattr(getattr(tf, "base", None), "matrix"):
            src_m = Matrix(tf.base.matrix)
          elif hasattr(getattr(tf, "base", None), "transform"):
            src_m = Matrix(getattr(tf.base, "transform"))
        if src_m is not None:
          src_rows = []
          for r in range(3):
            src_rows.append([round(float(src_m[r][c]), 4) for c in range(3)])
          node.blender["_iedm_dbg_src_rows"] = str(src_rows)
      except Exception:
        pass
  except Exception:
    pass

  # --- Render-only positioning & v10 basis handling ---
  _is_render_only_positioning = (
    node.render and node.blender and node.blender.type == "MESH"
    and (not node.transform or isinstance(node.transform, ArgVisibilityNode))
  )

  if _is_render_only_positioning:
    shared_parent = getattr(node.render, "shared_parent", None)
    _used_shared_parent_fallback = False
    
    # Parity: Always factor in shared_parent transform for V10 chunks.
    if shared_parent is not None:
      apply_node_transform(shared_parent, node.blender, used_shared_parent=True)
      _used_shared_parent_fallback = True

    if shared_parent is not None and node.parent and node.parent.blender:
      src = getattr(shared_parent, "_blender_obj", None)
      dst = node.parent.blender
      if src is not None:
        try:
          node.blender.matrix_local = dst.matrix_world.inverted() @ src.matrix_world
        except Exception:
          pass
      else:
        # If the shared parent transform was collapsed (no Blender object was created
        # for it), apply that EDM transform directly to the render-only mesh object so
        # the exporter control node does not become an identity wrapper under v_*.
        try:
          if getattr(shared_parent, "category", None) is NodeCategory.transform:
            apply_node_transform(shared_parent, node.blender, used_shared_parent=True)
            _used_shared_parent_fallback = True
        except Exception:
          pass

      if _edm_version >= 10:
        # If a visibility wrapper is acting as a control node, keep mesh aligned.
        if (
          node.parent and isinstance(node.parent.transform, ArgVisibilityNode) and node.parent.blender
          and not _used_shared_parent_fallback
        ):
          parent_loc = node.parent.blender.matrix_basis.decompose()[0]
          _offset_mesh_world(node.blender, parent_loc)

    # Optional pivot recentering (kept as you had it)
    if (
      _mesh_origin_mode == "APPROX"
      and node.parent and node.parent.blender
      and not isinstance(node.parent.transform, ArgVisibilityNode)
    ):
      if node.blender.location.length < 1e-9:
        _recenter_mesh_object_to_geometry(node.blender)
        if _edm_version >= 10:
          node.blender.location = (node.parent.blender.matrix_basis @ node.blender.location.to_4d()).to_3d()

  # Mark render-only Blender-suffixed duplicate helpers that ended up as
  # identity transforms after importer positioning. Exporter can skip their
  # no-op object transform wrapper safely.
  try:
    if node.blender and node.render is not None and node.transform is None:
      ob = node.blender
      ob_name = getattr(ob, "name", "") or ""
      has_parent = getattr(ob, "parent", None) is not None
      has_blender_dup_suffix = (len(ob_name) > 4 and ob_name[-4] == "." and ob_name[-3:].isdigit())
      parent_is_vis = isinstance(getattr(node.parent, "transform", None), ArgVisibilityNode)
      render_cls_name = type(node.render).__name__
      is_fake_light_render = render_cls_name in {"FakeOmniLightsNode", "FakeSpotLightsNode"}
      if (
        has_parent
        and _ob_local_is_identity(ob)
        and (
          has_blender_dup_suffix
          or (parent_is_vis and is_fake_light_render)
        )
      ):
        ob["_iedm_identity_passthrough"] = True
        if has_blender_dup_suffix and _is_narrow_safe_identity_helper_name(ob_name):
          ob["_iedm_narrow_identity_passthrough"] = True
        if parent_is_vis and is_fake_light_render:
          ob["_iedm_narrow_identity_passthrough"] = True
  except Exception:
    pass

  # Also handle combined transform+render fake-light helper nodes that are
  # identity under a visibility wrapper (e.g. v_* -> Omni_r001 -> Fake Light Transform).
  try:
    if node.blender and node.render is not None and node.transform is not None:
      ob = node.blender
      parent_is_vis = isinstance(getattr(node.parent, "transform", None), ArgVisibilityNode)
      is_anim_tf = isinstance(node.transform, AnimatingNode)
      render_cls_name = type(node.render).__name__
      is_fake_light_render = render_cls_name in {"FakeOmniLightsNode", "FakeSpotLightsNode"}
      if parent_is_vis and is_fake_light_render and (not is_anim_tf) and _ob_local_is_identity(ob):
        ob["_iedm_identity_passthrough"] = True
        ob["_iedm_narrow_identity_passthrough"] = True
  except Exception:
    pass

  # --- Visibility animation hookup ---
  _vis_source = None
  if isinstance(node.transform, ArgVisibilityNode):
    _vis_source = node.transform
  elif (
    node.parent
    and isinstance(node.parent.transform, ArgVisibilityNode)
    and node.parent.blender == node.blender
  ):
    # Only inherit visibility from parent wrapper if that wrapper was collapsed
    # onto this same Blender object.
    _vis_source = node.parent.transform

  vis_actions = get_actions_for_node(_vis_source) if _vis_source else []

  # --- Materials ---
  material_target = node.render_blender if node.render_blender else node.blender
  if (
    hasattr(node.render, "material")
    and node.render.material
    and hasattr(node.render.material, "blender_material")
    and node.render.material.blender_material
    and material_target
    and hasattr(material_target, "data")
    and material_target.data
  ):
    material_target.data.materials.append(node.render.material.blender_material)

  # --- Apply transform animation / base transforms (non-visibility transforms only) ---
  if node.transform and not isinstance(node.transform, ArgVisibilityNode):
    all_transforms = getattr(node, "_collapsed_transforms", [node.transform] if node.transform else [])
    anim_transforms = [tf for tf in all_transforms if isinstance(tf, AnimatingNode)]
    bone_anim_source_nodes = ctx.get("bone_anim_source_nodes", set())
    bone_anim_source_transforms = ctx.get("bone_anim_source_transforms", set())
    skip_object_anim = (
      node in bone_anim_source_nodes
      or any(tf in bone_anim_source_transforms for tf in anim_transforms)
    )

    if anim_transforms and not skip_object_anim:
      actions = get_actions_for_node(node.transform)
      for extra_tf in anim_transforms:
        if extra_tf is not node.transform:
          actions.extend(get_actions_for_node(extra_tf))

      if actions:
        if vis_actions and len(actions) == 1:
          actions = list(actions)
          actions[0] = _merge_visibility_action_into_transform_action(actions[0], vis_actions[0], node.blender.name)
        node.blender.animation_data_create()
        if len(actions) == 1:
          node.blender.animation_data.action = actions[0]
        else:
          nla_pushed = 0
          for action in actions:
            if _push_action_to_nla(node.blender, action):
              nla_pushed += 1
          node.blender.animation_data.action = None if nla_pushed > 0 else actions[0]

    apply_node_transform(node, node.blender, used_shared_parent=_used_shared_parent_fallback)

    if node.blender.type == "EMPTY":
      distFromScale = node.blender.scale - Vector((1, 1, 1))
      if distFromScale.length < 0.01:
          node.blender.empty_display_size = 0.01
    elif vis_actions:
      node.blender.animation_data_create()
      node.blender.animation_data.action = vis_actions[0]

  elif vis_actions:
    node.blender.animation_data_create()
    node.blender.animation_data.action = vis_actions[0]

  _compact_visibility_identity_intermediate(node)

  # Diagnostic: Dump all numeric/vector attributes of the EDM node to Blender
  # This allows the user to see what is being ignored by the parser.
  if node.blender:
    def _dump_diag(target, prefix="EDM_RAW_"):
      for attr in dir(target):
        if attr.startswith("_") or attr == "blender" or attr == "children" or attr == "parent":
          continue
        val = getattr(target, attr)
        if isinstance(val, (int, float, str, bool)):
          try:
            node.blender[prefix + attr] = val
          except (OverflowError, ValueError):
            pass
        elif isinstance(val, (list, tuple)) and len(val) <= 16:
          try:
            # Try to store as property (Blender supports arrays)
            node.blender[prefix + attr] = val
          except Exception:
            pass

    if node.transform:
      _dump_diag(node.transform, "EDM_TF_")
      if hasattr(node.transform, "base"):
          _dump_diag(node.transform.base, "EDM_BASE_")
    if node.render:
      _dump_diag(node.render, "EDM_RN_")
    
    # Store the importer's calculated Blender local matrix
    if hasattr(node, "_local_bl"):
      node.blender["IEDM_LOCAL_BL_MAT"] = [v for row in node._local_bl for v in row]

  _debug_dump_node_transform(node)

  if isinstance(node.transform, LodNode):
    node._lod_post_children = True


def _process_lod_post_children(node):
  """Apply LodNode properties after children have been processed."""
  if not getattr(node, "_lod_post_children", False):
    return
  assert node.blender.type == "EMPTY"
  node.blender.edm.is_lod_root = True
  for (start, end), child in zip(node.transform.level, node.children):
    child.blender.edm.lod_min_distance = start
    child.blender.edm.lod_max_distance = end
    child.blender.edm.nouse_lod_distance = end > 1e6


def _apply_shadeless(mat):
  """Make a material 'shadeless' by routing the Base Color texture to Emission Color
  on the Principled BSDF node, so it renders without lighting."""
  if not mat.use_nodes:
    return
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links
  # Find the Principled BSDF node
  principled = None
  for node in nodes:
    if node.type == 'BSDF_PRINCIPLED':
      principled = node
      break
  if not principled:
    return
  # Find what's connected to Base Color and duplicate that link to Emission Color
  base_color_input = principled.inputs['Base Color']
  if base_color_input.links:
    source_socket = base_color_input.links[0].from_socket
    links.new(source_socket, principled.inputs['Emission Color'])
    principled.inputs['Emission Strength'].default_value = 1.0


def _recenter_mesh_object_to_geometry(obj):
  """Move mesh object origin to local bounds center, preserving world mesh."""
  if obj.type != "MESH" or not obj.data or not obj.data.vertices:
    return
  min_v = Vector((float('inf'), float('inf'), float('inf')))
  max_v = Vector((float('-inf'), float('-inf'), float('-inf')))
  for v in obj.data.vertices:
    min_v.x = min(min_v.x, v.co.x)
    min_v.y = min(min_v.y, v.co.y)
    min_v.z = min(min_v.z, v.co.z)
    max_v.x = max(max_v.x, v.co.x)
    max_v.y = max(max_v.y, v.co.y)
    max_v.z = max(max_v.z, v.co.z)
  center = (min_v + max_v) * 0.5
  if center.length < 1e-8:
    return
  for v in obj.data.vertices:
    v.co -= center
  # Mesh vertex coordinates are already in Blender local space here.
  obj.location += center
  obj.data.update()


def _transform_mesh_data(obj, matrix):
  """Apply a mesh-space transform to object data in-place."""
  if obj.type != "MESH" or not obj.data:
    return
  obj.data.transform(matrix)
  obj.data.update()


def _offset_mesh_world(obj, delta_world):
  """Translate mesh vertices by a world-space vector, preserving object transforms."""
  if delta_world.length < 1e-9:
    return
  if obj.type != "MESH" or not obj.data:
    return
  local_delta = obj.matrix_world.inverted().to_3x3() @ delta_world
  _transform_mesh_data(obj, Matrix.Translation(local_delta))


def _is_identity_matrix_approx(mat, eps=1e-6):
  try:
    ident = Matrix.Identity(4)
    for r in range(4):
      for c in range(4):
        if abs(float(mat[r][c]) - float(ident[r][c])) > eps:
          return False
    return True
  except Exception:
    return False


def _is_fake_light_render_node(render):
  if render is None:
    return False
  return type(render).__name__ in ("FakeOmniLightsNode", "FakeSpotLightsNode", "FakeALSNode")


def _compact_visibility_identity_intermediate(node):
  """Hoist fake-light helper transforms toward the semantic node under a v_* wrapper.

  Handles both patterns:
    v_* -> identity helper -> fake-light
    v_* -> semantic(identity) -> transformed helper -> fake-light
  """
  if node is None or not getattr(node, "blender", None):
    return
  _renderless_helper_mode = (
    getattr(node, "render", None) is None
    and isinstance(getattr(node, "transform", None), TransformNode)
  )
  if getattr(node, "render", None) is None and not _renderless_helper_mode:
    return

  fake_obj = node.blender
  def _trace(msg):
    return

  if _renderless_helper_mode:
    helper_obj = node.blender
    fake_obj = None
    _trace("renderless-helper mode")
  else:
    helper_obj = getattr(fake_obj, "parent", None)
    if helper_obj is None or getattr(helper_obj, "type", None) != "EMPTY":
      _trace("skip helper_obj parent missing/non-empty type={}".format(getattr(helper_obj, "type", None) if helper_obj else None))
      return
  _trace("objs helper={} helper_parent={}".format(getattr(helper_obj, "name", None), getattr(getattr(helper_obj, "parent", None), "name", None)))
  # Graph chain from helper transform node (either current renderless node, or parent of render node).
  helper_graph = node if _renderless_helper_mode else getattr(node, "parent", None)
  if helper_graph is None:
    _trace("skip helper_graph missing")
    return
  if not isinstance(getattr(helper_graph, "transform", None), (TransformNode, AnimatingNode)):
    _trace("skip helper_graph type={}".format(type(getattr(helper_graph, "transform", None)).__name__))
    return
  _trace("graph helper_tf={} parent_tf={}".format(type(getattr(helper_graph, "transform", None)).__name__, type(getattr(getattr(helper_graph, "parent", None), "transform", None)).__name__ if getattr(helper_graph, "parent", None) else None))
  def _has_object_animation(obj):
    ad = getattr(obj, "animation_data", None)
    if ad is None:
      return False
    if getattr(ad, "action", None) is not None:
      return True
    try:
      return len(getattr(ad, "nla_tracks", [])) > 0
    except Exception:
      return False

  def _get_local(obj):
    try:
      return obj.matrix_basis.copy()
    except Exception:
      try:
        return obj.matrix_local.copy()
      except Exception:
        return None

  def _set_local(obj, mat):
    try:
      obj.matrix_basis = mat
      return True
    except Exception:
      try:
        loc, rot, scale = mat.decompose()
        obj.location = loc
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = rot.to_euler("XYZ")
        obj.scale = scale
        return True
      except Exception:
        return False

  def _collapse_redundant_helper_empty(helper_obj, semantic_obj, preferred_child=None):
    """Remove a static identity helper EMPTY after its transform was hoisted."""
    # Only collapse in the render-node path. If we delete the current node's
    # object during renderless-helper processing, later debug/output paths still
    # in process_node can touch a dead reference.
    if _renderless_helper_mode:
      return False
    if helper_obj is None or semantic_obj is None:
      return False
    if getattr(helper_obj, "type", None) != "EMPTY" or getattr(semantic_obj, "type", None) != "EMPTY":
      return False
    if _has_object_animation(helper_obj):
      return False
    helper_name = str(getattr(helper_obj, "name", "") or "")
    helper_local = _get_local(helper_obj)
    if helper_local is None or not _is_identity_matrix_approx(helper_local):
      return False
    children = list(getattr(helper_obj, "children", []) or [])
    if len(children) != 1:
      _trace("skip collapse helper '{}' child_count={}".format(helper_name, len(children)))
      return False
    child = children[0]
    if preferred_child is not None and child is not preferred_child:
      return False
    try:
      child_world = child.matrix_world.copy()
    except Exception:
      child_world = None
    try:
      child.parent = semantic_obj
      if child_world is not None:
        child.matrix_world = child_world
      else:
        child.matrix_parent_inverse = semantic_obj.matrix_world.inverted()
      bpy.data.objects.remove(helper_obj, do_unlink=True)
      _trace("collapsed redundant helper empty '{}'".format(helper_name))
      return True
    except Exception as e:
      _trace("failed collapsing helper '{}' ({})".format(helper_name, e))
      return False

  # Case 1: v_* -> identity helper -> fake-light (legacy helper compaction)
  helper_parent_graph = getattr(helper_graph, "parent", None)
  if isinstance(getattr(helper_parent_graph, "transform", None), ArgVisibilityNode):
    vis_name = str(getattr(getattr(helper_parent_graph, "transform", None), "name", "") or "")
    if not vis_name.startswith(("Cylinder", "Fspot", "Omni", "Omni_l", "Box", "ChamferBox", "Object")):
      _trace("skip case1 vis_name={!r}".format(vis_name))
      return
    semantic_obj = getattr(helper_obj, "parent", None)
    helper_local = _get_local(helper_obj)
    if (
      semantic_obj is not None
      and getattr(semantic_obj, "type", None) == "EMPTY"
      and not _has_object_animation(helper_obj)
      and helper_local is not None
      and not _is_identity_matrix_approx(helper_local)
      and _is_identity_matrix_approx(semantic_obj.matrix_basis)
    ):
      if _set_local(semantic_obj, helper_local):
        _trace("case1 moved helper_local to semantic")
        _set_local(helper_obj, Matrix.Identity(4))
        _collapse_redundant_helper_empty(helper_obj, semantic_obj, preferred_child=fake_obj)
        return
    if not _has_object_animation(helper_obj) and _is_identity_matrix_approx(helper_obj.matrix_basis):
      fake_local = _get_local(fake_obj) if fake_obj is not None else None
      if fake_local is not None and not _is_identity_matrix_approx(fake_local):
        if _set_local(helper_obj, fake_local):
          _trace("case1 moved fake_local to helper")
          _set_local(fake_obj, Matrix.Identity(4))
      else:
        _trace("case1 fake_local none/identity")
    else:
      _trace("case1 helper animated or non-identity helper_basis")
    return

  # Case 2: v_* -> semantic(identity) -> transformed helper -> fake-light
  semantic_graph = helper_parent_graph
  if semantic_graph is None or not isinstance(getattr(semantic_graph, "transform", None), (TransformNode, AnimatingNode)):
    _trace("skip case2 semantic_graph tf={}".format(type(getattr(semantic_graph, "transform", None)).__name__ if semantic_graph else None))
    return
  vis_graph = getattr(semantic_graph, "parent", None)
  if vis_graph is None or not isinstance(getattr(vis_graph, "transform", None), ArgVisibilityNode):
    _trace("skip case2 vis_graph tf={}".format(type(getattr(vis_graph, "transform", None)).__name__ if vis_graph else None))
    return
  vis_name = str(getattr(getattr(vis_graph, "transform", None), "name", "") or "")
  if not vis_name.startswith(("Cylinder", "Fspot", "Omni", "Omni_l", "Box", "ChamferBox", "Object")):
    _trace("skip case2 vis_name={!r}".format(vis_name))
    return

  semantic_obj = getattr(helper_obj, "parent", None)
  if semantic_obj is None or getattr(semantic_obj, "type", None) != "EMPTY":
    _trace("skip case2 semantic_obj missing/non-empty type={}".format(getattr(semantic_obj, "type", None) if semantic_obj else None))
    return
  if _has_object_animation(helper_obj):
    _trace("skip case2 helper animated")
    return

  helper_local = _get_local(helper_obj)
  if helper_local is None or _is_identity_matrix_approx(helper_local):
    _trace("skip case2 helper_local none/identity")
    return
  if not _is_identity_matrix_approx(semantic_obj.matrix_basis):
    _trace("skip case2 semantic basis non-identity")
    return

  if _set_local(semantic_obj, helper_local):
    _trace("case2 moved helper_local to semantic")
    _set_local(helper_obj, Matrix.Identity(4))
    _collapse_redundant_helper_empty(helper_obj, semantic_obj, preferred_child=fake_obj)


def _visibility_wrapper_needs_own_object(node):
  """Return True if a renderless ArgVisibilityNode must stay as its own object.

  We preserve explicit visibility wrapper empties when collapsing them would
  either:
  - force visibility and transform animation onto the same Blender object (the
    official exporter can only read one active action per object), or
  - remove important explicit v_* wrappers for common helper/light primitive
    families where DOT parity depends on the wrapper node existing.
  """
  if node is None or not isinstance(getattr(node, "transform", None), ArgVisibilityNode):
    return False

  # Preserve common helper/light wrapper families even when static. These are
  # frequent parity regressions when collapsed (e.g. Cylinder*/Fspot*), and
  # many visual-effect/light wrappers in the FA-18 asset are static but still
  # semantically important (`v_*` nodes in DOT).
  vis_name = str(getattr(node.transform, "name", "") or getattr(node, "name", "") or "")
  vis_base = vis_name[2:] if vis_name.startswith("v_") else vis_name

  # Light/fake-light descendants often need an explicit visibility wrapper even
  # when the wrapper itself is static; collapsing them drops many original v_*
  # nodes (Object*, Omni*, flame/afterburn effect controls, etc.).
  for child in getattr(node, "children", []) or []:
    rend = getattr(child, "render", None)
    if rend is None:
      continue
    rcls = type(rend).__name__
    if any(tok in rcls for tok in ("Fake", "Light")):
      return True

  # Universal connector-control case: visibility wrappers that guard connector
  # transforms/descendants (attach points, probe points, etc.) are semantically
  # authored controls and should stay explicit. This catches cases where the
  # connector is nested under an intermediate transform/helper child.
  try:
    def _has_connector_descendant(n, depth=0):
      if n is None or depth > 3:
        return False
      r = getattr(n, "render", None)
      if isinstance(r, Connector):
        return True
      for ch in getattr(n, "children", []) or []:
        if _has_connector_descendant(ch, depth + 1):
          return True
      return False
    if any(_has_connector_descendant(ch) for ch in (getattr(node, "children", []) or [])):
      return True
  except Exception:
    pass

  static_wrapper_prefixes = (
    # helper/light primitive families (existing regressions)
    "Cylinder", "Fspot", "Line", "Box", "ChamferBox", "Plane",
    # common effect/light wrapper families seen in FA-18
    "Object", "Omni", "Flame", "Formation_Lights_", "BANO_", "RN_",
    "oreol", "afterburn_", "forsag", "banno_", "lamp_", "Stv_",
    "RefuelingProbe_", "Refueling_", "Pilon_", "Pylon_", "ANA_SQ_",
    # airframe/damage/LOD visibility controls frequently authored as explicit wrappers
    "Wing_", "Stab_", "Aileron_", "Flap_", "Fin_", "Cockpit_", "KIL_", "kil_",
    "Pilot_F18_", "pilot_F18_",
  )
  if vis_base.startswith(static_wrapper_prefixes):
    return True

  # Small targeted allowlist for residual FA-18 wrappers still missing after
  # broad family preservation. These names are semantically-authored visibility
  # controls but do not share strong prefixes with other preserved families.
  if vis_base in {
    "Aileron_L",
    "AirBrake_part1",
    "Flap_L",
    "Group051",
    "Group052",
    "Merge",
    "Merge001",
    "Texturte_Cockpit001",
    "c1",
    "chehly",
    "f29_4-6",
    "f29_6-10",
    "f_4-6",
    "f_6-10",
    "merge",
    "merge001",
  }:
    return True

  if any(tok in vis_base for tok in ("_dam", "_DAM", "_On", "_Inv", "_LOD")):
    return True

  # Visibility controls with multiple args are usually semantic controls rather
  # than disposable wrappers; keep them explicit.
  vis_args = getattr(node.transform, "args", None)
  try:
    if vis_args is not None and len(vis_args) > 1:
      return True
  except Exception:
    pass

  # Universal static visibility-wrapper case:
  # keep explicit v_* wrapper when it is a renderless visibility node wrapping
  # a single static render child with the same semantic name (common on decal /
  # number meshes such as F15E_numbers_* in su-27_lod3).
  try:
    children = list(getattr(node, "children", []) or [])
    if len(children) == 1:
      ch = children[0]
      ch_tf = getattr(ch, "transform", None)
      ch_r = getattr(ch, "render", None)
      if ch_tf is None and ch_r is not None:
        ch_r_name = str(getattr(ch_r, "name", "") or "")
        if ch_r_name and ch_r_name == vis_base:
          return True
  except Exception:
    pass

  # Broader variant of the same semantic pattern: a renderless visibility
  # wrapper may contain multiple render-only children where one child carries the
  # authored semantic name and the rest are helper/split chunks.
  try:
    children = list(getattr(node, "children", []) or [])
    if len(children) >= 2:
      any_same_named_render = False
      all_children_render_only = True
      for ch in children:
        if getattr(ch, "transform", None) is not None:
          all_children_render_only = False
          break
        ch_r = getattr(ch, "render", None)
        if ch_r is None:
          all_children_render_only = False
          break
        ch_r_name = str(getattr(ch_r, "name", "") or "")
        if ch_r_name == vis_base:
          any_same_named_render = True
      if all_children_render_only and any_same_named_render:
        return True
  except Exception:
    pass

  # Narrow connector wrapper case (same-name static connector child):
  # keep explicit visibility wrapper when it wraps a single same-named static
  # connector transform (common on authored attach/connector visibility controls
  # such as su27p_* in su-27_lod3).
  try:
    children = list(getattr(node, "children", []) or [])
    if len(children) == 1:
      ch = children[0]
      ch_tf = getattr(ch, "transform", None)
      ch_r = getattr(ch, "render", None)
      if isinstance(ch_tf, TransformNode) and ch_r is not None and isinstance(ch_r, Connector):
        ch_tf_name = str(getattr(ch_tf, "name", "") or "")
        if ch_tf_name and ch_tf_name == vis_base:
          ch_actions = get_actions_for_node(ch_tf)
          if not ch_actions:
            return True
  except Exception:
    pass

  # Preserve some renderless Dummy* visibility groups when they wrap nested
  # control/visibility structure; collapsing them often drops semantic v_* nodes
  # (seen in SU-27) without providing meaningful wrapper reduction.
  try:
    if vis_base.startswith("Dummy"):
      children = list(getattr(node, "children", []) or [])
      if len(children) >= 1:
        for ch in children:
          ch_tf = getattr(ch, "transform", None)
          if isinstance(ch_tf, (ArgVisibilityNode, AnimatingNode)):
            return True
      # Also preserve Dummy* wrappers that keep generic split render chunks
      # together (e.g. `_0`, `_1`) even when they do not wrap nested transform
      # controllers directly.
      if len(children) >= 2:
        generic_split_render_count = 0
        for ch in children:
          if getattr(ch, "transform", None) is not None:
            continue
          ch_r = getattr(ch, "render", None)
          ch_r_name = str(getattr(ch_r, "name", "") or "") if ch_r else ""
          if re.match(r"^_[0-9]+$", ch_r_name):
            generic_split_render_count += 1
        if generic_split_render_count >= 2:
          return True
      # Single generic split render child can still be semantically important
      # when grouped under a Dummy* visibility control inside a larger authored
      # visibility subtree (e.g. SU-27 Dummy605 under Dummy610).
      if len(children) == 1:
        ch = children[0]
        if getattr(ch, "transform", None) is None:
          ch_r = getattr(ch, "render", None)
          ch_r_name = str(getattr(ch_r, "name", "") or "") if ch_r else ""
          if re.match(r"^_[0-9]+$", ch_r_name):
            return True
      # Universal Dummy* wrapper case: a Dummy visibility control nested under
      # another visibility wrapper with a single render child is often an
      # authored semantic visibility node (e.g. SU-27 Dummy605), not an
      # implementation-detail wrapper. Preserve it explicitly.
      if len(children) == 1 and isinstance(getattr(getattr(node, "parent", None), "transform", None), ArgVisibilityNode):
        ch = children[0]
        if getattr(ch, "transform", None) is None and getattr(ch, "render", None) is not None:
          return True
  except Exception:
    pass

  for child in getattr(node, "children", []) or []:
    all_tfs = getattr(child, "_collapsed_transforms", [child.transform] if child.transform else [])
    for tf in all_tfs:
      if tf is None:
        continue
      if isinstance(tf, ArgVisibilityNode):
        continue
      if isinstance(tf, AnimatingNode):
        try:
          if get_actions_for_node(tf):
            return True
        except Exception:
          pass
  return False


def _channel_slices_from_vertex_format(vertex_format):
  """Return channel->(start,end) slices based on vertex format packed layout."""
  offsets = {}
  if not vertex_format or not hasattr(vertex_format, "data"):
    return offsets
  cursor = 0
  for i, count in enumerate(vertex_format.data):
    n = int(count)
    if n <= 0:
      continue
    offsets[i] = (cursor, cursor + n)
    cursor += n
  return offsets


def _copy_fcurve_to_action(src_curve, dst_action, dst_path, action_group):
  dst_curve = dst_action.fcurves.new(
    data_path=dst_path,
    index=src_curve.array_index,
    action_group=action_group,
  )
  dst_curve.extrapolation = src_curve.extrapolation
  for key in src_curve.keyframe_points:
    new_key = dst_curve.keyframe_points.insert(key.co[0], key.co[1], options={"FAST"})
    new_key.interpolation = key.interpolation
    try:
      new_key.handle_left_type = key.handle_left_type
      new_key.handle_right_type = key.handle_right_type
      new_key.handle_left = key.handle_left
      new_key.handle_right = key.handle_right
    except Exception:
      pass
  dst_curve.update()


def _action_has_fcurve(action, data_path, index=None):
  if action is None:
    return False
  for fcu in action.fcurves:
    if fcu.data_path != data_path:
      continue
    if index is None or fcu.array_index == index:
      return True
  return False


def _merge_visibility_action_into_transform_action(transform_action, vis_action, obj_name):
  """Clone a transform action and append VISIBLE fcurves from a visibility action."""
  if transform_action is None or vis_action is None:
    return transform_action
  if _action_has_fcurve(transform_action, "VISIBLE"):
    return transform_action

  vis_curves = [fcu for fcu in vis_action.fcurves if fcu.data_path == "VISIBLE"]
  if not vis_curves:
    return transform_action

  # Official exporter extracts the EDM argument from the action name prefix.
  # Preserve the visibility action's name prefix (e.g. "10_*") so VISIBLE
  # wrappers are emitted for merged visibility+transform actions.
  merged_name = (getattr(vis_action, "name", "") or "").strip() or transform_action.name
  merged = bpy.data.actions.new(merged_name)
  for src_curve in transform_action.fcurves:
    group_name = src_curve.group.name if getattr(src_curve, "group", None) else "Transform"
    _copy_fcurve_to_action(src_curve, merged, src_curve.data_path, group_name)
  for src_curve in vis_curves:
    if merged.fcurves.find("VISIBLE", index=src_curve.array_index) is not None:
      continue
    group_name = src_curve.group.name if getattr(src_curve, "group", None) else "Visibility"
    _copy_fcurve_to_action(src_curve, merged, "VISIBLE", group_name)
  return merged


def _transfer_bone_actions_to_armature(graph, arm_obj, node_to_bone_name):
  """Retarget per-node ArgAnimatedBone actions to armature pose-bone actions."""
  action_map = {}
  source_graph_nodes = set()
  source_transforms = set()
  bone_nodes = sorted(
    node_to_bone_name.keys(),
    key=lambda n: getattr(n.transform, "_graph_idx", 1 << 30),
  )
  for node in bone_nodes:
    bone_name = node_to_bone_name[node]
    # EDM bone animation is typically encoded on wrapper nodes above the Bone.
    seen_sources = set()
    current = node
    while current is not None and current.parent is not None and current.render is None and current.transform is not None:
      tfnode = current.transform
      if isinstance(tfnode, AnimatingNode) and not isinstance(tfnode, ArgVisibilityNode):
        if id(tfnode) not in seen_sources:
          seen_sources.add(id(tfnode))
          src_actions = get_actions_for_node(tfnode)
          if src_actions:
            source_graph_nodes.add(current)
            source_transforms.add(tfnode)
          for src_action in src_actions:
            dst_action = action_map.get(src_action.name)
            if dst_action is None:
              dst_action = bpy.data.actions.new(src_action.name)
              action_map[src_action.name] = dst_action
            for src_curve in src_action.fcurves:
              dst_path = 'pose.bones["{}"].{}'.format(bone_name, src_curve.data_path)
              # Skip if this FCurve already exists (prevents crash on re-import)
              if dst_action.fcurves.find(dst_path, index=src_curve.array_index) is not None:
                continue
              _copy_fcurve_to_action(src_curve, dst_action, dst_path, bone_name)
      parent = current.parent
      if parent is None or _is_bone_transform(parent.transform):
        break
      current = parent

  if not action_map:
    return source_graph_nodes, source_transforms

  arm_obj.animation_data_create()
  ad = arm_obj.animation_data
  ad.use_nla = True
  ad.action = None
  for track in list(ad.nla_tracks):
    ad.nla_tracks.remove(track)

  for action_name in sorted(action_map.keys()):
    action = action_map[action_name]
    track = ad.nla_tracks.new()
    track.name = action.name
    strip = track.strips.new(action.name, 1, action)
    strip.name = action.name

  return source_graph_nodes, source_transforms


def _prepare_bone_import(graph, parent_obj=None):
  """Create a single armature for EDM Bone/ArgAnimatedBone nodes."""
  global _bone_import_ctx
  _bone_import_ctx = None

  bone_nodes = [n for n in graph.nodes if _is_bone_transform(n.transform)]
  if not bone_nodes:
    return
  # Only map actual Bone/ArgAnimatedBone nodes to the synthesized armature.
  # Swallowing non-bone ancestors (especially ArgVisibilityNode wrappers)
  # collapses authored control/visibility transforms into `s_iEDM_Armature`
  # and causes major DOT hierarchy drift on round-trip.
  bone_chain_nodes = set(bone_nodes)

  arm_name = "iEDM_Armature"
  if bpy.data.objects.get(arm_name) is not None:
    i = 1
    while bpy.data.objects.get("{}.{:03d}".format(arm_name, i)) is not None:
      i += 1
    arm_name = "{}.{:03d}".format(arm_name, i)

  arm_data = bpy.data.armatures.new(arm_name)
  arm_obj = bpy.data.objects.new(arm_name, arm_data)
  bpy.context.collection.objects.link(arm_obj)
  if parent_obj is not None:
    basis = arm_obj.matrix_basis.copy()
    arm_obj.parent = parent_obj
    arm_obj.matrix_parent_inverse = Matrix.Identity(4)
    arm_obj.matrix_basis = basis

  view_layer = bpy.context.view_layer
  prev_active = view_layer.objects.active
  try:
    for obj in bpy.context.selected_objects:
      obj.select_set(False)
    arm_obj.select_set(True)
    view_layer.objects.active = arm_obj
    if bpy.context.mode != "OBJECT":
      bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = arm_data.edit_bones
    node_to_bone_name = {}
    used_names = set()
    sorted_nodes = sorted(
      bone_nodes,
      key=lambda n: getattr(n.transform, "_graph_idx", 1 << 30),
    )

    def _unique_bone_name(base):
      name = base or "Bone"
      if name not in used_names:
        used_names.add(name)
        return name
      i = 1
      while True:
        candidate = "{}.{:03d}".format(name, i)
        if candidate not in used_names:
          used_names.add(candidate)
          return candidate
        i += 1

    for node in sorted_nodes:
      bone_name = _unique_bone_name(_transform_display_name(node.transform))
      edit_bones.new(bone_name)
      node_to_bone_name[node] = bone_name

    def _bone_bind_matrix(tfnode):
      """Extract the bone's own bind-pose matrix from EDM Bone or ArgAnimatedBone."""
      if isinstance(tfnode, Bone) and hasattr(tfnode, "data") and len(tfnode.data) >= 1:
        return Matrix(tfnode.data[0])
      if isinstance(tfnode, ArgAnimatedBone) and hasattr(tfnode, "boneTransform"):
        return Matrix(tfnode.boneTransform)
      return None

    def _bone_rest_matrix(node):
      """Compute a bone's full rest matrix in Blender/armature space.

      The exporter writes mat_inv = pbone.matrix.inverted() as the bind matrix:
        - Bone:            data[1] = mat_inv
        - ArgAnimatedBone: boneTransform = mat_inv
      Inverting gives pbone.matrix — the exact armature-space rest matrix we need
      for edit-bone placement.  This avoids precision loss from accumulating
      parent-chain matrix multiplications.
      """
      tf = node.transform

      # Bone (non-animated): data[1] is the inverse bind matrix.
      if isinstance(tf, Bone) and not isinstance(tf, ArgAnimatedBone):
        if hasattr(tf, "data") and len(tf.data) >= 2:
          inv_bind = Matrix(tf.data[1])
          if not inv_bind.is_identity:
            try:
              return inv_bind.inverted()
            except ValueError:
              pass
        # Fall back to accumulated world matrix
        world_mat = getattr(node, "_world_bl", None)
        if world_mat is None:
          world_mat = getattr(node, "_local_bl", Matrix.Identity(4))
        return world_mat

      # ArgAnimatedBone: boneTransform is the inverse bind matrix.
      if isinstance(tf, ArgAnimatedBone) and hasattr(tf, "boneTransform"):
        bone_bind = Matrix(tf.boneTransform)
        if not bone_bind.is_identity:
          try:
            return bone_bind.inverted()
          except ValueError:
            pass
        # Fall back to accumulated world matrix
        world_mat = getattr(node, "_world_bl", None)
        if world_mat is None:
          world_mat = getattr(node, "_local_bl", Matrix.Identity(4))
        return world_mat

      world_mat = getattr(node, "_world_bl", None)
      if world_mat is None:
        world_mat = getattr(node, "_local_bl", Matrix.Identity(4))
      return world_mat

    # Debug: report bone bind matrix detection
    _bone_bind_found = 0
    _bone_bind_missing = 0
    for node in sorted_nodes:
      tf = node.transform
      bb = _bone_bind_matrix(tf)
      tf_type = type(tf).__name__
      tf_name = getattr(tf, "name", "?")
      if bb is not None:
        _bone_bind_found += 1
        if not bb.is_identity:
          rest = _bone_rest_matrix(node)
          print("  [bone-bind] {} '{}' -> non-identity bind matrix, rest translation=({:.3f},{:.3f},{:.3f})".format(
            tf_type, tf_name, rest[0][3], rest[1][3], rest[2][3]))
      else:
        _bone_bind_missing += 1
        print("  [bone-bind] {} '{}' -> NO bind matrix (type={}, has_data={}, has_boneTransform={})".format(
          tf_type, tf_name, tf_type,
          hasattr(tf, "data") and len(getattr(tf, "data", [])) >= 1,
          hasattr(tf, "boneTransform")))
    print("Info: Bone bind matrices: {} found, {} missing".format(_bone_bind_found, _bone_bind_missing))

    bone_node_set = set(sorted_nodes)
    for node in sorted_nodes:
      eb = edit_bones[node_to_bone_name[node]]
      bone_rest = _bone_rest_matrix(node)

      head = bone_rest.to_translation()
      rot3 = bone_rest.to_3x3()
      y_axis = rot3 @ Vector((0.0, 1.0, 0.0))
      z_axis = rot3 @ Vector((0.0, 0.0, 1.0))

      if y_axis.length < 1e-8:
        y_axis = Vector((0.0, 0.01, 0.0))
      y_axis.normalize()

      length = 0.05
      for child in node.children:
        if child in bone_node_set:
          child_rest = _bone_rest_matrix(child)
          child_head = child_rest.to_translation()
          dist = (child_head - head).length
          if dist > 1e-5:
            length = dist
            break

      eb.head = head
      eb.tail = head + y_axis * max(length, 0.01)
      if z_axis.length > 1e-8:
        eb.align_roll(z_axis)

    for node in sorted_nodes:
      parent = node.parent
      while parent is not None and parent not in node_to_bone_name:
        parent = parent.parent
      if parent is None:
        continue
      eb = edit_bones[node_to_bone_name[node]]
      eb.parent = edit_bones[node_to_bone_name[parent]]
  finally:
    try:
      bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
      pass
    if prev_active is not None:
      view_layer.objects.active = prev_active

  # Preserve bind/rest transforms when exporter computes static bone matrices.
  arm_data.pose_position = "REST"
  _bone_anim_sources = _transfer_bone_actions_to_armature(graph, arm_obj, node_to_bone_name)
  bone_anim_source_nodes = set()
  bone_anim_source_transforms = set()
  if _bone_anim_sources:
    if isinstance(_bone_anim_sources, tuple) and len(_bone_anim_sources) == 2:
      bone_anim_source_nodes = _bone_anim_sources[0] or set()
      bone_anim_source_transforms = _bone_anim_sources[1] or set()
    else:
      bone_anim_source_nodes = _bone_anim_sources or set()
  bone_name_by_transform = {}
  for tnode, bname in node_to_bone_name.items():
    if tnode.transform is not None:
      bone_name_by_transform[tnode.transform] = bname

  _bone_import_ctx = {
    "armature": arm_obj,
    "bone_name_by_node": node_to_bone_name,
    "bone_name_by_transform": bone_name_by_transform,
    "bone_chain_nodes": bone_chain_nodes,
    "bone_anim_source_nodes": bone_anim_source_nodes,
    "bone_anim_source_transforms": bone_anim_source_transforms,
  }


def _bind_skin_object(mesh_obj, skin_node):
  """Attach a SkinNode mesh to imported armature with vertex groups."""
  ctx = _bone_import_ctx or {}
  arm_obj = ctx.get("armature")
  bone_name_by_transform = ctx.get("bone_name_by_transform", {})
  if arm_obj is None or mesh_obj is None or mesh_obj.type != "MESH":
    return

  skin_bones = []
  for bone_node in getattr(skin_node, "bones", []):
    name = bone_name_by_transform.get(bone_node)
    if name:
      skin_bones.append(name)
  if not skin_bones:
    return

  group_map = {}
  for bone_name in skin_bones:
    vg = mesh_obj.vertex_groups.get(bone_name)
    if vg is None:
      vg = mesh_obj.vertex_groups.new(name=bone_name)
    group_map[bone_name] = vg

  arm_mod = mesh_obj.modifiers.get("Armature")
  if arm_mod is None:
    arm_mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
  arm_mod.object = arm_obj

  # Parity: Keep original hierarchy parent if one exists. Reparenting to the 
  # armature root (the legacy path) breaks local transform inheritance from 
  # ancestors like wings or pylons.
  if mesh_obj.parent is None:
    mesh_obj.parent = arm_obj

  nverts = len(mesh_obj.data.vertices)
  if nverts == 0:
    return

  # For SkinNode, the mesh data was built using the full original vertexData 
  # pool (no compaction) to maintain 1:1 parity with the EDM bone weights.
  slice21 = _channel_slices_from_vertex_format(skin_node.material.vertex_format).get(21)
  per_vertex_group_count = [0] * nverts
  used_bone_indices = set()

  for vi in range(nverts):
    # Skinned meshes use the original EDM vertex index directly.
    if vi >= len(skin_node.vertexData):
      continue
    src = skin_node.vertexData[vi]
    weights = []
    if slice21:
      weights = [float(x) for x in src[slice21[0]:slice21[1]]]

    assigned = False
    for bi, weight in enumerate(weights[:4]):
      if bi >= len(skin_bones):
        break
      if weight <= 1e-6:
        continue
      group_map[skin_bones[bi]].add([vi], weight, "REPLACE")
      per_vertex_group_count[vi] += 1
      used_bone_indices.add(bi)
      assigned = True

    if not assigned:
      group_map[skin_bones[0]].add([vi], 1.0, "REPLACE")
      per_vertex_group_count[vi] += 1
      used_bone_indices.add(0)

  # Ensure every SkinNode bone appears at least once so exporter keeps palette.
  cursor = 0
  for bi, bone_name in enumerate(skin_bones):
    if bi in used_bone_indices:
      continue
    attempts = 0
    while attempts < nverts and per_vertex_group_count[cursor] >= 4:
      cursor = (cursor + 1) % nverts
      attempts += 1
    if attempts >= nverts:
      break
    group_map[bone_name].add([cursor], 0.01, "ADD")
    per_vertex_group_count[cursor] += 1
    cursor = (cursor + 1) % nverts

def _print_import_diagnostics(edm, graph):
  """Print summary of EDM node types found vs Blender objects created."""
  from collections import Counter
  edm_counts = Counter()
  created_counts = Counter()
  skipped_empty = 0

  for node in iterate_all_objects(edm):
    type_name = type(node).__name__
    edm_counts[type_name] += 1

  def _count_created(tnode):
    nonlocal skipped_empty
    if tnode.render:
      type_name = type(tnode.render).__name__
      if tnode.blender and tnode.blender.type == 'MESH':
        created_counts[type_name] += 1
      elif tnode.blender and tnode.blender.type == 'EMPTY':
        created_counts[type_name + " (empty)"] += 1
      elif tnode.blender and tnode.blender.type == 'LIGHT':
        created_counts[type_name] += 1
      elif not tnode.blender or (tnode.parent and tnode.blender == tnode.parent.blender):
        skipped_empty += 1

  graph.walk_tree(_count_created)

  print("\n--- Import Diagnostics ---")
  print("EDM render nodes by type:")
  for name, count in sorted(edm_counts.items()):
    print(f"  {name}: {count}")
  print("Created Blender objects by type:")
  for name, count in sorted(created_counts.items()):
    print(f"  {name}: {count}")
  if skipped_empty:
    print(f"  Skipped (empty indexData / no handler): {skipped_empty}")
  print(f"Total Blender objects: {len([o for o in bpy.data.objects])}")
  print("--- End Diagnostics ---\n")


def read_file(filename, options=None):
  options = options or {}
  _reset_transform_debug(options)
  _reset_mesh_origin_mode(options)
  global _bone_import_ctx
  _bone_import_ctx = None

  # Parse the EDM file
  edm = EDMFile(filename)

  print("Raw file graph:")
  print_edm_graph(edm.transformRoot)

  # Store EDM version globally early so importer heuristics can use it before
  # addon-export patch toggles are configured.
  global _edm_version
  _edm_version = edm.version

  # Detect v10 owner-encoded split render graphs (shared-parent render chunks).
  # These small control-node assets (e.g. Geomerty+Animation+Collision.edm)
  # are sensitive to mesh recentering and plain-root basis suppression.
  has_bones = any(_is_bone_transform(n) for n in edm.nodes)
  has_owner_encoded_split_renders = False
  has_generic_render_chunks = False
  has_shell_nodes = False
  try:
    if _edm_version >= 10:
      for obj in iterate_all_objects(edm):
        if isinstance(obj, RenderNode) and _is_generic_render_name(getattr(obj, "name", "")):
          has_generic_render_chunks = True
        elif isinstance(obj, ShellNode):
          has_shell_nodes = True
        if getattr(obj, "shared_parent", None) is not None:
          has_owner_encoded_split_renders = True
        if has_owner_encoded_split_renders and has_generic_render_chunks and has_shell_nodes:
          break
  except Exception:
    has_owner_encoded_split_renders = False
    has_generic_render_chunks = False
    has_shell_nodes = False
  plain_root_v10 = bool(_edm_version >= 10 and type(getattr(edm, "transformRoot", None)) is Node)
  auto_geometry_safe_v10_split = bool(
    plain_root_v10
    and not has_bones
    and (
      has_owner_encoded_split_renders
      or (has_generic_render_chunks and has_shell_nodes)
    )
  )

  # Preserve authored pivots on v10 split control-node assets by disabling
  # mesh-origin recentering. Blender operator defaults may pass
  # mesh_origin_mode='APPROX' explicitly, so treat APPROX as auto-overridable.
  user_mesh_origin_mode = str(options.get("mesh_origin_mode", "") or "").upper()
  can_auto_override_mesh_origin = (user_mesh_origin_mode in {"", "APPROX"})
  if auto_geometry_safe_v10_split and can_auto_override_mesh_origin and _mesh_origin_mode != "RAW":
    globals()["_mesh_origin_mode"] = "RAW"
    print("Info: Auto-selected mesh origin mode RAW for v10 split control-node asset (plain root, no bones).")
  elif (
    can_auto_override_mesh_origin
    and _mesh_origin_mode != "RAW"
    and plain_root_v10
    and not has_bones
  ):
    # F-117-style plain-root non-skeletal v10 assets preserve authored object
    # transforms only when mesh origin recentering is disabled. APPROX mode adds
    # per-mesh local offsets (e.g. F-117_paint.012 +0.007624 X / +0.438309 Z).
    globals()["_mesh_origin_mode"] = "RAW"
    print("Info: Auto-selected mesh origin mode RAW for v10 plain-root non-skeletal asset (preserve authored transforms).")

  # Set the required blender preferences
  bpy.context.preferences.edit.use_negative_frames = False
  bpy.context.scene.use_preview_range = False
  bpy.context.scene.frame_start = 0
  bpy.context.scene.frame_end = FRAME_SCALE
  bpy.context.scene.frame_set(FRAME_SCALE // 2)

  disable_export_optimizations = options.get("disable_export_optimizations", None)
  if disable_export_optimizations is None:
    # Default to *not* disabling exporter optimizations for the Geometry-style
    # v10 split control-node assets; forcing them off explodes wrappers and
    # changes transform factorization. Keep legacy behavior elsewhere.
    disable_export_optimizations = not auto_geometry_safe_v10_split

  if disable_export_optimizations:
    if _disable_official_export_optimizations():
      print("Info: Disabled io_scene_edm export optimizations for round-trip parity")
  if _enable_official_visibility_name_alias():
    print("Info: Enabled io_scene_edm visibility-name alias support for round-trip parity")

  # These exporter transform-shaping patches materially reduce DOT wrapper noise,
  # but can also change rendering semantics in ModelViewer2 (blank scene despite
  # valid DOT export). Keep them opt-in until we re-establish a visually correct
  # baseline and reintroduce them with tighter guards.
  enable_visibility_wrapper_passthrough = options.get("enable_visibility_wrapper_passthrough", None)
  if enable_visibility_wrapper_passthrough is None:
    # Geometry-style v10 split control-node assets are our visual canary and
    # already achieve compact parity without this exporter patch. For other
    # assets (e.g. FA-18), enable the narrow visibility-wrapper passthrough by
    # default to avoid an extra identity TransformNode under preserved `v_*`
    # wrappers.
    enable_visibility_wrapper_passthrough = not auto_geometry_safe_v10_split

  if enable_visibility_wrapper_passthrough:
    if _enable_official_visibility_wrapper_transform_passthrough():
      print("Info: Enabled io_scene_edm visibility-wrapper transform passthrough for round-trip parity")
  # Default-on: identity prune removes no-op tl_/tr_/s_/_mat/_bmat_inv wrappers
  # emitted by animation.extract_transform_anim when the matrices are identity.
  # The geometry canary is already at parity and unaffected (nothing left to
  # prune).  The armature export path is explicitly excluded inside the patch
  # (see note in _enable_official_identity_fallback_transform_prune), so bones
  # are never affected.  Pass enable_identity_fallback_transform_prune=False to
  # the importer options to disable if needed for debugging.
  if options.get("enable_identity_fallback_transform_prune", True):
    if _enable_official_identity_fallback_transform_prune():
      print("Info: Enabled io_scene_edm identity fallback transform pruning + orig-name passthrough for round-trip parity")
  if _enable_official_connector_transform_naming():
    print("Info: Enabled io_scene_edm connector transform naming for round-trip parity")

  # Convert the materials (cwd matters for search)
  with chdir(os.path.dirname(os.path.abspath(filename))):
    for material in edm.root.materials:
      material.blender_material = create_material(material)
      if material.blender_material and options.get("shadeless", False):
        _apply_shadeless(material.blender_material)

  if _mesh_origin_mode == "RAW":
    print("Info: Mesh origin mode RAW (no geometry-center recenter on render-only nodes)")

  # Store version in scene for export
  bpy.context.scene.edm_version = edm.version

  global _file_has_bones
  _file_has_bones = bool(has_bones)

  # For ALL .edm files, we MUST apply _ROOT_BASIS_FIX (the inverse of the DCS 
  # Y-up coordinate swap) to the root children so that they import into Blender 
  # as Z-up. When re-exported, io_scene_edm will apply its ROOT_TRANSFORM_MATRIX 
  # (which is the forward DCS swap) and restore the file to its original state, 
  # whether it was originally exported from 3ds Max (with a root swap matrix) 
  # or from Blender (with optimized meshes in Y-up space).

  # Import bounding box / user box empties if requested.
  import_bbox = bool(options.get("import_bounding_box", False))
  auto_bone_bbox = bool(options.get("auto_bone_bounding_box", True))
  if import_bbox or (has_bones and auto_bone_bbox):
    if not _has_special_box("BOUNDING_BOX"):
      create_bounding_box_from_root(edm.root)
  if options.get("import_user_box", False):
    _create_user_box_from_root(edm.root)

  # Build the translation graph with hierarchy simplification
  graph = build_graph(edm)
  graph.print_tree()

  # For plain model::Node roots, avoid forcing a Blender empty:
  # the official round-trip can preserve a non-transform root and forcing an
  # authored object turns it into `TransformNode '_'` on export.
  # Keep an explicit root object only when the file root actually carries
  # transform payload.
  root_tf = getattr(graph.root, "transform", None)
  has_root_transform_payload = isinstance(root_tf, (TransformNode, AnimatingNode))
  
  # For V10+, we ALWAYS create a root Blender object to carry the global 
  # coordinate basis fix. This ensures that ALL elements in the file (even 
  # those attached directly to the file root) are correctly rotated.
  force_root_obj = (_edm_version >= 10)

  if has_root_transform_payload or force_root_obj:
    root_name = getattr(root_tf, "name", "") or ""
    if not root_name.strip():
      root_name = "_EDMFileRoot"

    root_obj = bpy.data.objects.new(root_name, None)
    root_obj.empty_display_size = 0.1
    bpy.context.collection.objects.link(root_obj)
    graph.root.blender = root_obj

    # For V10, apply the global basis fix to the root object.
    # If the root carries its own transform (e.g. 3ds Max/Blender_ioEDM swap),
    # multiply them so they cancel out to Identity in Blender space.
    if _edm_version >= 10:
      m_root = Matrix.Identity(4)
      if hasattr(root_tf, "matrix"):
        m_root = Matrix(root_tf.matrix)
      elif hasattr(root_tf, "base") and hasattr(root_tf.base, "matrix"):
        m_root = Matrix(root_tf.base.matrix)
      
      # Apply the fix: Blender_World = _ROOT_BASIS_FIX * EDM_Root_Matrix
      root_obj.matrix_basis = _ROOT_BASIS_FIX @ m_root
    elif has_root_transform_payload:
      apply_node_transform(graph.root, graph.root.blender, used_shared_parent=False)
  else:
    graph.root.blender = None

  # Reconstruct armature data for Bone/ArgAnimatedBone transforms
  _prepare_bone_import(graph, graph.root.blender)

  # Walk through every node, and do the node processing
  graph.walk_tree(process_node)

  # Post-children pass: apply LodNode properties after children are created.
  graph.walk_tree(_process_lod_post_children)

  # Diagnostic: count EDM nodes vs created Blender objects
  _print_import_diagnostics(edm, graph)

  # Optional post-import collection organization (disabled by default for parity)
  if options.get("assign_collections", False):
    _assign_collections(graph)

  if _transform_debug.get("enabled"):
    if _transform_debug["emitted"] >= _transform_debug["limit"]:
      print("Info: Transform debug reached output limit of {}".format(_transform_debug["limit"]))
    print("Info: Transform debug emitted {} record(s)".format(_transform_debug["emitted"]))

  bpy.context.view_layer.update()

def create_visibility_actions(visNode):
  """Creates visibility actions from an ArgVisibilityNode"""
  def _vis_arg_to_frame(value):
    # EDM visibility ranges are authored on the same normalized timeline as
    # other v10 animation channels.
    return int(round((value + 1.0) * FRAME_SCALE / 2.0))

  actions = []
  for (arg, ranges) in visNode.visData:
    # Creates a visibility animation track
    vis_name = visNode.name or "node"
    # Strip visibility "v_" prefix and animation prefixes from name
    if vis_name.startswith("v_"):
      vis_name = vis_name[2:]
    vis_name = _strip_anim_prefix(vis_name)
    action = bpy.data.actions.new("{}_{}_Visib".format(arg, vis_name))
    actions.append(action)
    if hasattr(action, "argument"):
      action.argument = arg
    # Create f-curves for official exporter visibility property and Blender-native
    # hide flags so viewport/render visibility matches imported EDM behavior.
    curve_visible = action.fcurves.new(data_path="VISIBLE")
    curve_hide_render = action.fcurves.new(data_path="hide_render")
    curve_hide_viewport = action.fcurves.new(data_path="hide_viewport")

    def _add_constant_key(curve, frame, value):
      curve.keyframe_points.add(1)
      key = curve.keyframe_points[-1]
      key.co = (frame, float(value))
      key.interpolation = 'CONSTANT'

    # Create the keyframe data
    # Visibility ranges in EDM are [start, end) where end is the first OFF
    # frame boundary. Reconstruct official scene key style:
    # Emit:
    #   start -> 1
    #   end-1 -> 1
    #   end   -> 0
    for (start, end) in ranges:
      frameStart = max(0, min(FRAME_SCALE, _vis_arg_to_frame(start)))
      frameOff = FRAME_SCALE + 1 if end > 1.0 else _vis_arg_to_frame(end)
      if frameOff <= frameStart:
        frameOff = frameStart + 1
      frameEndVisible = frameOff - 1

      _add_constant_key(curve_visible, frameStart, 1.0)
      _add_constant_key(curve_hide_render, frameStart, 0.0)
      _add_constant_key(curve_hide_viewport, frameStart, 0.0)

      if frameOff <= FRAME_SCALE:
        _add_constant_key(curve_visible, frameEndVisible, 1.0)
        _add_constant_key(curve_hide_render, frameEndVisible, 0.0)
        _add_constant_key(curve_hide_viewport, frameEndVisible, 0.0)

        _add_constant_key(curve_visible, frameOff, 0.0)
        _add_constant_key(curve_hide_render, frameOff, 1.0)
        _add_constant_key(curve_hide_viewport, frameOff, 1.0)
  return actions

def add_position_fcurves(action, keys, transform_left, transform_right):
  "Adds position fcurve data to an animation action"
  # Create an fcurve for every component (reuse existing if present)
  curves = []
  for i in range(3):
    curve = action.fcurves.find("location", index=i)
    if curve is None:
      curve = action.fcurves.new(data_path="location", index=i)
    curves.append(curve)

  # Loop over every keyframe in this animation
  for framedata in keys:
    frame = _anim_frame_to_scene_frame(framedata.frame)

    # Calculate the position transformation (convert keyframe from EDM Y-up to Blender Z-up)
    newPosMat = transform_left @ Matrix.Translation(_anim_vector_to_blender(framedata.value)) @ transform_right
    newPos = newPosMat.decompose()[0]

    for curve, component in zip(curves, newPos):
      curve.keyframe_points.add(1)
      curve.keyframe_points[-1].co = (frame, component)
      curve.keyframe_points[-1].interpolation = 'LINEAR'

def add_rotation_fcurves(action, keys, transform_left, transform_right):
  "Adds rotation fcurve action to an animation action"
  # Keep rotation keys in quaternion form so official exporter can consume
  # them directly without Euler re-conversion drift.
  curves = []
  for i in range(4):
    curve = action.fcurves.find("rotation_quaternion", index=i)
    if curve is None:
      curve = action.fcurves.new(data_path="rotation_quaternion", index=i)
    curves.append(curve)

  # Loop over every keyframe in this animation
  previous_quat = None
  for framedata in keys:
    frame = _anim_frame_to_scene_frame(framedata.frame)
    
    # Calculate the rotation transformation (convert keyframe from EDM Y-up to Blender Z-up)
    newRotQuat = transform_left @ _anim_quaternion_to_blender(framedata.value) @ transform_right
    if previous_quat is not None and previous_quat.dot(newRotQuat) < 0.0:
      newRotQuat = -newRotQuat
    previous_quat = newRotQuat.copy()

    for curve, component in zip(curves, newRotQuat):
      curve.keyframe_points.add(1)
      curve.keyframe_points[-1].co = (frame, component)
      curve.keyframe_points[-1].interpolation = 'LINEAR'

def add_scale_fcurves(action, keys):
  "Adds scale fcurve action to an animation action"
  if not keys:
    return
  curves = []
  for i in range(3):
    curve = action.fcurves.find("scale", index=i)
    if curve is None:
      curve = action.fcurves.new(data_path="scale", index=i)
    curves.append(curve)

  for framedata in keys:
    frame = _anim_frame_to_scene_frame(framedata.frame)
    value = framedata.value
    comps = _anim_scale_components(value)

    for curve, component in zip(curves, comps):
      curve.keyframe_points.add(1)
      curve.keyframe_points[-1].co = (frame, component)
      curve.keyframe_points[-1].interpolation = 'LINEAR'

def create_arganimation_actions(node):
  "Creates a set of actions to represent an ArgAnimationNode"
  actions = []

  # Work out the transformation chain for this object.
  # V10 arg transform payloads are already authored in Blender-local terms,
  # except top-level nodes which include the file root basis wrapper.
  aabS_raw = MatrixScale(Vector((node.base.scale[0], node.base.scale[1], node.base.scale[2])))
  aabT_raw = Matrix.Translation(Vector(node.base.position))
  q1_raw = node.base.quat_1 if hasattr(node.base.quat_1, "to_matrix") else Quaternion(node.base.quat_1)
  q1m_raw = q1_raw.to_matrix().to_4x4()
  base_mat = Matrix(node.base.matrix)
  
  # Parity Fix: ArgAnimatedBone has an extra boneTransform that must be 
  # included in the static part of the animation chain.
  if type(node).__name__ == "ArgAnimatedBone" and hasattr(node, "boneTransform"):
    base_mat = base_mat @ Matrix(node.boneTransform)

  mat_bl = base_mat
  aabS = aabS_raw
  aabT = Matrix.Translation(_anim_vector_to_blender(node.base.position))
  q1m = _anim_base_quaternion_q1_to_blender(node, node.base.quat_1).to_matrix().to_4x4()
  q1 = q1m.to_quaternion()
  q1_rot = q1
  mat_anim = mat_bl
  matQuat = mat_bl.decompose()[1]

  # With:
  #   aabT = Matrix.Translation(argNode.base.position)
  #   aabS = argNode.base.scale as a Scale Matrix
  #   q1m  = argNode.base.quat_1 as a Matrix (e.g. fixQuat1)
  #   q2m  = argNode.base.quat_2 as a Matrix (e.g. fixQuat2)
  #   mat  = argNode.base.matrix
  #   m2b  = matrix_to_blender e.g. swapping Z -> -Y

  # Save the 'null' transform onto the node, because we will need to
  # apply it to any object using this node's animation.

  #               Geometry is in Blender-space in-file already, and the import
  #                  procedure over-rotated it into an undefined space. Rotate 
  #                                                    back into Blender-space
  #                                                                      |
  #                                Transforms SAVED in blender-space     |
  #                                                   |     |      |     |
  #             Rotates geometry into file-space      |     |      |     |
  #                                           |       |     |      |     |
  # Reconstruct the neutral local transform from the same chain used at export.
  # Parity Fix: Include aabT (base position) in the static transform. 
  # This correctly places the animation controller empty in Blender.
  zero_mat = (mat_bl @ aabT @ q1m @ aabS)
  
  bmat_inv = getattr(node, "bmat_inv", None)
  if bmat_inv is not None:
    # bmat_inv is often a coordinate system correction (e.g. 3ds Max vs Blender)
    # for that specific animation node.
    try:
      zero_mat = zero_mat @ bmat_inv.inverted()
    except Exception:
      pass

  node.zero_transform_local_matrix = zero_mat
  dcLoc, dcRot, dcScale = zero_mat.decompose()
  node.zero_transform_matrix = zero_mat
  node.zero_transform = dcLoc, dcRot, dcScale

  # Split a single node into separate actions based on the argument
  for arg in node.get_all_args():
    # Filter the animation data for this node to just this action
    posData = [x[1] for x in node.posData if x[0] == arg]
    rotData = [x[1] for x in node.rotData if x[0] == arg]
    scaleData = [x[1] for x in node.scaleData if x[0] == arg]
    # Create the action
    action_name = "{}_{}".format(arg, _strip_anim_prefix(node.name or "anim"))
    action = bpy.data.actions.new(action_name)
    actions.append(action)
    if hasattr(action, "argument"):
      action.argument = arg
    
    # Calculate the pre and post-animation-value transforms.
    # EDM data is stored in Y-up; convert into Blender (Z-up) by
    # reusing the same conversion applied to zero_transform.
    leftRotation = matQuat @ q1_rot
    rightRotation = Quaternion((1, 0, 0, 0))
    
    # STRIP TRANSLATION from the Action Basis (leftPosition).
    # The translation is handled by the Static Object Transform (zero_mat).
    # If we include it here, the Action creates translation F-curves, leading to
    # Double Translation on export (TransformNode + ArgAnimationNode both translating).
    leftPosition = (mat_anim @ aabT).to_3x3().to_4x4()
    
    # Factor in bmat_inv to the animation basis if present.
    rightPosition = q1m @ aabS
    if bmat_inv is not None:
      try:
        rightPosition = rightPosition @ bmat_inv.inverted()
        # Rotation basis also needs correction
        leftRotation = leftRotation @ bmat_inv.to_quaternion().inverted()
      except Exception:
        pass

    # Build the f-curves for the action
    for pos in posData:
      add_position_fcurves(action, pos, leftPosition, rightPosition)
    for rot in rotData:
      add_rotation_fcurves(action, rot, leftRotation, rightRotation)
    for sca_pair in scaleData:
      # Scale entries are stored as (4-component keys, 3-component keys);
      # use the 3-component set for object XYZ scale curves.
      keys3 = sca_pair[1] if isinstance(sca_pair, tuple) and len(sca_pair) > 1 else []
      add_scale_fcurves(action, keys3)
  # Return these new actions
  return actions


def get_actions_for_node(node):
  """Accepts a node and gets or creates actions to apply their animations"""
  
  # Don't do this twice
  if hasattr(node, "actions") and node.actions:
    actions = node.actions
  else:
    actions = []
    if isinstance(node, ArgVisibilityNode):
      actions = create_visibility_actions(node)
    if isinstance(node, ArgAnimationNode):
      actions = create_arganimation_actions(node)
    # Save these actions on the node
    node.actions = actions

  return actions


def _find_texture_file(name):
  """
  Searches for a texture file given a basename without extension.

  The current working directory will be searched, as will any
  subdirectories called "textures/", for any file starting with the
  designated name '{name}.'
  """
  files = glob.glob(name+".*")
  if not files:
    matcher = re.compile(fnmatch.translate(name+".*"), re.IGNORECASE)
    files = [x for x in glob.glob("*.*") if matcher.match(x)]
    if not files:
      files = glob.glob("textures/"+name+".*")
      if not files:
        matcher = re.compile(fnmatch.translate("textures/"+name+".*"), re.IGNORECASE)
        files = [x for x in glob.glob("textures/*.*") if matcher.match(x)]
        if not files:
          print("Warning: Could not find texture named {}".format(name))
          return None
  # print("Found {} as: {}".format(name, files))
  if len(files) > 1:
    print("Warning: Found more than one possible match for texture named {}. Using {}".format(name, files[0]))
    files = [files[0]]
  textureFilename = files[0]
  return os.path.abspath(textureFilename)

def create_material(material):
  """Create a blender node-based PBR material from an EDM one"""
  # Create a new material
  mat = bpy.data.materials.new(name=material.name)
  mat.use_nodes = True
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links

  # Clear existing nodes
  for node in nodes:
    nodes.remove(node)

  # Create Principled BSDF and Output nodes
  principled_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
  principled_bsdf.location = (0, 0)
  material_output = nodes.new('ShaderNodeOutputMaterial')
  material_output.location = (400, 0)
  links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

  # --- Handle Textures ---
  texture_nodes = {}
  for tex_def in material.textures:
    filename = _find_texture_file(tex_def.name)
    if not filename:
      continue

    # Create image texture node
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(filename)
    tex_image.location = (-400, tex_def.index * -300)
    texture_nodes[tex_def.index] = tex_image

  # Connect textures to Principled BSDF
  if 0 in texture_nodes: # Diffuse
    tex_image = texture_nodes[0]
    tex_image.image.colorspace_settings.name = 'sRGB'
    links.new(tex_image.outputs['Color'], principled_bsdf.inputs['Base Color'])

  if 1 in texture_nodes: # Normal
    tex_image = texture_nodes[1]
    tex_image.image.colorspace_settings.name = 'Non-Color'
    normal_map_node = nodes.new('ShaderNodeNormalMap')
    normal_map_node.location = (-200, -300)
    links.new(tex_image.outputs['Color'], normal_map_node.inputs['Color'])
    links.new(normal_map_node.outputs['Normal'], principled_bsdf.inputs['Normal'])

  if 2 in texture_nodes: # Specular
    tex_image = texture_nodes[2]
    tex_image.image.colorspace_settings.name = 'Non-Color'
    links.new(tex_image.outputs['Color'], principled_bsdf.inputs['Specular IOR Level'])

  # --- Handle Uniforms ---
  # Metallic
  if material.material_name == 'chrome_material':
    principled_bsdf.inputs['Metallic'].default_value = 1.0
  else:
    principled_bsdf.inputs['Metallic'].default_value = 0.0

  # Roughness from specPower
  specPower = material.uniforms.get("specPower", None)
  if specPower is not None:
    # This is a guess, might need tweaking. Assuming specPower is in a range of 0-1024 (common for older shaders)
    # Higher specPower means a smaller, more intense highlight, which means lower roughness.
    roughness = (1.0 - (specPower / 1024.0))**0.5 # Using sqrt for a more perceptually linear mapping
    principled_bsdf.inputs['Roughness'].default_value = max(0.0, min(1.0, roughness))
  else:
    principled_bsdf.inputs['Roughness'].default_value = 0.5

  # Specular from specFactor
  specFactor = material.uniforms.get("specFactor", None)
  if specFactor is not None:
    principled_bsdf.inputs['Specular IOR Level'].default_value = specFactor

  # --- Handle Blending ---
  if material.blending in (1, 2): # Blend or Alpha Test
    # Connect diffuse texture alpha to Principled BSDF Alpha input
    if 0 in texture_nodes:
      links.new(texture_nodes[0].outputs['Alpha'], principled_bsdf.inputs['Alpha'])
    mat.surface_render_method = 'DITHERED'

  # Add official exporter-compatible EDM group node when available.
  official_attached = _attach_official_material_bridge(mat, material, texture_nodes)
  if not official_attached:
    print(f"Warning: Material bridge not attached for '{material.material_name}' "
          f"(material '{material.name}') — re-export may not recognise this material")
  if official_attached:
    # Keep imported material node layout close to official reference scenes:
    # Output + official EDM group (+ texture nodes), without extra Principled.
    try:
      nodes.remove(principled_bsdf)
    except Exception:
      pass

  # Set other material properties
  try:
    mat.edm_material = material.material_name
  except TypeError:
    print(f"Warning: Unknown material type '{material.material_name}', defaulting to 'def_material'")
    mat.edm_material = "def_material"
  mat.edm_blending = str(material.blending)

  return mat


def create_connector(connector):
  """Create an empty object representing a connector"""
  ob = bpy.data.objects.new(connector.name, None)
  ob.empty_display_type = "CUBE"
  ob.edm.is_connector = True
  _set_official_special_type(ob, 'CONNECTOR')

  # Preserve connector properties for round-trip export
  if hasattr(connector, 'props') and connector.props:
    lines = []
    for key, value in connector.props.items():
      if key.startswith("__"):
        continue
      if isinstance(value, str):
        lines.append("{} = '{}'".format(key, value.replace("'", "\\'")))
      elif isinstance(value, (int, float)):
        lines.append("{} = {}".format(key, repr(value)))
    if lines:
      _set_edmprop(ob, "CONNECTOR_EXT", "\n".join(lines))

  bpy.context.collection.objects.link(ob)
  return ob

def create_segments(segments_node):
  """Create a mesh with edges from a SegmentsNode (collision lines)"""
  # Create a new mesh
  mesh = bpy.data.meshes.new(segments_node.name)

  # Extract vertices and edges from segment data
  verts = []
  edges = []

  for i, segment in enumerate(segments_node.data):
    # Each segment is 6 floats: [x1, y1, z1, x2, y2, z2]
    v1 = Vector((segment[0], segment[1], segment[2]))
    v2 = Vector((segment[3], segment[4], segment[5]))

    v1 = Vector(v1)
    v2 = Vector(v2)

    # Add vertices
    verts.extend([v1, v2])
    # Add edge connecting these two vertices
    edges.append([i*2, i*2+1])

  # Create mesh from vertex and edge data
  mesh.from_pydata(verts, edges, [])
  mesh.update()

  # Create object and set properties
  ob = bpy.data.objects.new(segments_node.name, mesh)
  ob.edm.is_collision_line = True
  ob.edm.is_renderable = False
  _set_official_special_type(ob, 'COLLISION_LINE')
  bpy.context.collection.objects.link(ob)

  return ob

def apply_node_transform(node, obj, used_shared_parent=False):
  """Assigns the transform to a given node. node can be a TranslationNode or raw EDM node."""
  # Unpack TranslationNode if provided
  tnode = node if hasattr(node, "transform") else None
  tfnode = tnode.transform if tnode else node
  
  if tfnode is None:
    return

  obj.rotation_mode = "XYZ"

  def _debug_set_trace(source_label, local_mat):
    if not _transform_debug.get("enabled"):
      return
    needle = _transform_debug.get("filter")
    node_name = getattr(tfnode, "name", "") or type(tfnode).__name__
    if needle:
      n = needle.lower()
      if n not in node_name.lower() and n not in obj.name.lower():
        return
    loc, rot, scale = local_mat.decompose()
    print("[iEDM][TFSET] {} src={} obj={} loc={} rot_deg={} scale={}".format(
      node_name, source_label, obj.name, _debug_fmt_vec3(loc), _debug_fmt_rot_deg(rot), _debug_fmt_vec3(scale)
    ))

  def _set_local_matrix(local_mat):
    try:
      obj.matrix_basis = local_mat
      return
    except Exception:
      pass
    loc, rot, scale = local_mat.decompose()
    obj.location = loc
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = rot.to_euler('XYZ')
    obj.scale = scale

  def _compose_with_parent_if_enabled(local_mat):
    if _legacy_v10_parent_compose and obj.parent is not None and _edm_version >= 10:
      try:
        return obj.parent.matrix_basis @ local_mat
      except Exception:
        return local_mat
    return local_mat

  # 1. Base Transform from Node.transform
  final_local = Matrix.Identity(4)
  if isinstance(tfnode, TransformNode):
    is_connector_obj = _is_connector_transform(tfnode, obj)
    local_mat = Matrix(tfnode.matrix)
    if is_connector_obj:
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

  # 2. Inherit RenderNode local offset if present
  # In EDM, RenderNodes can have their own pos/matrix separate from the transform node.
  render = tnode.render if tnode else None
  if render:
    m_rn = Matrix.Identity(4)
    if hasattr(render, "pos"):
      m_rn = Matrix.Translation(Vector(render.pos[:3]))
    elif hasattr(render, "matrix"):
      m_rn = Matrix(render.matrix)
    
    if not m_rn.is_identity:
      final_local = final_local @ m_rn

  _set_local_matrix(final_local)
  _debug_set_trace(type(tfnode).__name__, final_local)


def _create_mesh(vertexData, indexData, vertexFormat, compact=True):
  """Creates a blender mesh object from vertex, index and format data"""

  if compact:
    # Build compact per-part meshes from referenced vertices only.
    # Keeping full shared pools (e.g. 3k+ verts for every split chunk) causes
    # exporter merge/reindex behavior to diverge from official blend exports.
    used_indices = sorted(set(indexData))
    new_vertices = [vertexData[i] for i in used_indices]
    index_lookup = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
    new_indices = [index_lookup[i] for i in indexData]
  else:
    # For skinned meshes, we MUST use the full vertex pool to keep indices
    # in sync with the bone weights.
    new_vertices = vertexData
    new_indices = indexData

  # Make sure we have the right number of indices...
  assert len(new_indices) % 3 == 0

  bm = bmesh.new()
  # For strict official-export parity, rely on Blender-generated split normals.
  # Importing per-vertex custom normals here causes small vertex-split drift on
  # re-export (e.g. +2 verts on merged render chunks in this test asset).
  import_custom_normals = False
  vertex_normals = [] if (import_custom_normals and vertexFormat.normal_indices) else None

  # Extract where the indices are
  posIndex = vertexFormat.position_indices
  normIndex = vertexFormat.normal_indices
  uvIndexSets = []
  uv_offset = sum(vertexFormat.data[:4]) if hasattr(vertexFormat, "data") else 0
  if hasattr(vertexFormat, "data"):
    for channel in (4, 5, 6):
      if channel >= len(vertexFormat.data):
        continue
      count = int(vertexFormat.data[channel])
      if count >= 2:
        uvIndexSets.append(list(range(uv_offset, uv_offset + 2)))
      uv_offset += max(0, count)

  # Create the BMesh vertices, optionally with normals
  for i, vtx in enumerate(new_vertices):
    pos = Vector(vtx[x] for x in posIndex)
    vert = bm.verts.new(pos)
    if normIndex and vertex_normals is not None:
      norm = Vector(vtx[x] for x in normIndex)
      if norm.length_squared > 1e-12:
        norm.normalize()
      vert.normal = norm
      vertex_normals.append(norm.copy())

  bm.verts.ensure_lookup_table()

  # Prepare for texture information
  uv_layers = []
  for uv_set_idx, _ in enumerate(uvIndexSets):
    layer_name = "UVMap" if uv_set_idx == 0 else "UVMap.{:03d}".format(uv_set_idx)
    uv_layer = bm.loops.layers.uv.get(layer_name)
    if uv_layer is None:
      uv_layer = bm.loops.layers.uv.new(layer_name)
    uv_layers.append(uv_layer)
  
  # Generate faces, with texture coordinate information.
  # Some EDM files include degenerate or duplicate triangles that bmesh rejects.
  skipped_degenerate = 0
  skipped_duplicate = 0
  for face in [new_indices[i:i+3] for i in range(0, len(new_indices), 3)]:
    if len(set(face)) < 3:
      skipped_degenerate += 1
      continue
    try:
      f = bm.faces.new([bm.verts[i] for i in face])
      # Add UV data if we have any
      if uvIndexSets:
        for loop, v_idx in zip(f.loops, face):
          vtx = new_vertices[v_idx]
          for uv_layer, uv_idx in zip(uv_layers, uvIndexSets):
            if len(uv_idx) < 2:
              continue
            if uv_idx[1] >= len(vtx):
              continue
            loop[uv_layer].uv = (vtx[uv_idx[0]], 1 - vtx[uv_idx[1]])
    except ValueError as e:
      # BMesh rejects exact duplicate faces; keep track and continue.
      skipped_duplicate += 1

  if skipped_degenerate or skipped_duplicate:
    print("Info: Skipped {} degenerate and {} duplicate faces while building mesh".format(
      skipped_degenerate, skipped_duplicate
    ))

  # Create the mesh object
  mesh = bpy.data.meshes.new("Mesh")
  bm.to_mesh(mesh)
  bm.free()

  # EDM meshes are authored for smooth shading in most cases; keep curved
  # surfaces visually faithful by enabling smooth polygons and importing
  # provided normals as custom split normals.
  for poly in mesh.polygons:
    poly.use_smooth = True

  if vertex_normals and len(vertex_normals) == len(mesh.vertices):
    try:
      normals = [tuple(n) for n in vertex_normals]
      if hasattr(mesh, "normals_split_custom_set_from_vertices"):
        mesh.normals_split_custom_set_from_vertices(normals)
      else:
        loop_normals = [normals[loop.vertex_index] for loop in mesh.loops]
        mesh.normals_split_custom_set(loop_normals)
    except Exception as e:
      print("Warning: Failed to assign custom normals: {}".format(e))

  if hasattr(mesh, "use_auto_smooth"):
    mesh.use_auto_smooth = True

  mesh.update()

  return mesh


# Mapping from EDM animated-uniform names to EDMProps field names.
_ANIMATED_UNIFORM_TO_EDMPROPS = {
  "emissiveValue":  "EMISSIVE_ARG",
  "colorShift":     "COLOR_ARG",
  "opacityValue":   "OPACITY_VALUE_ARG",
}


def _map_animated_uniforms_to_edmprops(ob, node):
  """Extract animated uniform argument numbers and set EDMProps ARG fields."""
  if not hasattr(ob, "EDMProps"):
    return
  mat = getattr(node, "material", None)
  if mat is None:
    return
  anim_uniforms = getattr(mat, "animated_uniforms", None)
  if not anim_uniforms:
    return
  from .edm.typereader import AnimatedProperty
  for name, prop in anim_uniforms.items():
    if not isinstance(prop, AnimatedProperty):
      continue
    edmprops_field = _ANIMATED_UNIFORM_TO_EDMPROPS.get(name)
    if edmprops_field and hasattr(ob.EDMProps, edmprops_field):
      if prop.argument is not None and prop.argument >= 0:
        try:
          # Clamp to 32-bit signed int limit for Blender's IntProperty
          val = min(2147483647, int(prop.argument))
          setattr(ob.EDMProps, edmprops_field, val)
        except (OverflowError, ValueError):
          pass


def create_object(node):
  """Accepts an edm renderable and returns a blender object"""

  if not isinstance(node, (RenderNode, ShellNode, NumberNode, SkinNode)):
    print("WARNING: Do not understand creating types {} yet".format(type(node)))
    return

  if isinstance(node, (RenderNode, NumberNode, SkinNode)):
    vertexFormat = node.material.vertex_format
  elif isinstance(node, ShellNode):
    vertexFormat = node.vertex_format

  # Skip empty split children (owner-encoded splits with no triangles)
  if not node.indexData:
    return None

  # Create the mesh. Skinned meshes must NOT use compaction as their vertex
  # weights rely on the original EDM vertex indices.
  compact = not isinstance(node, SkinNode)
  mesh = _create_mesh(node.vertexData, node.indexData, vertexFormat, compact=compact)
  mesh.name = node.name

  # Owner-channel extraction needs index remap from source vertexData to mesh points.
  # Keep import stable for now; recover owner metadata in a dedicated mesh-mapping pass.

  # Create and link the object
  ob = bpy.data.objects.new(node.name, mesh)
  ob.edm.is_collision_shell = isinstance(node, ShellNode)
  ob.edm.is_renderable      = isinstance(node, (RenderNode, NumberNode, SkinNode))
  ob.edm.damage_argument = -1 if not isinstance(node, (RenderNode, NumberNode)) else node.damage_argument
  if isinstance(node, ShellNode):
    _set_official_special_type(ob, 'COLLISION_SHELL')
    # Match original collision-shell viewport style from official scenes.
    ob.display_type = 'BOUNDS'
  elif isinstance(node, NumberNode):
    _set_official_special_type(ob, 'NUMBER_TYPE')
    raw = getattr(node, "number_params_raw", None)
    if raw and hasattr(ob, "EDMProps"):
      # Layout observed in v10 NumberNode payloads:
      # [unknown, xArg, xScale-int, yArg, yScale-float]
      ob.EDMProps.NUMBER_UV_X_ARG = int(raw[1])
      ob.EDMProps.NUMBER_UV_X_SCALE = float(raw[2])
      ob.EDMProps.NUMBER_UV_Y_ARG = int(raw[3])
      ob.EDMProps.NUMBER_UV_Y_SCALE = float(raw[4])
  else:
    _set_official_special_type(ob, 'UNKNOWN_TYPE')

  # Map damage_argument → EDMProps.DAMAGE_ARG for official exporter.
  if hasattr(ob, "EDMProps") and isinstance(node, (RenderNode, NumberNode)):
    dmg = getattr(node, "damage_argument", -1)
    if dmg is not None and dmg >= 0:
      ob.EDMProps.DAMAGE_ARG = int(dmg)

  # Map animated uniforms from EDM material → EDMProps ARG fields.
  _map_animated_uniforms_to_edmprops(ob, node)

  bpy.context.collection.objects.link(ob)

  return ob

def _extract_light_property(prop_value):
  """Return (argument, keyframes, static_value) for a light property payload."""
  argument = -1
  keys = None
  static_value = None

  if hasattr(prop_value, "keys") and hasattr(prop_value, "argument"):
    try:
      argument = int(prop_value.argument)
    except Exception:
      argument = -1
    keys = list(getattr(prop_value, "keys", []) or [])
    if keys:
      static_value = getattr(keys[0], "value", None)
  elif isinstance(prop_value, list):
    keys = prop_value
    if keys:
      static_value = getattr(keys[0], "value", None)
  else:
    static_value = prop_value

  return argument, keys, static_value


def _to_float(value, default=0.0):
  try:
    return float(value)
  except Exception:
    return float(default)


def _to_vec3(value, default=(1.0, 1.0, 1.0)):
  if value is None:
    return default
  try:
    return (float(value[0]), float(value[1]), float(value[2]))
  except Exception:
    return default


def _edm_light_brightness_to_blender_energy(brightness_value, light_type):
  """
  Invert the official exporter conversion path from Blender light energy to EDM
  Brightness property, so importing then exporting preserves values.
  """
  brightness = max(0.0, _to_float(brightness_value, 0.0))
  if brightness <= 0.0:
    return 0.0

  base = max(0.0, brightness) ** (1.0 / _BLENDER_LAMP_WEAK_COEFFICIENT)
  denom = _PBR_WATTS_TO_LUMENS * _BLENDER_LAMP_ENERGY_COEFFICIENT
  if denom <= 0.0:
    return 0.0
  energy = base / denom
  if light_type != 'SUN':
    energy *= (4.0 * math.pi)
  return max(0.0, energy)


def _set_edmprop(obj, prop_name, value):
  if not hasattr(obj, "EDMProps"):
    return
  edm_props = obj.EDMProps
  if hasattr(edm_props, prop_name):
    try:
      # Clamp integers to 32-bit signed limit for Blender's IntProperty
      if isinstance(value, int):
        value = min(2147483647, max(-2147483648, value))
      setattr(edm_props, prop_name, value)
    except (OverflowError, ValueError):
      pass
    except Exception:
      pass


def _new_fcurve(action, data_path, array_index=None):
  idx = 0 if array_index is None else int(array_index)
  existing = action.fcurves.find(data_path, index=idx)
  if existing is not None:
    return existing
  if array_index is None:
    return action.fcurves.new(data_path=data_path)
  return action.fcurves.new(data_path=data_path, index=int(array_index))


def _add_light_keyframes(action, data_path, keys, value_fn, array_index=None):
  curve = _new_fcurve(action, data_path, array_index)
  for framedata in keys:
    frame = _anim_frame_to_scene_frame(getattr(framedata, "frame", 0.0))
    value = value_fn(getattr(framedata, "value", 0.0))
    key = curve.keyframe_points.insert(frame, float(value), options={'FAST'})
    key.interpolation = 'LINEAR'
  return curve


def _import_light_properties(node, obj, light_data, light_type):
  props = getattr(node, "lightProps", None) or {}
  if not props:
    return

  color_arg, color_keys, color_static = _extract_light_property(props.get("Color"))
  bright_arg, bright_keys, bright_static = _extract_light_property(props.get("Brightness"))
  dist_arg, dist_keys, dist_static = _extract_light_property(props.get("Distance"))
  phi_arg, phi_keys, phi_static = _extract_light_property(props.get("Phi"))
  theta_arg, theta_keys, theta_static = _extract_light_property(props.get("Theta"))
  spec_arg, spec_keys, spec_static = _extract_light_property(props.get("Specular"))
  soft_arg, soft_keys, soft_static = _extract_light_property(props.get(""))
  vol_radius_arg, vol_radius_keys, vol_radius_static = _extract_light_property(props.get("VolumeRadiusFactor"))
  vol_density_arg, vol_density_keys, vol_density_static = _extract_light_property(props.get("VolumeDensityFactor"))
  vol_near_arg, vol_near_keys, vol_near_static = _extract_light_property(props.get("VolumeNearDistance"))
  vol_type_arg, vol_type_keys, vol_type_static = _extract_light_property(props.get("VolumeType"))

  if color_static is not None:
    light_data.color = _to_vec3(color_static, light_data.color)
  if bright_static is not None:
    light_data.energy = _edm_light_brightness_to_blender_energy(bright_static, light_type)
  if dist_static is not None:
    light_data.use_custom_distance = True
    light_data.cutoff_distance = max(0.0, _to_float(dist_static, 0.0))
  if spec_static is not None and hasattr(light_data, "specular_factor"):
    light_data.specular_factor = max(0.0, _to_float(spec_static, light_data.specular_factor))

  if light_type == 'SPOT':
    if phi_static is not None:
      light_data.spot_size = min(math.radians(170.0), max(0.0, _to_float(phi_static, light_data.spot_size)))
    if theta_static is not None:
      light_data.spot_blend = min(1.0, max(0.0, _to_float(theta_static, light_data.spot_blend)))

  # Map EDM animated-property arguments into official exporter EDMProps fields.
  if color_arg >= 0:
    _set_edmprop(obj, "LIGHT_COLOR_ARG", int(color_arg))
  if bright_arg >= 0:
    _set_edmprop(obj, "LIGHT_POWER_ARG", int(bright_arg))
  if dist_arg >= 0:
    _set_edmprop(obj, "LIGHT_DISTANCE_ARG", int(dist_arg))
  if spec_arg >= 0:
    _set_edmprop(obj, "LIGHT_SPECULAR_ARG", int(spec_arg))
  if soft_static is not None:
    _set_edmprop(obj, "LIGHT_SOFTNESS", max(0.0, _to_float(soft_static, 0.0)))
  if vol_radius_static is not None:
    _set_edmprop(obj, "LIGHT_VOLUME_RADIUS_FACTOR", min(1.0, max(0.0, _to_float(vol_radius_static, 0.0))))
  if vol_density_static is not None:
    _set_edmprop(obj, "LIGHT_VOLUME_DENSITY_FACTOR", min(1.0, max(0.0, _to_float(vol_density_static, 0.0))))
  if vol_near_static is not None:
    _set_edmprop(obj, "LIGHT_VOLUME_NEAR_DISTANCE", max(0.0, _to_float(vol_near_static, 0.0)))
  if vol_type_static is not None:
    try:
      vol_type_int = int(round(_to_float(vol_type_static, 4)))
      volume_types = {0: "LANDING", 1: "NAV", 2: "TAXI", 3: "BANO"}
      _set_edmprop(obj, "LIGHT_VOLUME_TYPE", volume_types.get(vol_type_int, "NONE"))
    except Exception:
      pass
  if light_type == 'SPOT':
    spot_arg = phi_arg if phi_arg >= 0 else theta_arg
    if spot_arg >= 0:
      _set_edmprop(obj, "LIGHT_SPOT_SHAPE_ARG", int(spot_arg))

  # Recreate light-data animation curves for exporter parity.
  has_anim = any(keys for keys in (color_keys, bright_keys, dist_keys, phi_keys, theta_keys, spec_keys))
  if not has_anim:
    return

  action = bpy.data.actions.new("Light_{}".format(obj.name))
  if hasattr(action, "argument"):
    first_arg = next((a for a in (color_arg, bright_arg, dist_arg, phi_arg, theta_arg, spec_arg, soft_arg) if a >= 0), -1)
    if first_arg >= 0:
      action.argument = int(first_arg)

  if color_keys:
    for idx in range(3):
      _add_light_keyframes(action, "color", color_keys, lambda v, c=idx: _to_vec3(v)[c], array_index=idx)
  if bright_keys:
    _add_light_keyframes(
      action,
      "energy",
      bright_keys,
      lambda v: _edm_light_brightness_to_blender_energy(v, light_type),
    )
  if dist_keys:
    light_data.use_custom_distance = True
    _add_light_keyframes(action, "cutoff_distance", dist_keys, lambda v: max(0.0, _to_float(v, 0.0)))
  if spec_keys and hasattr(light_data, "specular_factor"):
    _add_light_keyframes(action, "specular_factor", spec_keys, lambda v: max(0.0, _to_float(v, 0.0)))
  if light_type == 'SPOT':
    if phi_keys:
      _add_light_keyframes(
        action,
        "spot_size",
        phi_keys,
        lambda v: min(math.radians(170.0), max(0.0, _to_float(v, 0.0))),
      )
    if theta_keys:
      _add_light_keyframes(
        action,
        "spot_blend",
        theta_keys,
        lambda v: min(1.0, max(0.0, _to_float(v, 0.0))),
      )

  anim_data = light_data.animation_data_create()
  anim_data.action = action


def create_lamp(node):
  """Creates a blender lamp from an edm renderable LampNode"""
  light_props = getattr(node, "lightProps", None) or {}
  light_type = 'SPOT' if ("Phi" in light_props or "Theta" in light_props) else 'POINT'
  light_data = bpy.data.lights.new(name=node.name, type=light_type)
  obj = bpy.data.objects.new(name=node.name, object_data=light_data)
  _import_light_properties(node, obj, light_data, light_type)
  bpy.context.collection.objects.link(obj)
  return obj


def _create_fake_light_material(obj_name, kind="fake_omni"):
  """Create a Blender material with the correct EDM node group for fake lights.

  Args:
    obj_name: Name for the material (will be used as-is).
    kind: "fake_omni" or "fake_spot" — determines which node group to create.

  Returns:
    A bpy.types.Material with the correct EDM node group attached, or None.
  """
  bridge = _ensure_official_material_bridge()
  if not bridge.get("available"):
    return None

  names = bridge.get("names", {})
  official_name = names.get(kind, names.get("fake_omni", "EDM_Fake_Omni_Material"))

  mat = bpy.data.materials.new(name=obj_name)
  mat.use_nodes = True
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links

  # Clear default nodes
  for node in list(nodes):
    nodes.remove(node)

  # Create Material Output
  output_node = nodes.new('ShaderNodeOutputMaterial')
  output_node.location = (400, 0)

  # Create the official EDM group node
  group_node = _create_official_group_node(nodes, bridge, kind, official_name)
  if group_node is None:
    # Last resort: create a bare ShaderNodeGroup with a named node tree
    try:
      group_node = nodes.new("ShaderNodeGroup")
      group_node.node_tree = bpy.data.node_groups.get(official_name)
      if group_node.node_tree is None:
        group_node.node_tree = bpy.data.node_groups.new(official_name, "ShaderNodeTree")
    except Exception:
      return mat  # Return material without group — better than nothing

  group_node.location = (0, 0)

  # Link group output to material output (if there's an output socket)
  if group_node.outputs:
    try:
      links.new(group_node.outputs[0], output_node.inputs['Surface'])
    except Exception:
      pass

  return mat


def _create_fake_light_mesh(name, positions):
  """Create a mesh with one vertex per light position (already in Blender coords)."""
  mesh = bpy.data.meshes.new(name)
  mesh.vertices.add(len(positions))
  for i, pos in enumerate(positions):
    mesh.vertices[i].co = pos
  mesh.update()
  return mesh


def _unpack_uv_double(dval):
  """Decode a double that packs two floats (UV pair) via reinterpretation."""
  import struct
  raw = struct.pack('<d', dval)
  return struct.unpack('<2f', raw)


def create_fake_omni_lights(node):
  """Create a mesh object for FakeOmniLightsNode with one vertex per light."""
  name = node.name or "FakeOmniLights"

  # Each entry: 6 doubles = [x, y, z, size, uv_lb_packed, uv_rt_packed]
  # The last two doubles each pack two floats (UV coordinates).
  positions = []
  sizes = []
  uv_lb = None
  uv_rt = None
  for entry in node.data:
    pos_edm = (entry[0], entry[1], entry[2])
    positions.append(vector_to_blender(pos_edm))
    sizes.append(entry[3])
    if uv_lb is None:
      uv_lb = _unpack_uv_double(entry[4])
      uv_rt = _unpack_uv_double(entry[5])

  if not positions:
    # No light entries — create an empty placeholder
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "SPHERE"
    ob.empty_display_size = 0.1
    _set_official_special_type(ob, 'FAKE_LIGHT')
    bpy.context.collection.objects.link(ob)
    return ob

  mesh = _create_fake_light_mesh(name, positions)
  ob = bpy.data.objects.new(name, mesh)
  _set_official_special_type(ob, 'FAKE_LIGHT')

  # Assign dedicated fake omni material with correct EDM node group
  fake_mat = _create_fake_light_material(name, kind="fake_omni")
  if fake_mat is not None:
    mesh.materials.append(fake_mat)

  # Set EDMProps for fake light export
  if hasattr(ob, "EDMProps"):
    ob.EDMProps.SIZE = float(sizes[0]) if sizes else 3.0
    if uv_lb is not None:
      ob.EDMProps.UV_LB = (float(uv_lb[0]), float(uv_lb[1]))
      ob.EDMProps.UV_RT = (float(uv_rt[0]), float(uv_rt[1]))

  bpy.context.collection.objects.link(ob)
  return ob


def create_fake_spot_lights(node):
  """Create a mesh object for FakeSpotLightsNode.

  We reconstruct these as surface-mode quads (one quad per light) so the
  official exporter emits the same v10 FakeSpotLightsNode layout as reference
  assets (no extra trailing direction vector).
  """
  name = node.name or "FakeSpotLights"

  positions = []
  dirs_bl = []
  sizes = []
  for i, entry in enumerate(node.data):
    pos_edm = entry['position']
    positions.append(vector_to_blender(pos_edm))

    # Direction is more reliable in parentData than the currently-partial raw
    # fake-spot entry parser.
    dir_edm = None
    if i < len(getattr(node, "parentData", [])):
      pd = node.parentData[i]
      if len(pd) >= 3:
        cand = pd[2]
        try:
          if all(math.isfinite(float(v)) for v in cand):
            dir_edm = tuple(float(v) for v in cand)
        except Exception:
          dir_edm = None
    if dir_edm is None:
      cand = getattr(node, "trailing_direction", None)
      if cand is not None:
        try:
          if all(math.isfinite(float(v)) for v in cand):
            dir_edm = tuple(float(v) for v in cand)
        except Exception:
          dir_edm = None
    if dir_edm is None:
      cand = entry.get('direction')
      try:
        if cand is not None and all(math.isfinite(float(v)) for v in cand):
          dir_edm = tuple(float(v) for v in cand)
      except Exception:
        dir_edm = None
    if dir_edm is None:
      dir_edm = (1.0, 0.0, 0.0)

    dvec = vector_to_blender(dir_edm)
    if dvec.length <= 1e-8:
      dvec = Vector((1.0, 0.0, 0.0))
    else:
      dvec.normalize()
    dirs_bl.append(dvec)

    s = entry.get('size', 0.0)
    try:
      s = abs(float(s))
    except Exception:
      s = 0.0
    if not math.isfinite(s) or s <= 1e-6:
      s = 0.1
    sizes.append(s)

  if not positions:
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "SPHERE"
    ob.empty_display_size = 0.1
    _set_official_special_type(ob, 'FAKE_LIGHT')
    bpy.context.collection.objects.link(ob)
    return ob

  # Build one quad per light center, oriented by the light direction. Exporter
  # surface-mode fake spot parsing expects faces + UVs.
  verts = []
  faces = []
  uv_quads = []
  for i, (center, normal, size_val) in enumerate(zip(positions, dirs_bl, sizes)):
    up = Vector((0.0, 0.0, 1.0))
    if abs(normal.dot(up)) > 0.999:
      up = Vector((0.0, 1.0, 0.0))
    tangent = normal.cross(up)
    if tangent.length <= 1e-8:
      tangent = Vector((1.0, 0.0, 0.0))
    tangent.normalize()
    bitangent = normal.cross(tangent)
    if bitangent.length <= 1e-8:
      bitangent = Vector((0.0, 1.0, 0.0))
    bitangent.normalize()

    # parse_faces derives "light size" from the max distance from vertex0 to the
    # other quad vertices (diagonal dominates), so choose side length so the
    # resulting value is approximately size_val.
    half_side = max(0.001, float(size_val) / (2.0 * math.sqrt(2.0)))
    quad = [
      center - tangent * half_side - bitangent * half_side,
      center + tangent * half_side - bitangent * half_side,
      center + tangent * half_side + bitangent * half_side,
      center - tangent * half_side + bitangent * half_side,
    ]
    base = len(verts)
    verts.extend([tuple(v) for v in quad])
    faces.append((base + 0, base + 1, base + 2, base + 3))
    uv_quads.append(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))

  mesh = bpy.data.meshes.new(name)
  mesh.from_pydata(verts, [], faces)
  mesh.update()
  if faces:
    try:
      uv_layer = mesh.uv_layers.new(name="UVMap")
      loop_uv = uv_layer.data
      for poly in mesh.polygons:
        q = uv_quads[poly.index]
        for li, uv in zip(poly.loop_indices, q):
          loop_uv[li].uv = uv
    except Exception:
      pass

  ob = bpy.data.objects.new(name, mesh)
  _set_official_special_type(ob, 'FAKE_LIGHT')

  # Assign dedicated fake spot material with correct EDM node group
  fake_mat = _create_fake_light_material(name, kind="fake_spot")
  if fake_mat is not None:
    mesh.materials.append(fake_mat)

  if hasattr(ob, "EDMProps"):
    ob.EDMProps.SURFACE_MODE = True
    ob.EDMProps.TWO_SIDED = False
    ob.EDMProps.SIZE = float(sizes[0]) if sizes else 0.1

  bpy.context.collection.objects.link(ob)
  return ob


def create_fake_als_lights(node):
  """Create a mesh object for FakeALSNode with one vertex per light."""
  name = node.name or "FakeALSLights"

  positions = []
  for entry in node.data:
    pos_edm = entry['position']
    positions.append(vector_to_blender(pos_edm))

  if not positions:
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "SPHERE"
    ob.empty_display_size = 0.1
    _set_official_special_type(ob, 'FAKE_LIGHT')
    bpy.context.collection.objects.link(ob)
    return ob

  mesh = _create_fake_light_mesh(name, positions)
  ob = bpy.data.objects.new(name, mesh)
  _set_official_special_type(ob, 'FAKE_LIGHT')

  # ALS lights map to fake_omni material kind
  fake_mat = _create_fake_light_material(name, kind="fake_omni")
  if fake_mat is not None:
    mesh.materials.append(fake_mat)

  bpy.context.collection.objects.link(ob)
  return ob
