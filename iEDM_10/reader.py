"""Compatibility aggregation module for the importer package."""

from .blender_importer import prelude as _prelude
from .blender_importer import graph_build as _graph_build
from .blender_importer import graph_pipeline as _graph_pipeline
from .blender_importer import import_capabilities as _import_capabilities
from .blender_importer import materials_bridge as _materials_bridge
from .blender_importer import mesh_create as _mesh_create
from .blender_importer import object_create as _object_create
from .blender_importer import bbox_utils as _bbox_utils
from .blender_importer import material_setup as _material_setup
from .blender_importer import animation as _animation
from .blender_importer import graph_postprocess as _graph_postprocess
from .blender_importer import session as _session
from .blender_importer import anim_actions as _anim_actions
from .blender_importer import orient_scale as _orient_scale
from .blender_importer import vis_rewrites as _vis_rewrites
from .blender_importer import skin_rewrites as _skin_rewrites
from .blender_importer import orient_fixes as _orient_fixes
from .blender_importer import ctrl_splits as _ctrl_splits
from .blender_importer import node_transform as _node_transform
from .blender_importer import import_pipeline as _import_pipeline
from .blender_importer import lights as _lights
from .blender_importer.nodes import armature as _node_armature
from .blender_importer.nodes import core as _node_core
from .blender_importer.nodes import diagnostics as _node_diagnostics
from .blender_importer.nodes import mesh as _node_mesh
from .blender_importer.nodes import visibility as _node_visibility

_MODULES = (
  _prelude,
  _import_capabilities,
  _materials_bridge,
  _graph_build,
  _graph_pipeline,
  _mesh_create,
  _object_create,
  _bbox_utils,
  _material_setup,
  _animation,
  _graph_postprocess,
  _anim_actions,
  _orient_scale,
  _vis_rewrites,
  _skin_rewrites,
  _orient_fixes,
  _ctrl_splits,
  _node_transform,
  _node_mesh,
  _node_visibility,
  _node_armature,
  _node_core,
  _node_diagnostics,
  _session,
  _import_pipeline,
  _lights,
)


def _exportable_items(module):
  for name, value in vars(module).items():
    if name.startswith("__") and name.endswith("__"):
      continue
    yield name, value


_SHARED = {}
for _module in _MODULES:
  _SHARED.update(dict(_exportable_items(_module)))

# Mirror prior single-file behavior: each part can resolve names that were
# originally defined in other sections of the monolithic reader.py.
for _module in _MODULES:
  _module.__dict__.update(_SHARED)

globals().update(_SHARED)

del _SHARED
del _MODULES
del _prelude
del _graph_build
del _import_capabilities
del _materials_bridge
del _graph_pipeline
del _mesh_create
del _object_create
del _bbox_utils
del _material_setup
del _animation
del _graph_postprocess
del _anim_actions
del _orient_scale
del _vis_rewrites
del _skin_rewrites
del _orient_fixes
del _ctrl_splits
del _node_transform
del _node_mesh
del _node_visibility
del _node_armature
del _node_core
del _node_diagnostics
del _session
del _import_pipeline
del _lights
del _exportable_items
