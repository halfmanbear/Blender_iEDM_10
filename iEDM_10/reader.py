"""Compatibility aggregation module for the importer package."""

from .blender_importer import prelude as _prelude
from .blender_importer import graph_pipeline as _graph_pipeline
from .blender_importer import classifier as _classifier
from .blender_importer import import_capabilities as _import_capabilities
from .blender_importer import import_profiles as _import_profiles
from .blender_importer import materials_bridge as _materials_bridge
from .blender_importer import mesh_create as _mesh_create
from .blender_importer import object_create as _object_create
from .blender_importer import material_setup as _material_setup
from .blender_importer import animation as _animation
from .blender_importer import graph_postprocess as _graph_postprocess
from .blender_importer import import_pipeline as _import_pipeline
from .blender_importer import lights as _lights
from .blender_importer.nodes import armature as _node_armature
from .blender_importer.nodes import core as _node_core
from .blender_importer.nodes import diagnostics as _node_diagnostics
from .blender_importer.nodes import mesh as _node_mesh
from .blender_importer.nodes import visibility as _node_visibility

_MODULES = (
  _prelude,
  _classifier,
  _import_capabilities,
  _import_profiles,
  _materials_bridge,
  _graph_pipeline,
  _mesh_create,
  _object_create,
  _material_setup,
  _animation,
  _graph_postprocess,
  _node_mesh,
  _node_visibility,
  _node_armature,
  _node_core,
  _node_diagnostics,
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
del _classifier
del _import_capabilities
del _import_profiles
del _materials_bridge
del _graph_pipeline
del _mesh_create
del _object_create
del _material_setup
del _animation
del _graph_postprocess
del _node_mesh
del _node_visibility
del _node_armature
del _node_core
del _node_diagnostics
del _import_pipeline
del _lights
del _exportable_items
