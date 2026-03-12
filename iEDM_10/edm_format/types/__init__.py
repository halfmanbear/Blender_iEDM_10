"""Compatibility aggregation module for split EDM type definitions."""

from . import core as _core
from . import lights as _lights
from . import number as _number
from . import render_shell as _render_shell

_MODULES = (_core, _render_shell, _lights, _number)


def _exportable_items(module):
  for name, value in vars(module).items():
    if name.startswith("__") and name.endswith("__"):
      continue
    yield name, value


_SHARED = {}
for _module in _MODULES:
  _SHARED.update(dict(_exportable_items(_module)))

# Mirror the original single-module behavior: all symbols are visible in each
# part module's global namespace, regardless of where they were originally defined.
for _module in _MODULES:
  _module.__dict__.update(_SHARED)

globals().update(_SHARED)

del _SHARED
del _MODULES
del _core
del _render_shell
del _lights
del _number
del _exportable_items
