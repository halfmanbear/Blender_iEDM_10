"""Scope importer compatibility patches to each official EDM export.

Wraps io_scene_edm.run_edm_export so each installer below patches the
exporter just before an export and everything is restored afterwards, even
when the export fails. Installers return (owner, attribute, original) lists.
The exporter's own always-on profiler is also switched off for the export.
"""

import functools
import sys
import types

from .export_damage_mask import install_damage_builder
from .export_skin_bind import install_skin_bind


class _NoProfile:
    """Stand-in for cProfile.Profile; the exporter never reads its stats."""

    def enable(self):
        pass

    def disable(self):
        pass


def _skip_exporter_profiler():
    """The exporter profiles every export (~10% of export time) and discards it."""
    walker = getattr(sys.modules.get("io_scene_edm"), "collection_walker", None)
    profile = getattr(walker, "cProfile", None)
    if profile is None:
        return []
    walker.cProfile = types.SimpleNamespace(Profile=_NoProfile)
    return [(walker, "cProfile", profile)]


class _IntSocket:
    """A float socket read as the integer socket the exporter asked for."""

    def __init__(self, socket):
        self._socket = socket

    @property
    def default_value(self):
        return int(round(self._socket.default_value))

    def __getattr__(self, name):
        return getattr(self._socket, name)


def _int_socket_lookup():
    """The exporter builds NodeSocketInt group inputs as NodeSocketFloat
    (node_tree_tools.create_inodesocket_input) but material_wrap looks them
    up as NodeSocketInt, so DecalId was never read and every decal id
    exported as 0 (fa-18c/F-16C numbers, F-15E refuel number)."""
    wrap = sys.modules.get("materials.material_wrap")
    search = getattr(wrap, "search_in_socket", None)
    if search is None:
        return []

    @functools.wraps(search)
    def lookup(node_group, name, socket_type):
        socket = search(node_group, name, socket_type)
        if socket is None and socket_type == "NodeSocketInt":
            socket = search(node_group, name, "NodeSocketFloat")
            if socket is not None:
                return _IntSocket(socket)
        return socket

    wrap.search_in_socket = lookup
    return [(wrap, "search_in_socket", search)]


def _keep_zero_socket_values():
    """material_wrap.ValueDesk.update reads ``socket.default_value if socket
    and socket.default_value else def_value``, so a socket set to 0 exported
    its default instead (F-15E visor opacityValue 0 -> 1)."""
    wrap = sys.modules.get("materials.material_wrap")
    desk = getattr(wrap, "ValueDesk", None)
    update = getattr(desk, "update", None)
    if update is None:
        return []

    @functools.wraps(update)
    def keep_zero(self):
        update(self)
        socket = getattr(self, "socket", None)
        value = getattr(socket, "default_value", None)
        if type(value) in (int, float) and not value:
            self.value = value

    desk.update = keep_zero
    return [(desk, "update", update)]


_INSTALLERS = (
    install_damage_builder,
    install_skin_bind,
    _skip_exporter_profiler,
    _int_socket_lookup,
    _keep_zero_socket_values,
)


def _wrap_export(run_edm_export):
    @functools.wraps(run_edm_export)
    def run(*args, **kwargs):
        swapped = []
        try:
            for install in _INSTALLERS:
                swapped.extend(install())
            return run_edm_export(*args, **kwargs)
        finally:
            for owner, name, original in reversed(swapped):
                setattr(owner, name, original)

    run._iedm_original = run_edm_export
    return run


def install_export_hook():
    """Wrap the exporter's export entry point. Idempotent."""
    exporter = sys.modules.get("io_scene_edm")
    run = getattr(exporter, "run_edm_export", None)
    if run is None or hasattr(run, "_iedm_original"):
        return run is not None
    exporter.run_edm_export = _wrap_export(run)
    return True


def remove_export_hook():
    exporter = sys.modules.get("io_scene_edm")
    run = getattr(exporter, "run_edm_export", None)
    original = getattr(run, "_iedm_original", None)
    if original is not None:
        exporter.run_edm_export = original
