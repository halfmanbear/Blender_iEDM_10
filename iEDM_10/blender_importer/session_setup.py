"""Configure import options, runtime context, materials, and scene metadata."""

import os as os
from dataclasses import dataclass, field
from typing import Mapping

import bpy as bpy

from ..edm_format import EDMFile
from ..utils import chdir, print_edm_graph
from .graph_diagnostics import _reset_mesh_origin_mode, _reset_transform_debug
from .import_capabilities import derive_import_capabilities, inspect_import_graph
from .import_context import (
    DEFAULT_PROFILE,
    FRAME_SCALE,
    _import_capability_detail,
    _import_capability_name,
    _import_ctx,
    _import_profile_detail,
    _import_profile_flag,
    _import_profile_name,
    _log,
)
from .material_setup import create_material
from .nodes.core import _apply_shadeless


@dataclass(frozen=True)
class SceneBoxOptions:
    bounding_box: bool = True
    user_box: bool = True
    light_box: bool = True

    def any_enabled(self):
        return self.bounding_box or self.user_box or self.light_box


@dataclass(frozen=True)
class ImportOptions:
    raw: Mapping[str, object] = field(default_factory=dict)
    scene_boxes: SceneBoxOptions = field(default_factory=SceneBoxOptions)

    @classmethod
    def from_mapping(cls, options):
        if isinstance(options, cls):
            return options
        raw = dict(options or {})
        return cls(raw=raw, scene_boxes=_resolve_scene_box_options(raw))

    def get(self, key, default=None):
        return self.raw.get(key, default)


def _resolve_scene_box_options(options):
    """Resolve scene-box creation flags from legacy and per-box options."""
    has_individual_options = any(
        key in options
        for key in ("import_bounding_box", "import_user_box", "import_light_box")
    )

    if "preserve_scene_boxes" in options:
        default_enabled = bool(options.get("preserve_scene_boxes"))
    else:
        default_enabled = not has_individual_options

    return SceneBoxOptions(
        bounding_box=bool(options.get("import_bounding_box", default_enabled)),
        user_box=bool(options.get("import_user_box", default_enabled)),
        light_box=bool(options.get("import_light_box", default_enabled)),
    )


def _start_import_session(filename, options):
    # Re-initialize the global import context for this new import session
    _import_ctx.__init__()
    _reset_transform_debug(options)
    _import_ctx.render_split_debug = {
        "enabled": bool(options.get("debug_render_splits", False))
    }
    _reset_mesh_origin_mode(options)
    _import_ctx.bone_import_ctx = None
    _import_ctx.source_dir = os.path.dirname(os.path.abspath(filename))

    # Parse the EDM file
    edm = EDMFile(filename)

    if getattr(_import_ctx, "verbosity", 0) >= 1:
        print("[iEDM] Debug[1]: Raw file graph:")
        print_edm_graph(edm.transformRoot)

    # Store EDM version globally early so importer heuristics can use it before
    # addon-export patch toggles are configured.
    _import_ctx.edm_version = edm.version
    _import_ctx.import_profile = DEFAULT_PROFILE
    _import_ctx.import_capabilities = derive_import_capabilities(edm)
    features = inspect_import_graph(edm)

    _log.info(
        "EDM import capabilities = {} ({})".format(
            _import_capability_name(),
            getattr(
                getattr(_import_ctx, "import_capabilities", None), "description", ""
            ),
        )
    )
    _log.info("EDM import capability detail = {}".format(_import_capability_detail()))

    # Dump active flags so flag-related bugs are immediately visible
    caps = getattr(_import_ctx, "import_capabilities", None)
    flags = getattr(caps, "flags", None) or {}
    if flags:
        active = {k: v for k, v in flags.items() if v}
        if active:
            _log.debug(
                "Active flags: {}".format(
                    ", ".join("{}={}".format(k, v) for k, v in sorted(active.items()))
                ),
                level=1,
            )

    return edm, features


def _configure_mesh_origin_mode(options, features):
    # Detect v10 owner-encoded split render graphs (shared-parent render chunks).
    has_bones = features.has_bones
    has_owner_encoded_split_renders = features.has_owner_encoded_split_renders
    has_generic_render_chunks = features.has_generic_render_chunks
    has_shell_nodes = features.has_shell_nodes
    plain_root_v10 = features.plain_root_v10
    auto_geometry_safe_v10_split = bool(
        plain_root_v10
        and not has_bones
        and (
            has_owner_encoded_split_renders
            or (has_generic_render_chunks and has_shell_nodes)
        )
    )

    user_mesh_origin_mode = str(options.get("mesh_origin_mode", "") or "").upper()
    can_auto_override_mesh_origin = user_mesh_origin_mode in {"", "APPROX"}
    if (
        _import_profile_flag("auto_raw_mesh_origin_v10_split")
        and auto_geometry_safe_v10_split
        and can_auto_override_mesh_origin
        and _import_ctx.mesh_origin_mode != "RAW"
    ):
        _import_ctx.mesh_origin_mode = "RAW"
        _log.info(
            "Auto-selected RAW origin mode for a v10 split control-node "
            "asset (plain root, no bones)."
        )
    elif (
        False  # no capability enables plain-root non-skeletal RAW mode
        and can_auto_override_mesh_origin
        and _import_ctx.mesh_origin_mode != "RAW"
        and plain_root_v10
        and not has_bones
    ):
        _import_ctx.mesh_origin_mode = "RAW"
        _log.info(
            "Auto-selected RAW origin mode for a plain-root v10 asset "
            "to preserve authored transforms."
        )


def _configure_blender_import_scene():
    bpy.context.preferences.edit.use_negative_frames = False
    bpy.context.scene.use_preview_range = False
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = FRAME_SCALE
    bpy.context.scene.frame_set(0)


def _create_import_materials(filename, edm, options):
    with chdir(os.path.dirname(os.path.abspath(filename))):
        for material in edm.root.materials:
            material.blender_material = create_material(material)
            if material.blender_material and options.get("shadeless", False):
                _apply_shadeless(material.blender_material)

    if _import_ctx.mesh_origin_mode == "RAW":
        _log.info(
            "Mesh origin mode RAW (no geometry-center recenter on render-only nodes)"
        )


def _store_import_metadata(edm):
    bpy.context.scene.edm_version = edm.version
    try:
        bpy.context.scene["_iedm_import_profile"] = str(_import_profile_name())
        bpy.context.scene["_iedm_import_profile_detail"] = str(_import_profile_detail())
        bpy.context.scene["_iedm_import_capabilities"] = str(_import_capability_name())
        bpy.context.scene["_iedm_import_capability_detail"] = str(
            _import_capability_detail()
        )
    except Exception as e:
        _log.warn("store import metadata", exc=e)
