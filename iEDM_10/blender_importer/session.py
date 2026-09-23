import os
from dataclasses import dataclass, field
from typing import Mapping

import bpy

from ..edm_format import EDMFile
from ..edm_format.mathtypes import Matrix
from ..edm_format.types import AnimatingNode, LodNode, TransformNode
from ..edm_format.types.render_shell import SkinNode
from ..utils import action_fcurves, chdir, print_edm_graph
from .bbox_utils import (
    _cache_root_aabb_payloads_on_scene,
    _create_light_box_from_root,
    _create_user_box_from_root,
    _has_special_box,
    create_bounding_box_from_root,
)
from .ctrl_splits import (
    _rename_control_wrapper_mesh_pairs,
    _split_multi_arg_rotation_controls,
)
from .graph_build import build_graph
from .graph_pipeline import (
    _assign_collections,
    _debug_filter_terms,
    _debug_fmt_rot_deg,
    _debug_fmt_vec3,
    _reset_mesh_origin_mode,
    _reset_transform_debug,
)
from .graph_postprocess import (
    _apply_root_visibility_pair_wrapper_basis_fix,
    _apply_static_root_visibility_wrapper_basis_fix,
    _zero_render_child_mesh_locals_under_transform,
)
from .import_capabilities import derive_import_capabilities, inspect_import_graph
from .material_setup import create_material
from .node_transform import apply_node_transform
from .nodes.armature import _prepare_bone_import
from .nodes.core import _apply_shadeless, _process_lod_post_children, process_node
from .nodes.diagnostics import _print_import_diagnostics
from .orient_scale import _rewrite_oriented_scale_controls

# Session orchestration fragment.
# Functions here drive the top-level import sequence; the pipeline detail
# (graph construction, node processing, animations) lives in import_pipeline.py.
from .prelude import (
    _ROOT_BASIS_FIX,
    DEFAULT_PROFILE,
    FRAME_SCALE,
    _assign_action,
    _import_capability_detail,
    _import_capability_name,
    _import_ctx,
    _import_profile_detail,
    _import_profile_flag,
    _import_profile_name,
    _log,
    _visibility_scene_keys,
    _visibility_scene_ranges,
)
from .skin_rewrites import _resolve_skin_parent_overrides_by_bind_rest
from .vis_rewrites import (
    _apply_argvis_chain_basis_fix,
    _apply_plain_root_visibility_basis_fix,
    _apply_visibility_pair_wrapper_object_basis_fix,
    _fix_inverse_scaled_visibility_rest_offset,
    _restore_skin_visibility_transform_basis,
    _split_multi_arg_visibility_controls,
)


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


def _detect_bonetransform_prefix_compound(graph, root_tf):
    """Walk leading Bonetransform chain and return the compound prefix matrix.

    Some community-exported EDMs place one or more TransformNodes named
    'Bonetransform' between the graph root and the actual scene content.
    Without intervention _EDMFileRoot gets RBF, creating a world chain of
      RBF @ M1 @ M2 @ ...  which over-rotates the aircraft.

    This function walks the single-child chain from graph.root, collecting each
    consecutive TransformNode named 'Bonetransform', and returns their compound
    matrix  compound = M1 @ M2 @ ...  as a mathutils.Matrix, or None if no such
    chain exists.

    The call site then sets:
      _EDMFileRoot = RBF @ inv(compound)
    so the effective world transform collapses:
      (RBF @ inv(compound)) @ compound @ content = RBF @ content
    """
    # For plain-root v10 the graph root carries a bare unnamed Node (no matrix/base).
    # Only bail if root_tf actually carries transform data.
    if root_tf is not None and (hasattr(root_tf, "matrix") or hasattr(root_tf, "base")):
        return None

    MBl = type(_ROOT_BASIS_FIX)  # mathutils.Matrix
    compound = MBl.Identity(4)
    node = graph.root
    found_count = 0

    while True:
        children = getattr(node, "children", []) or []
        if len(children) != 1:
            break
        child = children[0]
        child_tf = getattr(child, "transform", None)
        if not isinstance(child_tf, TransformNode):
            break
        child_name = (getattr(child_tf, "name", "") or "").lower()
        if child_name != "bonetransform":
            break
        if not hasattr(child_tf, "matrix"):
            break
        m = Matrix(child_tf.matrix)
        m_bl = MBl([[float(m[r][c]) for c in range(4)] for r in range(4)])
        compound = compound @ m_bl
        found_count += 1
        node = child

    if found_count == 0:
        return None

    return compound


def _create_graph_root_object(graph, options, features):
    root_tf = getattr(graph.root, "transform", None)
    has_root_transform_payload = isinstance(root_tf, (TransformNode, AnimatingNode))
    has_shell_nodes = features.has_shell_nodes
    has_segments_nodes = features.has_segments_nodes
    wants_embedded_collision_root_basis_object = bool(
        _import_ctx.edm_version >= 10
        and _import_profile_flag("embedded_collision_scene_root_basis_object")
        and (has_shell_nodes or has_segments_nodes)
    )
    _import_ctx.collision_geometry_basis_fix = bool(
        _import_ctx.edm_version >= 10
        and _import_profile_flag("collision_geometry_basis_fix")
        and (has_shell_nodes or has_segments_nodes)
    )
    wants_lod_root_basis_object = bool(
        _import_ctx.edm_version >= 10
        and _import_profile_flag("lod_root_scene_basis_object")
        and isinstance(root_tf, LodNode)
    )

    force_root_obj = bool(options.get("force_root_object", False))

    wants_profile_root_basis_object = bool(
        _import_ctx.edm_version >= 10
        and _import_profile_flag("v10_root_object_basis_fix")
        and _import_profile_flag("implicit_scene_root_basis_object")
        and not has_root_transform_payload
    )

    _import_ctx.use_scene_root_basis_object = bool(
        force_root_obj
        or wants_embedded_collision_root_basis_object
        or wants_lod_root_basis_object
        or (
            _import_profile_flag("implicit_scene_root_basis_object")
            and has_root_transform_payload
        )
        or wants_profile_root_basis_object
        or (
            has_root_transform_payload
            and _import_ctx.edm_version >= 10
            and _import_profile_flag("v10_root_object_basis_fix")
        )
    )

    if (
        has_root_transform_payload
        or force_root_obj
        or wants_embedded_collision_root_basis_object
        or wants_lod_root_basis_object
        or wants_profile_root_basis_object
    ):
        root_name = getattr(root_tf, "name", "") or ""
        if not root_name.strip():
            root_name = "_EDMFileRoot"

        root_obj = bpy.data.objects.new(root_name, None)
        root_obj.empty_display_size = 0.1
        bpy.context.collection.objects.link(root_obj)
        graph.root.blender = root_obj

        if _import_ctx.edm_version >= 10 and _import_profile_flag(
            "v10_root_object_basis_fix"
        ):
            m_root = Matrix.Identity(4)
            if hasattr(root_tf, "matrix"):
                m_root = Matrix(root_tf.matrix)
            elif hasattr(root_tf, "base") and hasattr(root_tf.base, "matrix"):
                m_root = Matrix(root_tf.base.matrix)
            # Detect a non-standard Bonetransform prefix: a single TransformNode child
            # chain before the standard ROOT_BASIS_FIX node.
            _bonetransform_compound = _detect_bonetransform_prefix_compound(
                graph, root_tf
            )
            if _bonetransform_compound is not None:
                _import_ctx.bonetransform_prefix_matrix = _bonetransform_compound
                # The compound (M1@M2) collapses via inv(compound), leaving RBF as the
                # effective world transform.  For Bonetransform-prefix EDMs the raw
                # geometry has its nose along local -Z, which RBF maps to Blender +Y.
                # An extra Rz(-90°) rotates +Y to +X for the DCS/Blender convention.
                # Rz(-90°) = [[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]]
                _MBl = type(_ROOT_BASIS_FIX)
                _Rz_neg90 = _MBl(
                    ((0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
                )
                root_obj.matrix_basis = (
                    _Rz_neg90 @ _ROOT_BASIS_FIX @ _bonetransform_compound.inverted()
                )
            else:
                root_obj.matrix_basis = _ROOT_BASIS_FIX @ m_root
        elif has_root_transform_payload:
            apply_node_transform(
                graph.root, graph.root.blender, used_shared_parent=False
            )
    else:
        graph.root.blender = None


def _create_import_scene_boxes(edm_root, box_options):
    if box_options.any_enabled():
        bbox_coord_fix = (
            _ROOT_BASIS_FIX if _import_ctx.use_scene_root_basis_object else None
        )
        if box_options.bounding_box and not _has_special_box("BOUNDING_BOX"):
            create_bounding_box_from_root(edm_root, coord_fix=bbox_coord_fix)
        if box_options.user_box:
            _create_user_box_from_root(edm_root, coord_fix=bbox_coord_fix)
        if box_options.light_box:
            _create_light_box_from_root(edm_root, coord_fix=bbox_coord_fix)


def _debug_dump_stage_objects(stage_name):
    dbg = getattr(_import_ctx, "transform_debug", {}) or {}
    if not dbg.get("enabled"):
        return
    filter_terms = _debug_filter_terms()
    if not filter_terms:
        return

    matches = []
    for ob in list(getattr(bpy.data, "objects", []) or []):
        name = getattr(ob, "name", "") or ""
        lowered = name.lower()
        if any(term in lowered for term in filter_terms):
            matches.append(ob)
    if not matches:
        return

    print("[iEDM][STAGE] begin stage={} matched={}".format(stage_name, len(matches)))
    for ob in sorted(matches, key=lambda item: getattr(item, "name", "")):
        try:
            basis_loc, basis_rot, basis_scale = ob.matrix_basis.decompose()
            world_loc, world_rot, world_scale = ob.matrix_world.decompose()
            action = getattr(getattr(ob, "animation_data", None), "action", None)
            curves = [fc.data_path for fc in action_fcurves(action)] if action else []
            print(
                "[iEDM][STAGE] stage={} obj={} type={} parent={} action={} curves={} "
                "basis_loc={} basis_rot_deg={} basis_scale={} world_loc={} "
                "world_rot_deg={} world_scale={}".format(
                    stage_name,
                    ob.name,
                    getattr(ob, "type", ""),
                    getattr(getattr(ob, "parent", None), "name", None),
                    getattr(action, "name", None),
                    curves,
                    _debug_fmt_vec3(basis_loc),
                    _debug_fmt_rot_deg(basis_rot),
                    _debug_fmt_vec3(basis_scale),
                    _debug_fmt_vec3(world_loc),
                    _debug_fmt_rot_deg(world_rot),
                    _debug_fmt_vec3(world_scale),
                )
            )
        except Exception as e:
            print(
                "[iEDM][STAGE] stage={} obj={} error={}".format(
                    stage_name, getattr(ob, "name", "<unknown>"), e
                )
            )
    print("[iEDM][STAGE] end stage={}".format(stage_name))


def _run_render_offset_postprocess(graph):
    _zero_render_child_mesh_locals_under_transform(graph)
    _debug_dump_stage_objects("after_process_node")


def _run_visibility_basis_postprocess(graph):
    _apply_plain_root_visibility_basis_fix(graph)
    _debug_dump_stage_objects("after_plain_root_visibility_basis_fix")
    _apply_static_root_visibility_wrapper_basis_fix(graph)
    _debug_dump_stage_objects("after_static_root_visibility_wrapper_basis_fix")
    _apply_root_visibility_pair_wrapper_basis_fix(graph)
    _debug_dump_stage_objects("after_root_visibility_pair_wrapper_basis_fix")


def _run_control_rewrite_postprocess(graph, options):
    _apply_argvis_chain_basis_fix()
    _debug_dump_stage_objects("after_argvis_chain_basis_fix")
    _rewrite_oriented_scale_controls(graph)
    _debug_dump_stage_objects("after_rewrite_oriented_scale_controls")
    graph.walk_tree(_process_lod_post_children)
    _debug_dump_stage_objects("after_lod_post_children")

    _split_multi_arg_rotation_controls(graph)
    # The official exporter only reads active visibility actions on objects.
    # Visibility splits are required even without transform helper rewrites.
    _split_multi_arg_visibility_controls(graph)
    _debug_dump_stage_objects("after_multi_arg_rewrite")

    _apply_visibility_pair_wrapper_object_basis_fix()
    _debug_dump_stage_objects("after_visibility_pair_wrapper_object_basis_fix")
    _rename_control_wrapper_mesh_pairs(graph)
    _debug_dump_stage_objects("after_rename_control_wrapper_mesh_pairs")


def _run_orientation_postprocess():
    # A world Euler angle cannot identify missing coordinate conversion: valid
    # authored child rotations can cancel the root basis. Preserve the graph's
    # composed transforms instead of rotating meshes and then their parents.
    _restore_skin_visibility_transform_basis()
    _debug_dump_stage_objects("after_restore_skin_visibility_transform_basis")
    _resolve_skin_parent_overrides_by_bind_rest()
    _debug_dump_stage_objects("after_resolve_skin_parent_overrides_by_bind_rest")
    _fix_inverse_scaled_visibility_rest_offset()
    _debug_dump_stage_objects("after_fix_inverse_scaled_visibility_rest_offset")


def _finish_import_postprocess(edm, graph, options):
    _print_import_diagnostics(edm, graph)

    if options.get("assign_collections", True):
        _assign_collections(graph)

    if _import_ctx.transform_debug.get("enabled"):
        if (
            _import_ctx.transform_debug["emitted"]
            >= _import_ctx.transform_debug["limit"]
        ):
            _log.info(
                "Transform debug reached output limit of {}".format(
                    _import_ctx.transform_debug["limit"]
                )
            )
        _log.info(
            "Transform debug emitted {} record(s)".format(
                _import_ctx.transform_debug["emitted"]
            )
        )

    _propagate_visibility_hide_to_render_nodes(graph)

    bpy.context.scene.frame_set(100)
    bpy.context.view_layer.update()


def _run_skin_transform_postprocess():
    """Apply object-level transforms on all skinned meshes bound to an armature.

    SkinNode meshes may carry a localized matrix_basis after bind-rest resolution.
    Applying it bakes vertex positions to the new rest and resets the object to
    identity so the official exporter reads clean transforms on re-export.
    """
    view_layer = bpy.context.view_layer
    prev_active = view_layer.objects.active
    for o in list(bpy.context.selected_objects):
        o.select_set(False)
    applied = 0
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            continue
        if not any(m.type == "ARMATURE" for m in obj.modifiers):
            continue
        try:
            if obj.matrix_basis == Matrix.Identity(4):
                continue
        except Exception:
            continue
        try:
            obj.select_set(True)
            view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            obj.select_set(False)
            applied += 1
        except Exception as e:
            _log.warn(
                "transform_apply on skin mesh '{}': {}".format(
                    getattr(obj, "name", "?"), e
                ),
                exc=e,
            )
            try:
                obj.select_set(False)
            except Exception as exc:
                _log.debug(
                    "Optional operation failed: {}".format(exc), level=2
                )
    if prev_active is not None:
        view_layer.objects.active = prev_active
    if applied:
        _log.debug("Applied transforms on {} skinned mesh object(s).".format(applied))


def _run_import_postprocess(edm, graph, options):
    _run_render_offset_postprocess(graph)
    _run_visibility_basis_postprocess(graph)
    _run_control_rewrite_postprocess(graph, options)
    _run_orientation_postprocess()
    bpy.context.scene.frame_set(100)
    bpy.context.view_layer.update()
    _finalize_skin_bind_space(graph)
    _run_skin_transform_postprocess()
    _finish_import_postprocess(edm, graph, options)
    _finalize_render_origins(graph)
    from .bone_controls import build_bone_control_graph

    build_bone_control_graph(graph)


def read_file(filename, options=None):
    options = ImportOptions.from_mapping(options)

    edm, features = _start_import_session(filename, options)
    _configure_mesh_origin_mode(options, features)
    _configure_blender_import_scene()
    _create_import_materials(filename, edm, options)
    _store_import_metadata(edm)

    _import_ctx.file_has_bones = bool(features.has_bones)

    _cache_root_aabb_payloads_on_scene(edm.root)

    graph = build_graph(edm)
    graph.print_tree()

    _create_graph_root_object(graph, options, features)
    _create_import_scene_boxes(edm.root, options.scene_boxes)

    has_skin_nodes = any(
        isinstance(getattr(n, "render", None), SkinNode) for n in graph.nodes
    )
    if has_skin_nodes:
        _prepare_bone_import(graph, graph.root.blender)

    graph.walk_tree(process_node)

    _run_import_postprocess(edm, graph, options)


def _finalize_skin_bind_space(graph):
    """Localize absolute skin vertices against the settled neutral-pose parent.

    Initial binding subtracts the bind translation, before helper rewrites have
    settled. Undo that translation and use the full object inverse here; otherwise
    rotated/scaled helpers rotate/scale absolute skeleton vertices a second time.
    Only meshes localized by this import participate.
    """
    from mathutils import Matrix as BlenderMatrix
    from mathutils import Vector

    seen = set()
    for node in graph.nodes:
        obj = node.blender
        if obj is None or obj in seen or obj.type != "MESH":
            continue
        seen.add(obj)
        if not obj.get("_iedm_skin_localized_to_bind") or obj.get(
            "_iedm_skin_bind_space_finalized"
        ):
            continue
        loc = obj.get("_iedm_skin_bind_target_loc")
        if loc is None:
            continue
        try:
            correction = obj.matrix_world.inverted() @ BlenderMatrix.Translation(
                Vector(loc)
            )
        except ValueError:
            _log.warn(
                "Cannot localize skin '{}': singular neutral transform".format(obj.name)
            )
            continue
        obj.data.transform(correction)
        obj.data.update()
        obj["_iedm_skin_bind_space_finalized"] = True


def _visibility_frame_intervals(vis_data):
    """Intersect controls and union each control's preview-timeline ranges."""
    intervals = [(0, FRAME_SCALE + 1)]
    for _arg, ranges in vis_data:
        control = _visibility_scene_ranges(ranges)
        intervals = [
            (max(a, c), min(b, d))
            for a, b in intervals
            for c, d in control
            if max(a, c) < min(b, d)
        ]
    return intervals


def _propagate_visibility_hide_to_render_nodes(graph):
    """Preview inherited visibility using actions that obey exporter argument muting.

    Each distinct argument/range set has a custom-property action. The official
    exporter ignores that property, while its argument tools mute the action just
    like the authored VISIBLE actions. Mesh hide drivers read these evaluated
    properties instead of bypassing action muting with direct frame expressions.
    """
    controllers = {}

    for node in graph.nodes:
        obj = node.blender
        if obj is None or obj.type != "MESH" or node.render is None:
            continue
        controls = []
        ancestor = node
        seen = set()
        while ancestor is not None:
            transforms = [ancestor.transform] + list(
                getattr(ancestor, "_collapsed_transforms", None) or []
            )
            for transform in transforms:
                if transform is not None and id(transform) not in seen:
                    seen.add(id(transform))
                    for arg, ranges in getattr(transform, "visData", None) or []:
                        control = _visibility_controller_for(arg, ranges, controllers)
                        if control not in controls:
                            controls.append(control)
            ancestor = ancestor.parent
        if not controls:
            continue
        # Keep expressions below Blender's driver length limit for deep hierarchies.
        while len(controls) > 24:
            combined = []
            for offset in range(0, len(controls), 24):
                helper = bpy.data.objects.new("IEDM_VisibilityIntersection", None)
                bpy.context.collection.objects.link(helper)
                helper.empty_display_size = 0.01
                helper["_iedm_visible"] = 1.0
                _add_visibility_hide_driver(
                    helper,
                    '["_iedm_visible"]',
                    controls[offset : offset + 24],
                    invert=False,
                )

                combined.append(helper)
            controls = combined
        for path in ("hide_viewport", "hide_render"):
            _add_visibility_hide_driver(obj, path, controls)


def _visibility_controller_for(arg, ranges, controllers):
    """Create or reuse the action-backed visibility controller for an argument."""
    signature = (arg, tuple(tuple(pair) for pair in ranges))
    if signature in controllers:
        return controllers[signature]
    helper = bpy.data.objects.new("IEDM_Visibility_{}".format(arg), None)
    bpy.context.collection.objects.link(helper)
    helper.empty_display_size = 0.01
    helper["_iedm_visibility_preview_control"] = True
    intervals = sorted(_visibility_frame_intervals([(arg, ranges)]))
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    helper["_iedm_visible"] = float(any(a <= 100 < b for a, b in merged))
    action = bpy.data.actions.new("{}_IEDM_VisibilityPreview".format(arg))
    if hasattr(action, "argument"):
        action.argument = arg
    curve = action_fcurves(action).new(data_path='["_iedm_visible"]')
    for frame, value in _visibility_scene_keys(ranges):
        curve.keyframe_points.add(1)
        key = curve.keyframe_points[-1]
        key.co = (frame, value)
        key.interpolation = "CONSTANT"
    curve.update()
    _assign_action(helper, action)
    controllers[signature] = helper
    return helper

def _add_visibility_hide_driver(obj, path, controls, invert=True):
    """Drive an object's hide state from one or more visibility controllers."""
    driver = obj.driver_add(path).driver
    driver.type = "SCRIPTED"
    for variable in list(driver.variables):
        driver.variables.remove(variable)
    for index, control in enumerate(controls):
        variable = driver.variables.new()
        variable.name = "v{}".format(index)
        variable.type = "SINGLE_PROP"
        variable.targets[0].id = control
        variable.targets[0].data_path = '["_iedm_visible"]'
    expression = " and ".join("v{}".format(i) for i in range(len(controls)))
    driver.expression = "not (" + expression + ")" if invert else expression

def _finalize_render_origins(graph):
    """Change editing origins only after authored transforms are finalized."""
    from .nodes.mesh import _recenter_mesh_object_to_geometry

    transform_paths = {
        "location",
        "rotation_euler",
        "rotation_quaternion",
        "rotation_axis_angle",
        "scale",
        "delta_location",
        "delta_rotation_euler",
        "delta_rotation_quaternion",
        "delta_scale",
    }
    for node in graph.nodes:
        obj = node.blender
        if not getattr(node, "_recenter_render_origin", False) or obj is None:
            continue
        # Preserve children, animated pivots, and unevaluated hidden meshes.
        if obj.children or obj.type != "MESH" or not obj.visible_get():
            continue
        ad = obj.animation_data
        if ad:
            actions = ([ad.action] if ad.action else []) + [
                strip.action
                for track in ad.nla_tracks
                for strip in track.strips
                if strip.action
            ]
            if any(
                fc.data_path in transform_paths
                for action in actions
                for fc in action_fcurves(action)
            ):
                continue
            if any(fc.data_path in transform_paths for fc in ad.drivers):
                continue
        _recenter_mesh_object_to_geometry(obj)
    bpy.context.view_layer.update()
