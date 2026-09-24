"""Run import postprocessing for controls, skin transforms, and export."""

import bpy as bpy

from ..edm_format.mathtypes import Matrix
from ..utils import action_fcurves
from .ctrl_splits import (
    _rename_control_wrapper_mesh_pairs,
    _split_multi_arg_rotation_controls,
)
from .graph_collections import _assign_collections
from .graph_diagnostics import _debug_filter_terms, _debug_fmt_rot_deg, _debug_fmt_vec3
from .graph_postprocess import (
    _apply_root_visibility_pair_wrapper_basis_fix,
    _apply_static_root_visibility_wrapper_basis_fix,
    _zero_render_child_mesh_locals_under_transform,
)
from .import_context import _import_ctx, _log
from .nodes.core import _process_lod_post_children
from .nodes.diagnostics import _print_import_diagnostics
from .orient_scale import _rewrite_oriented_scale_controls
from .session_visibility import _propagate_visibility_hide_to_render_nodes
from .skin_rewrites import _resolve_skin_parent_overrides_by_bind_rest
from .skin_visibility import (
    _fix_inverse_scaled_visibility_rest_offset,
    _restore_skin_visibility_transform_basis,
)
from .vis_rewrites import (
    _apply_argvis_chain_basis_fix,
    _apply_plain_root_visibility_basis_fix,
    _apply_visibility_pair_wrapper_object_basis_fix,
    _split_multi_arg_visibility_controls,
)


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
                _log.debug("Optional operation failed: {}".format(exc), level=2)
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
    from .export_hold_keys import protect_hold_keys
    from .export_hooks import install_export_hook
    from .export_matrices import evaluate_hidden_objects
    from .export_visibility import place_export_visibility
    from .exporter_frames import undo_exporter_frame_rotations
    from .node_transform import restore_sheared_frames
    from .skin_space import bake_skins_to_armature_space

    undo_exporter_frame_rotations()
    build_bone_control_graph(graph)
    place_export_visibility(graph)
    bake_skins_to_armature_space()
    restore_sheared_frames()
    protect_hold_keys()
    evaluate_hidden_objects()
    install_export_hook()


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
