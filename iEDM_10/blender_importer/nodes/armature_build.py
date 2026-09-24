"""Create armatures, build edit bones, and finalize the import context."""

import bpy as bpy

from ...edm_format.mathtypes import Matrix
from ..export_skin_bind import mark_imported_rig
from ..import_context import _ROOT_BASIS_FIX, _import_ctx, _import_profile_flag, _log
from ..import_logging import _log_bone_debug_event, _matrix_trs_summary
from ..node_identity import _transform_display_name
from .bone_rest import (
    _bone_rest_matrix_for_node,
    _create_edit_bone,
    _debug_bone_bind_matrix_summary,
    _unique_bone_name,
)


def _create_armature_object(bone_nodes, parent_obj):
    """Create and parent the armature object; compute basis-fix flags.

    Returns (arm_obj, arm_data, apply_bone_root_fix, arm_carries_basis_fix).
    """
    arm_name = "iEDM_Armature"
    if bpy.data.objects.get(arm_name) is not None:
        i = 1
        while bpy.data.objects.get("{}.{:03d}".format(arm_name, i)) is not None:
            i += 1
        arm_name = "{}.{:03d}".format(arm_name, i)

    arm_data = bpy.data.armatures.new(arm_name)
    arm_obj = bpy.data.objects.new(arm_name, arm_data)
    bpy.context.collection.objects.link(arm_obj)

    # Determine Y-up to Z-up basis correction strategy.
    #
    # scene-root v10 (v10_root_object_basis_fix=True):
    #   Parent carries _ROOT_BASIS_FIX; armature cancels it so arm.world ~= Identity.
    #   Bone rests use Blender Z-up (apply_bone_root_fix=True, arm_carries=False).
    #
    # BLOB_RENDER (v10_root_object_basis_fix=False):
    #   Parent does NOT carry _ROOT_BASIS_FIX. Armature carries it directly.
    #   Skinned meshes get _ROOT_BASIS_FIX baked into matrix_basis by core.py.
    #   (apply_bone_root_fix=False, arm_carries=True)
    _profile_needs_root_fix = _import_profile_flag("bone_rest_requires_root_basis_fix")
    _parent_carries_root_fix = (parent_obj is not None) and _import_profile_flag(
        "v10_root_object_basis_fix"
    )
    arm_carries_basis_fix = _profile_needs_root_fix and not _parent_carries_root_fix
    apply_bone_root_fix = _profile_needs_root_fix and _parent_carries_root_fix

    if parent_obj is not None:
        arm_obj.parent = parent_obj
        arm_obj.matrix_parent_inverse = Matrix.Identity(4)
        if arm_carries_basis_fix:
            try:
                arm_obj.matrix_basis = (
                    parent_obj.matrix_basis.inverted() @ _ROOT_BASIS_FIX
                )
            except Exception:
                arm_obj.matrix_basis = _ROOT_BASIS_FIX
        else:
            try:
                arm_obj.matrix_basis = parent_obj.matrix_basis.inverted()
            except Exception:
                arm_obj.matrix_basis = Matrix.Identity(4)
    elif arm_carries_basis_fix:
        arm_obj.matrix_basis = _ROOT_BASIS_FIX

    _log_bone_debug_event(
        "armature-parent",
        {
            "armature": arm_name,
            "parent": getattr(parent_obj, "name", None)
            if parent_obj is not None
            else None,
            "armature_basis": _matrix_trs_summary(arm_obj.matrix_basis),
            "parent_basis": _matrix_trs_summary(
                getattr(parent_obj, "matrix_basis", None)
            )
            if parent_obj is not None
            else None,
        },
        arm_name,
        getattr(parent_obj, "name", None) if parent_obj is not None else None,
    )

    return arm_obj, arm_data, apply_bone_root_fix, arm_carries_basis_fix


def _build_edit_bones(arm_obj, arm_data, bone_nodes, apply_bone_root_fix):
    """Enter Blender edit mode and create bones from EDM bind/rest matrices.

    Returns node_to_bone_name mapping (TranslationNode -> bone name string).
    """
    view_layer = bpy.context.view_layer
    prev_active = view_layer.objects.active
    node_to_bone_name = {}
    try:
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        arm_obj.select_set(True)
        view_layer.objects.active = arm_obj
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.mode_set(mode="EDIT")

        edit_bones = arm_data.edit_bones
        mark_imported_rig(arm_data)
        used_names = set()
        sorted_nodes = sorted(
            bone_nodes,
            key=lambda n: getattr(n.transform, "_graph_idx", 1 << 30),
        )

        for node in sorted_nodes:
            bone_name = _unique_bone_name(
                _transform_display_name(node.transform), used_names
            )
            edit_bones.new(bone_name)
            node_to_bone_name[node] = bone_name

        _debug_bone_bind_matrix_summary(sorted_nodes, apply_bone_root_fix)

        bone_node_set = set(sorted_nodes)
        for node in sorted_nodes:
            _create_edit_bone(
                node,
                edit_bones,
                node_to_bone_name,
                bone_node_set,
                apply_bone_root_fix,
            )

        for node in sorted_nodes:
            parent = node.parent
            while parent is not None and parent not in node_to_bone_name:
                parent = parent.parent
            if parent is None:
                continue
            eb = edit_bones[node_to_bone_name[node]]
            eb.parent = edit_bones[node_to_bone_name[parent]]

        for node in sorted_nodes:
            bone_name = node_to_bone_name[node]
            tf_name = (
                getattr(node.transform, "name", "") or type(node.transform).__name__
            )
            eb = edit_bones[bone_name]
            parent_matrix = eb.parent.matrix.copy() if eb.parent else None
            matrix_local = eb.matrix.copy()
            if parent_matrix is not None:
                try:
                    matrix_local = parent_matrix.inverted() @ eb.matrix
                except Exception:
                    matrix_local = eb.matrix.copy()
            _log_bone_debug_event(
                "edit-bone-final",
                {
                    "bone_name": bone_name,
                    "source_name": tf_name,
                    "source_type": type(node.transform).__name__,
                    "parent_bone": eb.parent.name if eb.parent else None,
                    "matrix": _matrix_trs_summary(eb.matrix),
                    "matrix_local": _matrix_trs_summary(matrix_local),
                    "head": [round(float(v), 6) for v in eb.head],
                    "tail": [round(float(v), 6) for v in eb.tail],
                },
                bone_name,
                tf_name,
            )
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception as e:
            print(f"Warning in blender_importer/nodes/armature.py: {e}")
        if prev_active is not None:
            view_layer.objects.active = prev_active

    return node_to_bone_name


def _finalize_bone_import_ctx(
    arm_obj,
    node_to_bone_name,
    bone_chain_nodes,
    arm_carries_basis_fix,
    graph,
    apply_bone_root_fix,
):
    """Build import context, retarget bone actions, and store on _import_ctx."""
    bone_name_by_transform = {}
    bone_rest_matrix_by_name = {}
    for tnode, bname in node_to_bone_name.items():
        if tnode.transform is not None:
            bone_name_by_transform[tnode.transform] = bname
        try:
            bone_rest_matrix_by_name[bname] = _bone_rest_matrix_for_node(
                tnode, apply_bone_root_fix
            ).copy()
        except Exception as exc:
            _log.debug("Optional operation failed: {}".format(exc), level=2)

    _import_ctx.bone_import_ctx = {
        "armature": arm_obj,
        "bone_name_by_node": node_to_bone_name,
        "bone_name_by_transform": bone_name_by_transform,
        "bone_rest_matrix_by_name": bone_rest_matrix_by_name,
        "bone_chain_nodes": bone_chain_nodes,
        "bone_anim_source_nodes": set(),
        "bone_anim_source_transforms": set(),
        "arm_carries_basis_fix": arm_carries_basis_fix,
    }

    arm_obj.data.pose_position = "REST"


def _apply_armature_transforms(arm_obj):
    """Apply the armature's object-level transforms into its bone rest positions."""
    if arm_obj is None:
        return
    try:
        if arm_obj.matrix_basis == Matrix.Identity(4):
            return
    except Exception as exc:
        _log.debug("Optional operation failed: {}".format(exc), level=2)
    view_layer = bpy.context.view_layer
    prev_active = view_layer.objects.active
    try:
        for o in list(bpy.context.selected_objects):
            o.select_set(False)
        arm_obj.select_set(True)
        view_layer.objects.active = arm_obj
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    except Exception as e:
        _log.warn(
            "transform_apply on armature '{}': {}".format(
                getattr(arm_obj, "name", "?"), e
            ),
            exc=e,
        )
    finally:
        try:
            arm_obj.select_set(False)
        except Exception as exc:
            _log.debug("Optional operation failed: {}".format(exc), level=2)
        if prev_active is not None:
            view_layer.objects.active = prev_active
