"""Rewrite oriented-scale controls and retain geometry helper imports."""

import bpy as bpy

from ..edm_format.mathtypes import Matrix as Matrix
from ..edm_format.mathtypes import MatrixScale as MatrixScale
from ..edm_format.mathtypes import Quaternion as Quaternion
from ..edm_format.mathtypes import Vector as Vector
from ..edm_format.types import ArgAnimationNode as ArgAnimationNode
from ..utils import action_fcurves as action_fcurves
from . import action_curves as _action_curves_compat
from . import orient_scale_geometry as _orient_scale_geometry_compat
from .action_curves import _build_arganimation_action as _build_arganimation_action
from .action_curves import _clone_action_filtered as _clone_action_filtered
from .action_values import (
    _compose_oriented_scale_matrix as _compose_oriented_scale_matrix,
)
from .anim_actions import (
    _clear_object_animation_tracks as _clear_object_animation_tracks,
)
from .animation import _quat_is_identity as _quat_is_identity
from .animation import add_scale_fcurves as add_scale_fcurves
from .import_context import _ROOT_BASIS_FIX as _ROOT_BASIS_FIX
from .import_context import _import_ctx as _import_ctx
from .import_context import _log as _log
from .node_identity import _assign_action as _assign_action
from .node_transform import (
    _transform_uses_quaternion_rotation as _transform_uses_quaternion_rotation,
)
from .orient_scale_candidates import (
    _oriented_scale_rewrite_inputs as _oriented_scale_rewrite_inputs,
)
from .orient_scale_geometry import (
    _arganimation_key_rotation_basis_local as _arganimation_key_rotation_basis_local,
)
from .orient_scale_geometry import (
    _arganimation_prerotation_basis_local as _arganimation_prerotation_basis_local,
)
from .orient_scale_geometry import (
    _arganimation_rotation_basis_local as _arganimation_rotation_basis_local,
)
from .orient_scale_geometry import (
    _insert_parent_wrapper_object as _insert_parent_wrapper_object,
)
from .orient_scale_geometry import (
    _matrix_transform_roundtrip_error as _matrix_transform_roundtrip_error,
)
from .orient_scale_geometry import (
    _needs_prerotation_affine_split as _needs_prerotation_affine_split,
)
from .orient_scale_geometry import (
    _object_collection_targets as _object_collection_targets,
)
from .orient_scale_geometry import (
    _promote_oriented_scale_top_name as _promote_oriented_scale_top_name,
)
from .orient_scale_geometry import (
    _reconstruct_edm_local_matrix as _reconstruct_edm_local_matrix,
)
from .orient_scale_geometry import (
    _render_local_matrix_for_graph_node as _render_local_matrix_for_graph_node,
)
from .visibility_graph import _is_child_of_file_root as _is_child_of_file_root
from .visibility_graph import (
    _is_top_level_visibility_authored_pair as _is_top_level_visibility_authored_pair,
)

_create_scale_orientation_rotation_action = (
    _action_curves_compat._create_scale_orientation_rotation_action
)

_scale_orientation_source_for_graph_node = (
    _orient_scale_geometry_compat._scale_orientation_source_for_graph_node
)


def _finish_oriented_scale_rewrite(
    node, source_tf, ob, top_wrapper, render_local, leaf_vis_action
):
    _clear_object_animation_tracks(ob)
    if leaf_vis_action is not None:
        _assign_action(ob, leaf_vis_action)
    ob.matrix_basis = render_local
    ob["_iedm_oriented_scale_leaf"] = True
    _promote_oriented_scale_top_name(node, top_wrapper, ob)
    source_tf._iedm_oriented_scale_wrapped = True
    source_tf._iedm_oriented_scale_wrapped = True


def _rewrite_oriented_scale_controls(graph):
    bone_ctx = _import_ctx.bone_import_ctx or {}
    bone_anim_source_nodes = bone_ctx.get("bone_anim_source_nodes", set())
    bone_anim_source_transforms = bone_ctx.get("bone_anim_source_transforms", set())
    for node in getattr(graph, "nodes", []) or []:
        inputs = _oriented_scale_rewrite_inputs(
            node,
            bone_anim_source_nodes,
            bone_anim_source_transforms,
            _scale_orientation_source_for_graph_node,
        )
        if inputs is None:
            continue
        (
            ob,
            source_tf,
            active_action,
            action_arg,
            scale_arg,
            keys4,
            keys3,
            has_anim_scale,
            has_anim_orient,
            q2_raw,
            base_scale_vec,
            has_base_scale,
        ) = inputs
        zero_transform = getattr(source_tf, "zero_transform_local_matrix", None)
        if zero_transform is None:
            continue

        static_scale_mat = (
            _compose_oriented_scale_matrix(base_scale_vec, q2_raw)
            if has_base_scale
            else Matrix.Identity(4)
        )
        try:
            top_local = Matrix(zero_transform) @ static_scale_mat.inverted()
        except Exception:
            continue
        prerotation_parent_local = None
        top_wrapper_local = top_local
        prerotation_basis_local = _arganimation_prerotation_basis_local(source_tf)
        split_rotation_basis_local = _arganimation_key_rotation_basis_local(source_tf)
        rotation_basis_local = _arganimation_rotation_basis_local(source_tf)
        if _needs_prerotation_affine_split(
            top_local, prerotation_basis_local, split_rotation_basis_local
        ):
            prerotation_parent_local = prerotation_basis_local
            top_wrapper_local = split_rotation_basis_local
            rotation_basis_local = split_rotation_basis_local

        frame_mapper = None  # default EDM argument -> scene frame mapping

        top_action = None
        if active_action is not None and action_arg is not None:
            try:
                top_action = _build_arganimation_action(
                    source_tf,
                    int(action_arg),
                    top_wrapper_local,
                    frame_mapper=frame_mapper,
                    include_scale=False,
                    action_name="{}_iedm_tf".format(active_action.name),
                    rotation_basis_local=rotation_basis_local,
                    position_prefix_lifted=prerotation_parent_local is not None,
                )
            except Exception:
                top_action = None
        if top_action is None and active_action is not None:
            top_action = _clone_action_filtered(
                active_action,
                "_iedm_tf",
                exclude_paths={"scale", "VISIBLE", "hide_viewport"},
            )
        leaf_vis_action = (
            _clone_action_filtered(
                active_action,
                "_iedm_vis",
                include_paths={"VISIBLE", "hide_viewport"},
            )
            if active_action is not None
            else None
        )
        render_local = _render_local_matrix_for_graph_node(node)

        top_wrapper = _insert_parent_wrapper_object(
            ob, "{}_iedm_tf".format(ob.name), top_wrapper_local, reset_child_local=False
        )
        top_wrapper["_iedm_oriented_scale_helper"] = True
        top_wrapper["_iedm_oriented_scale_top"] = True
        if prerotation_parent_local is not None:
            prerotation_parent = _insert_parent_wrapper_object(
                top_wrapper,
                "{}_iedm_tprefix".format(ob.name),
                prerotation_parent_local,
                reset_child_local=False,
            )
            prerotation_parent["_iedm_oriented_scale_helper"] = True
            prerotation_parent["_iedm_oriented_scale_top_prefix"] = True
        if top_action is not None:
            _assign_action(top_wrapper, top_action)
            top_wrapper.rotation_mode = (
                "QUATERNION"
                if _transform_uses_quaternion_rotation(source_tf, top_wrapper)
                else "XYZ"
            )

        child = ob
        if has_base_scale:
            if not _quat_is_identity(q2_raw):
                post_base = _insert_parent_wrapper_object(
                    child,
                    "{}_iedm_sbase_post".format(ob.name),
                    q2_raw.conjugated().to_matrix().to_4x4(),
                    reset_child_local=False,
                )
                post_base.rotation_mode = "QUATERNION"
                post_base["_iedm_oriented_scale_helper"] = True
                child = post_base
            base_scale = _insert_parent_wrapper_object(
                child,
                "{}_iedm_sbase".format(ob.name),
                MatrixScale(base_scale_vec),
                reset_child_local=False,
            )
            base_scale["_iedm_oriented_scale_helper"] = True
            child = base_scale
            if not _quat_is_identity(q2_raw):
                pre_base = _insert_parent_wrapper_object(
                    child,
                    "{}_iedm_sbase_pre".format(ob.name),
                    q2_raw.to_matrix().to_4x4(),
                    reset_child_local=False,
                )
                pre_base.rotation_mode = "QUATERNION"
                pre_base["_iedm_oriented_scale_helper"] = True
                child = pre_base

        child = _create_animated_scale_chain(
            child, ob, scale_arg, keys3, keys4, has_anim_scale, has_anim_orient
        )

        _finish_oriented_scale_rewrite(
            node, source_tf, ob, top_wrapper, render_local, leaf_vis_action
        )


def _create_animated_scale_chain(
    child, obj, scale_arg, scale_keys, orientation_keys, has_scale, has_orientation
):
    """Wrap an object in the authored animated scale/orientation action chain."""
    if not has_scale:
        return child
    argument_prefix = "{}_".format(int(scale_arg)) if scale_arg is not None else ""
    if has_orientation:
        post_anim = _insert_parent_wrapper_object(
            child,
            "{}_iedm_sanim_post".format(obj.name),
            Matrix.Identity(4),
            reset_child_local=False,
        )
        post_anim.rotation_mode = "QUATERNION"
        post_anim["_iedm_oriented_scale_helper"] = True
        post_action = _create_scale_orientation_rotation_action(
            "{}{}_iedm_sanim_post".format(argument_prefix, obj.name),
            orientation_keys,
            frame_mapper=None,
            invert=True,
        )
        _assign_oriented_scale_action(post_anim, post_action, scale_arg)
        child = post_anim

    anim_scale = _insert_parent_wrapper_object(
        child,
        "{}_iedm_sanim".format(obj.name),
        Matrix.Identity(4),
        reset_child_local=False,
    )
    anim_scale["_iedm_oriented_scale_helper"] = True
    anim_scale_action = bpy.data.actions.new(
        "{}{}_iedm_sanim".format(argument_prefix, obj.name)
    )
    if scale_arg is not None and hasattr(anim_scale_action, "argument"):
        anim_scale_action.argument = int(scale_arg)
    add_scale_fcurves(anim_scale_action, scale_keys, frame_mapper=None, base_scale=None)
    if len(action_fcurves(anim_scale_action)):
        _assign_action(anim_scale, anim_scale_action)
    else:
        try:
            bpy.data.actions.remove(anim_scale_action)
        except Exception as exc:
            _log.debug("Optional operation failed: {}".format(exc), level=2)
    child = anim_scale

    if has_orientation:
        pre_anim = _insert_parent_wrapper_object(
            child,
            "{}_iedm_sanim_pre".format(obj.name),
            Matrix.Identity(4),
            reset_child_local=False,
        )
        pre_anim.rotation_mode = "QUATERNION"
        pre_anim["_iedm_oriented_scale_helper"] = True
        pre_action = _create_scale_orientation_rotation_action(
            "{}{}_iedm_sanim_pre".format(argument_prefix, obj.name),
            orientation_keys,
            frame_mapper=None,
            invert=False,
        )
        _assign_oriented_scale_action(pre_anim, pre_action, scale_arg)
        child = pre_anim
    return child


def _assign_oriented_scale_action(obj, action, scale_arg):
    if action is None:
        return
    if scale_arg is not None and hasattr(action, "argument"):
        action.argument = int(scale_arg)
    _assign_action(obj, action)
