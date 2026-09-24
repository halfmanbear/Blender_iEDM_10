"""Create the graph root and scene boxes with the required coordinate basis."""

import bpy as bpy

from ..edm_format.mathtypes import Matrix
from ..edm_format.types import AnimatingNode, LodNode, TransformNode
from .bbox_utils import (
    _create_light_box_from_root,
    _create_user_box_from_root,
    _has_special_box,
    create_bounding_box_from_root,
)
from .import_context import _ROOT_BASIS_FIX, _import_ctx, _import_profile_flag
from .node_transform import apply_node_transform


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
        # Header boxes are in final (Y-up) EDM space and the exporter always
        # converts Blender's Z-up world back, with or without a root basis
        # object (f-117 has none; its boxes came back with y and z swapped).
        bbox_coord_fix = _ROOT_BASIS_FIX
        if box_options.bounding_box and not _has_special_box("BOUNDING_BOX"):
            create_bounding_box_from_root(edm_root, coord_fix=bbox_coord_fix)
        if box_options.user_box:
            _create_user_box_from_root(edm_root, coord_fix=bbox_coord_fix)
        if box_options.light_box:
            _create_light_box_from_root(edm_root, coord_fix=bbox_coord_fix)
