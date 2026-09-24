"""Compute bone rest frames and edit-bone geometry."""

from ...edm_format.mathtypes import Matrix, Vector
from ...edm_format.types import ArgAnimatedBone, Bone
from ..import_context import _ROOT_BASIS_FIX
from ..import_logging import _log_bone_debug_event, _matrix_trs_summary


def _bone_bind_matrix(tfnode):
    """Extract the bone's own bind-pose matrix from EDM Bone or ArgAnimatedBone."""
    if isinstance(tfnode, Bone) and hasattr(tfnode, "bone_matrix"):
        return Matrix(tfnode.bone_matrix)
    if isinstance(tfnode, ArgAnimatedBone) and hasattr(tfnode, "inv_base_bone_matrix"):
        return Matrix(tfnode.inv_base_bone_matrix)
    return None


def _effective_root_basis_fix():
    """Return the basis fix matrix for bone bind-matrix conversion (Y-up → Z-up).

    Always returns plain _ROOT_BASIS_FIX.  When a Bonetransform prefix M1 is
    present the root object carries inv(M1), so the effective world chain is
    inv(M1)@M1@M2(=RBF) = RBF — bone rests stay in the same RBF space.
    """
    return _ROOT_BASIS_FIX


def _bone_rest_matrix_for_node(node, apply_root_fix):
    """Compute a bone's full rest matrix in Blender/armature space.

    The exporter writes mat_inv = pbone.matrix.inverted() as the bind matrix:
      - Bone:            bone_matrix = mat_inv  (in addition to Bone.matrix)
      - ArgAnimatedBone: inv_base_bone_matrix = mat_inv  (separate field at node+488)
    Inverting gives pbone.matrix - the exact armature-space rest matrix needed for
    edit-bone placement.

    For Blender-exported EDMs the bind matrix is in Blender Z-up space.
    For 3ds Max-exported EDMs we apply _ROOT_BASIS_FIX to convert Y-up to Z-up,
    controlled by the bone_rest_requires_root_basis_fix profile flag.
    """
    tf = node.transform
    basis_fix = _effective_root_basis_fix()

    if isinstance(tf, Bone) and not isinstance(tf, ArgAnimatedBone):
        if hasattr(tf, "bone_matrix"):
            inv_bind = Matrix(tf.bone_matrix)
            if not inv_bind.is_identity:
                try:
                    rest = inv_bind.inverted()
                    if apply_root_fix:
                        rest = basis_fix @ rest
                    return rest
                except ValueError:
                    pass
        world_mat = getattr(node, "_world_bl", None)
        if world_mat is None:
            world_mat = getattr(node, "_local_bl", Matrix.Identity(4))
        return world_mat

    if isinstance(tf, ArgAnimatedBone) and hasattr(tf, "inv_base_bone_matrix"):
        bone_bind = Matrix(tf.inv_base_bone_matrix)
        if not bone_bind.is_identity:
            try:
                rest = bone_bind.inverted()
                if apply_root_fix:
                    rest = basis_fix @ rest
                return rest
            except ValueError:
                pass
        world_mat = getattr(node, "_world_bl", None)
        if world_mat is None:
            world_mat = getattr(node, "_local_bl", Matrix.Identity(4))
        return world_mat

    world_mat = getattr(node, "_world_bl", None)
    if world_mat is None:
        world_mat = getattr(node, "_local_bl", Matrix.Identity(4))
    return world_mat


def _unique_bone_name(base, used_names):
    name = base or "Bone"
    if name not in used_names:
        used_names.add(name)
        return name
    i = 1
    while True:
        candidate = "{}.{:03d}".format(name, i)
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        i += 1


def _debug_bone_bind_matrix_summary(nodes, apply_bone_root_fix):
    """Report which source bones carry explicit bind matrices."""
    found = missing = 0
    for node in nodes:
        tf = node.transform
        bind = _bone_bind_matrix(tf)
        tf_type = type(tf).__name__
        tf_name = getattr(tf, "name", "?")
        if bind is not None:
            found += 1
            if not bind.is_identity:
                rest = _bone_rest_matrix_for_node(node, apply_bone_root_fix)
                print(
                    "  [bone-bind] {} '{}' -> non-identity bind matrix, "
                    "rest translation=({:.3f},{:.3f},{:.3f})".format(
                        tf_type, tf_name, rest[0][3], rest[1][3], rest[2][3]
                    )
                )
        else:
            missing += 1
            print(
                "  [bone-bind] {} '{}' -> NO bind matrix (type={}, "
                "has_bone_matrix={}, has_inv_base_bone_matrix={})".format(
                    tf_type,
                    tf_name,
                    tf_type,
                    hasattr(tf, "bone_matrix"),
                    hasattr(tf, "inv_base_bone_matrix"),
                )
            )
    print("Info: Bone bind matrices: {} found, {} missing".format(found, missing))


def _create_edit_bone(node, edit_bones, node_to_bone_name, bone_nodes, apply_root_fix):
    """Set one edit bone's rest head, tail, roll and debug metadata."""
    bone_name = node_to_bone_name[node]
    bone = edit_bones[bone_name]
    rest = _bone_rest_matrix_for_node(node, apply_root_fix)
    bind = _bone_bind_matrix(node.transform)
    source_name = getattr(node.transform, "name", "") or type(node.transform).__name__
    head = rest.to_translation()
    rotation = rest.to_3x3()
    y_axis = rotation @ Vector((0.0, 1.0, 0.0))
    z_axis = rotation @ Vector((0.0, 0.0, 1.0))
    if y_axis.length < 1e-8:
        y_axis = Vector((0.0, 0.01, 0.0))
    y_axis.normalize()

    length = _edit_bone_length(node, bone_nodes, head, apply_root_fix)
    bone.head = head
    bone.tail = head + y_axis * max(length, 0.01)
    if z_axis.length > 1e-8:
        bone.align_roll(z_axis)
    parent_name = node_to_bone_name.get(getattr(node, "parent", None))
    source_type = type(node.transform).__name__
    _log_bone_debug_event(
        "bone-bind-rest",
        {
            "bone_name": bone_name,
            "source_name": source_name,
            "source_type": source_type,
            "bind_matrix": _matrix_trs_summary(bind) if bind is not None else None,
            "rest_matrix": _matrix_trs_summary(rest),
            "head_src": [round(float(v), 6) for v in head],
            "y_axis_src": [round(float(v), 6) for v in y_axis],
            "z_axis_src": [round(float(v), 6) for v in z_axis],
            "derived_length": round(float(max(length, 0.01)), 6),
            "parent_bone": parent_name,
        },
        bone_name,
        source_name,
    )
    _log_bone_debug_event(
        "bone-rest",
        {
            "bone_name": bone_name,
            "source_name": source_name,
            "source_type": source_type,
            "rest_matrix": _matrix_trs_summary(rest),
            "head": [round(float(v), 6) for v in bone.head],
            "tail": [round(float(v), 6) for v in bone.tail],
            "parent_bone": parent_name,
        },
        bone_name,
        source_name,
    )


def _edit_bone_length(node, bone_nodes, head, apply_root_fix):
    """Use the first non-coincident child bone to derive a stable length."""
    for child in node.children:
        if child not in bone_nodes:
            continue
        child_rest = _bone_rest_matrix_for_node(child, apply_root_fix)
        distance = (child_rest.to_translation() - head).length
        if distance > 1e-5:
            return distance
    return 0.05
