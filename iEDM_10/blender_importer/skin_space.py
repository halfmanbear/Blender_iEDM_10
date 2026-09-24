"""Put skinned mesh vertices in armature space for the exporter.

The exporter writes skin vertices in the mesh object's local space and puts
the object transform only on the skin's fake control bone, which DCS does not
use to deform weighted vertices. Any transform between the armature and the
skinned mesh is therefore baked into the mesh data.
"""

import bpy
from mathutils import Matrix

_TOLERANCE = 1e-6


def _relative_to(obj, ancestor):
    """Object-to-ancestor matrix from local data (world matrices may be stale)."""
    matrix = obj.matrix_parent_inverse @ obj.matrix_basis
    parent = obj.parent
    while parent is not None and parent is not ancestor:
        matrix = parent.matrix_parent_inverse @ parent.matrix_basis @ matrix
        parent = parent.parent
    return matrix if parent is ancestor else None


def _is_identity(matrix):
    return all(
        abs(matrix[r][c] - (1.0 if r == c else 0.0)) < _TOLERANCE
        for r in range(4)
        for c in range(4)
    )


def _bake(obj, matrix):
    mesh = obj.data
    if mesh.users > 1:
        mesh = mesh.copy()
        obj.data = mesh
    # Vertex order is kept, so world-space winding (what DCS culls on) is
    # unchanged even when the baked matrix mirrors.
    mesh.transform(matrix)
    mesh.update()
    obj.matrix_basis = Matrix.Identity(4)
    obj.matrix_parent_inverse = Matrix.Identity(4)


def bake_skins_to_armature_space():
    """Bake each skinned mesh's armature-relative transform into its vertices."""
    baked = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        rigs = [m.object for m in obj.modifiers if m.type == "ARMATURE" and m.object]
        if not rigs:
            continue
        matrix = _relative_to(obj, rigs[0])
        if matrix is None or _is_identity(matrix):
            continue
        # Only identity helpers may remain between rig and mesh after the bake.
        helper = obj.parent
        if helper is not rigs[0] and _relative_to(helper, rigs[0]) is not None:
            if not _is_identity(_relative_to(helper, rigs[0])):
                continue
        _bake(obj, matrix)
        baked += 1
    if baked:
        # Refresh matrix_world; later readers (and the exporter) use it.
        bpy.context.view_layer.update()
    return baked
