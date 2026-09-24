"""Export imported skins in the rest space Blender deforms them from.

The exporter writes each bone's inverse bind as the inverse of its pose at
the export frame, and skinned vertices from the mesh deformed at that frame.
That is exact only when every bone's export-frame pose is a rigid move of its
rest. Source files whose bind differs from the arg-0 pose (Bones.edm, AH-6J
gunners, C-130J DAM_ bones) come back with multi-weight vertices displaced
whenever an arg moves.

Blender deforms a skin as pose @ inverse(rest) @ vertex, and import places
rest and vertices so that this reproduces the source; binds with scale that
an edit bone cannot hold are absorbed into the mesh data. During an EDM
export, imported rigs therefore get inverse(rest) as each bone's inverse
bind, and meshes deformed only by imported rigs are written undeformed, so
DCS computes exactly what Blender shows.
"""

import sys

RIG_PROP = "_iedm_rig"


def mark_imported_rig(armature_data):
    armature_data[RIG_PROP] = True


def _is_imported_rig(armature):
    return bool(getattr(armature, "data", None) and armature.data.get(RIG_PROP))


def _patched_update_matrices(original):
    def update_matrices(self):
        original(self)
        if _is_imported_rig(self.armature):
            self.mat_inv = self.bone.matrix_local.inverted()

    update_matrices._iedm_original = original
    return update_matrices


def _patched_get_mesh(original):
    def get_mesh(obj):
        modifiers = list(getattr(obj, "modifiers", ()))
        if (
            modifiers
            and all(
                m.type == "ARMATURE" and _is_imported_rig(m.object) for m in modifiers
            )
            and obj.type == "MESH"
            and not obj.data.is_editmode
        ):
            return obj.data  # rest space, matching inverse(rest) above
        return original(obj)

    get_mesh._iedm_original = original
    return get_mesh


def install_skin_bind():
    """Swap in rest-space bone matrices and skin meshes; return restore list."""
    armature = sys.modules.get("export_armature")
    builder = sys.modules.get("mesh_builder")
    bone_node = getattr(armature, "BoneNode", None)
    if bone_node is None or builder is None:
        return []
    original_update = bone_node.update_matrices
    original_get_mesh = builder.get_mesh
    if hasattr(original_update, "_iedm_original"):
        return []
    bone_node.update_matrices = _patched_update_matrices(original_update)
    builder.get_mesh = _patched_get_mesh(original_get_mesh)
    return [
        (bone_node, "update_matrices", original_update),
        (builder, "get_mesh", original_get_mesh),
    ]
