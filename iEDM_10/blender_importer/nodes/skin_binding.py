"""Resolve skin bind targets, vertex weights, and mesh transforms."""

import struct as struct

import bpy as bpy

from ...edm_format.mathtypes import Matrix, Vector


def _channel_slices_from_vertex_format(vertex_format):
    """Return channel->(start,end) slices based on vertex format packed layout."""
    offsets = {}
    if not vertex_format or not hasattr(vertex_format, "data"):
        return offsets
    cursor = 0
    for i, count in enumerate(vertex_format.data):
        n = int(count)
        if n <= 0:
            continue
        offsets[i] = (cursor, cursor + n)
        cursor += n
    return offsets


def _decode_packed_bone_indices(value):
    """Decode four uint8 bone palette indices packed into a float channel."""
    try:
        packed = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    except Exception:
        return None
    return tuple((packed >> (8 * i)) & 0xFF for i in range(4))


def _mesh_bounds_center_and_extent(mesh_obj):
    try:
        verts = list(getattr(getattr(mesh_obj, "data", None), "vertices", []) or [])
    except Exception:
        return None, 0.0
    if not verts:
        return None, 0.0
    min_v = Vector((float("inf"), float("inf"), float("inf")))
    max_v = Vector((float("-inf"), float("-inf"), float("-inf")))
    for vert in verts:
        co = vert.co
        min_v.x = min(min_v.x, co.x)
        min_v.y = min(min_v.y, co.y)
        min_v.z = min(min_v.z, co.z)
        max_v.x = max(max_v.x, co.x)
        max_v.y = max(max_v.y, co.y)
        max_v.z = max(max_v.z, co.z)
    size = max_v - min_v
    return (min_v + max_v) * 0.5, max(abs(size.x), abs(size.y), abs(size.z))


def _choose_skin_bind_target(
    control_bone,
    weight_bones,
    bone_rest_matrix_by_name,
    mesh_obj,
    skin_node,
    channel_slices,
):
    """Select the bind/rest bone used to localize absolute SkinNode vertices."""
    if not control_bone:
        return "", None

    default_name = control_bone
    default_matrix = bone_rest_matrix_by_name.get(default_name)
    default_loc = (
        default_matrix.to_translation() if default_matrix is not None else None
    )

    center, _extent = _mesh_bounds_center_and_extent(mesh_obj)
    if center is None or default_loc is None:
        return default_name, default_loc

    slice21 = channel_slices.get(21)
    pos_slice = channel_slices.get(0)
    packed_bone_index_offset = None
    if pos_slice is not None and (pos_slice[1] - pos_slice[0]) >= 4:
        packed_bone_index_offset = pos_slice[0] + 3

    weight_sums = _skin_bone_weight_sums(
        skin_node, slice21, packed_bone_index_offset, len(weight_bones)
    )

    best_name = default_name
    best_loc = default_loc
    best_distance = (center - default_loc).length
    default_distance = best_distance

    for bone_index, weight_sum in weight_sums.items():
        if weight_sum <= 0.0:
            continue
        bone_name = weight_bones[bone_index]
        mat = bone_rest_matrix_by_name.get(bone_name)
        if mat is None:
            continue
        loc = mat.to_translation()
        distance = (center - loc).length
        if distance < best_distance:
            best_name = bone_name
            best_loc = loc
            best_distance = distance

    # Most skins use the first palette bone as the wrapper/bind target.  Some head
    # meshes are an exception: the first bone is a control/helper high above the
    # actual weighted head/eye bind cluster.  Only override clear outliers.
    if (
        best_name != default_name
        and default_distance > 1.0
        and best_distance < default_distance * 0.5
    ):
        return best_name, best_loc

    return default_name, default_loc


def _skin_bone_weight_sums(skin_node, weight_slice, packed_index_offset, bone_count):
    """Accumulate vertex weights by palette index, honoring packed indices."""
    totals = {}
    if not weight_slice:
        return totals
    for source in getattr(skin_node, "vertexData", []) or []:
        decoded = None
        if packed_index_offset is not None and packed_index_offset < len(source):
            decoded = _decode_packed_bone_indices(source[packed_index_offset])
            if not decoded or not any(0 <= int(idx) < bone_count for idx in decoded):
                decoded = None
        weights = [float(value) for value in source[weight_slice[0] : weight_slice[1]]]
        for slot, weight in enumerate(weights[:4]):
            if weight <= 1e-6:
                continue
            index = int(decoded[slot]) if decoded and slot < len(decoded) else slot
            if 0 <= index < bone_count:
                totals[index] = totals.get(index, 0.0) + weight
    return totals


def _localize_skin_mesh_to_bind_target(mesh_obj, bind_target_loc):
    if (
        bind_target_loc is None
        or mesh_obj is None
        or getattr(mesh_obj, "type", "") != "MESH"
    ):
        return False
    if bool(mesh_obj.get("_iedm_skin_localized_to_bind")):
        return False
    center, extent = _mesh_bounds_center_and_extent(mesh_obj)
    if center is None:
        return False
    try:
        # SkinNode geometry is in skeleton space, including meshes and bind
        # anchors near the origin. Distance cannot identify a coordinate space.
        # A palette anchor can be far from its geometry; distance is not evidence
        # that vertices are already local.
        for vert in mesh_obj.data.vertices:
            vert.co -= bind_target_loc
        mesh_obj.data.update()
        mesh_obj["_iedm_skin_localized_to_bind"] = True
        return True
    except Exception:
        return False


def _attach_skin_mesh_to_parent(mesh_obj, arm_obj, skin_node, bind_target_loc):
    """Choose a named wrapper when available, otherwise preserve current parent."""
    name = getattr(skin_node, "name", "") or ""
    candidate = _find_skin_parent_candidate(mesh_obj, arm_obj, name, bind_target_loc)
    if (
        candidate is not None
        and candidate not in {mesh_obj, arm_obj}
        and (mesh_obj.parent is None or mesh_obj.parent == arm_obj)
    ):
        mesh_obj.parent = candidate
        mesh_obj.matrix_parent_inverse = Matrix.Identity(4)
        mesh_obj["_iedm_skin_parent_override"] = True
    elif mesh_obj.parent is None:
        mesh_obj.parent = arm_obj
    else:
        mesh_obj.matrix_parent_inverse = Matrix.Identity(4)


def _find_skin_parent_candidate(mesh_obj, arm_obj, name, bind_target_loc):
    """Find the best matching non-mesh wrapper for a skin object."""
    if not name:
        return None
    candidates = []
    for obj in list(getattr(bpy.data, "objects", []) or []):
        if obj in {mesh_obj, arm_obj} or getattr(obj, "type", "") == "MESH":
            continue
        debug_name = str(obj.get("_iedm_dbg_tf_name", "") or "")
        debug_type = str(obj.get("_iedm_dbg_tf_cls", "") or "")
        if debug_name != name and getattr(obj, "name", "") != name:
            continue
        score = _skin_parent_candidate_score(obj, debug_type, name, bind_target_loc)
        candidates.append((score, getattr(obj, "name", ""), obj))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2] if candidates else None


def _skin_parent_candidate_score(obj, debug_type, name, bind_target_loc):
    """Score a potential wrapper by semantic type, exact name and bind proximity."""
    score = {
        "TransformNode": 40,
        "ArgVisibilityNode": 5,
    }.get(debug_type, 20 if debug_type else 0)
    if getattr(obj, "name", "") == name:
        score += 10
    try:
        world_loc = obj.matrix_world.to_translation()
        if bind_target_loc is not None:
            score -= min((world_loc - bind_target_loc).length * 1000.0, 1000.0)
    except Exception:
        pass
    return score


def _assign_skin_vertex_weights(
    mesh_obj, skin_node, control_bone, weight_bones, group_map, channel_slices
):
    nverts = len(mesh_obj.data.vertices)
    if nverts == 0:
        return

    # Skin meshes retain the original vertex pool for 1:1 EDM weight indexing.
    slice21 = channel_slices.get(21)
    pos_slice = channel_slices.get(0)
    packed_bone_index_offset = None
    if pos_slice is not None and (pos_slice[1] - pos_slice[0]) >= 4:
        packed_bone_index_offset = pos_slice[0] + 3

    for vi in range(nverts):
        # Skinned meshes use the original EDM vertex index directly.
        if vi >= len(skin_node.vertexData):
            continue
        src = skin_node.vertexData[vi]
        weights = []
        if slice21:
            weights = [float(x) for x in src[slice21[0] : slice21[1]]]
        bone_indices = None
        if packed_bone_index_offset is not None and packed_bone_index_offset < len(src):
            decoded = _decode_packed_bone_indices(src[packed_bone_index_offset])
            if decoded and any(0 <= int(idx) < len(weight_bones) for idx in decoded):
                bone_indices = decoded

        # Merge repeated packed slots before normalizing actual influences.
        influences = {}
        for bi, weight in enumerate(weights[:4]):
            bone_index = (
                int(bone_indices[bi]) if bone_indices and bi < len(bone_indices) else bi
            )
            in_range = 0 <= bone_index < len(weight_bones)
            name = weight_bones[bone_index] if in_range else None
            if not name or weight <= 0.0:
                continue
            influences[name] = influences.get(name, 0.0) + weight
        total = sum(influences.values())
        # Match the exporter's 0.001 influence filter before normalizing.
        influences = {n: w for n, w in influences.items() if w >= 0.001 * total}
        # Weight missing from 1.0 follows the control bone (palette[0]); the
        # exporter caps influences at four, so only add it when it fits.
        rest = 1.0 - sum(influences.values())
        if rest >= 0.001 and len(influences) < 4:
            influences[control_bone] = influences.get(control_bone, 0.0) + rest
        total = sum(influences.values())
        for name, weight in influences.items():
            group_map[name].add([vi], weight / total, "REPLACE")


def _bake_skin_mesh_object_transforms(mesh_obj):
    """Bake non-identity matrix_basis into vertices so the basis becomes identity.

    The exporter requires skinned meshes to have applied transforms. After
    _bind_skin_object vertex positions are in the correct local space but the
    mesh may still carry a non-identity matrix_basis from apply_node_transform.
    Baking it here satisfies the exporter without disturbing vertex group weights.
    """
    if mesh_obj is None or mesh_obj.type != "MESH" or not mesh_obj.data:
        return
    try:
        mat = mesh_obj.matrix_basis
        is_identity = all(
            abs(float(mat[r][c]) - (1.0 if r == c else 0.0)) < 1e-6
            for r in range(4)
            for c in range(4)
        )
        if is_identity:
            return
        mat = mat.copy()
        for vert in mesh_obj.data.vertices:
            vert.co = mat @ vert.co
        mesh_obj.data.update()
        mesh_obj.matrix_basis = Matrix.Identity(4)
    except Exception as e:
        print(
            "Warning: _bake_skin_mesh_object_transforms failed for "
            f"'{getattr(mesh_obj, 'name', '?')}': {e}"
        )
