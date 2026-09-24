import struct

import bpy

from ...edm_format.mathtypes import (
    Matrix,
    Quaternion,
    Vector,
)
from ...edm_format.types import (
    AnimatingNode,
    ArgAnimatedBone,
    ArgVisibilityNode,
    Bone,
)
from ...utils import action_fcurves, new_grouped_fcurve
from ..anim_actions import get_actions_for_node
from ..prelude import (
    _ROOT_BASIS_FIX,
    _import_ctx,
    _import_profile_flag,
    _is_bone_transform,
    _log,
    _log_bone_debug_event,
    _matrix_trs_summary,
    _transform_display_name,
)


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
    control_bone, weight_bones, bone_rest_matrix_by_name, mesh_obj, skin_node,
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

    # Most skins use the first palette bone as the wrapper/bind target.  A-10's
    # head mesh is an exception: the first bone is a control/helper high above the
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
        # that vertices are already local (for example Su-27 Object10934817).
        for vert in mesh_obj.data.vertices:
            vert.co -= bind_target_loc
        mesh_obj.data.update()
        mesh_obj["_iedm_skin_localized_to_bind"] = True
        return True
    except Exception:
        return False

def _copy_fcurve_to_action(src_curve, dst_action, dst_path, action_group):
    dst_curve = new_grouped_fcurve(
        dst_action,
        data_path=dst_path,
        index=src_curve.array_index,
        action_group=action_group,
    )
    dst_curve.extrapolation = src_curve.extrapolation
    for key in src_curve.keyframe_points:
        new_key = dst_curve.keyframe_points.insert(
            key.co[0], key.co[1], options={"FAST"}
        )
        new_key.interpolation = key.interpolation
        try:
            new_key.handle_left_type = key.handle_left_type
            new_key.handle_right_type = key.handle_right_type
            new_key.handle_left = key.handle_left
            new_key.handle_right = key.handle_right
        except Exception as e:
            print(f"Warning in blender_importer/nodes/armature.py: {e}")
    dst_curve.update()

def _action_has_fcurve(action, data_path, index=None):
    if action is None:
        return False
    for fcu in action_fcurves(action):
        if fcu.data_path != data_path:
            continue
        if index is None or fcu.array_index == index:
            return True
    return False

def _copy_bone_rotation_curves(src_action, dst_action, bone_name, rest_quat_inv):
    """Copy rotation_quaternion curves from world to bone-local space.

    Object-level actions bake leftRot (the bone's world-space rest rotation) into
    every keyframe so that rotation_quaternion=leftRot at the rest frame. Pose bone
    rotation_quaternion is relative to the edit-bone orientation, so the rest frame
    must be identity. This removes leftRot by premultiplying each keyframe quaternion
    by inv(rest_rot), giving rotation_quaternion=identity at rest.
    """
    src_curves = {
        fcu.array_index: fcu
        for fcu in action_fcurves(src_action)
        if fcu.data_path == "rotation_quaternion"
    }
    if not src_curves:
        return
    dst_path = 'pose.bones["{}"].rotation_quaternion'.format(bone_name)
    if any(
        action_fcurves(dst_action).find(dst_path, index=i) is not None for i in range(4)
    ):
        return

    dst_curves = []
    for i in range(4):
        dc = new_grouped_fcurve(
            dst_action, data_path=dst_path, index=i, action_group=bone_name
        )
        if i in src_curves:
            dc.extrapolation = src_curves[i].extrapolation
        dst_curves.append(dc)

    all_frames = sorted(
        {kp.co[0] for fcu in src_curves.values() for kp in fcu.keyframe_points}
    )
    for frame in all_frames:
        comps = []
        for i in range(4):
            fcu = src_curves.get(i)
            if fcu is None:
                comps.append(1.0 if i == 0 else 0.0)
                continue
            val = next(
                (
                    kp.co[1]
                    for kp in fcu.keyframe_points
                    if abs(kp.co[0] - frame) < 0.001
                ),
                None,
            )
            comps.append(float(val) if val is not None else (1.0 if i == 0 else 0.0))
        local_q = rest_quat_inv @ Quaternion((comps[0], comps[1], comps[2], comps[3]))
        for i, component in enumerate([local_q.w, local_q.x, local_q.y, local_q.z]):
            new_kp = dst_curves[i].keyframe_points.insert(
                frame, component, options={"FAST"}
            )
            src_fcu = src_curves.get(i)
            if src_fcu:
                src_kp = next(
                    (
                        k
                        for k in src_fcu.keyframe_points
                        if abs(k.co[0] - frame) < 0.001
                    ),
                    None,
                )
                if src_kp:
                    new_kp.interpolation = src_kp.interpolation
                    try:
                        new_kp.handle_left_type = src_kp.handle_left_type
                        new_kp.handle_right_type = src_kp.handle_right_type
                    except Exception as exc:
                        _log.debug("Optional operation failed: {}".format(exc), level=2)
    for dc in dst_curves:
        dc.update()

def _merge_visibility_action_into_transform_action(
    transform_action, vis_action, obj_name
):
    """Clone a transform action and append VISIBLE fcurves from a visibility action."""
    if transform_action is None or vis_action is None:
        return transform_action
    if _action_has_fcurve(transform_action, "VISIBLE"):
        return transform_action

    vis_curves = [
        fcu for fcu in action_fcurves(vis_action) if fcu.data_path == "VISIBLE"
    ]
    if not vis_curves:
        return transform_action

    # Official exporter extracts the EDM argument from the action name prefix.
    # Preserve the visibility action's name prefix (e.g. "10_*") so VISIBLE
    # wrappers are emitted for merged visibility+transform actions.
    merged_name = (
        getattr(vis_action, "name", "") or ""
    ).strip() or transform_action.name
    merged = bpy.data.actions.new(merged_name)
    for src_curve in action_fcurves(transform_action):
        group_name = (
            src_curve.group.name if getattr(src_curve, "group", None) else "Transform"
        )
        _copy_fcurve_to_action(src_curve, merged, src_curve.data_path, group_name)
    for src_curve in vis_curves:
        if (
            action_fcurves(merged).find("VISIBLE", index=src_curve.array_index)
            is not None
        ):
            continue
        group_name = (
            src_curve.group.name if getattr(src_curve, "group", None) else "Visibility"
        )
        _copy_fcurve_to_action(src_curve, merged, "VISIBLE", group_name)
    return merged

def _transfer_bone_actions_to_armature(graph, arm_obj, node_to_bone_name):
    """Retarget per-node ArgAnimatedBone actions to armature pose-bone actions."""
    action_map = {}
    source_graph_nodes = set()
    source_transforms = set()
    bone_nodes = sorted(
        node_to_bone_name.keys(),
        key=lambda n: getattr(n.transform, "_graph_idx", 1 << 30),
    )
    _bone_rest_mats = (_import_ctx.bone_import_ctx or {}).get(
        "bone_rest_matrix_by_name", {}
    )
    for node in bone_nodes:
        bone_name = node_to_bone_name[node]
        rest_mat = _bone_rest_mats.get(bone_name)
        rest_quat_inv = (
            Matrix(rest_mat).to_quaternion().inverted()
            if rest_mat is not None
            else Quaternion()
        )
        # EDM bone animation is typically encoded on wrapper nodes above the Bone.
        seen_sources = set()
        current = node
        while (
            current is not None
            and current.parent is not None
            and current.render is None
            and current.transform is not None
        ):
            tfnode = current.transform
            if isinstance(tfnode, AnimatingNode) and not isinstance(
                tfnode, ArgVisibilityNode
            ):
                if id(tfnode) not in seen_sources:
                    seen_sources.add(id(tfnode))
                    src_actions = get_actions_for_node(tfnode)
                    if src_actions:
                        source_graph_nodes.add(current)
                        source_transforms.add(tfnode)
                    for src_action in src_actions:
                        _log_bone_debug_event(
                            "retarget-source",
                            {
                                "graph_node": getattr(tfnode, "name", "")
                                or type(tfnode).__name__,
                                "graph_node_type": type(tfnode).__name__,
                                "bone_name": bone_name,
                                "action_name": src_action.name,
                                "fcurves": [
                                    fcu.data_path for fcu in action_fcurves(src_action)
                                ],
                            },
                            getattr(tfnode, "name", "") or type(tfnode).__name__,
                            bone_name,
                            src_action.name,
                        )
                        dst_action = action_map.get(src_action.name)
                        if dst_action is None:
                            dst_action = bpy.data.actions.new(src_action.name)
                            action_map[src_action.name] = dst_action
                        # rotation_quaternion curves carry leftRot baked into the rest value.
                        # Pose bone rotation is relative to edit-bone orientation, so
                        # rest frame must be identity. Use the dedicated
                        # helper to remove the rest rotation from each keyframe.
                        _copy_bone_rotation_curves(
                            src_action, dst_action, bone_name, rest_quat_inv
                        )
                        for src_curve in action_fcurves(src_action):
                            if src_curve.data_path == "rotation_quaternion":
                                continue
                            dst_path = 'pose.bones["{}"].{}'.format(
                                bone_name, src_curve.data_path
                            )
                            # Skip existing FCurves to prevent a crash on re-import.
                            if (
                                action_fcurves(dst_action).find(
                                    dst_path, index=src_curve.array_index
                                )
                                is not None
                            ):
                                continue
                            _copy_fcurve_to_action(
                                src_curve, dst_action, dst_path, bone_name
                            )
            parent = current.parent
            if parent is None or _is_bone_transform(parent.transform):
                break
            current = parent

    if not action_map:
        return source_graph_nodes, source_transforms

    _log_bone_debug_event(
        "retarget-summary",
        {
            "armature": getattr(arm_obj, "name", None),
            "actions": sorted(action_map.keys()),
            "source_graph_nodes": sorted(
                (
                    getattr(getattr(n, "transform", None), "name", "")
                    or type(getattr(n, "transform", None)).__name__
                )
                for n in source_graph_nodes
            ),
        },
        getattr(arm_obj, "name", None),
    )

    _attach_retargeted_bone_actions(arm_obj, action_map)
    return source_graph_nodes, source_transforms

def _attach_retargeted_bone_actions(arm_obj, action_map):
    """Attach copied bone actions to the armature using matching NLA tracks."""
    arm_obj.animation_data_create()
    ad = arm_obj.animation_data
    ad.use_nla = True
    ad.action = None
    for track in list(ad.nla_tracks):
        ad.nla_tracks.remove(track)

    for action_name in sorted(action_map.keys()):
        action = action_map[action_name]
        track = ad.nla_tracks.new()
        track.name = action.name
        # Start the strip at the action's first key so keys keep their scene frames.
        start = float(action.frame_range[0])
        strip = track.strips.new(action.name, int(start), action)
        if getattr(strip, "action_slot", False) is None and action.slots:
            strip.action_slot = action.slots[0]
        if abs(strip.frame_start - start) > 1e-6 and hasattr(strip, "frame_start_ui"):
            strip.frame_start_ui = start
        strip.name = action.name
        strip.extrapolation = "NOTHING"



# ---------------------------------------------------------------------------
# Helpers lifted from _prepare_bone_import for standalone readability
# ---------------------------------------------------------------------------

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
    # Animated ancestors also drive ordinary meshes. Keep their object actions.
    # The full source hierarchy drives bones after mesh bind-space finalization.


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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

def _prepare_bone_import(graph, parent_obj=None):
    """Create a single armature for EDM Bone/ArgAnimatedBone nodes."""
    _import_ctx.bone_import_ctx = None

    bone_nodes = [n for n in graph.nodes if _is_bone_transform(n.transform)]
    if not bone_nodes:
        return

    # Only map actual Bone/ArgAnimatedBone nodes to the synthesized armature.
    # Swallowing non-bone ancestors (ArgVisibilityNode wrappers etc.) collapses
    # authored control/visibility transforms and causes DOT hierarchy drift.
    bone_chain_nodes = set(bone_nodes)

    arm_obj, arm_data, apply_bone_root_fix, arm_carries_basis_fix = (
        _create_armature_object(
            bone_nodes,
            parent_obj,
        )
    )
    node_to_bone_name = _build_edit_bones(
        arm_obj,
        arm_data,
        bone_nodes,
        apply_bone_root_fix,
    )
    _finalize_bone_import_ctx(
        arm_obj,
        node_to_bone_name,
        bone_chain_nodes,
        arm_carries_basis_fix,
        graph,
        apply_bone_root_fix,
    )
    _apply_armature_transforms(arm_obj)

def _bind_skin_object(mesh_obj, skin_node):
    """Attach a SkinNode mesh to imported armature with vertex groups."""
    ctx = _import_ctx.bone_import_ctx or {}
    arm_obj = ctx.get("armature")
    bone_name_by_transform = ctx.get("bone_name_by_transform", {})
    bone_rest_matrix_by_name = ctx.get("bone_rest_matrix_by_name", {})
    if arm_obj is None or mesh_obj is None or mesh_obj.type != "MESH":
        return

    palette = [bone_name_by_transform.get(b) for b in getattr(skin_node, "bones", [])]
    skin_bones = [name for name in palette if name]
    if not skin_bones:
        return
    # Vertex bone index i is palette[i + 1]; palette[0] is the control bone.
    control_bone = palette[0] or skin_bones[0]
    weight_bones = palette[1:]

    group_map = {}
    for bone_name in dict.fromkeys(skin_bones):
        vg = mesh_obj.vertex_groups.get(bone_name)
        if vg is None:
            vg = mesh_obj.vertex_groups.new(name=bone_name)
        group_map[bone_name] = vg

    channel_slices = _channel_slices_from_vertex_format(
        skin_node.material.vertex_format
    )

    arm_mod = mesh_obj.modifiers.get("Armature")
    if arm_mod is None:
        arm_mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    arm_mod.object = arm_obj

    bind_target_name, bind_target_loc = _choose_skin_bind_target(
        control_bone, weight_bones, bone_rest_matrix_by_name, mesh_obj, skin_node,
        channel_slices,
    )
    try:
        mesh_obj["_iedm_skin_bind_target_bone"] = str(bind_target_name or "")
        if bind_target_loc is not None:
            mesh_obj["_iedm_skin_bind_target_loc"] = [
                float(bind_target_loc.x),
                float(bind_target_loc.y),
                float(bind_target_loc.z),
            ]
    except Exception as exc:
        _log.debug("Optional operation failed: {}".format(exc), level=2)

    _attach_skin_mesh_to_parent(mesh_obj, arm_obj, skin_node, bind_target_loc)

    _localize_skin_mesh_to_bind_target(mesh_obj, bind_target_loc)

    if not mesh_obj.data.vertices:
        return
    _assign_skin_vertex_weights(
        mesh_obj, skin_node, control_bone, weight_bones, group_map, channel_slices
    )
    _bake_skin_mesh_object_transforms(mesh_obj)

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
