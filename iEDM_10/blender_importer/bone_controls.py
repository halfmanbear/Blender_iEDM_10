"""Evaluate EDM bone transforms without flattening their animated ancestors.

The helper graph uses ordinary Blender actions and constraints, so saved scenes
continue to animate without Python handlers or the importer being installed.
"""

import bpy
from mathutils import Matrix

from ..edm_format.types import ArgAnimationNode
from ..utils import action_fcurves
from .anim_actions import _scale_orientation_quaternion
from .bone_nla import bake_bone_nla
from .prelude import _ROOT_BASIS_FIX, _anim_frame_to_scene_frame, _import_ctx


def _empty(collection, name, parent=None, matrix=None):
    obj = bpy.data.objects.new(name, None)
    collection.objects.link(obj)
    obj.parent = parent
    obj.empty_display_size = 0.01
    obj.hide_render = True
    obj["_iedm_bone_control"] = True
    if matrix is not None:
        obj.matrix_basis = matrix
    return obj


def _affine(collection, name, parent, matrix):
    """Keep shear and signed scale; an Object's single TRS cannot store shear."""
    loc, rot, scale = matrix.decompose()
    rebuilt = Matrix.LocRotScale(loc, rot, scale)
    error = max(abs(matrix[r][c] - rebuilt[r][c]) for r in range(4) for c in range(4))
    if error < 1e-6:
        return _empty(collection, name, parent, matrix)
    import numpy as np

    left, values, right = np.linalg.svd(np.asarray(matrix.to_3x3(), dtype=float))
    # Put reflections in the diagonal, leaving two proper rotations.
    if np.linalg.det(left) < 0:
        left[:, -1] *= -1
        values[-1] *= -1
    if np.linalg.det(right) < 0:
        right[-1, :] *= -1
        values[-1] *= -1
    first = Matrix(left.tolist()).to_4x4()
    first.translation = loc
    parent = _empty(collection, name + "_affine", parent, first)
    parent = _empty(collection, name + "_scale", parent, Matrix.Diagonal((*values, 1)))
    return _empty(collection, name, parent, Matrix(right.tolist()).to_4x4())


def _keys_action(obj, arg, path, keys, convert=lambda value: value):
    action = bpy.data.actions.new(f"{arg}_{obj.name}")
    if hasattr(action, "argument"):
        action.argument = arg
    samples = [(float(k.frame), convert(k.value)) for k in keys]
    if path == "rotation_quaternion":
        obj.rotation_mode = "QUATERNION"
        previous = None
        for index, (frame, quat) in enumerate(samples):
            quat = quat.normalized()
            if previous is not None and previous.dot(quat) < 0:
                quat = -quat
            samples[index] = (frame, quat)
            previous = quat
    for index in range(len(samples[0][1])):
        curve = action_fcurves(action).new(path, index=index)
        curve.extrapolation = "CONSTANT"
        curve.keyframe_points.add(len(samples))
        for point, (frame, value) in zip(curve.keyframe_points, samples, strict=False):
            point.co = (_anim_frame_to_scene_frame(frame), value[index])
            point.interpolation = "LINEAR"
        curve.update()
    # Blender 4.4+ binds a slot only when the assigned action already has one.
    obj.animation_data_create()
    obj.animation_data.action = action
    if getattr(obj.animation_data, "action_slot", False) is None and action.slots:
        obj.animation_data.action_slot = action.slots[0]


def _scale_chain(collection, name, parent, scale, orient):
    parent = _empty(collection, name + "_orient", parent, orient.to_matrix().to_4x4())
    parent = _empty(collection, name + "_scale", parent, Matrix.Diagonal((*scale, 1)))
    return _empty(
        collection, name + "_unorient", parent, orient.inverted().to_matrix().to_4x4()
    )


def _animated_node(collection, name, parent, source):
    base = source.base
    parent = _affine(
        collection,
        name + "_base",
        parent,
        Matrix(base.matrix) @ Matrix.Translation(base.position),
    )
    # Translations precede default rotation; separate arguments add here.
    for index, (arg, keys) in enumerate(source.posData):
        if keys:
            parent = _empty(collection, f"{name}_pos{index}", parent)
            _keys_action(parent, arg, "location", keys)
    parent = _empty(
        collection, name + "_rotation", parent, base.quat_1.to_matrix().to_4x4()
    )
    for index, (arg, keys) in enumerate(source.rotData):
        if keys:
            parent = _empty(collection, f"{name}_rot{index}", parent)
            _keys_action(parent, arg, "rotation_quaternion", keys)
    parent = _scale_chain(collection, name + "_base", parent, base.scale, base.quat_2)
    for index, (arg, (orient_keys, scale_keys)) in enumerate(source.scaleData):
        if not orient_keys and not scale_keys:
            continue
        prefix = f"{name}_scale{index}"
        parent = _empty(collection, prefix + "_orient", parent)
        if orient_keys:
            _keys_action(
                parent,
                arg,
                "rotation_quaternion",
                orient_keys,
                _scale_orientation_quaternion,
            )
        parent = _empty(collection, prefix, parent)
        if scale_keys:
            _keys_action(parent, arg, "scale", scale_keys)
        parent = _empty(collection, prefix + "_unorient", parent)
        if orient_keys:
            _keys_action(
                parent,
                arg,
                "rotation_quaternion",
                orient_keys,
                lambda value: _scale_orientation_quaternion(value).inverted(),
            )
    return parent


def build_bone_control_graph(graph):
    """Drive each pose by world(source) @ inverse_bind, retaining the full chain."""
    ctx = _import_ctx.bone_import_ctx or {}
    rig = ctx.get("armature")
    mapping = ctx.get("bone_name_by_node", {})
    if rig is None or not mapping:
        return
    collection = bpy.data.collections.new("EDM Bone Controls")
    bpy.context.scene.collection.children.link(collection)
    bpy.context.view_layer.update()
    basis = _ROOT_BASIS_FIX
    root = _empty(
        collection,
        "_EDMBoneControls",
        graph.root.blender,
        graph.root.blender.matrix_world.inverted() @ basis,
    )
    sources = {}

    def build(source):
        if source is None:
            return root
        if source in sources:
            return sources[source]
        parent = build(source.parent)
        name = f"_iedm_bone_{getattr(source, '_graph_idx', 0)}_{source.name}"
        if isinstance(source, ArgAnimationNode):
            obj = _animated_node(collection, name, parent, source)
        else:
            obj = _affine(
                collection,
                name,
                parent,
                Matrix(getattr(source, "matrix", Matrix.Identity(4))),
            )
        sources[source] = obj
        return obj

    for node, name in mapping.items():
        source = node.transform
        parent = build(source)
        rest_world = rig.matrix_world @ rig.data.bones[name].matrix_local
        inv_bind = getattr(
            source, "bone_matrix", getattr(source, "inv_base_bone_matrix", None)
        )
        if inv_bind is None:
            raise ValueError(f"Missing inverse bind for {name}")
        correction = Matrix(inv_bind) @ basis.inverted() @ rest_world
        target = _affine(collection, f"_iedm_bind_{name}", parent, correction)
        constraint = rig.pose.bones[name].constraints.new("COPY_TRANSFORMS")
        constraint.name = "EDM source transform"
        constraint.target = target
        constraint.owner_space = "WORLD"
        constraint.target_space = "WORLD"
    # Bones carry every animated ancestor; an animated mesh parent would apply it twice.
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.parent in (None, rig):
            continue
        if not any(m.type == "ARMATURE" and m.object == rig for m in obj.modifiers):
            continue
        world = obj.matrix_world.copy()
        obj.parent = rig
        obj.parent_type = "OBJECT"
        obj.matrix_world = world
    rig.data.pose_position = "POSE"
    rig["_iedm_source_pose_graph"] = True
    ctx["source_pose_objects"] = sources
    bpy.context.view_layer.update()
    # The exporter cannot read constraints; give it per-argument NLA strips.
    bake_bone_nla(rig, collection)
    for obj in collection.objects:
        obj.hide_set(True)
    bpy.context.view_layer.update()
