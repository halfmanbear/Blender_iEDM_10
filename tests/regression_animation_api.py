"""Check action evaluation in Blender 4.5 and 5.2, without external assets."""

import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import iEDM_10
from iEDM_10.blender_importer.graph_pipeline import _push_action_to_nla
from iEDM_10.blender_importer.material_setup import _mat_set_linear_on_path
from iEDM_10.blender_importer.prelude import _assign_action
from iEDM_10.utils import action_fcurves, new_grouped_fcurve


def keyed(curve):
    for frame, value in ((1, 2), (11, 12)):
        key = curve.keyframe_points.insert(frame, value)
        key.interpolation = "LINEAR"
    curve.update()


def check(owner, path, index=None, id_type="OBJECT", nla=False):
    action = bpy.data.actions.new("API_" + owner.name)
    keyed(action_fcurves(action, id_type).new(path, index=index or 0))
    if nla:
        assert _push_action_to_nla(owner, action)
    else:
        _assign_action(owner, action)
    bpy.context.scene.frame_set(6)
    bpy.context.view_layer.update()
    value = owner.path_resolve(path)
    if index is not None:
        value = value[index]
    assert abs(value - 7) < 1e-5, (owner.name, path, value)


iEDM_10.register()
for nla in (False, True):
    obj = bpy.data.objects.new("Transform", None)
    bpy.context.collection.objects.link(obj)
    check(obj, "location", 0, nla=nla)
    bpy.context.scene.frame_set(1)

light = bpy.data.lights.new("AnimatedLight", "POINT")
obj = bpy.data.objects.new("LightObject", light)
bpy.context.collection.objects.link(obj)
check(light, "energy", id_type="LIGHT")

mat = bpy.data.materials.new("AnimatedMaterial")
mat.use_nodes = True
socket = mat.node_tree.nodes.get("Principled BSDF").inputs["Roughness"]
for frame, value in ((1, 0), (11, 1)):
    socket.default_value = value
    socket.keyframe_insert("default_value", frame=frame)
action = mat.node_tree.animation_data.action
_mat_set_linear_on_path(action, socket.path_from_id())
assert all(
    k.interpolation == "LINEAR"
    for fc in action_fcurves(action)
    for k in fc.keyframe_points
)
assert abs(action_fcurves(action)[0].evaluate(6) - 0.5) < 1e-5

action = bpy.data.actions.new("GroupedBone")
curve = new_grouped_fcurve(action, 'pose.bones["Bone"].location', 0, "Bone")
keyed(curve)
assert curve.group.name == "Bone"
assert abs(curve.evaluate(6) - 7) < 1e-5
iEDM_10.unregister()
print("PASS animation API:", bpy.app.version_string)
