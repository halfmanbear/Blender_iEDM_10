"""Blender background regression for skin bind space and visibility inheritance.
Pass EDM paths after --. Saves clean diagnostic scenes alongside the reports.
"""
import contextlib
import io
from pathlib import Path
import sys
import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import iEDM_10
from iEDM_10 import reader
from iEDM_10.blender_importer import session, orient_scale, vis_rewrites
from iEDM_10.blender_importer.nodes import armature

iEDM_10.register()
original_localize = armature._localize_skin_mesh_to_bind_target
original_build = session.build_graph
captured = []
expected_skin = {}

def localize(obj, loc):
    points = [v.co.copy() for v in obj.data.vertices]
    result = original_localize(obj, loc)
    if result:
        expected_skin[obj] = points
    return result

def build(edm):
    graph = original_build(edm)
    captured[:] = [graph]
    return graph

armature._localize_skin_mesh_to_bind_target = localize
session.build_graph = build

def check_split(original):
    def run(graph):
        children = {child: (child.parent, child.matrix_basis.copy())
                    for node in graph.nodes if node.blender is not None
                    for child in node.blender.children}
        original(graph)
        for child, (parent, basis) in children.items():
            if child.parent != parent:
                error = max(abs(basis[r][c] - child.matrix_basis[r][c])
                            for r in range(4) for c in range(4))
                assert error < 1e-6, ('split changed child local transform', child.name, error)
    return run

session._split_multi_arg_rotation_controls = check_split(session._split_multi_arg_rotation_controls)
session._split_multi_arg_nonarmature_controls = check_split(session._split_multi_arg_nonarmature_controls)

# Inserting an identity wrapper must preserve a real bone-parent relationship.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.armature_add(location=(2, 3, 4))
rig = bpy.context.object
child = bpy.data.objects.new('BoneChild', None)
bpy.context.collection.objects.link(child)
child.parent = rig
child.parent_type = 'BONE'
child.parent_bone = rig.data.bones[0].name
child.location = (1, 2, 3)
bpy.context.view_layer.update()
world = child.matrix_world.copy()
wrapper = orient_scale._insert_parent_wrapper_object(child, 'Wrapper', Matrix.Identity(4))
bpy.context.view_layer.update()
assert wrapper.parent_type == 'BONE' and wrapper.parent == rig
assert child.parent_type == 'OBJECT' and not child.parent_bone
assert max(abs(world[r][c]-child.matrix_world[r][c]) for r in range(4) for c in range(4)) < 1e-6
print('PASS bone-parent wrapper')

# Visibility-only control splits must also preserve a translated child local.
from types import SimpleNamespace
from unittest.mock import patch
control = bpy.data.objects.new('VisibilityControl', None)
bpy.context.collection.objects.link(control)
control.location = (4, 5, 6)
child.parent = control
child.location = (1, 2, 3)
bpy.context.view_layer.update()
basis = child.matrix_basis.copy()
actions = [bpy.data.actions.new('1_Visibility'), bpy.data.actions.new('2_Visibility')]
with patch.multiple(vis_rewrites,
                    _visibility_source_for_graph_node=lambda node: object(),
                    get_actions_for_node=lambda source: actions,
                    _collect_merged_transform_actions_for_graph_node=lambda node, ctx: []):
    vis_rewrites._split_multi_arg_visibility_controls(SimpleNamespace(nodes=[SimpleNamespace(blender=control)]))
assert child.parent != control and child.matrix_basis == basis
print('PASS visibility-control child transform')

# File structure must never change the argument timeline. Include narrow,
# fractional, overlapping and negative damage windows in both visibility paths.
from iEDM_10.blender_importer import anim_actions
for prefix_matrix in (None, Matrix.Identity(4)):
    session._import_ctx.bonetransform_prefix_matrix = prefix_matrix
    ranges = [(0.0025, 0.0075), (-0.755, -0.25), (0.3, 0.6), (0.5, 1.01)]
    action = anim_actions.create_visibility_actions(SimpleNamespace(name='ThresholdTest', visData=[(153, ranges)]))[0]
    curve = action.fcurves.find('VISIBLE')
    probe = bpy.data.objects.new('ThresholdProbe', bpy.data.meshes.new('ThresholdProbe'))
    bpy.context.collection.objects.link(probe)
    node = SimpleNamespace(blender=probe, render=object(), parent=None,
                           transform=SimpleNamespace(visData=[(153, ranges)]))
    session._propagate_visibility_hide_to_render_nodes(SimpleNamespace(nodes=[node]))
    for frame in (0, 24.4, 24.6, 74.9, 75.1, 99.9, 100, 100.2, 100.3, 100.7, 100.8, 129.9, 130.1, 159.9, 160.1, 200):
        value = frame / 100.0 - 1.0
        visible = any(a <= value < b for a, b in ranges)
        bpy.context.scene.frame_set(int(frame), subframe=frame-int(frame))
        bpy.context.view_layer.update()
        assert bool(curve.evaluate(frame)) == visible, ('export visibility threshold', frame)
        assert probe.hide_viewport == (not visible), ('preview visibility threshold', frame)
    assert abs(anim_actions._anim_frame_to_scene_frame(0.0025) - 100.25) < 1e-6
    assert [anim_actions._anim_frame_to_scene_frame(v) for v in (-1, 0, 1)] == [0, 100, 200]
session._import_ctx.bonetransform_prefix_matrix = None
print('PASS normalized argument timeline and fractional visibility thresholds')

# A distant palette anchor must not skip absolute skin-space conversion.
probe = bpy.data.objects.new('DistantSkinAnchor', bpy.data.meshes.new('DistantSkinAnchor'))
bpy.context.collection.objects.link(probe)
probe.data.from_pydata([(-3.0, -6.2, 0.0), (-1.5, -6.2, 0.1), (-2.0, -6.3, 0.0)], [], [(0,1,2)])
original = [v.co.copy() for v in probe.data.vertices]
anchor = Vector((5.2, -2.5, 0.9))
assert original_localize(probe, anchor), 'Distant bind anchor incorrectly treated as local geometry'
assert max((v.co + anchor - p).length for v,p in zip(probe.data.vertices, original)) < 1e-5
print('PASS distant skin anchor')

# Deep inherited visibility must not exceed the driver expression limit.
mesh = bpy.data.objects.new('DeepVisibilityMesh', bpy.data.meshes.new('DeepVisibilityMesh'))
bpy.context.collection.objects.link(mesh)
ancestor = None
for arg in range(30):
    ancestor = SimpleNamespace(transform=SimpleNamespace(visData=[(arg, [(0.0, 1.01)])]), parent=ancestor)
leaf = SimpleNamespace(blender=mesh, render=object(), transform=None, parent=ancestor)
session._propagate_visibility_hide_to_render_nodes(SimpleNamespace(nodes=[leaf]))
for frame, hidden in ((100, False), (0, True), (200, False)):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    assert mesh.hide_viewport == hidden and mesh.hide_render == hidden
assert all(fc.driver.is_valid for o in bpy.data.objects if o.animation_data for fc in o.animation_data.drivers)
print('PASS deep inherited visibility')

for filename in sys.argv[sys.argv.index('--')+1:]:
    path = Path(filename).resolve()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    expected_skin.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        reader.read_file(str(path), options={'mesh_origin_mode':'APPROX', 'preserve_scene_boxes':True})
    graph = captured[0]
    # Every retained graph object must remain live through postprocessing.
    for node in graph.nodes:
        if node.blender is not None:
            assert bpy.data.objects.get(node.blender.name) == node.blender
    # Match the official exporter influence cutoff, not just Blender group totals.
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        modifiers = [m for m in obj.modifiers if m.type == 'ARMATURE' and m.object]
        if not modifiers:
            continue
        bone_names = set(modifiers[0].object.data.bones.keys())
        for vertex in obj.data.vertices:
            weights = [g.weight for g in vertex.groups if obj.vertex_groups[g.group].name in bone_names]
            exported = [w for w in weights if w >= 0.001]
            assert len(weights) <= 4 and abs(sum(weights)-1.0) < 1e-6, (path.name,obj.name,vertex.index,weights)
            assert abs(sum(exported)-1.0) < 1e-6, (path.name,obj.name,vertex.index,'export cutoff',weights)
    print('PASS graph lifetime and export skin weights', path.name)
    # Every visibility-only argument must survive as an active action, since
    # the official exporter ignores NLA visibility on non-armature objects.
    visibility_controls = 0
    for node in graph.nodes:
        obj = node.blender
        if obj is None or obj.type != 'EMPTY':
            continue
        source = vis_rewrites._visibility_source_for_graph_node(node)
        if source is None:
            continue
        actions = vis_rewrites.get_actions_for_node(source)
        if len(actions) < 2 or vis_rewrites._collect_merged_transform_actions_for_graph_node(
                node, session._import_ctx.bone_import_ctx or {}):
            continue
        candidates = [obj]
        stack = list(obj.children)
        while stack:
            helper = stack.pop()
            if helper.get('_iedm_vis_passthrough'):
                candidates.append(helper)
                stack.extend(helper.children)
        active = [o.animation_data.action for o in candidates if o.animation_data]
        assert all(action in active for action in actions), (
            path.name, obj.name, 'visibility argument missing from active export actions')
        visibility_controls += 1
    print('PASS active visibility controls', path.name, visibility_controls)
    error = max(((obj.matrix_world @ v.co - point).length
                 for obj, points in expected_skin.items()
                 for v, point in zip(obj.data.vertices, points)), default=0)
    assert error < 0.0001, (path.name, 'skin neutral placement', error)
    invalid = [o.name for o in bpy.data.objects if o.parent_type == 'BONE'
               and (o.parent is None or o.parent.type != 'ARMATURE' or o.parent_bone not in o.parent.data.bones)]
    assert not invalid, invalid
    checked = 0
    controls_by_object = {}
    for frame in (0, 50, 100, 150, 200):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for node in graph.nodes:
            obj = node.blender
            if obj is None or obj.type != 'MESH' or node.render is None:
                continue
            ancestor = node
            controls = []
            seen = set()
            while ancestor:
                for tf in [ancestor.transform] + list(getattr(ancestor, '_collapsed_transforms', None) or []):
                    if tf is not None and id(tf) not in seen:
                        seen.add(id(tf))
                        controls.extend(getattr(tf, 'visData', None) or [])
                ancestor = ancestor.parent
            if not controls:
                continue
            controls_by_object[obj] = controls
            # Compare independently against source argument values.
            value = frame / 100.0 - 1.0
            visible = all(any(start <= value < end for start, end in ranges) for arg, ranges in controls)
            if obj.hide_viewport != (not visible):
                print('VIS DEBUG', obj.name, [(fc.data_path, fc.driver.expression, fc.driver.is_valid) for fc in obj.animation_data.drivers])
            assert obj.hide_viewport == (not visible), (path.name, obj.name, frame, controls, obj.hide_viewport)
            assert obj.hide_render == (not visible)
            checked += 1
    bpy.context.scene.frame_set(100)
    bpy.context.view_layer.update()
    # Selecting one argument must leave other damage arguments at rest.
    previews = [o for o in bpy.data.objects if o.get('_iedm_visibility_preview_control')]
    if previews:
        arguments = {int(o.animation_data.action.name.split('_')[0]) for o in previews}
        selected_arg = 153 if 153 in arguments else min(arguments)
        for preview in previews:
            action = preview.animation_data.action
            for curve in action.fcurves:
                curve.mute = int(action.name.split('_')[0]) != selected_arg
        bpy.context.scene.frame_set(150)
        bpy.context.view_layer.update()
        for obj, controls in controls_by_object.items():
            visible = all(any(start <= (0.5 if arg == selected_arg else 0.0) < end
                              for start, end in ranges) for arg, ranges in controls)
            assert obj.hide_viewport == (not visible), (path.name, obj.name, 'muted damage argument advanced')
        for preview in previews:
            for curve in preview.animation_data.action.fcurves:
                curve.mute = False
        bpy.context.scene.frame_set(100)
        bpy.context.view_layer.update()
        print('PASS selective visibility preview', path.name, selected_arg)
    invalid_drivers = [(o.name, fc.data_path) for o in bpy.data.objects if o.animation_data
                       for fc in o.animation_data.drivers if not fc.driver.is_valid]
    assert not invalid_drivers, invalid_drivers[:10]
    print('PASS', path.name, 'skins', len(expected_skin), 'max_error', error,
          'visibility_checks', checked, 'hidden_meshes', sum(o.hide_viewport for o in bpy.data.objects if o.type == 'MESH'))
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / 'diagnostics' / ('corrected_' + path.stem + '.blend')))


