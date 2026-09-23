"""Small Blender checks that isolate wrapper and visibility invariants."""

from types import SimpleNamespace
from unittest.mock import patch


def run_prechecks(bpy, Matrix, orient_scale, vis_rewrites, session, action_fcurves):
    from iEDM_10.blender_importer import anim_actions

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.armature_add(location=(2, 3, 4))
    rig = bpy.context.object
    child = bpy.data.objects.new("BoneChild", None)
    bpy.context.collection.objects.link(child)
    child.parent = rig
    child.parent_type = "BONE"
    child.parent_bone = rig.data.bones[0].name
    child.location = (1, 2, 3)
    bpy.context.view_layer.update()
    world = child.matrix_world.copy()
    wrapper = orient_scale._insert_parent_wrapper_object(
        child, "Wrapper", Matrix.Identity(4)
    )
    bpy.context.view_layer.update()
    assert wrapper.parent_type == "BONE" and wrapper.parent == rig
    assert child.parent_type == "OBJECT" and not child.parent_bone
    assert (
        max(
            abs(world[r][c] - child.matrix_world[r][c])
            for r in range(4)
            for c in range(4)
        )
        < 1e-6
    )
    print("PASS bone-parent wrapper")

    control = bpy.data.objects.new("VisibilityControl", None)
    bpy.context.collection.objects.link(control)
    control.location = (4, 5, 6)
    child.parent = control
    child.location = (1, 2, 3)
    bpy.context.view_layer.update()
    basis = child.matrix_basis.copy()
    actions = [
        bpy.data.actions.new("1_Visibility"),
        bpy.data.actions.new("2_Visibility"),
    ]
    with patch.multiple(
        vis_rewrites,
        _visibility_source_for_graph_node=lambda node: object(),
        get_actions_for_node=lambda source: actions,
        _collect_merged_transform_actions_for_graph_node=lambda node, ctx: [],
    ):
        vis_rewrites._split_multi_arg_visibility_controls(
            SimpleNamespace(nodes=[SimpleNamespace(blender=control)])
        )
    assert child.parent != control and child.matrix_basis == basis
    print("PASS visibility-control child transform")

    ranges = [(0.0025, 0.0075), (-0.755, -0.25), (0.3, 0.6), (0.5, 1.01)]
    for prefix_matrix in (None, Matrix.Identity(4)):
        session._import_ctx.bonetransform_prefix_matrix = prefix_matrix
        action = anim_actions.create_visibility_actions(
            SimpleNamespace(name="ThresholdTest", visData=[(153, ranges)])
        )[0]
        curve = action_fcurves(action).find("VISIBLE")
        probe = bpy.data.objects.new(
            "ThresholdProbe", bpy.data.meshes.new("ThresholdProbe")
        )
        bpy.context.collection.objects.link(probe)
        node = SimpleNamespace(
            blender=probe,
            render=object(),
            parent=None,
            transform=SimpleNamespace(visData=[(153, ranges)]),
        )
        session._propagate_visibility_hide_to_render_nodes(
            SimpleNamespace(nodes=[node])
        )
        frames = (
            0,
            24.4,
            24.6,
            74.9,
            75.1,
            99.9,
            100,
            100.2,
            100.3,
            100.7,
            100.8,
            129.9,
            130.1,
            159.9,
            160.1,
            200,
        )
        for frame in frames:
            value = frame / 100.0 - 1.0
            visible = any(a <= value < b for a, b in ranges)
            bpy.context.scene.frame_set(int(frame), subframe=frame - int(frame))
            bpy.context.view_layer.update()
            assert bool(curve.evaluate(frame)) == visible
            assert probe.hide_viewport == (not visible)
        assert abs(anim_actions._anim_frame_to_scene_frame(0.0025) - 100.25) < 1e-6
        assert [anim_actions._anim_frame_to_scene_frame(v) for v in (-1, 0, 1)] == [
            0,
            100,
            200,
        ]
    session._import_ctx.bonetransform_prefix_matrix = None
    print("PASS normalized argument timeline and fractional visibility thresholds")
