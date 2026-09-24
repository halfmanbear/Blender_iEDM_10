"""Small Blender checks that isolate wrapper and visibility invariants."""

from types import SimpleNamespace
from unittest.mock import patch


def run_prechecks(bpy, Matrix, orient_scale, vis_rewrites, session, action_fcurves):
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

    _check_visibility_thresholds(bpy, Matrix, session, action_fcurves)
    _check_lifted_prefix(Matrix, orient_scale, action_fcurves)
    _check_light_shear(Matrix)
    _check_rotation_signs(bpy, action_fcurves)
    _check_mesh_shear(bpy, Matrix)
    _check_export_values()


def _check_visibility_thresholds(bpy, Matrix, session, action_fcurves):
    from iEDM_10.blender_importer import anim_actions

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


def _check_lifted_prefix(Matrix, orient_scale, action_fcurves):
    # Oriented-scale rebuilds lift base.matrix @ T(base.position) into a parent
    # helper; the wrapper's position keys must then not be rotated again.
    from mathutils import Euler, Vector

    from iEDM_10.blender_importer import anim_actions

    q1 = Euler((0.3, -0.7, 1.1)).to_quaternion()
    base = SimpleNamespace(
        matrix=Matrix.LocRotScale((0.02, 0.4, 0.9), Euler((1.2, 0.1, -0.4)), None),
        position=(-0.3, 0.57, 0.0),
        quat_1=q1,
        quat_2=q1.copy(),
        scale=(1.0, 1.003, 0.997),
    )
    keys = [
        SimpleNamespace(frame=frame, value=value)
        for frame, value in ((0.0, (0, 0, 0)), (1.0, (0.0446, -0.08, 0.01)))
    ]
    node = SimpleNamespace(
        name="LiftedPrefix", base=base, posData=[(4, keys)], rotData=[], scaleData=[]
    )
    prefix = orient_scale._arganimation_prerotation_basis_local(node)
    q1_basis = q1.to_matrix().to_4x4()
    action = anim_actions._build_arganimation_action(
        node,
        4,
        q1_basis,
        include_scale=False,
        rotation_basis_local=q1_basis,
        position_prefix_lifted=True,
    )
    for key in keys:
        frame = anim_actions._anim_frame_to_scene_frame(key.frame)
        location = Vector(
            [
                action_fcurves(action).find("location", index=i).evaluate(frame)
                for i in range(3)
            ]
        )
        got = (prefix @ Matrix.Translation(location)).translation
        want = (
            Matrix(base.matrix)
            @ Matrix.Translation(Vector(base.position) + Vector(key.value))
        ).translation
        assert (got - want).length < 1e-6, ("lifted prefix position key", got, want)
    print("PASS lifted oriented-scale prefix position keys")


def _check_light_shear(Matrix):
    # Sheared light frames lose shear in Blender; the beam axis (EDM +X, which
    # the exporter's Ry(+90) maps to the Blender light's -Z) must stay exact.
    from iEDM_10.blender_importer import node_transform

    sheared = Matrix(
        (
            (0.31, 0.02, 0.0, 1.0),
            (0.0, 0.3, 0.01, 2.0),
            (0.05, -0.01, 0.31, 3.0),
            (0, 0, 0, 1),
        )
    )
    fixed = node_transform._light_frame_keeping_beam(sheared)
    beam = sheared.col[0].to_3d().normalized()
    assert (fixed.col[0].to_3d().normalized() - beam).length < 1e-6
    assert (fixed.translation - sheared.translation).length < 1e-9
    rotation = fixed.to_3x3().normalized()
    assert (
        max(
            abs(rotation.col[i].dot(rotation.col[j]))
            for i, j in ((0, 1), (0, 2), (1, 2))
        )
        < 1e-6
    )
    mirrored = Matrix.Diagonal((1.0, 1.0, -1.0, 1.0)) @ Matrix.Rotation(0.4, 4, "Z")
    for untouched in (Matrix.Rotation(0.7, 4, "Y"), mirrored):
        assert node_transform._light_frame_keeping_beam(untouched) == untouched
    print("PASS sheared light frame keeps beam axis")


def _check_rotation_signs(bpy, action_fcurves):
    # Shortest-path signs follow the file keys, not the basis-changed ones:
    # orthogonal keys (+/-90 degrees, dot 0) must not be flipped by basis
    # noise, or arg 0 (the midpoint) turns 180 degrees, while a file dot of
    # -1e-7 (a half turn from identity) must still flip.
    import math

    from mathutils import Quaternion

    from iEDM_10.blender_importer.animation import add_rotation_fcurves

    def blender_midpoint(keys, left):
        action = bpy.data.actions.new("half_turn_precheck")
        sweep = [SimpleNamespace(frame=f, value=q) for f, q in keys]
        add_rotation_fcurves(
            action, sweep, left, Quaternion(), quat_to_blender=Quaternion
        )
        curves = [
            action_fcurves(action).find("rotation_quaternion", index=i)
            for i in range(4)
        ]
        middle = Quaternion([c.evaluate(100.0) for c in curves]).normalized()
        first, last = (
            Quaternion([c.evaluate(f) for c in curves]) for f in (0.0, 200.0)
        )
        bpy.data.actions.remove(action)
        # DCS negates the second key when the stored dot is negative, so the
        # float32 keys the exporter reads must keep a positive dot.
        assert first.dot(last) > 1e-7, first.dot(last)
        return left.inverted() @ middle

    # Bases whose float noise makes the basis-changed dot slightly negative.
    lefts = [
        Quaternion(
            (
                0.8372837901115417,
                0.18225619196891785,
                0.3645123839378357,
                -0.3645123839378357,
            )
        ),
        Quaternion(
            (
                0.5982347130775452,
                0.1542142778635025,
                -0.7710714340209961,
                -0.1542142778635025,
            )
        ),
    ]
    a, b = Quaternion((0, 1, 0), math.pi / 2), Quaternion((0, 1, 0), -math.pi / 2)
    assert all((left @ a).dot(left @ b) < 0.0 for left in lefts)
    for left in lefts:
        middle = blender_midpoint([(-1.0, a), (1.0, b)], left)
        assert abs(middle.w) > 0.999, middle
    half_turn = Quaternion((-1e-7, 0.0, -0.1125144, 0.9936502))
    middle = blender_midpoint([(-1.0, Quaternion()), (1.0, half_turn)], Quaternion())
    assert middle.z < -0.5, middle
    print("PASS half-turn rotation keys keep their file signs")


def _check_mesh_shear(bpy, Matrix):
    from iEDM_10.blender_importer import node_transform

    # A slightly sheared mesh frame (AH-6J tail rotor lever) must reach the
    # exporter's matrix_local exactly, although the basis cannot hold shear.
    frame = Matrix(
        (
            (-0.8833, 0.1746, 0.4297, -0.1755),
            (0.0171, -0.9134, 0.409, -4.6893),
            (0.4686, 0.3676, 0.805, -0.1555),
            (0, 0, 0, 1),
        )
    )
    obj = bpy.data.objects.new("shear_precheck", None)
    obj.matrix_basis = frame

    def frame_error():
        # matrix_local of a static object (the exporter's Transform)
        local = obj.matrix_parent_inverse @ obj.matrix_basis
        return max(
            abs(a - b)
            for ra, rb in zip(local, frame, strict=True)
            for a, b in zip(ra, rb, strict=True)
        )

    assert frame_error() > 1e-4, "frame should need shear kept"
    node_transform._remember_shear(obj, frame)
    node_transform.restore_sheared_frames()
    assert frame_error() < 1e-5 and "_iedm_shear_residual" not in obj
    bpy.data.objects.remove(obj)
    print("PASS sheared mesh frame kept in matrix_parent_inverse")


def _check_export_values():
    import math

    # Light and material values must invert the exporter's own conversions.
    from iEDM_10.blender_importer import light_values
    from iEDM_10.blender_importer.light_real import _theta_to_spot_blend
    from iEDM_10.blender_importer.materials_bridge import _official_transparency_enum

    for phi, theta in ((1.0, 0.58), (0.88, 0.46), (1.3, 0.25)):
        blend = _theta_to_spot_blend(theta, phi)
        # pyedm writes theta = 2 * atan(tan(phi / 2) * (1 - spot_blend))
        written = 2.0 * math.atan(math.tan(phi / 2.0) * (1.0 - blend))
        assert abs(written - theta) < 1e-6, (phi, theta, written)
    for kind in ("POINT", "SPOT"):
        energy = light_values._edm_light_brightness_to_blender_energy(
            4.0, kind, animated=True
        )
        # export_lights.light_power_to_energy, used for animated keys
        written = (
            energy
            * light_values._PBR_WATTS_TO_LUMENS
            * light_values._BLENDER_LAMP_ENERGY_COEFFICIENT
        ) ** light_values._BLENDER_LAMP_WEAK_COEFFICIENT
        assert abs(written - 4.0) < 1e-6, (kind, written)
    # the exporter's TransparencyEnumItems identifier for alpha test (2)
    assert _official_transparency_enum(2) == "Z_TEST"
    print("PASS light and material values invert the exporter's conversions")
