"""Check pyedm hold retention and rotation error with the installed exporter.

Run with Blender --background --factory-startup --python-exit-code 1
--python tests/regression_export_hold_keys.py. No aircraft assets are required.
"""

import math
import sys
import tempfile
from pathlib import Path

import addon_utils
import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iEDM_10.blender_importer.export_hold_keys import protect_hold_keys
from iEDM_10.edm_format.types import EDMFile
from iEDM_10.utils import action_fcurves

TIMES = (-0.5, -0.1, 0.0, 0.24, 0.3)


def cases():
    for path in ("location", "scale"):
        for magnitude in (0, 1, 100, -100, 1000):
            a = Vector((magnitude, 0, 0))
            yield (
                f"{path}_{magnitude}",
                path,
                (a, a + Vector((0, 1, 0)), a, a, a + Vector((0, 0, 1))),
            )
    rotations = [Quaternion((0.5, 0.5, 0.5, 0.5))]
    rotations.extend(
        Quaternion(axis, angle)
        for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3))
        for angle in (0, 0.7, math.pi, -2.3)
    )
    for i, rotation in enumerate(rotations):
        a = rotation
        b = a @ Quaternion((0, 0, 1), 0.3)
        c = a @ Quaternion((0, 0, 1), -0.5)
        values = (a, b, a, a, c)
        yield f"quaternion_{i}", "rotation_quaternion", values
        yield (
            f"negative_quaternion_{i}",
            "rotation_quaternion",
            tuple(-q for q in values),
        )
        yield f"euler_{i}", "rotation_euler", tuple(q.to_euler() for q in values)


def make_curves(name, path, values):
    action = bpy.data.actions.new(name)
    curves = [action_fcurves(action).new(path, index=i) for i in range(len(values[0]))]
    for i, curve in enumerate(curves):
        for t, value in zip(TIMES, values, strict=True):
            key = curve.keyframe_points.insert(100 + t * 100, value[i])
            key.interpolation = "LINEAR"
        curve.update()
    return curves


def export_keys(path, curves):
    keys = []
    for t in TIMES:
        value = [fc.evaluate(100 + t * 100) for fc in curves]
        if path == "rotation_euler":
            value = Euler(value).to_quaternion()
        elif path == "rotation_quaternion":
            value = Quaternion(value)
        else:
            value = Vector(value)
        keys.append((t, value))
    return keys


def add_channel(pyedm, model, name, path, keys):
    node = pyedm.AnimationNode(name)
    if path.startswith("rotation"):
        node.setRotationAnimation([[1, keys]])
    elif path == "location":
        node.setPositionAnimation([[1, keys]])
    else:
        node.setScaleAnimation([[1, keys]])
    model.getRootTransform().addChild(node)
    transform = pyedm.Transform(name + "_transform", Matrix.Identity(4))
    node.addChild(transform)
    connector = pyedm.Connector(name)
    connector.setControlNode(transform)
    model.addConnector(connector)


def read_keys(connector, path):
    attr = (
        "rotData"
        if path.startswith("rotation")
        else ("posData" if path == "location" else "scaleData")
    )
    node = connector.parent
    result = []
    while node is not None and not isinstance(node, (int, list)):
        for _, keys in getattr(node, attr, []):
            result.extend(keys[1] if attr == "scaleData" else keys)
        node = getattr(node, "parent", None)
    return sorted(result, key=lambda key: key.frame)


def sample_rotation(keys, t):
    for left, right in zip(keys, keys[1:], strict=False):
        if left.frame <= t <= right.frame:
            u = (t - left.frame) / (right.frame - left.frame)
            a, b = left.value, right.value
            if a.dot(b) < 0:
                b = -b
            return Quaternion(
                tuple(x * (1 - u) + y * u for x, y in zip(a, b, strict=True))
            ).normalized()
    raise AssertionError((t, [key.frame for key in keys]))


def check_model(model, expected):
    worst = 0.0
    checked = set()
    for connector in model.connectors:
        protected, name = connector.name.split("__", 1)
        path, reference = expected[name]
        keys = read_keys(connector, path)
        kept = any(abs(key.frame) < 1e-6 for key in keys)
        assert kept == (protected == "protected"), (connector.name, kept)
        checked.add(connector.name)
        if protected == "protected" and path.startswith("rotation"):
            # A distant vertex reveals excessive rotation nudges even when
            # the writer successfully retains the key. Check inside the hold too.
            for t in (0.0, 0.12, 0.24):
                rotation = sample_rotation(keys, t)
                for point in ((3, 0, 0), (0, 3, 0), (0, 0, 3)):
                    point = Vector(point)
                    error = (rotation @ point - reference @ point).length
                    assert error < 0.0014, (name, t, error)
                    worst = max(worst, error)
    assert len(checked) == 2 * len(expected), len(checked)
    return worst


def check_distant_interpolation(pyedm):
    """A key on slerp(first key, next key) but off its neighbours' slerp is
    dropped by pyedm unless protected (F4U-1D gear strut pivot arg 5)."""
    angles = (0.0, 0.2, 0.1, 0.74, 0.8)  # 0.74 = slerp(0.0 @ -0.5, 0.8 @ 0.3)
    values = tuple(Quaternion((0, 0, 1), angle) for angle in angles)
    kept = {}
    for protected in (False, True):
        name = f"distant_{protected}"
        curves = make_curves(name, "rotation_quaternion", values)
        if protected:
            assert protect_hold_keys() >= 1
        model = pyedm.Model()
        add_channel(
            pyedm,
            model,
            name,
            "rotation_quaternion",
            export_keys("rotation_quaternion", curves),
        )
        with tempfile.TemporaryDirectory(prefix="iedm_hold_keys_") as directory:
            output = str(Path(directory) / "distant.edm")
            model.save(output, 10)
            keys = read_keys(EDMFile(output).connectors[0], "rotation_quaternion")
        kept[protected] = any(abs(key.frame - 0.24) < 1e-6 for key in keys)
        bpy.data.actions.remove(curves[0].id_data)
    assert kept == {False: False, True: True}, kept


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("io_scene_edm", default_set=False)
    from pyedm_platform_selector import pyedm

    model = pyedm.Model()
    channels = {}
    expected = {}
    for name, path, values in cases():
        curves = make_curves(name, path, values)
        keys = export_keys(path, curves)
        add_channel(pyedm, model, "unprotected__" + name, path, keys)
        channels[name] = (path, curves)
        expected[name] = (path, keys[2][1])
    assert protect_hold_keys() == len(channels)
    for name, (path, curves) in channels.items():
        add_channel(pyedm, model, "protected__" + name, path, export_keys(path, curves))
    with tempfile.TemporaryDirectory(prefix="iedm_hold_keys_") as directory:
        output = str(Path(directory) / "holds.edm")
        model.save(output, 10)
        worst = check_model(EDMFile(output), expected)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    check_distant_interpolation(pyedm)
    print(
        "PASS export hold keys:",
        len(channels),
        "cases; max rotation displacement",
        worst,
        "m; Blender",
        bpy.app.version_string,
    )


if __name__ == "__main__":
    main()
