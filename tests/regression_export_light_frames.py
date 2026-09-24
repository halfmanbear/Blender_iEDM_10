"""Round-trip light frames, including lights directly under the file root.

Run with Blender --background --factory-startup --python-exit-code 1
--python tests/regression_export_light_frames.py -- [model.edm ...].

Defaults to tests/assets/Lighting_Real.edm, whose lights sit in root-level
transform nodes. Imports each model, exports with edm.export and checks every
light's arg-0 frame (the static transform chain above it) matches the source
light of the same name.
"""

import re
import sys
import tempfile
from pathlib import Path

import addon_utils
import bpy
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import iEDM_10
from iEDM_10 import reader
from iEDM_10.edm_format.types import EDMFile

TOLERANCE = 1e-4
_SUFFIX = re.compile(r"(?:\.\d{3})+$")


def _frame(node):
    matrix = Matrix.Identity(4)
    while node is not None and not isinstance(node, (int, list)):
        local = getattr(node, "matrix", None)
        if local is not None:
            matrix = Matrix(local) @ matrix
        node = getattr(node, "parent", None)
    return matrix


def light_frames(path):
    frames = {}
    for light in EDMFile(str(path)).lightNodes or []:
        name = _SUFFIX.sub("", light.name or "")
        frames.setdefault(name, []).append(_frame(light.parent))
    return frames


def _difference(a, b):
    return max(
        abs(x - y)
        for ra, rb in zip(a, b, strict=True)
        for x, y in zip(ra, rb, strict=True)
    )


def worst_difference(source, output):
    expected, actual = light_frames(source), light_frames(output)
    assert expected, (source.name, "has no light nodes")
    assert sorted(expected) == sorted(actual), (sorted(expected), sorted(actual))
    worst = 0.0
    for name, frames in expected.items():
        for frame in frames:
            worst = max(worst, min(_difference(frame, other) for other in actual[name]))
    return worst, sum(len(f) for f in expected.values())


def round_trip(source, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("io_scene_edm", default_set=True)
    iEDM_10.register()
    reader.read_file(str(source), {})
    assert bpy.ops.edm.export(filepath=str(output)) == {"FINISHED"}
    iEDM_10.unregister()


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    paths = [Path(p).resolve() for p in argv] or [
        ROOT / "tests/assets/Lighting_Real.edm"
    ]
    with tempfile.TemporaryDirectory(prefix="iedm_light_frames_") as directory:
        for path in paths:
            output = Path(directory) / (path.stem + ".edm")
            round_trip(path, output)
            worst, count = worst_difference(path, output)
            assert worst < TOLERANCE, (path.name, worst)
            print("PASS", path.name, "lights", count, "worst", worst)
    print("Blender", bpy.app.version_string)


if __name__ == "__main__":
    main()
