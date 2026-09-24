"""Round-trip animated UV shifts through import and the official exporter.

Run with Blender --background --factory-startup --python-exit-code 1
--python tests/regression_export_uv_shift.py -- model.edm [...].

Every animated diffuse/emissive/decal/AO shift in the source (uniform name,
argument and keys) must come back from edm.export unchanged. Emission.edm
from tests/assets covers diffuseShift and emissiveShift.
"""

import sys
import tempfile
from collections import Counter
from pathlib import Path

import addon_utils
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import iEDM_10
from iEDM_10 import reader
from iEDM_10.blender_importer.material_uv_shift import _UV_SHIFT_SLOTS
from iEDM_10.edm_format.types import EDMFile


def _rounded(value):
    return tuple(round(float(v), 4) + 0.0 for v in value)


def uv_shifts(path):
    shifts = Counter()
    for material in EDMFile(str(path)).root.materials:
        for name, prop in (material.animated_uniforms or {}).items():
            if name in _UV_SHIFT_SLOTS:
                keys = tuple(
                    (round(float(k.frame), 4), _rounded(k.value)) for k in prop.keys
                )
                shifts[(name, int(prop.argument), keys)] += 1
    return shifts


def round_trip(source, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("io_scene_edm", default_set=True)
    iEDM_10.register()
    reader.read_file(str(source), options={"mesh_origin_mode": "RAW"})
    assert bpy.ops.edm.export(filepath=str(output)) == {"FINISHED"}
    iEDM_10.unregister()
    return uv_shifts(source), uv_shifts(output)


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    assert argv, "pass at least one EDM after --"
    with tempfile.TemporaryDirectory(prefix="iedm_uv_shift_") as directory:
        for arg in argv:
            path = Path(arg).resolve()
            before, after = round_trip(path, Path(directory) / (path.stem + ".edm"))
            assert before, (path.name, "has no animated UV shifts")
            lost = set(before) - set(after)
            extra = set(after) - set(before)
            assert not lost and not extra, (path.name, sorted(lost), sorted(extra))
            names = Counter(name for name, _, _ in before.elements())
            print("PASS", path.name, "UV shifts:", dict(names))
    print("Blender", bpy.app.version_string)


if __name__ == "__main__":
    main()
