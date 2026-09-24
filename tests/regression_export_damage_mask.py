"""Round-trip damage masks through import and the installed official exporter.

Run with Blender --background --factory-startup --python-exit-code 1
--python tests/regression_export_damage_mask.py [-- model.edm ...].

Builds three triangles whose damage uses the implicit legacy volume mask
(slot 5 only), an explicit volume mask (slot 15) and an RGBA mask (slot 18),
imports them, exports with edm.export and checks each mask keeps its slot,
name and damage argument. Optional EDM paths are round-tripped too: every
source damage material must come back with the same mask slot and name.
"""

import sys
import tempfile
from collections import Counter
from pathlib import Path

import addon_utils
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import iEDM_10
from iEDM_10 import reader
from iEDM_10.blender_importer.export_damage_mask import damage_mask_payload
from iEDM_10.edm_format.types import EDMFile

ARG = 150
CASES = {
    "legacy": None,
    "volume": ("setMask", "probe_damage_map"),
    "rgba": ("setMaskRGBA", "probe_damage_rgba"),
}


def build_source(pyedm, path):
    model = pyedm.Model()
    uv = np.array([0, 0, 1, 0, 0, 1], dtype=np.float32)
    for offset, (name, mask) in enumerate(CASES.items()):
        node = pyedm.PBRNode(name, name)
        node.setIndices(np.array([0, 1, 2], dtype=np.uint32))
        base = pyedm.BaseBlock()
        positions = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
        positions[0::3] += 3 * offset
        base.setPositions(positions)
        base.setNormals(np.array([0, 0, 1] * 3, dtype=np.float32))
        base.setAlbedoMapUV(uv)
        base.setAlbedoMap(name + "_base")
        node.addBlock(base)
        damage = pyedm.DamageBlock()
        damage.setAlbedoMapUV(uv)
        damage.setAlbedoMap("probe_damage")
        if mask is not None:
            getattr(damage, mask[0])(mask[1])
        damage.setArgument(ARG)
        node.addBlock(damage)
        node.setControlNode(model.getRootTransform())
        model.addRenderNode(node)
    model.save(str(path), 10)


def damage_masks(path):
    """(damage colour, mask kind, mask name) per damage material."""
    edm = EDMFile(str(path))
    masks = Counter()
    for material in edm.root.materials:
        payload = damage_mask_payload(material)
        if payload is not None:
            color = next(t.name for t in material.textures if t.index == 5)
            masks[(color, payload["kind"], payload["name"])] += 1
    args = {getattr(n, "damage_argument", -1) for n in edm.renderNodes}
    return masks, args


def round_trip(source, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("io_scene_edm", default_set=True)
    iEDM_10.register()
    reader.read_file(str(source), options={"mesh_origin_mode": "RAW"})
    common = sys.modules["materials.materials_common"]
    builder = common.make_def_damage_block
    assert bpy.ops.edm.export(filepath=str(output)) == {"FINISHED"}
    # The volume-aware builder only lives for the duration of an export.
    assert common.make_def_damage_block is builder
    assert sys.modules["materials.material_default"].make_def_damage_block is builder
    iEDM_10.unregister()
    return damage_masks(source), damage_masks(output)


def check_synthetic(directory):
    source = directory / "damage_source.edm"
    addon_utils.enable("io_scene_edm", default_set=True)
    from pyedm_platform_selector import pyedm

    build_source(pyedm, source)
    (before, _), (after, args) = round_trip(source, directory / "damage_out.edm")
    expected = Counter(
        {
            ("probe_damage", "volume", "probe_damage_map"): 2,
            ("probe_damage", "rgba", "probe_damage_rgba"): 1,
        }
    )
    assert before == expected, before
    assert after == expected, after
    assert ARG in args, args
    print("PASS synthetic damage masks:", dict(after))


def check_model(path, directory):
    (before, _), (after, _) = round_trip(path, directory / (path.stem + "_out.edm"))
    lost = set(before) - set(after)
    assert not lost, (path.name, sorted(lost))
    print("PASS", path.name, "damage masks:", sorted(before))


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    with tempfile.TemporaryDirectory(prefix="iedm_damage_mask_") as name:
        directory = Path(name)
        check_synthetic(directory)
        for path in argv:
            check_model(Path(path).resolve(), directory)
    print("Blender", bpy.app.version_string)


if __name__ == "__main__":
    main()
