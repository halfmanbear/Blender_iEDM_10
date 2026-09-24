"""Material socket values the exporter used to drop must reach the file.

Run with Blender --background --factory-startup --python-exit-code 1
--python tests/regression_export_socket_values.py -- [model.edm].

The exporter builds integer group inputs (DecalId) as float sockets but looks
them up as integer sockets, so every decal id exported as 0, and it replaced
any socket value of 0 with the socket's default (opacity 0 -> 1). export_hooks
patches both. Imports the model, sets DecalId and a zero Opacity Value on
every material, exports with edm.export and checks the written values.
"""

import sys
import tempfile
from pathlib import Path

import addon_utils
import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import iEDM_10  # noqa: E402
from iEDM_10 import reader  # noqa: E402
from iEDM_10.edm_format.types import EDMFile  # noqa: E402

DECAL_ID = 3


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    source = Path(argv[0]).resolve() if argv else ROOT / "tests/assets/Emission.edm"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("io_scene_edm", default_set=True)
    iEDM_10.register()
    reader.read_file(str(source), {})
    tagged = set()
    for material in bpy.data.materials:
        for node in material.node_tree.nodes if material.node_tree else ():
            socket = node.inputs.get("DecalId") if hasattr(node, "inputs") else None
            if socket is not None:
                socket.default_value = DECAL_ID
                tagged.add(material.name.replace(".", "_"))
            opacity = (
                node.inputs.get("Opacity Value") if hasattr(node, "inputs") else None
            )
            if opacity is not None:
                opacity.default_value = 0.0
    assert tagged, "no material with a DecalId input"
    with tempfile.TemporaryDirectory(prefix="iedm_decal_") as directory:
        output = Path(directory) / "decal.edm"
        assert bpy.ops.edm.export(filepath=str(output)) == {"FINISHED"}
        materials = [
            node.material
            for node in EDMFile(str(output)).renderNodes
            if node.material is not None and node.material.name in tagged
        ]
    iEDM_10.unregister()
    decals = {m.name: m.decal for m in materials}
    assert decals and set(decals.values()) == {DECAL_ID}, decals
    opacities = {
        m.name: m.uniforms["opacityValue"]
        for m in materials
        if "opacityValue" in m.uniforms
    }
    assert opacities and set(opacities.values()) == {0.0}, opacities
    print(
        "PASS socket values exported:",
        len(decals),
        "decal ids,",
        len(opacities),
        "zero opacities",
    )
    print("Blender", bpy.app.version_string)


if __name__ == "__main__":
    main()
