"""Import one EDM with the installed official exporter; pass its path after --.

Run each asset in a fresh Blender process to isolate exporter registration.
"""

import sys
from pathlib import Path

import addon_utils
import bpy

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import iEDM_10
from iEDM_10 import reader
from iEDM_10.blender_importer.prelude import _ensure_official_material_bridge
from iEDM_10.utils import action_fcurves

for path in [Path(sys.argv[-1]).resolve()]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("io_scene_edm", default_set=False)
    assert addon_utils.check("io_scene_edm")[1]
    iEDM_10.register()
    reader.read_file(str(path), options={"mesh_origin_mode": "RAW"})
    for owner in (
        list(bpy.data.objects) + list(bpy.data.lights) + list(bpy.data.node_groups)
    ):
        ad = owner.animation_data
        if ad is None:
            continue
        if ad.action and len(action_fcurves(ad.action)):
            assert ad.action_slot is not None, owner.name
        for track in ad.nla_tracks:
            for strip in track.strips:
                if strip.action and len(action_fcurves(strip.action)):
                    assert strip.action_slot is not None, (owner.name, strip.name)
    for frame in (0, 50, 100, 150, 200):
        bpy.context.scene.frame_set(frame)
    groups = sum(
        1
        for mat in bpy.data.materials
        if mat.node_tree
        for node in mat.node_tree.nodes
        if getattr(node, "node_tree", None) is not None
    )
    assert groups > 0, "No official material groups created"
    for mat in bpy.data.materials:
        if mat.node_tree:
            for node in mat.node_tree.nodes:
                tree = getattr(node, "node_tree", None)
                desc = (
                    _ensure_official_material_bridge()["material_descs"].get(tree.name)
                    if tree is not None
                    else None
                )
                # Some exporter material kinds are interface-only placeholders.
                if desc is not None and desc.nodes:
                    assert len(tree.nodes) >= len(desc.nodes), (
                        tree.name,
                        len(tree.nodes),
                        len(desc.nodes),
                    )
                    if desc.links:
                        assert len(tree.links) > 0, (tree.name, len(desc.links))
    print(
        "INTEGRATION_PASS",
        path.name,
        "objects",
        len(bpy.data.objects),
        "shader_groups",
        groups,
    )
    iEDM_10.unregister()
