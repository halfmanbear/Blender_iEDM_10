"""Export legacy volume damage masks through pyedm's setMask().

DCS has two damage mask slots: 15 (legacy volume mask, four stages in the
red channel of a 3D texture) and 18 (RGBA mask, one stage per channel). When
a material has a damage colour (slot 5) but neither mask, DCS loads
"<damage colour>_map" into slot 15 (model::GRootNodeMT::loadTextures).

The official exporter only builds damage blocks with setMaskRGBA() and skips
them when no mask texture is linked, so imported legacy materials lost their
damage. Converting a volume mask to RGBA changes stage timing and the
roughness/metallic response, and a white mask punches holes (> 0.85).

Import records the mask kind and name on the material. During an EDM export
the damage-block builder is swapped for one that calls setMask() on tagged
volume materials without a linked mask texture; every other material goes
through the exporter's own builder. export_hooks restores it after export.
"""

import functools
import json
import sys

DAMAGE_MASK_PROP = "_iedm_damage_mask"

_VOLUME_SLOT = 15
_RGBA_SLOT = 18
_DAMAGE_COLOR_SLOT = 5
_DAMAGE_NORMAL_SLOT = 10
_BUILDER = "make_def_damage_block"
_BUILDER_MODULES = (
    "materials.materials_common",
    "materials.material_default",
    "materials.material_glass",
)


def damage_mask_payload(material):
    """Describe the source damage mask, or None without a damage colour."""
    if (getattr(material, "material_name", "") or "").lower() == "deck_material":
        return None  # deck damage uses slots 7/9 and its own exporter path
    slots = {
        int(getattr(tex, "index", -1)): str(getattr(tex, "name", "") or "")
        for tex in getattr(material, "textures", None) or []
    }
    color = slots.get(_DAMAGE_COLOR_SLOT)
    if not color:
        return None
    payload = {"damage_normal": _DAMAGE_NORMAL_SLOT in slots}
    if slots.get(_RGBA_SLOT):
        payload.update(kind="rgba", name=slots[_RGBA_SLOT], implicit=False)
    elif slots.get(_VOLUME_SLOT):
        payload.update(kind="volume", name=slots[_VOLUME_SLOT], implicit=False)
    else:
        payload.update(kind="volume", name=color + "_map", implicit=True)
    return payload


def _volume_mask(bpy_material, textures):
    """Volume mask name for a tagged material, unless a mask is now linked."""
    if bpy_material is None or textures.damage_mask.texture:
        return None
    try:
        payload = json.loads(bpy_material.get(DAMAGE_MASK_PROP) or "{}")
    except (TypeError, ValueError):
        return None
    if payload.get("kind") != "volume" or not payload.get("name"):
        return None
    return payload


def _volume_damage_block(pyedm, mesh_storage, textures, edm_props, bpy_material):
    payload = _volume_mask(bpy_material, textures)
    if payload is None or not textures.damage_color.texture:
        return None
    if edm_props.DAMAGE_ARG < 0 and not mesh_storage.has_dmg_group:
        return None

    block = pyedm.DamageBlock()
    if mesh_storage.has_dmg_group:
        block.setPerVertexArguments(mesh_storage.damage_arguments)

    color = textures.damage_color.texture
    block.setAlbedoMapUV(
        mesh_storage.get_uv(color.get_uv_map(mesh_storage.uv_active), bpy_material.name)
    )
    block.setAlbedoMap(color.texture_name)

    # Import wires the base normal into Damage Normal; only keep it when the
    # source had its own damage normal.
    normal = textures.damage_normal.texture
    if normal and payload.get("damage_normal"):
        block.setNormalMapUV(
            mesh_storage.get_uv(
                normal.get_uv_map(mesh_storage.uv_active), bpy_material.name
            )
        )
        block.setNormalMap(normal.texture_name)

    block.setMask(payload["name"])
    block.setArgument(edm_props.DAMAGE_ARG)
    return block


def _adapted_builder(original, pyedm):
    @functools.wraps(original)
    def builder(mesh_storage, textures, edm_props, bpy_material):
        if _volume_mask(bpy_material, textures) is None:
            return original(mesh_storage, textures, edm_props, bpy_material)
        return _volume_damage_block(
            pyedm, mesh_storage, textures, edm_props, bpy_material
        )

    builder._iedm_original = original
    return builder


def install_damage_builder():
    """Swap in the volume-aware builder; return (owner, name, original) list."""
    common = sys.modules.get(_BUILDER_MODULES[0])
    original = getattr(common, _BUILDER, None)
    if original is None or hasattr(original, "_iedm_original"):
        return []
    adapted = _adapted_builder(original, common.pyedm)
    swapped = []
    for name in _BUILDER_MODULES:
        module = sys.modules.get(name)
        if getattr(module, _BUILDER, None) is original:
            setattr(module, _BUILDER, adapted)
            swapped.append((module, _BUILDER, original))
    return swapped
