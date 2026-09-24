"""Map EDM material families to exporter material kinds."""

_SELF_ILLUM_MATERIALS = frozenset({
    "self_illum_material",
    "transparent_self_illum_material",
    "additive_self_illum_material",
    "additive_self_illum_color_material",
    "additive_self_illum_tex_material",
})

def _map_edm_material_to_official_kind(edm_material_name):
    mat = (edm_material_name or "").lower()
    if mat in {"glass_material", "glass_instrumental_material"}:
        return "glass"
    if mat in {"mirror_material"}:
        return "mirror"
    # Keep BANO materials on regular render meshes unless the source node itself
    # is explicitly a Fake* light node. Mapping bano_material to fake_omni causes
    # RenderNode -> FakeOmniLightsNode type drift on round-trip.
    if mat in {
        "fake_omni_lights",
        "fake_omni_lights2",
        "fake_als_lights",
        "animated_fake_omni_lights2",
    }:
        return "fake_omni"
    if mat in {
        "fake_spot_lights",
        "fake_spot_lights2",
        "fake_spot_lights2_wdir",
        "animated_fake_spot_lights2",
    }:
        return "fake_spot"
    if mat in {"deck_material"}:
        return "deck"
    # Self-illum materials are emissive render meshes. The official fake-omni
    # path converts meshes into point lights and drops the geometry, so export
    # them through the default material's emissive block instead.
    if mat in _SELF_ILLUM_MATERIALS:
        return "default"
    # Explicit default-family names resolve to "default" rather than falling
    # through, so unknown future names still get the fallback warning path.
    if mat in {
        "def_material",
        "color_material",
        "chrome_material",
        "aluminium_material",
    }:
        return "default"
    return "default"


