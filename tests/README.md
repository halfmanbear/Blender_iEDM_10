# tests

Headless Blender checks for the importer. Run everything from the repository root with Blender 4.5 LTS or 5.2 LTS.

## Test assets

Test assets are ED `Learning_Demo` files and are **not** stored in this repository (`tests/assets/` is git-ignored).

1. Get the official ED exporter add-on (`io_scene_edm`) from https://github.com/EagleDynamics/Blender-EDM-Exporter.
2. Copy the nine `.blend` files and the `textures/` folder from `io_scene_edm/Learning_Demo/` into `tests/assets/`.
3. Open each `.blend` and export it with the official exporter to a same-named `.edm` in `tests/assets/`.

```
tests/assets/
├── Bones.blend / Bones.edm
├── ...
└── textures/
```

| Name | Description |
|---|---|
| `Bones` | Armature with bone animations |
| `Damage` | Damage argument driven visibility |
| `Deck` | Deck material |
| `Emission` | Self-illumination / emissive material |
| `Fake_Lights` | Fake omni and spot lights |
| `Geomerty+Animation+Collision` | Combined geometry, animation, and collision shell |
| `Glass` | Glass material |
| `LightMap` | AO / lightmap UV channel |
| `Lighting_Real` | Real light nodes |

## Transform regression checks

These checks use the importer directly; they do not require the official exporter or test material/export parity.

```powershell
& 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' --background --factory-startup --python-exit-code 1 --python tests/audit_transform_postprocess.py
& 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' --background --factory-startup --python-exit-code 1 --python tests/compare_reference_geometry.py -- 'tests/assets/Geomerty+Animation+Collision.edm'
```

The audit imports every `.edm` in `tests/assets/` and asserts that an existing scene mesh
retains its transform and action. Pass additional EDM paths after `--` to audit
external assets. `--report path.json` records postprocessing changes.

The geometry check compares evaluated world-space vertices of matching mesh
names at frame 100 against the paired `.blend`, with a tolerance of `1e-5`.
It does not check unmatched meshes, materials, or animation across other frames.
The collision example includes four front-wheel meshes that were incorrectly
rotated by the former angle-based orientation postprocessing.

## Aircraft import regressions

Aircraft assets are external and never bundled. Pass paths to your own copies:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' --background --factory-startup --python-exit-code 1 --python tests/regression_aircraft_import.py -- '<model_a.edm>' '<model_b.edm>'
```

Checks neutral skin vertex placement, valid bone parenting, child local transforms
through control splits, and inherited visibility against source ranges at frames
0, 50, 100, 150 and 200. Saves `local/diagnostics/corrected_<asset>.blend` for inspection
(`local/` is git-ignored).

Visibility preview uses custom-property actions that follow the official
exporter's selective argument muting. With all actions enabled, all argument
values advance together on the import timeline. Authored `VISIBLE` export
actions remain separate; visibility-only multi-argument controls are split
into active actions so the exporter does not lose NLA-only visibility. These checks do not validate an EDM export
round trip, materials, or every animated pose.

Argument timing is always EDM -1/0/1 to Blender frames 0/100/200, including
Bonetransform-prefix files. Visibility and transform keys retain fractional
frames. The aircraft regression also checks negative, overlapping, and narrow
fractional visibility windows against both the VISIBLE action and mesh preview
drivers, with and without the skeletal-prefix context. These tests do not
establish native EDM round-trip fidelity.

## Default-origin regression

The aircraft visibility regression uses the UI default APPROX origin mode.
Compare it with RAW to ensure changing an editing origin cannot move aircraft
panels, lights or damage geometry:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' --background --factory-startup --python-exit-code 1 --python tests/regression_origin_modes.py -- '<model_a.edm>' '<model_b.edm>'
```

This compares every mesh vertex transformed by its object matrix at frame 100,
plus sampled vertices at frames 0, 150 and 200, and visibility in both modes.
It does not evaluate modifier-deformed vertices or establish EDM export fidelity.

## Source-matrix mesh regression

Rebuilds each static RenderNode's world transform from the parsed EDM node chain
(base matrix, position, rotation and oriented-scale keys) and compares sampled
world vertices against Blender at frames 100, 150 and 200 (tolerance 2 mm):

```powershell
& 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' --background --factory-startup --python-exit-code 1 --python tests/regression_source_meshes.py -- '<model_a.edm>' '<model_b.edm>'
```

Skinned meshes and render nodes split into several Blender objects are not
checked; the skin paths are covered by the aircraft regression above. The
reference model is the importer's own reading of the EDM transform order, not
DCS ModelViewer output.

## Blender version compatibility

`regression_export_hold_keys.py` requires the installed official exporter but
no aircraft assets. Run it with the same background/factory-startup flags on
both Blender versions. It writes synthetic location, scale, quaternion and
Euler hold curves through pyedm, verifies that protected hold-start keys survive
(and unprotected ones are dropped), and bounds rotation displacement at a
three-metre radius to 1.4 mm. Cases include balanced and negative quaternions;
a nudge that works only near identity is insufficient. This checks finite
examples, not DCS behavior or a universal geometry-error bound.

`regression_export_damage_mask.py` also needs only the installed official
exporter. It builds triangles with an implicit legacy volume mask (slot 5 only,
which DCS resolves to `<damage>_map`), an explicit volume mask (slot 15) and an
RGBA mask (slot 18), imports them, exports with `edm.export` and checks that
volume masks come back through `setMask()` (slot 15), RGBA masks through
`setMaskRGBA()` (slot 18), and the damage argument survives. Pass source EDMs
after `--` to check that every source mask kind/name survives a round trip.
This is structural only; it does not check DCS rendering.

`regression_export_uv_shift.py` round-trips animated UV shifts
(`diffuseShift`, `emissiveShift`, `decalShift`, `ambientOcclusionShift`) and
checks each uniform, argument and key list survives `edm.export`. Pass EDMs
after `--`; `tests/assets/Emission.edm` covers the diffuse and emissive shifts.
DCS samples diffuse, decal and AO shifts only (`Bazar/shaders/model/common/uniforms.hlsl`);
`emissiveShift` is kept for file parity but has no in-game effect.

`regression_export_skin_bind.py` round-trips skins whose bind is not the
arg-0 pose (default `tests/assets/Bones.edm`; pass other EDMs after `--`). It
moves one skin argument at a time over -1..1 and deforms every skin the DCS
way (packed index + 1, missing weight to palette[0]); each source vertex must
have a round-trip vertex within 1 mm. Without the importer's export bind
bridge Bones.edm is off by ~38 mm.

`regression_export_light_frames.py` round-trips light nodes (default
`tests/assets/Lighting_Real.edm`; pass other EDMs after `--`) and checks each
light's static frame matches the source. Lights directly under the file root
need the same Y-up basis as root meshes; without it they come back rotated
Rx(-90) about the origin (~48 m off in Lighting_Real).

Run `regression_animation_api.py` with each Blender executable using the same
background/factory-startup flags above. It checks evaluated object/NLA/light
animation, material curve interpolation, bone groups, and registration.

`regression_blender_integration.py -- tests/assets/Deck.edm` additionally enables
the installed official exporter and checks bound animation slots and nonempty
material groups. Run each asset in a fresh Blender process. This checks import
and shader construction, not visual render parity or an EDM export round trip.
