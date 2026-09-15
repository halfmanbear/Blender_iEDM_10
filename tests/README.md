# tests

## Transform regression checks

Run from the repository root with Blender 4.5.6. These checks use the importer
directly; they do not require the official exporter or test material/export parity.

```powershell
& 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' --background --factory-startup --python-exit-code 1 --python tests/audit_transform_postprocess.py
& 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' --background --factory-startup --python-exit-code 1 --python tests/compare_reference_geometry.py -- 'tests/Geomerty+Animation+Collision.edm'
```

The audit imports all nine bundled EDMs and asserts that an existing scene mesh
retains its transform and action. Pass additional EDM paths after `--` to audit
external assets. `--report path.json` records postprocessing changes.

The geometry check compares evaluated world-space vertices of matching mesh
names at frame 100 against the paired `.blend`, with a tolerance of `1e-5`.
It does not check unmatched meshes, materials, or animation across other frames.
The collision example includes four front-wheel meshes that were incorrectly
rotated by the former angle-based orientation postprocessing.

## Aircraft import regressions

```powershell
& 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' --background --factory-startup --python-exit-code 1 --python tests/regression_aircraft_import.py -- '../EDM_Files/SU-27/su-27.edm' '../EDM_Files/F-15E_Suite4.EDM' '../EDM_Files/A-10.EDM' '../EDM_Files/F-117/f-117.edm'
```

Checks neutral skin vertex placement, valid bone parenting, child local transforms
through control splits, and inherited visibility against source ranges at frames
0, 50, 100, 150 and 200. Saves `diagnostics/corrected_<asset>.blend` for inspection.
Aircraft assets are external and are not bundled with the repository.

Visibility preview uses custom-property actions that follow the official
exporter's selective argument muting. With all actions enabled, all argument
values advance together on the import timeline. Authored `VISIBLE` export
actions remain separate; visibility-only multi-argument controls are split
into active actions so the exporter does not lose NLA-only visibility. These checks do not validate an EDM export
round trip, materials, or every animated pose.

Reference test assets used by the scripts in `utils/`.

Each test case consists of a paired `.blend` (reference scene) and `.edm` (EDM model file).
The utility scripts import each EDM and compare the result against its reference blend.

## Prerequisite: io_scene_edm

Testing requires the official ED Blender EDM exporter plugin (`io_scene_edm`).

1. Download the plugin from: https://mods.eagle.ru/blender_plugin/index.html
2. Extract the archive so that the `io_scene_edm` folder sits **directly in the root of this project**:

```
Blender_iEDM_10/
├── io_scene_edm/       <-- extracted here
├── iEDM_10/
├── tests/
├── utils/
└── ...
```

## Test Cases

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

## textures/

Shared texture files referenced by the test `.blend` and `.edm` files.

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
& 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' --background --factory-startup --python-exit-code 1 --python tests/regression_origin_modes.py -- '../EDM_Files/A-10.EDM' '../EDM_Files/SU-27/su-27.edm' '../EDM_Files/F-15E_Suite4.EDM' '../EDM_Files/F-117/f-117.edm'
```

This compares every mesh vertex transformed by its object matrix at frame 100,
plus sampled vertices at frames 0, 150 and 200, and visibility in both modes.
It does not evaluate modifier-deformed vertices or establish EDM export fidelity.
