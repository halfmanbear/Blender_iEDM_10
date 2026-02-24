# tests

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
