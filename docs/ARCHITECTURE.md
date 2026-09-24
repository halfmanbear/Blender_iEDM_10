# iEDM Addon - Python Scripts Index

All project Python files are at most 400 physical lines. Older entry modules
retain compatibility exports, while implementation modules import their
dependencies explicitly. The shared `ImportContext` instance lives in
`blender_importer/import_context.py`; compatibility imports reference that
same instance.

## Responsibility splits

| Entry module | Focused implementation modules |
|--------------|--------------------------------|
| `blender_importer/anim_actions.py` | `action_values.py` interprets keys and action order; `action_curves.py` builds and copies curves |
| `blender_importer/graph_pipeline.py` | `graph_diagnostics.py` handles transform diagnostics; `graph_collections.py` assigns collections |
| `blender_importer/material_setup.py` | `material_animation.py` handles uniform/UV animation; `material_payload.py` preserves exporter metadata |
| `blender_importer/materials_bridge.py` | `material_mapping.py` maps material families; `material_sockets.py` resolves sockets, UV channels, and enums |
| `blender_importer/nodes/armature.py` | `bone_actions.py`, `bone_rest.py`, `armature_build.py`, and `skin_binding.py` separate animation, rest geometry, rig creation, and skin data |
| `blender_importer/object_create.py` | `morph_objects.py` handles morph payloads and shape keys |
| `blender_importer/orient_scale.py` | `orient_scale_geometry.py` computes bases and creates wrappers |
| `blender_importer/prelude.py` | `import_context.py`, `import_logging.py`, `visibility_graph.py`, `visibility_timeline.py`, `node_identity.py`, and `exporter_properties.py` own the shared helpers |
| `blender_importer/session.py` | `session_setup.py`, `scene_root.py`, `session_postprocess.py`, and `session_visibility.py` implement import phases |
| `blender_importer/vis_rewrites.py` | `skin_visibility.py` restores skin transforms and inverse-scale offsets |
| `edm_format/types/lights.py` | `light_nodes.py`, `fake_spot_nodes.py`, `fake_omni_nodes.py`, `light_parsing.py`, and `light_records.py` implement readers and decoding |
| `edm_format/types/render_shell.py` | `render_nodes.py`, `render_special_nodes.py`, and `render_payload.py` implement registered readers and binary payload helpers |

Importing the original parser entry modules still registers all supported
node readers. Regression coverage in `tests/test_module_boundaries.py`
checks those registrations, legacy helper imports, and shared context identity.

## Root Directory (`iEDM_10/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Blender addon registration; defines `bl_info` and registers import operator + RNA properties |
| `reader.py` | Compatibility entry point; re-exports the `blender_importer` package's public surface (each sub-module now imports explicitly what it needs, no cross-module namespace injection) |
| `io_operators.py` | Defines the `ImportEDM` Blender operator (File > Import > DCS World), including debug options and import settings UI |
| `utils.py` | Utility functions: `chdir` context manager for directory switching, `get_root_object` to find top-level parents, `print_edm_graph` for debugging the EDM node tree |
| `rna.py` | Extends Blender data model with EDM-specific properties (`EDMProps`, `EDMObjectSettings`); handles registration of custom properties for objects, materials, and actions |
| `translation.py` | Defines `TranslationNode` and `TranslationGraph` classes that map EDM transform/render nodes to Blender objects; handles tree walking, node insertion, and parent/child relationships |

## `edm_format/` - EDM File Format Parsing

| File | Purpose |
|------|---------|
| `__init__.py` | Exports `EDMFile` class and model format identification utilities (`ModelFormatInfo`, `UnsupportedModelFormatError`) |
| `basereader.py` | Low-level binary stream reader (`BaseReader`); provides `read_float`, `read_uint`, `read_string`, `read_matrixf`, `read_quaternion`, etc. for parsing EDM binary data |
| `typereader.py` | Type reader registry system; decorators like `@reads_type`, `@allow_properties`, `@animatable` register functions to read specific EDM types (vectors, matrices, animated properties, keyframes) |
| `mathtypes.py` | Math type definitions (`Vector`, `Matrix`, `Quaternion`); coordinate conversion helpers (`matrix_to_blender`, `quaternion_to_blender`, `vector_to_blender`) for Y-up (EDM) to Z-up (Blender) conversion |
| `propertiesset.py` | Reads, writes, and audits EDM `PropertiesSet` payloads; preserves scalar, string, vector, and animated property values |
| `material_types.py` | EDM material parsing: `Material`, `VertexFormat`, `Texture`, `ShadowSettings` classes; reads material properties, textures, uniforms, and animated uniforms from binary |
| `types/__init__.py` | Exports EDM node type classes (likely `RenderNode`, `ShellNode`, `ArgAnimationNode`, `Bone`, etc.) |
| `types/core.py` | Compatibility facade for the EDM v10 parser and node types |
| `types/core_support.py`, `core_file.py` | Type-reader support, indexes, and EDM file read/write orchestration |
| `types/core_nodes.py`, `core_animation.py` | Transform, root, and animation node definitions and their registered binary readers |
| `types/lights.py` | EDM light node types: `FakeOmniLightsNode`, `FakeSpotLightsNode`, `FakeALSNode`; includes decode functions for packed light data |
| `types/number.py` | EDM number display node type (`NumberNode`) for in-game numeric displays |
| `types/render_shell.py` | EDM render/collision shell node types (`RenderNode`, `ShellNode`, `SegmentsNode`) |
| `probe.py` | Model format detection (`identify_model_file`, `ModelFormatInfo`); identifies EDM version from file header |

## `blender_importer/` - Blender Scene Construction

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker; indicates this folder contains the importer logic |
| `prelude.py` | Core shared state and helpers: `ImportContext` (thread-local import state), `_ROOT_BASIS_FIX` matrix, logging (`_log`), debug helpers, visibility chain analysis, node classification (`is_skeleton_node`, `_is_child_of_file_root`) |
| `import_pipeline.py` | Pipeline coordinator; imports and re-exports functions from all fragment modules for callers that only import `import_pipeline` |
| `graph_build.py` | Phase 1: Builds `TranslationGraph` from EDM file data; attaches render nodes, collapses transform/render chains, eliminates artifact wrappers, sorts children |
| `graph_pipeline.py` | Phase 2: Graph debug utilities, animation helpers (`_anim_vector_to_blender`, `_anim_quaternion_to_blender`), action helpers, collection assignment (Vehicle/Collision/Texture_Animation) |
| `graph_postprocess.py` | Phase 3: Post-processing passes - fixes owner-encoded render offsets, zeros render child mesh locals, applies basis fixes for visibility/root/static wrapper nodes |
| `import_capabilities.py` | Inspects parsed EDM graph and derives import capabilities (`ImportCapabilities`) based on detected features (bones, shells, bano materials, etc.) |
| `animation.py` | Animation keyframe processing: position/rotation/scale FCurve creation, Euler normalization, morph node decoding, quaternion identity/orientation checks |
| `anim_actions.py` | Builds Blender actions for EDM animations: visibility actions (`create_visibility_actions`), ArgAnimation actions, multi-arg control splitting, oriented scale action creation |
| `mesh_create.py` | Creates Blender mesh objects from EDM vertex/index data; handles compacting, UV layers, normals, triangle/line primitives, merge by distance |
| `object_create.py` | Creates Blender objects for EDM nodes: `create_connector`, `create_segments`, `create_object`; handles materials, special types, shape keys for morph nodes |
| `material_setup.py` | Creates Blender PBR materials from EDM data; maps textures (diffuse/normal/specular), uniforms, animated uniforms; integrates with official EDM exporter material system |
| `lights.py` | Compatibility facade for light creation functions and helpers |
| `light_values.py`, `light_real.py`, `light_entry.py`, `light_textured.py`, `light_billboard.py` | Light property values, real and textured lights, and billboard surrogates |
| `light_materials.py`, `light_geometry.py`, `light_animation.py`, `light_fake.py` | Fake light materials, geometry, animation, and object creation |
| `orient_scale.py` | Oriented scale decomposition and rewrite pass; detects ArgAnimationNodes with scale+orientation keys and splits them into multiple Blender objects for proper round-trip |
| `vis_rewrites.py` | Visibility graph basis-fix passes: fixes multi-arg visibility controls, plain root visibility basis, skin visibility transforms, inverse-scaled visibility offsets |
| `skin_rewrites.py` | Late-stage skin parent binding using bind-rest world positions; resolves skin mesh parent overrides for proper armature attachment |
| `ctrl_splits.py` | Multi-arg control splitting and control/mesh-pair renaming; splits animations with multiple arguments into separate Blender objects, renames wrappers to preserve semantic names |
| `bbox_utils.py` | Creates Blender empties for EDM bounding boxes and user boxes; reads BoundingBoxNode/UserBoxNode data and generates Empty objects with proper size, location, orientation, and EDM property stamping |
| `session.py` | Orchestrates top-level import sequence: opens EDM via edm_format, builds TranslationGraph, derives import capabilities, runs node processing, applies post-processing passes, handles collection assignment and scene cleanup |
| `materials_bridge.py` | Maps EDM materials to official `io_scene_edm` exporter material types; converts EDM Material/Texture/Uniform data into Blender nodes that match the exporter's PBR workflow for DCS World compatibility |
| `node_transform.py` | Defines `apply_node_transform` function to apply EDM transform matrices to Blender objects |

## `blender_importer/nodes/` - Node Processing

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker for node processing modules |
| `processing.py` | Legacy aggregation wrapper for node processing helpers; re-exports core, mesh, visibility, armature, and diagnostics symbols |
| `core.py` | Compatibility facade for node processing helpers |
| `node_process.py`, `node_helpers.py` | `process_node` entry point, diagnostics, naming, and shared helper logic |
| `node_create.py`, `node_parent.py`, `node_properties.py` | Blender object creation, parenting, and EDM metadata stamping |
| `node_position.py`, `node_animation.py` | Render positioning and animation hookup |
| `armature.py` | Armature/skeleton import: creates armature object, builds edit bones from EDM Bone/ArgAnimatedBone nodes, transfers bone actions, binds skin meshes to armature with vertex groups |
| `mesh.py` | Mesh utility functions: `_recenter_mesh_object_to_geometry`, `_transform_mesh_data`, identity matrix checks |
| `visibility.py` | Visibility wrapper logic: helper compaction for fake lights under visibility wrappers (`_compact_visibility_identity_intermediate`) |
| `diagnostics.py` | Import diagnostics: `_print_import_diagnostics` prints summary of EDM node types vs Blender objects created, including shell layouts and render split details |

## Quick Reference by Function

### Import Pipeline Flow
1. `io_operators.py:ImportEDM.execute()` → calls `read_file()`
2. `reader.py` exposes the shared compatibility namespace, including `read_file()` from `session.py`
3. `session.py:read_file()` orchestrates:
   - `EDMFile(filename)` - Parse the binary EDM file through `edm_format`
   - `import_capabilities.py:derive_import_capabilities()` - Detect file features and select behavior flags
   - `graph_build.py:build_graph()` - Build the `TranslationGraph` from parsed transform/render nodes
   - `nodes/core.py:process_node()` (implemented in `nodes/node_process.py`) - Create Blender objects for each graph node
   - `graph_postprocess.py`, `vis_rewrites.py`, `orient_scale.py`, `ctrl_splits.py`, `skin_rewrites.py` - Apply post-processing, visibility/control rewrites, and skin parent resolution

### Key Concepts
- **TranslationGraph**: Maps EDM transform/render nodes to Blender objects
- **Supported model containers**: Classic `.edm` files are identified by the `EDM` header; `.edm2` / ClassReader20 containers are detected and rejected with a clear unsupported-format error
- **EDM v10 coordinate handling**: v10 position vectors are treated as Blender-compatible Z-up values, while matrix/quaternion helpers still apply basis conversion for rotation data
- **ArgAnimationNode**: Animated transform with position/rotation/scale keyframes
- **ArgVisibilityNode**: Controls object visibility based on argument values
- **SkinNode**: Mesh with bone weights for skeletal animation
- **Fake Light Nodes**: Non-mesh light representations (omni, spot, ALS)
