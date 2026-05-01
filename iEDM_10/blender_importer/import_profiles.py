"""Legacy exporter-family override profiles.

These profiles are no longer intended to define the importer's core EDM
handling. Binary semantics live in `edm_format`, while parsed-graph-derived
capabilities now drive primary importer behavior. The profiles in this module
remain as a fallback override layer for Blender-side reconstruction heuristics
that are not yet cleanly derivable from the parsed graph alone.
"""

from dataclasses import dataclass

from .classifier import (
  CLASS_3DSMAX_BANO,
  CLASS_3DSMAX_DEF,
  CLASS_BLENDER_DEF,
  CLASS_COLLISION_MESH,
  CLASS_BLOB_COLLISION,
  CLASS_BLOB_RENDER,
  CLASS_BLOB_SKELETAL,
  CLASS_UNKNOWN,
  CLASS_INVALID,
)


@dataclass(frozen=True)
class ImportProfile:
  name: str
  description: str
  # --- Transform & Basis Fixes ---
  # Apply the global v10 DCS->Blender basis fix when the EDM graph root is a
  # plain transform carrier rather than a fully authored scene object.
  graph_root_v10_basis_fix: bool = False
  # Create/use a Blender root object that carries the imported v10 basis fix.
  v10_root_object_basis_fix: bool = False
  # Correct top-level ArgRotation q1 quaternions that were authored in the
  # original -90deg X basis instead of Blender-local space.
  top_level_neg90x_q1_basis_fix: bool = False
  # Apply the root basis fix directly to top-level render-node local matrices
  # when the file root itself does not contribute a usable transform object.
  plain_root_render_local_basis_fix: bool = False
  # Apply the missing basis fix to Blender EMPTY objects created from plain-root
  # visibility/control wrappers.
  plain_root_visibility_object_basis_fix: bool = False
  # Apply the missing basis fix directly to mesh object transforms under
  # plain-root visibility wrappers.
  plain_root_visibility_mesh_basis_fix: bool = False
  # Fix child transforms beneath connector helpers when the plain root omitted a
  # scene-level basis object.
  plain_root_connector_child_basis_fix: bool = False
  # Apply the basis fix to connector helper objects themselves for plain-root
  # files.
  plain_root_connector_basis_fix: bool = False
  # Post-pass that fixes plain-root visibility wrapper transforms after the
  # normal graph walk has finished.
  plain_root_visibility_basis_fix: bool = False
  # Keep authored q1 local orientation instead of normalizing it away when that
  # local basis is part of the intended export.
  preserve_authored_q1_local_basis: bool = False
  # Correct render chunks that use shared-parent/owner-encoded attachment tables
  # so their imported Blender offsets match the authored hierarchy.
  owner_encoded_render_offset_fix: bool = False
  
  # --- Hierarchy & Scene Simplification ---
  # Remove known exporter-generated wrapper nodes that do not carry meaningful
  # authored structure.
  eliminate_exporter_artifact_wrappers: bool = False
  # Collapse chains of dummy transforms when they only forward transforms and do
  # not represent useful authored controls.
  collapse_dummy_transform_chains: bool = False
  # Collapse a transform node into its sole render child when the extra wrapper
  # is not semantically important.
  collapse_single_mesh_render_to_parent: bool = False
  # Collapse ArgVisibility->single mesh patterns into a tighter Blender scene
  # when that preserves the intended switch structure.
  collapse_single_argvis_mesh_wrapper: bool = False
  # Keep authored ArgVisibility/animation control pairs as separate Blender
  # objects so exporter-relevant control structure survives round-trip.
  preserve_authored_argvis_control_pairs: bool = False
  
  # --- Animation & Logic ---
  # Remap authored [0..1] style animation frame ranges into the scene frame
  # interval expected by the importer/exporter workflow.
  plain_root_unit_interval_frame_remap: bool = False
  # Special-case top-level ArgRotation nodes that directly control visibility
  # wrappers from the file root.
  root_direct_visibility_argrot: bool = False
  # Fix plain-root ArgRotation branches whose rest orientation behaves like a
  # Z half-turn in authored assets.
  plain_root_z_halfturn_argrot: bool = False
  # Fix negative-scale plain-root ArgRotation branches whose rest orientation
  # behaves like a Y half-turn in authored assets.
  plain_root_negscale_rest_y_halfturn_argrot: bool = False
  # Preserve ArgRotation imports as quaternion curves instead of converting them
  # to Euler curves.
  quaternion_argrotation_curves: bool = False
  # Split multi-argument helper controls into separate helper objects by
  # default during import.
  default_rewrite_multi_arg_helpers: bool = False
  # Legacy post-import repair for 3ds Max visibility-wrapper basis chains.
  # This is a heuristic world-space fix, not part of the DLL's read semantics.
  legacy_3dsmax_argvis_chain_basis_fix: bool = False
  # Legacy post-import world-orientation repair for 3ds Max empties/meshes.
  # This should stay off for strict format-aligned import paths.
  legacy_3dsmax_world_orientation_postfix: bool = False
  
  # --- Origins & Meta ---
  # Automatically switch v10 split/shared-parent meshes to RAW origin mode so
  # authored pivots are not recentered away.
  auto_raw_mesh_origin_v10_split: bool = False
  # Automatically keep RAW mesh origins for plain-root non-skeletal assets whose
  # geometry positions already encode the intended part offsets.
  auto_raw_mesh_origin_plain_root_non_skeletal: bool = False
  # Insert an implicit scene-level basis object when the file shape indicates
  # that one should have existed in the authored exporter.
  implicit_scene_root_basis_object: bool = False
  # Create a scene root basis object for embedded collision-style files that
  # otherwise import in the wrong global basis.
  embedded_collision_scene_root_basis_object: bool = False
  # Create a scene root basis object when the EDM root is a LodNode and needs a
  # stable Blender parent to carry the imported basis.
  lod_root_scene_basis_object: bool = False
  # Align mesh placement beneath ArgVisibility parents when exporter-authored
  # parent/child offsets would otherwise import slightly misregistered.
  argvis_parent_mesh_alignment_fix: bool = False
  # Organize imported objects into the standard collection layout by default.
  default_assign_collections: bool = False

  # --- Armature / Bone ---
  # Apply the root basis fix when deriving imported armature rest poses.
  bone_rest_requires_root_basis_fix: bool = False


DEFAULT_PROFILE = ImportProfile(
  name="DEFAULT",
  description="Conservative defaults; all specialized fixes disabled.",
)

LEGACY_IMPORT_PROFILES = {
  CLASS_3DSMAX_BANO: ImportProfile(
    name=CLASS_3DSMAX_BANO,
    description="3ds Max bano-material family (FA-18/C-130 style).",
    graph_root_v10_basis_fix = True,
    v10_root_object_basis_fix = True,
    eliminate_exporter_artifact_wrappers = False,
    collapse_dummy_transform_chains = False,
    collapse_single_mesh_render_to_parent = False,
    owner_encoded_render_offset_fix = True,
    preserve_authored_argvis_control_pairs = True,
    collapse_single_argvis_mesh_wrapper = False,
    default_assign_collections = True,
    auto_raw_mesh_origin_v10_split = True,
    auto_raw_mesh_origin_plain_root_non_skeletal = True,
    implicit_scene_root_basis_object = True,
    top_level_neg90x_q1_basis_fix = True,
    plain_root_render_local_basis_fix = True,
    plain_root_connector_basis_fix = True,
    plain_root_connector_child_basis_fix = True,
    plain_root_unit_interval_frame_remap = True,
    argvis_parent_mesh_alignment_fix = True,
    preserve_authored_q1_local_basis = True,
    plain_root_visibility_basis_fix = True,
    root_direct_visibility_argrot = True,
    plain_root_z_halfturn_argrot = False,
    plain_root_negscale_rest_y_halfturn_argrot = False,
    legacy_3dsmax_argvis_chain_basis_fix = False,
    legacy_3dsmax_world_orientation_postfix = False,
    bone_rest_requires_root_basis_fix = True,
  ),
  CLASS_3DSMAX_DEF: ImportProfile(
    name=CLASS_3DSMAX_DEF,
    description="3ds Max standard material family (Su-27 style).",
    graph_root_v10_basis_fix = True,
    v10_root_object_basis_fix = True,
    eliminate_exporter_artifact_wrappers = False,
    collapse_dummy_transform_chains = False,
    collapse_single_mesh_render_to_parent = False,
    owner_encoded_render_offset_fix = True,
    preserve_authored_argvis_control_pairs = True,
    collapse_single_argvis_mesh_wrapper = False,
    default_assign_collections = True,
    auto_raw_mesh_origin_v10_split = True,
    auto_raw_mesh_origin_plain_root_non_skeletal = True,
    implicit_scene_root_basis_object = True,
    top_level_neg90x_q1_basis_fix = True,
    plain_root_render_local_basis_fix = True,
    plain_root_connector_basis_fix = True,
    plain_root_connector_child_basis_fix = True,
    plain_root_unit_interval_frame_remap = True,
    argvis_parent_mesh_alignment_fix = True,
    preserve_authored_q1_local_basis = True,
    quaternion_argrotation_curves = True,
    plain_root_visibility_basis_fix = True,
    root_direct_visibility_argrot = False,
    plain_root_z_halfturn_argrot = False,
    plain_root_negscale_rest_y_halfturn_argrot = False,
    legacy_3dsmax_argvis_chain_basis_fix = False,
    legacy_3dsmax_world_orientation_postfix = False,
    bone_rest_requires_root_basis_fix = True,
  ),
  CLASS_BLENDER_DEF: ImportProfile( # Mostly correct some animation connectors animate in the wrong plane 
    name=CLASS_BLENDER_DEF,
    description="Official Blender exporter family.",
    plain_root_render_local_basis_fix = True,
    plain_root_visibility_object_basis_fix = True,
    plain_root_visibility_mesh_basis_fix = True,
    plain_root_connector_basis_fix = True,
    plain_root_connector_child_basis_fix = True,
    preserve_authored_q1_local_basis = True,
    quaternion_argrotation_curves = True,
    v10_root_object_basis_fix = True,
    embedded_collision_scene_root_basis_object = True,
    lod_root_scene_basis_object = True,
  ),
  CLASS_COLLISION_MESH: ImportProfile(
    name=CLASS_COLLISION_MESH,
    description="Modern collision-only node graph.",
    # Pure collision-only v10 graphs now use a geometry-space basis conversion
    # derived from the parsed graph instead of a synthetic +90 X root object.
  ),
  CLASS_BLOB_COLLISION: ImportProfile(
    name=CLASS_BLOB_COLLISION,
    description="Legacy blob collision family.",
    # TransformNode matrices encode Y-up positions (3ds Max exporter baked -90 X).
    # A root basis object with _ROOT_BASIS_FIX cancels the coordinate swap and
    # brings all collision geometry into Blender Z-up space.
    embedded_collision_scene_root_basis_object = True,
    v10_root_object_basis_fix = True,
    # Split multi-arg animation/visibility into single-action helper objects so
    # the official io_scene_edm exporter (which only reads the active action per
    # object) can reconstruct per-arg EDM nodes on re-export.
    default_rewrite_multi_arg_helpers = True,
  ),
  CLASS_BLOB_RENDER: ImportProfile(
    name=CLASS_BLOB_RENDER,
    description="Legacy blob render family (old 3ds Max / pre-DEF exporter / deprecated older ioEDM plugin).",
    # --- Basis / orientation ---
    # Plain Node root carries no matrix, so all children need the DCS→Blender
    # basis fix applied at the appropriate level.
    graph_root_v10_basis_fix = False, # No clear effect/change when enabled
    plain_root_render_local_basis_fix = True,
    plain_root_visibility_basis_fix = True, 
    plain_root_connector_child_basis_fix = True,
    root_direct_visibility_argrot = True,
    preserve_authored_q1_local_basis = True,
    # --- Animation ---
    plain_root_unit_interval_frame_remap = False, # No clear effect/change when enabled
    quaternion_argrotation_curves = True,
    # --- Hierarchy ---
    # Collapse a plain ArgVis→single-mesh chain into the ArgVis node so the
    # Blender scene matches the intended visibility-switch structure.
    collapse_single_argvis_mesh_wrapper = True,
    collapse_single_mesh_render_to_parent = True,
    # --- Origins ---
    # Non-skeletal files should keep authored mesh pivots rather than
    # recentering, since the geometry positions encode the part offsets.
    auto_raw_mesh_origin_plain_root_non_skeletal = False, # No clear effect/change when enabled
    # --- Scene ---
    default_assign_collections = True,
    # --- Armature ---
    bone_rest_requires_root_basis_fix = True,
  ),
  CLASS_BLOB_SKELETAL: ImportProfile(
    name=CLASS_BLOB_SKELETAL,
    description="Legacy blob render with skeletal animation (rigged blob files).",
    # --- Basis / orientation ---
    v10_root_object_basis_fix = True,
    implicit_scene_root_basis_object = True,
    plain_root_render_local_basis_fix = True,
    plain_root_visibility_basis_fix = True,
    plain_root_connector_basis_fix = True,
    plain_root_connector_child_basis_fix = True,
    root_direct_visibility_argrot = True,
    preserve_authored_q1_local_basis = True,
    top_level_neg90x_q1_basis_fix = True,
    plain_root_z_halfturn_argrot = True,
    plain_root_negscale_rest_y_halfturn_argrot = True,
    # --- Animation ---
    plain_root_unit_interval_frame_remap = True,
    quaternion_argrotation_curves = True,
    # --- Hierarchy ---
    # Blob files have many duplicate visibility nodes that need collapsing.
    eliminate_exporter_artifact_wrappers = True,
    collapse_dummy_transform_chains = True,
    collapse_single_argvis_mesh_wrapper = True,
    collapse_single_mesh_render_to_parent = True,
    preserve_authored_argvis_control_pairs = True,
    # --- Origins ---
    auto_raw_mesh_origin_v10_split = True,
    auto_raw_mesh_origin_plain_root_non_skeletal = False,
    # --- Scene ---
    default_assign_collections = True,
    argvis_parent_mesh_alignment_fix = True,
    # --- Armature ---
    bone_rest_requires_root_basis_fix = True,
    # --- Render ---
    owner_encoded_render_offset_fix = True,
  ),
}


def get_import_profile(exporter_class):
  return LEGACY_IMPORT_PROFILES.get(exporter_class, DEFAULT_PROFILE)


def get_legacy_import_profile(exporter_class):
  return get_import_profile(exporter_class)
