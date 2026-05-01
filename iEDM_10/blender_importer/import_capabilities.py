"""Graph-derived import capabilities.

These capabilities are inferred from the parsed EDM graph and control the
importer's core behavior. They intentionally sit above binary parsing and below
Blender-side scene heuristics:

1. `edm_format` handles serialized layout exactly as read from disk.
2. This module inspects the parsed graph and enables importer capabilities that
   follow from the observed file shape.
3. Legacy exporter-family profiles remain available as a fallback for Blender
   reconstruction heuristics that are not yet derivable from the parsed graph.
"""

from dataclasses import dataclass, field

from ..edm_format.types import (
  AnimatingNode,
  ArgAnimatedBone,
  Bone,
  LodNode,
  Node,
  RenderNode,
  SegmentsNode,
  ShellNode,
  TransformNode,
)


@dataclass(frozen=True)
class ImportGraphFeatures:
  edm_version: int
  root_name: str
  root_type: str
  transform_root_type: str
  root_name_is_scene_root: bool
  plain_root_v10: bool
  has_root_transform_payload: bool
  has_bones: bool
  has_arganimated_bones: bool
  has_owner_encoded_split_renders: bool
  has_generic_render_chunks: bool
  has_shell_nodes: bool
  has_segments_nodes: bool
  has_bano_material: bool
  render_node_count: int
  connector_count: int
  light_node_count: int


@dataclass(frozen=True)
class ImportCapabilities:
  name: str
  description: str
  detail: str = ""
  flags: dict = field(default_factory=dict)

  def get(self, flag_name, default=None):
    return self.flags.get(flag_name, default)


DEFAULT_CAPABILITIES = ImportCapabilities(
  name="DERIVED_DEFAULT",
  description="No derived importer capabilities.",
)


def _iter_all_objects(edm):
  for node in edm.renderNodes:
    yield node
  for node in edm.connectors:
    yield node
  for node in edm.shellNodes:
    yield node
  for node in edm.lightNodes:
    yield node


def inspect_import_graph(edm):
  root = getattr(edm, "root", None)
  root_tf = getattr(edm, "transformRoot", None)
  root_name = str(getattr(root, "name", "") or "")
  plain_root_v10 = bool(edm.version >= 10 and type(root_tf) is Node)
  has_root_transform_payload = isinstance(root_tf, (TransformNode, AnimatingNode))
  has_bones = any(isinstance(node, Bone) for node in edm.nodes)
  has_arganimated_bones = any(isinstance(node, ArgAnimatedBone) for node in edm.nodes)
  has_owner_encoded_split_renders = any(getattr(obj, "shared_parent", None) is not None for obj in _iter_all_objects(edm))
  has_generic_render_chunks = any(
    isinstance(obj, RenderNode) and str(getattr(obj, "name", "") or "").startswith("_")
    for obj in edm.renderNodes
  )
  has_shell_nodes = bool(edm.shellNodes)
  has_segments_nodes = any(isinstance(obj, SegmentsNode) for obj in _iter_all_objects(edm))
  has_bano_material = any(
    str(
      getattr(mat, "material_name", "")
      or getattr(mat, "materialName", "")
      or getattr(mat, "name", "")
      or ""
    ) == "bano_material"
    for mat in getattr(root, "materials", []) or []
  )
  return ImportGraphFeatures(
    edm_version=int(getattr(edm, "version", 0) or 0),
    root_name=root_name,
    root_type=type(root).__name__,
    transform_root_type=type(root_tf).__name__,
    root_name_is_scene_root=(root_name == "Scene Root"),
    plain_root_v10=plain_root_v10,
    has_root_transform_payload=has_root_transform_payload,
    has_bones=has_bones,
    has_arganimated_bones=has_arganimated_bones,
    has_owner_encoded_split_renders=has_owner_encoded_split_renders,
    has_generic_render_chunks=has_generic_render_chunks,
    has_shell_nodes=has_shell_nodes,
    has_segments_nodes=has_segments_nodes,
    has_bano_material=has_bano_material,
    render_node_count=len(getattr(edm, "renderNodes", []) or []),
    connector_count=len(getattr(edm, "connectors", []) or []),
    light_node_count=len(getattr(edm, "lightNodes", []) or []),
  )


def _capability_detail(features):
  parts = [
    f"v{features.edm_version}",
    f"root={features.root_type}:{features.root_name!r}",
    f"transform_root={features.transform_root_type}",
  ]
  if features.plain_root_v10:
    parts.append("plain_root_v10")
  if features.has_root_transform_payload:
    parts.append("root_transform_payload")
  if features.has_bones:
    parts.append("bones")
  if features.has_arganimated_bones:
    parts.append("arganimated")
  if features.root_name_is_scene_root:
    parts.append("scene_root_name")
  if features.has_owner_encoded_split_renders:
    parts.append("owner_encoded")
  if features.has_generic_render_chunks:
    parts.append("generic_render_chunks")
  if features.has_shell_nodes:
    parts.append("shell_nodes")
  if features.has_segments_nodes:
    parts.append("segments_nodes")
  if features.has_bano_material:
    parts.append("bano_material")
  return ", ".join(parts)


def derive_import_capabilities(edm):
  features = inspect_import_graph(edm)
  flags = {}
  authored_scene_root_v10 = bool(features.edm_version >= 10 and features.plain_root_v10 and features.root_name_is_scene_root)
  pure_collision_payload = bool(
    features.edm_version >= 10
    and features.plain_root_v10
    and not features.has_root_transform_payload
    and not features.render_node_count
    and not features.connector_count
    and not features.light_node_count
    and (features.has_shell_nodes or features.has_segments_nodes)
  )

  if features.edm_version >= 10:
    if authored_scene_root_v10 and not pure_collision_payload:
      flags["graph_root_v10_basis_fix"] = True
      flags["v10_root_object_basis_fix"] = True
      flags["implicit_scene_root_basis_object"] = True

    if features.has_root_transform_payload:
      flags["v10_root_object_basis_fix"] = True

    if features.plain_root_v10 and not pure_collision_payload:
      flags["plain_root_render_local_basis_fix"] = True
      flags["plain_root_visibility_basis_fix"] = True
      flags["plain_root_connector_basis_fix"] = True
      flags["plain_root_connector_child_basis_fix"] = True

      if features.has_bones:
        flags["bone_rest_requires_root_basis_fix"] = True
        flags["v10_root_object_basis_fix"] = True
        flags["implicit_scene_root_basis_object"] = True

      if features.has_owner_encoded_split_renders:
        flags["owner_encoded_render_offset_fix"] = True

      if not features.has_bones and (
        features.has_owner_encoded_split_renders
        or (features.has_generic_render_chunks and features.has_shell_nodes)
      ):
        flags["auto_raw_mesh_origin_v10_split"] = True

    if features.has_shell_nodes or features.has_segments_nodes:
      if pure_collision_payload:
        # Pure collision-only graphs already store shell/segment payloads in a
        # rootless, Y-up object space. Convert geometry directly instead of
        # inventing a synthetic +90 X scene object that leaves Blender objects
        # looking rotated even though the world-space mesh is correct.
        flags["collision_geometry_basis_fix"] = True
      else:
        flags["embedded_collision_scene_root_basis_object"] = True
        flags["v10_root_object_basis_fix"] = True

    if features.transform_root_type == "LodNode":
      flags["lod_root_scene_basis_object"] = True
      flags["v10_root_object_basis_fix"] = True

  if features.has_owner_encoded_split_renders:
    flags["owner_encoded_render_offset_fix"] = True

  if features.has_bones and features.edm_version >= 10:
    flags["bone_rest_requires_root_basis_fix"] = True

  if pure_collision_payload:
    name = "DERIVED_PURE_COLLISION_PAYLOAD"
    description = "Pure collision-only shell/segment graph with geometry-space basis conversion."
  elif features.has_shell_nodes or features.has_segments_nodes:
    name = "DERIVED_EMBEDDED_COLLISION"
    description = "Embedded collision-style graph derived from parsed shell/segment nodes."
  elif authored_scene_root_v10 and features.has_bones:
    name = "DERIVED_AUTHORED_SCENE_SKELETAL"
    description = "Scene-root-authored v10 graph with skeletal structure."
  elif authored_scene_root_v10:
    name = "DERIVED_AUTHORED_SCENE_RENDER"
    description = "Scene-root-authored v10 render/control graph."
  elif features.plain_root_v10 and features.has_bones:
    name = "DERIVED_PLAIN_ROOT_SKELETAL"
    description = "Plain-root v10 graph with bones."
  elif features.plain_root_v10:
    name = "DERIVED_PLAIN_ROOT_RENDER"
    description = "Plain-root v10 render/control graph."
  elif features.has_bano_material:
    name = "DERIVED_AUTHORED_SCENE_BANO"
    description = "Authored scene graph with bano material payload."
  else:
    name = "DERIVED_AUTHORED_SCENE"
    description = "Authored scene graph with explicit root payload."

  return ImportCapabilities(
    name=name,
    description=description,
    detail=_capability_detail(features),
    flags=flags,
  )
