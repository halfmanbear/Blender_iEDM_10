import math

import bpy
from mathutils import Matrix, Vector

from ..edm_format.types import (
  NumberNode,
  RenderNode,
  SegmentsNode,
  ShellNode,
  SkinNode,
)
from .material_setup import _map_animated_uniforms_to_edmprops
from .mesh_create import _create_mesh
from .prelude import (
  _ROOT_BASIS_FIX,
  _import_ctx,
  _import_profile_flag,
  _set_edmprop,
  _set_official_special_type,
)


def _connector_property_lines(connector):
  lines = []
  for key, value in (getattr(connector, "props", None) or {}).items():
    if key.startswith("__"):
      continue
    if isinstance(value, str):
      lines.append("{} = '{}'".format(key, value.replace("'", "\\'")))
    elif isinstance(value, (int, float)):
      lines.append("{} = {}".format(key, repr(value)))
  return lines


def _apply_connector_properties(ob, connector):
  lines = _connector_property_lines(connector)
  if lines:
    _set_edmprop(ob, "CONNECTOR_EXT", "\n".join(lines))


def _create_connector_empty(connector):
  ob = bpy.data.objects.new(connector.name, None)
  ob.empty_display_type = "CUBE"
  ob.edm.is_connector = True
  _set_official_special_type(ob, 'CONNECTOR')
  return ob


def create_connector(connector):
  """Create an empty object representing a connector."""
  ob = _create_connector_empty(connector)
  _apply_connector_properties(ob, connector)

  bpy.context.collection.objects.link(ob)
  return ob


def _segment_vertices_and_edges(segments_node):
  verts = []
  edges = []

  for i, segment in enumerate(segments_node.data):
    v1 = Vector((segment[0], segment[1], segment[2]))
    v2 = Vector((segment[3], segment[4], segment[5]))
    verts.extend([v1, v2])
    edges.append([i*2, i*2+1])
  return verts, edges


def _create_segments_mesh(segments_node):
  mesh = bpy.data.meshes.new(segments_node.name)
  verts, edges = _segment_vertices_and_edges(segments_node)
  mesh.from_pydata(verts, edges, [])
  mesh.update()
  return mesh


def create_segments(segments_node):
  """Create a mesh with edges from a SegmentsNode (collision lines)."""
  mesh = _create_segments_mesh(segments_node)
  ob = bpy.data.objects.new(segments_node.name, mesh)
  ob.edm.is_collision_line = True
  ob.edm.is_renderable = False
  _set_official_special_type(ob, 'COLLISION_LINE')
  bpy.context.collection.objects.link(ob)

  return ob


def _is_morph_node(render_node):
  if render_node is None:
    return False
  if hasattr(render_node, "_morph_payload"):
    return True
  source_type = str(getattr(render_node, "_source_type_name", "") or "")
  return source_type == "model::MorphNode" or type(render_node).__name__ == "MorphNode"


def _preserve_morph_payload(blender_obj, render_node):
  payload = getattr(render_node, "_morph_payload", None)
  if not payload or blender_obj is None:
    return
  try:
    blender_obj["_iedm_translation_status"] = "preserve_only"
    blender_obj["_iedm_translation_source"] = "MorphNode"
    blender_obj["_iedm_morph_payload_size"] = int(len(payload))
    blender_obj["_iedm_morph_payload_sha1"] = __import__("hashlib").sha1(payload).hexdigest()
    blender_obj["_iedm_morph_payload_preview_hex"] = bytes(payload[:64]).hex()
    blender_obj["_iedm_morph_payload_vertex_count"] = int(len(getattr(render_node, "vertexData", None) or []))
    if len(payload) <= 2048:
      blender_obj["_iedm_morph_payload_b64"] = __import__("base64").b64encode(payload).decode("ascii")
      blender_obj["_iedm_morph_payload_storage"] = "inline_b64"
    else:
      safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in (blender_obj.name or "Morph"))
      text_name = "IEDM_MorphPayload_{}".format(safe_name[:48])
      text_block = bpy.data.texts.get(text_name)
      if text_block is None:
        text_block = bpy.data.texts.new(text_name)
      text_block.clear()
      text_block.write(__import__("base64").b64encode(payload).decode("ascii"))
      blender_obj["_iedm_morph_payload_text"] = text_name
      blender_obj["_iedm_morph_payload_storage"] = "text_b64"
  except Exception as e:
    print(f"Warning preserving MorphNode payload on {getattr(blender_obj, 'name', '')}: {e}")


def _decode_morph_payload_to_shape_keys(blender_obj, render_node, transform):
  payload = getattr(render_node, "_morph_payload", None)
  if not payload or blender_obj is None or getattr(blender_obj, "type", "") != "MESH":
    return 0
  vertex_count = int(len(getattr(render_node, "vertexData", None) or []))
  if vertex_count <= 0 or len(payload) % 12 != 0:
    return 0
  if len(getattr(blender_obj.data, "vertices", []) or []) != vertex_count:
    return 0

  vec3_count = len(payload) // 12
  if vec3_count < vertex_count or (vec3_count % vertex_count) != 0:
    return 0

  morph_count = vec3_count // vertex_count
  if morph_count <= 0 or morph_count > 8:
    return 0

  try:
    floats = __import__("struct").unpack("<{}f".format(len(payload) // 4), payload)
  except Exception:
    return 0

  if any(not math.isfinite(v) for v in floats):
    return 0

  basis_positions = [vert.co.copy() for vert in blender_obj.data.vertices]
  if len(basis_positions) != vertex_count:
    return 0

  bbox_extent = 0.0
  for pos in basis_positions:
    bbox_extent = max(bbox_extent, abs(float(pos.x)), abs(float(pos.y)), abs(float(pos.z)))
  delta_limit = max(10.0, bbox_extent * 10.0, 1.0)

  delta_transform = None
  if transform is not None:
    try:
      delta_transform = Matrix(transform).to_3x3()
    except Exception:
      delta_transform = None

  decoded = 0
  if getattr(blender_obj.data, "shape_keys", None) is None:
    try:
      blender_obj.shape_key_add(name="Basis", from_mix=False)
    except Exception:
      return 0

  for morph_index in range(morph_count):
    start = morph_index * vertex_count * 3
    max_abs_delta = 0.0
    deltas = []
    for vertex_index in range(vertex_count):
      offset = start + (vertex_index * 3)
      delta = Vector((floats[offset], floats[offset + 1], floats[offset + 2]))
      if delta_transform is not None:
        try:
          delta = delta_transform @ delta
        except Exception:
          pass
      max_abs_delta = max(max_abs_delta, abs(float(delta.x)), abs(float(delta.y)), abs(float(delta.z)))
      deltas.append(delta)

    if max_abs_delta <= 1.0e-8 or max_abs_delta > delta_limit:
      continue

    try:
      key_block = blender_obj.shape_key_add(name="Morph_{:03d}".format(morph_index), from_mix=False)
    except Exception:
      continue
    for vertex_index, delta in enumerate(deltas):
      key_block.data[vertex_index].co = basis_positions[vertex_index] + delta
    decoded += 1

  if decoded:
    try:
      blender_obj["_iedm_translation_status"] = "approximate"
      blender_obj["_iedm_translation_source"] = "MorphNode"
      blender_obj["_iedm_morph_decode_mode"] = "raw_vec3_per_vertex"
      blender_obj["_iedm_morph_shape_key_count"] = int(decoded)
    except Exception:
      pass
  return decoded


def _preserve_source_metadata(ob, node):
  try:
    source_type = getattr(node, "_source_type_name", None)
    if source_type:
      ob["_iedm_source_node_type"] = str(source_type)
    layout_variant = getattr(node, "_layout_variant", None)
    if layout_variant:
      ob["_iedm_layout_variant"] = str(layout_variant)
    shell_family = getattr(node, "_shell_family", None)
    if shell_family:
      ob["_iedm_shell_family"] = str(shell_family)
  except Exception as e:
    print(f"Warning in blender_importer\\object_create.py: {e}")


def _apply_number_node_projection(ob, node):
  payload = dict(getattr(node, "number_payload", None) or {})
  uv_params = payload.get("uv_params")
  raw = getattr(node, "number_params_raw", None)
  if uv_params and hasattr(ob, "EDMProps"):
    ob.EDMProps.NUMBER_UV_X_ARG = int(uv_params.get("x_arg", -1))
    ob.EDMProps.NUMBER_UV_X_SCALE = float(uv_params.get("x_scale", 0.0))
    ob.EDMProps.NUMBER_UV_Y_ARG = int(uv_params.get("y_arg", -1))
    ob.EDMProps.NUMBER_UV_Y_SCALE = float(uv_params.get("y_scale", 0.0))
  elif raw and hasattr(ob, "EDMProps"):
    ob.EDMProps.NUMBER_UV_X_ARG = int(raw[1])
    ob.EDMProps.NUMBER_UV_X_SCALE = float(raw[2])
    ob.EDMProps.NUMBER_UV_Y_ARG = int(raw[3])
    ob.EDMProps.NUMBER_UV_Y_SCALE = float(raw[4])

  try:
    ob["IEDM_NUMBER_VALUE"] = float(getattr(node, "value", 0.0))
    if raw:
      ob["IEDM_NUMBER_PARAMS_RAW"] = [
        int(raw[0]), int(raw[1]), float(raw[2]), int(raw[3]), float(raw[4])
      ]
    if payload:
      payload_idprops = {}
      for key in ("unknown_start", "material", "parent", "damage_argument", "index_prefix"):
        if key in payload and payload[key] is not None:
          payload_idprops[key] = int(payload[key])
      uv_payload = payload.get("uv_params") or {}
      if uv_payload:
        payload_idprops["uv_params"] = {
          "unknown": int(uv_payload.get("unknown", 0)),
          "x_arg": int(uv_payload.get("x_arg", -1)),
          "x_scale": float(uv_payload.get("x_scale", 0.0)),
          "y_arg": int(uv_payload.get("y_arg", -1)),
          "y_scale": float(uv_payload.get("y_scale", 0.0)),
        }
      if payload_idprops:
        ob["IEDM_NUMBER_PAYLOAD"] = payload_idprops
  except Exception as e:
    print(f"Warning preserving NumberNode payload on {ob.name}: {e}")


def _apply_render_special_type(ob, node, material):
  if isinstance(node, ShellNode):
    _set_official_special_type(ob, 'COLLISION_SHELL')
    ob.display_type = 'BOUNDS'
  elif isinstance(node, NumberNode):
    _set_official_special_type(ob, 'NUMBER_TYPE')
    _apply_number_node_projection(ob, node)
  else:
    material_name = str(getattr(material, "material_name", "") or "").lower()
    if material_name == "water_mask_material":
      _set_official_special_type(ob, 'WATER_MASK')
    else:
      _set_official_special_type(ob, 'UNKNOWN_TYPE')


def _apply_render_edmprops(ob, node, material):
  if not (hasattr(ob, "EDMProps") and isinstance(node, (RenderNode, NumberNode))):
    return
  dmg = getattr(node, "damage_argument", -1)
  if dmg is not None and dmg >= 0:
    ob.EDMProps.DAMAGE_ARG = int(dmg)
  try:
    ob.EDMProps.TWO_SIDED = bool(int(getattr(material, "culling", 0)) != 0)
  except Exception:
    pass


def _is_supported_mesh_node(node):
  return isinstance(node, (RenderNode, ShellNode, NumberNode, SkinNode))


def _vertex_format_for_node(node):
  if isinstance(node, (RenderNode, NumberNode, SkinNode)):
    return node.material.vertex_format
  if isinstance(node, ShellNode):
    return node.vertex_format
  return None


def _mesh_transform_for_node(node):
  if isinstance(node, SkinNode) and _import_ctx.edm_version >= 10:
    if _import_profile_flag("skin_mesh_geometry_root_basis_fix"):
      return _ROOT_BASIS_FIX
  elif isinstance(node, ShellNode) and getattr(_import_ctx, "collision_geometry_basis_fix", False):
    return _ROOT_BASIS_FIX
  return None


def _create_render_mesh(node, vertex_format, compact, mesh_transform):
  mesh = _create_mesh(
    node.vertexData,
    node.indexData,
    vertex_format,
    compact=compact,
    join_triangles=isinstance(node, ShellNode),
    transform=mesh_transform,
  )
  mesh.name = node.name
  return mesh


def create_object(node):
  """Accepts an EDM renderable and returns a Blender mesh object."""

  if not _is_supported_mesh_node(node):
    print("WARNING: Do not understand creating types {} yet".format(type(node)))
    return

  # Skip empty split children (owner-encoded splits with no triangles)
  if not node.indexData:
    return None

  is_morph_node = _is_morph_node(node)

  # Skinned meshes and morph payloads rely on original EDM vertex ordering.
  compact = not isinstance(node, SkinNode) and not is_morph_node
  vertex_format = _vertex_format_for_node(node)
  mesh_transform = _mesh_transform_for_node(node)
  mesh = _create_render_mesh(node, vertex_format, compact, mesh_transform)

  ob = bpy.data.objects.new(node.name, mesh)
  ob.edm.is_collision_shell = isinstance(node, ShellNode)
  ob.edm.is_renderable = isinstance(node, (RenderNode, NumberNode, SkinNode))
  ob.edm.damage_argument = -1 if not isinstance(node, (RenderNode, NumberNode)) else node.damage_argument
  mat = getattr(node, "material", None)
  _preserve_source_metadata(ob, node)
  _apply_render_special_type(ob, node, mat)
  _apply_render_edmprops(ob, node, mat)

  _map_animated_uniforms_to_edmprops(ob, node)

  bpy.context.collection.objects.link(ob)

  if is_morph_node:
    _preserve_morph_payload(ob, node)
    _decode_morph_payload_to_shape_keys(ob, node, mesh_transform)

  return ob
