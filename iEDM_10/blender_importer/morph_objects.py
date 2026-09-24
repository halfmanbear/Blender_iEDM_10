"""Preserve morph payloads and decode them into Blender shape keys."""

import math as math

import bpy as bpy
from mathutils import Matrix, Vector

from .import_context import _log


def _is_morph_node(render_node):
    if render_node is None:
        return False
    if hasattr(render_node, "_morph_payload"):
        return True
    source_type = str(getattr(render_node, "_source_type_name", "") or "")
    return (
        source_type == "model::MorphNode" or type(render_node).__name__ == "MorphNode"
    )


def _preserve_morph_payload(blender_obj, render_node):
    payload = getattr(render_node, "_morph_payload", None)
    if not payload or blender_obj is None:
        return
    try:
        blender_obj["_iedm_translation_status"] = "preserve_only"
        blender_obj["_iedm_translation_source"] = "MorphNode"
        blender_obj["_iedm_morph_payload_size"] = int(len(payload))
        blender_obj["_iedm_morph_payload_sha1"] = (
            __import__("hashlib").sha1(payload).hexdigest()
        )
        blender_obj["_iedm_morph_payload_preview_hex"] = bytes(payload[:64]).hex()
        blender_obj["_iedm_morph_payload_vertex_count"] = int(
            len(getattr(render_node, "vertexData", None) or [])
        )
        if len(payload) <= 2048:
            blender_obj["_iedm_morph_payload_b64"] = (
                __import__("base64").b64encode(payload).decode("ascii")
            )
            blender_obj["_iedm_morph_payload_storage"] = "inline_b64"
        else:
            safe_name = "".join(
                ch if (ch.isalnum() or ch in "._-") else "_"
                for ch in (blender_obj.name or "Morph")
            )
            text_name = "IEDM_MorphPayload_{}".format(safe_name[:48])
            text_block = bpy.data.texts.get(text_name)
            if text_block is None:
                text_block = bpy.data.texts.new(text_name)
            text_block.clear()
            text_block.write(__import__("base64").b64encode(payload).decode("ascii"))
            blender_obj["_iedm_morph_payload_text"] = text_name
            blender_obj["_iedm_morph_payload_storage"] = "text_b64"
    except Exception as e:
        print(
            "Warning preserving MorphNode payload on "
            f"{getattr(blender_obj, 'name', '')}: {e}"
        )


def _decode_morph_payload_to_shape_keys(blender_obj, render_node, transform):
    payload = getattr(render_node, "_morph_payload", None)
    prepared = _prepare_morph_payload(blender_obj, render_node, payload)
    if prepared is None:
        return 0
    vertex_count, morph_count, floats, basis_positions, delta_limit = prepared

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
        max_abs_delta, deltas = _decode_morph_deltas(
            floats, start, vertex_count, delta_transform
        )

        if max_abs_delta <= 1.0e-8 or max_abs_delta > delta_limit:
            continue

        try:
            key_block = blender_obj.shape_key_add(
                name="Morph_{:03d}".format(morph_index), from_mix=False
            )
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
        except Exception as exc:
            _log.debug("Optional operation failed: {}".format(exc), level=2)
    return decoded


def _prepare_morph_payload(blender_obj, render_node, payload):
    """Validate packed shape-key deltas and collect the base mesh positions."""
    if not payload or blender_obj is None or getattr(blender_obj, "type", "") != "MESH":
        return None
    vertex_count = len(getattr(render_node, "vertexData", None) or [])
    if vertex_count <= 0 or len(payload) % 12 != 0:
        return None
    if len(getattr(blender_obj.data, "vertices", []) or []) != vertex_count:
        return None
    vector_count = len(payload) // 12
    if vector_count < vertex_count or vector_count % vertex_count:
        return None
    morph_count = vector_count // vertex_count
    if not 0 < morph_count <= 8:
        return None
    try:
        floats = __import__("struct").unpack("<{}f".format(len(payload) // 4), payload)
    except Exception:
        return None
    if any(not math.isfinite(value) for value in floats):
        return None
    positions = [vertex.co.copy() for vertex in blender_obj.data.vertices]
    extent = max(
        (abs(float(component)) for pos in positions for component in pos),
        default=0.0,
    )
    return vertex_count, morph_count, floats, positions, max(10.0, extent * 10.0)


def _decode_morph_deltas(floats, start, vertex_count, transform):
    """Decode and optionally transform one shape's per-vertex displacement."""
    deltas = []
    max_delta = 0.0
    for vertex_index in range(vertex_count):
        offset = start + vertex_index * 3
        delta = Vector((floats[offset], floats[offset + 1], floats[offset + 2]))
        if transform is not None:
            try:
                delta = transform @ delta
            except Exception as exc:
                _log.debug("Optional operation failed: {}".format(exc), level=2)
        max_delta = max(max_delta, *(abs(float(value)) for value in delta))
        deltas.append(delta)
    return max_delta, deltas
