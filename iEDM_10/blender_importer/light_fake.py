import json
import logging
import math

import bpy
from mathutils import Vector

from ..edm_format.mathtypes import vector_to_blender
from .light_animation import (
    _apply_animated_fake_omni_brightness,
    _apply_fake_light_animation_payload,
    _has_animated_fake_omni_payload,
)
from .light_geometry import (
    _add_fake_spot_direction_child,
    _classify_fake_spot_mode,
    _create_axis_box_mesh,
    _create_fake_light_mesh,
    _create_fake_omni_mesh,
    _default_fake_spot_uvs,
    _fake_light_world_from_edm,
)
from .light_materials import _material_for_fake_light
from .prelude import (
    _set_official_special_type,
)

_logger = logging.getLogger(__name__)


def create_fake_omni_lights(node):
    """Create a non-surface fake omni object compatible with the official exporter."""
    name = node.name or "FakeOmniLights"

    positions = []
    decoded_entries = list(getattr(node, "decoded_data", []) or [])
    if not decoded_entries:
        try:
            from ..edm_format.types.lights import decode_fake_omni_entry

            decoded_entries = [
                decode_fake_omni_entry(entry) for entry in getattr(node, "data", [])
            ]
        except Exception:
            decoded_entries = []

    size = None
    uv_lb = None
    uv_rt = None
    for decoded in decoded_entries:
        pos_edm = decoded.get("position", (0.0, 0.0, 0.0))
        positions.append(_fake_light_world_from_edm(pos_edm))
        if size is None:
            size = decoded.get("size")
            uv_lb = decoded.get("uv_lb")
            uv_rt = decoded.get("uv_rt")
    if not positions:
        for entry in getattr(node, "data", []) or []:
            positions.append(_fake_light_world_from_edm(entry[0:3]))

    if not positions:
        # No light entries — create an empty placeholder
        ob = bpy.data.objects.new(name, None)
        ob.empty_display_type = "SPHERE"
        ob.empty_display_size = 0.1
        _set_official_special_type(ob, "FAKE_LIGHT")
        bpy.context.collection.objects.link(ob)
        return ob

    mesh, location = _create_fake_omni_mesh(name, positions)
    ob = bpy.data.objects.new(name, mesh)
    ob.location = location
    _set_official_special_type(ob, "FAKE_LIGHT")

    # Assign dedicated fake omni material with correct EDM node group
    fake_mat = _material_for_fake_light(node, name, kind="fake_omni")
    if fake_mat is not None:
        mesh.materials.clear()
        mesh.materials.append(fake_mat)

    # Set EDMProps for fake light export
    if hasattr(ob, "EDMProps"):
        ob.EDMProps.SIZE = float(size) if size is not None else 3.0
        ob.EDMProps.SURFACE_MODE = False
        if uv_lb is not None:
            ob.EDMProps.UV_LB = (float(uv_lb[0]), float(uv_lb[1]))
        if uv_rt is not None:
            ob.EDMProps.UV_RT = (float(uv_rt[0]), float(uv_rt[1]))

    if _has_animated_fake_omni_payload(node):
        _apply_animated_fake_omni_brightness(ob, node, len(positions))
    else:
        _apply_fake_light_animation_payload(ob, getattr(node, "material", None))

    bpy.context.collection.objects.link(ob)
    return ob


def _create_surface_spot_mesh(name, positions, dirs_bl, sizes):
    verts = []
    faces = []
    uv_quads = []
    for center, normal, size_val in zip(positions, dirs_bl, sizes, strict=False):
        up = Vector((0.0, 0.0, 1.0))
        if abs(normal.dot(up)) > 0.999:
            up = Vector((0.0, 1.0, 0.0))
        tangent = normal.cross(up)
        if tangent.length <= 1e-8:
            tangent = Vector((1.0, 0.0, 0.0))
        tangent.normalize()
        bitangent = normal.cross(tangent)
        if bitangent.length <= 1e-8:
            bitangent = Vector((0.0, 1.0, 0.0))
        bitangent.normalize()

        half_side = max(0.001, float(size_val) / (2.0 * math.sqrt(2.0)))
        quad = [
            center - tangent * half_side - bitangent * half_side,
            center + tangent * half_side - bitangent * half_side,
            center + tangent * half_side + bitangent * half_side,
            center - tangent * half_side + bitangent * half_side,
        ]
        base = len(verts)
        verts.extend([tuple(v) for v in quad])
        faces.append((base + 0, base + 1, base + 2, base + 3))
        uv_quads.append(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    if faces:
        try:
            uv_layer = mesh.uv_layers.new(name="UVMap")
            loop_uv = uv_layer.data
            for poly in mesh.polygons:
                q = uv_quads[poly.index]
                for li, uv in zip(poly.loop_indices, q, strict=False):
                    loop_uv[li].uv = uv
        except Exception as e:
            print(f"Warning in blender_importer/lights.py: {e}")
    return mesh


def create_fake_spot_lights(node):
    """Create a mesh object for FakeSpotLightsNode."""
    name = node.name or "FakeSpotLights"

    positions = []
    dirs_bl = []
    sizes = []
    two_sided = False
    for i, entry in enumerate(node.data):
        pos_edm = entry["position"]
        positions.append(vector_to_blender(pos_edm))

        # Direction is more reliable in parentData than the currently-partial raw
        # fake-spot entry parser.
        dir_edm = None
        if i < len(getattr(node, "parentData", [])):
            pd = node.parentData[i]
            if len(pd) >= 3:
                cand = pd[2]
                try:
                    if all(math.isfinite(float(v)) for v in cand):
                        dir_edm = tuple(float(v) for v in cand)
                except Exception:
                    dir_edm = None
        if dir_edm is None:
            cand = getattr(node, "trailing_direction", None)
            if cand is not None:
                try:
                    if all(math.isfinite(float(v)) for v in cand):
                        dir_edm = tuple(float(v) for v in cand)
                except Exception:
                    dir_edm = None
        if dir_edm is None:
            cand = entry.get("direction")
            try:
                if cand is not None and all(math.isfinite(float(v)) for v in cand):
                    dir_edm = tuple(float(v) for v in cand)
            except Exception:
                dir_edm = None
        if dir_edm is None:
            dir_edm = (1.0, 0.0, 0.0)

        dvec = vector_to_blender(dir_edm)
        if dvec.length <= 1e-8:
            dvec = Vector((1.0, 0.0, 0.0))
        else:
            dvec.normalize()
        dirs_bl.append(dvec)

        s = entry.get("size", 0.0)
        try:
            s = abs(float(s))
        except Exception:
            s = 0.0
        if not math.isfinite(s) or s <= 1e-6:
            s = 0.1
        sizes.append(s)
        if entry.get("back_side") is True:
            two_sided = True
        elif "flag" in entry:
            try:
                two_sided = two_sided or bool(int(entry.get("flag", 0)) & 0x1)
            except Exception:
                _logger.debug("Ignoring optional operation failure", exc_info=True)

    if not positions:
        ob = bpy.data.objects.new(name, None)
        ob.empty_display_type = "SPHERE"
        ob.empty_display_size = 0.1
        _set_official_special_type(ob, "FAKE_LIGHT")
        bpy.context.collection.objects.link(ob)
        return ob

    mode = _classify_fake_spot_mode(positions, dirs_bl)
    if mode == "non_surface_box":
        mesh = _create_axis_box_mesh(name, positions)
        if mesh is None:
            mesh = _create_fake_light_mesh(name, positions)
    elif mode == "non_surface_points":
        mesh = _create_fake_light_mesh(name, positions)
    else:
        mesh = _create_surface_spot_mesh(name, positions, dirs_bl, sizes)

    ob = bpy.data.objects.new(name, mesh)
    _set_official_special_type(ob, "FAKE_LIGHT")

    # Assign dedicated fake spot material with correct EDM node group
    fake_mat = _material_for_fake_light(node, name, kind="fake_spot")
    if fake_mat is not None:
        mesh.materials.clear()
        mesh.materials.append(fake_mat)

    if hasattr(ob, "EDMProps"):
        ob.EDMProps.SURFACE_MODE = mode == "surface"
        ob.EDMProps.TWO_SIDED = bool(two_sided)
        ob.EDMProps.SIZE = float(sizes[0]) if sizes else 0.1
        if mode != "surface":
            front_lb, front_rt, back_lb, back_rt = _default_fake_spot_uvs(
                bool(two_sided)
            )
            ob.EDMProps.UV_LB = front_lb
            ob.EDMProps.UV_RT = front_rt
            ob.EDMProps.UV_LB_BACK = back_lb
            ob.EDMProps.UV_RT_BACK = back_rt
    if _has_animated_fake_omni_payload(node):
        _apply_animated_fake_omni_brightness(
            ob, node, len(node.data), verts_per_light=4 if mode == "surface" else 1
        )
    else:
        _apply_fake_light_animation_payload(ob, getattr(node, "material", None))

    bpy.context.collection.objects.link(ob)
    if mode != "surface":
        direction = dirs_bl[0] if dirs_bl else Vector((1.0, 0.0, 0.0))
        _add_fake_spot_direction_child(
            ob, direction, distance=max(1.0, float(sizes[0]) * 0.5 if sizes else 1.0)
        )
    return ob


def create_fake_als_lights(node):
    """Create a mesh object for FakeALSNode with one vertex per light."""

    def _preserve_fake_als_metadata(obj, als_node):
        if obj is None or als_node is None:
            return
        try:
            obj["_iedm_translation_status"] = "approximate"
            obj["_iedm_translation_source"] = "FakeALSNode"
            obj["_iedm_fake_als_extra_preserved"] = True
            obj["_iedm_fake_als_count"] = int(
                len(getattr(als_node, "data", None) or [])
            )
            header = list(getattr(als_node, "als_header", ()) or ())
            if header:
                obj["_iedm_fake_als_header"] = [int(v) for v in header[:3]]

            payload = {
                "header": [int(v) for v in header[:3]],
                "entries": [
                    {
                        "position": [
                            float(v) for v in (entry.get("position") or ())[:3]
                        ],
                        "extra": [float(v) for v in (entry.get("extra") or ())[:7]],
                    }
                    for entry in (getattr(als_node, "data", None) or [])
                ],
            }
            encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            if len(encoded) <= 60000:
                obj["_iedm_fake_als_payload"] = encoded
                obj["_iedm_fake_als_payload_storage"] = "inline_json"
            else:
                safe_name = "".join(
                    ch if (ch.isalnum() or ch in "._-") else "_"
                    for ch in (obj.name or "FakeALS")
                )
                text_name = "IEDM_FakeALS_{}".format(safe_name[:48])
                text_block = bpy.data.texts.get(text_name)
                if text_block is None:
                    text_block = bpy.data.texts.new(text_name)
                text_block.clear()
                text_block.write(encoded)
                obj["_iedm_fake_als_payload_text"] = text_name
                obj["_iedm_fake_als_payload_storage"] = "text_json"
        except Exception as e:
            print(
                "Warning preserving FakeALSNode payload on "
                f"{getattr(obj, 'name', '')}: {e}"
            )

    name = node.name or "FakeALSLights"

    positions = []
    for entry in node.data:
        pos_edm = entry["position"]
        positions.append(vector_to_blender(pos_edm))

    if not positions:
        ob = bpy.data.objects.new(name, None)
        ob.empty_display_type = "SPHERE"
        ob.empty_display_size = 0.1
        _set_official_special_type(ob, "FAKE_LIGHT")
        _preserve_fake_als_metadata(ob, node)
        bpy.context.collection.objects.link(ob)
        return ob

    mesh = _create_fake_light_mesh(name, positions)
    ob = bpy.data.objects.new(name, mesh)
    _set_official_special_type(ob, "FAKE_LIGHT")

    # ALS lights map to fake_omni material kind
    fake_mat = _material_for_fake_light(node, name, kind="fake_omni")
    if fake_mat is not None:
        mesh.materials.clear()
        mesh.materials.append(fake_mat)
    _apply_fake_light_animation_payload(ob, getattr(node, "material", None))
    _preserve_fake_als_metadata(ob, node)

    bpy.context.collection.objects.link(ob)
    return ob
