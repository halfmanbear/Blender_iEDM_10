"""Preserve EDM material payloads and exporter argument metadata."""

import json as json

from .export_damage_mask import DAMAGE_MASK_PROP, damage_mask_payload
from .material_animation import _logger, _resolve_animated_uniform_target


def _material_prop_scalar(value):
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return None


def _material_prop_value(value):
    scalar = _material_prop_scalar(value)
    if scalar is not None:
        return scalar
    if hasattr(value, "__iter__") and not isinstance(
        value, (str, bytes, bytearray, dict)
    ):
        try:
            return [_material_prop_value(item) for item in value]
        except Exception:
            return repr(value)
    return repr(value)


def _material_texture_payload(material):
    payload = []
    for tex in getattr(material, "textures", None) or []:
        entry = {
            "index": int(getattr(tex, "index", -1)),
            "name": str(getattr(tex, "name", "") or ""),
        }
        matrix = getattr(tex, "matrix", None)
        if matrix is not None:
            try:
                entry["matrix"] = [[float(v) for v in row] for row in matrix]
            except Exception:
                _logger.debug("Ignoring optional operation failure", exc_info=True)
        payload.append(entry)
    return payload


def _infer_material_texture_roles(material):
    textures = getattr(material, "textures", None) or []
    roles = {}
    material_name = str(getattr(material, "material_name", "") or "").lower()
    slot_names = {
        int(getattr(tex, "index", -1)): str(getattr(tex, "name", "") or "")
        for tex in textures
    }

    def _add(slot, role):
        if slot not in slot_names:
            return
        roles[role] = {"slot": int(slot), "name": slot_names[slot]}

    _add(0, "albedo")
    _add(1, "normal")
    _add(3, "decal")
    _add(8, "emissive")
    _add(9, "lightmap")
    if 10 in slot_names:
        _add(10, "damage_normal")
    elif 1 in slot_names:
        _add(1, "normal")
    if 13 in slot_names:
        _add(13, "roughmet")
    elif 2 in slot_names:
        _add(2, "roughmet")
    _add(14, "glass_filter")
    _add(18, "damage_mask")

    if material_name == "deck_material":
        _add(4, "decal_roughmet")
        _add(5, "wetmap")
        _add(7, "damage_albedo")
        if 9 in slot_names:
            roles.setdefault("damage_mask", {"slot": 9, "name": slot_names[9]})
    elif material_name == "glass_material":
        _add(5, "damage_albedo")
    else:
        _add(5, "damage_albedo")

    for slot, name in slot_names.items():
        lowered = name.lower()
        if "flir" in lowered and "flir" not in roles:
            roles["flir"] = {"slot": int(slot), "name": name}

    return roles


def _material_uniform_payload(props):
    return {
        str(key): _material_prop_value(value) for key, value in (props or {}).items()
    }


def _material_animated_uniform_payload(props):
    payload = {}
    for key, value in (props or {}).items():
        entry = {}
        arg = getattr(value, "argument", None)
        if arg is not None:
            try:
                entry["argument"] = int(arg)
            except Exception:
                entry["argument"] = _material_prop_value(arg)
        keys = []
        for framedata in getattr(value, "keys", []) or []:
            keys.append(
                {
                    "frame": _material_prop_value(getattr(framedata, "frame", None)),
                    "value": _material_prop_value(getattr(framedata, "value", None)),
                }
            )
        if keys:
            entry["keys"] = keys
        payload[str(key)] = entry or _material_prop_value(value)
    return payload


def _preserve_material_payload(mat, material):
    if mat is None or material is None:
        return
    try:
        mat["_iedm_material_name_raw"] = str(
            getattr(material, "material_name", "") or ""
        )
        mat["_iedm_material_label_raw"] = str(getattr(material, "name", "") or "")
        mat["_iedm_texture_channels"] = json.dumps(
            [
                int(v)
                for v in (getattr(material, "texture_coordinates_channels", None) or [])
            ],
            separators=(",", ":"),
        )
        mat["_iedm_texture_slots"] = json.dumps(
            _material_texture_payload(material), separators=(",", ":")
        )
        mat["_iedm_texture_roles"] = json.dumps(
            _infer_material_texture_roles(material),
            separators=(",", ":"),
            sort_keys=True,
        )
        mat["_iedm_uniforms"] = json.dumps(
            _material_uniform_payload(getattr(material, "uniforms", None) or {}),
            separators=(",", ":"),
            sort_keys=True,
        )
        mat["_iedm_animated_uniforms"] = json.dumps(
            _material_animated_uniform_payload(
                getattr(material, "animated_uniforms", None) or {}
            ),
            separators=(",", ":"),
            sort_keys=True,
        )
        damage_mask = damage_mask_payload(material)
        if damage_mask is not None:
            mat[DAMAGE_MASK_PROP] = json.dumps(damage_mask, sort_keys=True)
    except Exception as e:
        print(f"Warning in blender_importer/material_setup.py: {e}")


def _preserve_object_material_args(ob, node):
    mat = getattr(node, "material", None)
    anim_uniforms = getattr(mat, "animated_uniforms", None) or {}
    unknown_args = {}
    translated_args = {}
    for name, prop in anim_uniforms.items():
        arg = getattr(prop, "argument", None)
        if arg is None:
            continue
        target = _resolve_animated_uniform_target(name)
        if target is not None:
            translated_args[str(name)] = {
                "mode": str(target.get("mode", "")),
                "category": str(target.get("category", "") or ""),
                "socket": str(target.get("socket", "") or ""),
                "edmprop": str(target.get("edmprop", "") or ""),
            }
            if target.get("mode") == "exact":
                continue
        try:
            unknown_args[str(name)] = int(arg)
        except Exception:
            unknown_args[str(name)] = _material_prop_value(arg)
    if translated_args:
        try:
            ob["_iedm_anim_uniform_translation"] = json.dumps(
                translated_args, separators=(",", ":"), sort_keys=True
            )
        except Exception as e:
            print(f"Warning in blender_importer/material_setup.py: {e}")
    if unknown_args:
        try:
            ob["_iedm_anim_uniform_args"] = json.dumps(
                unknown_args, separators=(",", ":"), sort_keys=True
            )
        except Exception as e:
            print(f"Warning in blender_importer/material_setup.py: {e}")


def _map_animated_uniforms_to_edmprops(ob, node):
    """Extract animated uniform argument numbers and set EDMProps ARG fields."""
    if not hasattr(ob, "EDMProps"):
        return
    mat = getattr(node, "material", None)
    if mat is None:
        return
    anim_uniforms = getattr(mat, "animated_uniforms", None)
    if not anim_uniforms:
        return
    from ..edm_format.typereader import AnimatedProperty

    for name, prop in anim_uniforms.items():
        if not isinstance(prop, AnimatedProperty):
            continue
        target = _resolve_animated_uniform_target(name)
        edmprops_field = target.get("edmprop") if target else None
        if edmprops_field and hasattr(ob.EDMProps, edmprops_field):
            if prop.argument is not None and prop.argument >= 0:
                try:
                    # Clamp to 32-bit signed int limit for Blender's IntProperty
                    val = min(2147483647, int(prop.argument))
                    setattr(ob.EDMProps, edmprops_field, val)
                except (OverflowError, ValueError):
                    pass
    _preserve_object_material_args(ob, node)
