"""Resolve light material references and read shared fake-light payloads."""

from .core import logger


def _material_name_matches_kind(material, kind):
    mat_name = str(getattr(material, "material_name", "") or "").lower()
    if kind == "fake_omni":
        return mat_name in {"fake_omni_lights", "fake_omni_lights2", "fake_als_lights"}
    if kind == "fake_spot":
        return mat_name in {"fake_spot_lights"}
    if kind == "fake_als":
        return mat_name in {"fake_als_lights", "fake_omni_lights", "fake_omni_lights2"}
    return False


def _resolve_material_from_candidates(materials, candidates, kind):
    if not materials:
        return None
    valid = []
    for idx in candidates:
        try:
            idx_int = int(idx)
        except Exception:
            continue
        if 0 <= idx_int < len(materials):
            valid.append((idx_int, materials[idx_int]))

    for _idx, mat in valid:
        if _material_name_matches_kind(mat, kind):
            return mat

    matching = [mat for mat in materials if _material_name_matches_kind(mat, kind)]
    if len(matching) == 1:
        return matching[0]
    if valid:
        return valid[0][1]
    return matching[0] if matching else None


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _read_fake_omni_light(stream):
    return {
        "position": tuple(stream.read_doubles(3)),
        "uv0": tuple(stream.read_floats(2)),
        "uv1": tuple(stream.read_floats(2)),
        "size": stream.read_float(),
        "face_arg": stream.read_uint(),
    }


def _read_animated_fake_lights_payload(stream, node, node_label):
    # readAnimatedFakeOmniLights / readAnimatedFakeSpotLights read three fields
    # after the base node:
    #   uint32  arg_handle   — animation argument channel index (throws if 0xFFFFFFFF)
    #   uint32  sample_rate  — must equal 128
    #   uint32  data_count   — total float32 samples (= lightCount * 128)
    #   data_count * 4 bytes — raw float32 animation curve data
    node.anim_arg_handle = stream.read_uint()
    if node.anim_arg_handle == 0xFFFFFFFF:
        logger.warning("%s: invalid animation arg handle (0xFFFFFFFF)", node_label)
    node.anim_sample_rate = stream.read_uint()
    if node.anim_sample_rate != 128:
        logger.warning(
            "%s: unexpected sample rate %d (expected 128)",
            node_label,
            node.anim_sample_rate,
        )
    node.anim_data_count = stream.read_uint()
    node.anim_data_raw = stream.read(node.anim_data_count * 4)
    return node
