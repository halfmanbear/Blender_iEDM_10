"""Decode fake-light record values without scene effects."""

import struct


def _unpack_float_pair_from_double(dval):
    raw = struct.pack("<d", float(dval))
    return struct.unpack("<2f", raw)


def decode_fake_omni_entry(entry):
    """
    Decode a v10 fake-omni record into importer-friendly values.

    The official exporter-facing Blender representation does not match the raw
    on-disk layout one-to-one. In reference assets, the last three doubles behave
    like packed float pairs:
    - entry[3] -> UV_LB
    - entry[4].x -> UV_RT.x
    - entry[5].x -> SIZE
    """
    if isinstance(entry, dict):
        position = tuple(float(v) for v in (entry.get("position") or (0.0, 0.0, 0.0)))
        uv0 = entry.get("uv0")
        uv1 = entry.get("uv1")
        decoded = {
            "position": position,
            "size": float(entry.get("size", 0.0) or 0.0),
            "uv_lb": tuple(float(v) for v in uv0) if uv0 is not None else None,
            "uv_rt": tuple(float(v) for v in uv1) if uv1 is not None else None,
            "face_arg": int(entry.get("face_arg", 0) or 0),
            "layout": "v10_mixed",
        }
        decoded["legacy_size"] = decoded["size"]
        decoded["legacy_uv_lb"] = decoded["uv_lb"]
        decoded["legacy_uv_rt"] = decoded["uv_rt"]
        return decoded

    values = tuple(float(v) for v in entry)
    decoded = {
        "position": values[0:3],
        "legacy_size": values[3] if len(values) > 3 else 0.0,
        "legacy_uv_lb": _unpack_float_pair_from_double(values[4])
        if len(values) > 4
        else None,
        "legacy_uv_rt": _unpack_float_pair_from_double(values[5])
        if len(values) > 5
        else None,
        "layout": "legacy",
    }
    if len(values) < 6:
        decoded["size"] = decoded["legacy_size"]
        decoded["uv_lb"] = decoded["legacy_uv_lb"]
        decoded["uv_rt"] = decoded["legacy_uv_rt"]
        return decoded

    packed_lb = _unpack_float_pair_from_double(values[3])
    packed_rt = _unpack_float_pair_from_double(values[4])
    packed_size = _unpack_float_pair_from_double(values[5])

    uv_lb = (float(packed_lb[0]), float(packed_lb[1]))
    uv_rt_y = float(packed_rt[1])
    if abs(uv_rt_y) <= 1.0e-6 and abs(uv_lb[1]) > 1.0e-6:
        uv_rt_y = 1.0
    uv_rt = (float(packed_rt[0]), uv_rt_y)
    size = float(packed_size[0])

    if (
        size > 1.0e-6
        and all(abs(v) <= 16.0 for v in uv_lb + uv_rt)
        and (
            size > abs(decoded["legacy_size"]) or abs(decoded["legacy_size"]) <= 1.0e-4
        )
    ):
        decoded["layout"] = "packed_props"
        decoded["size"] = size
        decoded["uv_lb"] = uv_lb
        decoded["uv_rt"] = uv_rt
        decoded["packed_uv_lb"] = packed_lb
        decoded["packed_uv_rt"] = packed_rt
        decoded["packed_size"] = packed_size
        return decoded

    decoded["size"] = decoded["legacy_size"]
    decoded["uv_lb"] = decoded["legacy_uv_lb"]
    decoded["uv_rt"] = decoded["legacy_uv_rt"]
    return decoded


