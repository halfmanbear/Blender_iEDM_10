import struct

from .core import *  # noqa: F401,F403
from .render_shell import _read_index_data, _read_vertex_data  # noqa: F401


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


def _unpack_float_pair_from_double(dval):
  raw = struct.pack('<d', float(dval))
  return struct.unpack('<2f', raw)


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
  values = tuple(float(v) for v in entry)
  decoded = {
    "position": values[0:3],
    "legacy_size": values[3] if len(values) > 3 else 0.0,
    "legacy_uv_lb": _unpack_float_pair_from_double(values[4]) if len(values) > 4 else None,
    "legacy_uv_rt": _unpack_float_pair_from_double(values[5]) if len(values) > 5 else None,
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
    and (size > abs(decoded["legacy_size"]) or abs(decoded["legacy_size"]) <= 1.0e-4)
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


def summarize_billboard_payload(raw):
  """Return a small importer-facing summary for opaque BillboardNode payloads."""
  data = bytes(raw or b"")
  summary = {
    "payload_len": len(data),
    "payload_hex": data.hex(),
  }
  if len(data) >= 16:
    try:
      summary["head_u32"] = struct.unpack_from("<4I", data, 0)
    except Exception:
      pass
    try:
      summary["head_f32"] = struct.unpack_from("<4f", data, 0)
    except Exception:
      pass
  if len(data) >= 2:
    try:
      summary["tail_u16"] = struct.unpack_from("<H", data, len(data) - 2)[0]
    except Exception:
      pass
  return summary


@reads_type("model::BillboardNode")
class BillboardNode(Node):
  category = NodeCategory.render

  @classmethod
  def read(cls, stream):
    self = super(BillboardNode, cls).read(stream)
    self.data = stream.read(154)
    self.payload_summary = summarize_billboard_payload(self.data)
    return self

@reads_type("model::LightNode")
class LightNode(BaseNode):
  category = NodeCategory.light
  @classmethod
  def read(cls, stream):
    self = super(LightNode, cls).read(stream)
    self.parent = stream.read_uint()
    self.unknown = [stream.read_uchar()]
    self.pre_props_flag = self.unknown[0]
    # Preserve animation argument ids for light properties so importer can map
    # curves and EDMProps args exactly for official exporter round-trips.
    self.lightProps = PropertiesSet.read(stream, count=False, preserve_animated=True)
    self.unknown.append(stream.read_uchar())
    self.post_props_flag = self.unknown[1]
    return self

@reads_type("model::FakeSpotLightsNode")
class FakeSpotLightsNode(BaseNode):
  category = NodeCategory.render
  @classmethod
  def read(cls, stream):
    self = super(FakeSpotLightsNode, cls).read(stream)

    # Seems to start relatively similar to renderNode
    self.unknown_start = stream.read_uint()
    self.material_ish = stream.read_uint()

    # We have parent-like blocks of two uints + three floats
    controlNodeCount = stream.read_uint()

    self.parentData = []
    self.parentData_float_offsets = []
    for _ in range(controlNodeCount):
      _u0 = stream.read_uint()
      _u1 = stream.read_uint()
      _vec_pos = stream.tell()
      _vec = stream.read_floats(3)
      self.parentData_float_offsets.append(_vec_pos)
      self.parentData.append([
          _u0,
          _u1,
          _vec
        ])
    # Control node seems to follow same rules as RenderNode
    if controlNodeCount:
      stream.mark_type_read('model::FSLNControlNode', controlNodeCount-1)

    dataCount = stream.read_uint()
    self.raw_data = [stream.read(65) for _ in range(dataCount)]
    stream.mark_type_read("model::FakeSpotLight", dataCount)

    # Parse raw 65-byte entries: 8 doubles (64 bytes) + 1 byte flag
    # Layout: pos(3) + dir(3) + size(1) + unknown(1) + flag(1 byte)
    self.data = []
    for raw in self.raw_data:
      doubles = struct.unpack_from('<8d', raw, 0)
      flag = raw[64]
      self.data.append({
        'position': doubles[0:3],
        'direction': doubles[3:6],
        'size': doubles[6],
        'unknown': doubles[7],
        'flag': flag,
      })

    # Some v10 FakeSpotLightsNode records carry an extra trailing direction
    # vector (3 floats) after the light entries. If we leave it unread the
    # parser desynchronizes and the next v10 string-table lookup sees 1.0
    # (0x3f800000) as a bogus string index.
    self.trailing_direction = None
    self.trailing_direction_offset = None
    self.trailing_blob = b""
    self.trailing_blob_offset = None
    if getattr(stream, "v10", False):
      pos = stream.tell()
      try:
        next_u = stream.read_uint()
      except Exception:
        next_u = None
      finally:
        stream.seek(pos)

      if next_u is not None and getattr(stream, "strings", None):
        looks_like_next_type = (
          0 <= next_u < len(stream.strings)
          and isinstance(stream.strings[next_u], str)
          and stream.strings[next_u].startswith("model::")
        )
        if not looks_like_next_type:
          tpos = stream.tell()
          try:
            trailing = stream.read_floats(3)
            next_after = stream.read_uint()
            looks_aligned_after = (
              0 <= next_after < len(stream.strings)
              and isinstance(stream.strings[next_after], str)
              and stream.strings[next_after].startswith("model::")
            )
            if looks_aligned_after:
              self.trailing_direction = trailing
              self.trailing_direction_offset = tpos
              # Keep the next token unread; we only wanted a look-ahead probe.
              stream.seek(tpos + 12)
            else:
              stream.seek(tpos)
          except Exception:
            stream.seek(tpos)

      preserve_from = stream.tell()
      skip = _scan_to_next_v10_type_token(stream)
      stream.seek(preserve_from)
      if skip is not None and skip > 0:
        self.trailing_blob_offset = preserve_from
        self.trailing_blob = stream.read(skip)

    return self

  def prepare(self, nodes, materials):
    self.material = _resolve_material_from_candidates(
      materials,
      [getattr(self, "material_ish", -1)],
      "fake_spot",
    )

@reads_type("model::AnimatedFakeSpotLightsNode")
class AnimatedFakeSpotLightsNode(FakeSpotLightsNode):
  pass

@reads_type("model::FakeSpotLights3Node")
class FakeSpotLights3Node(FakeSpotLightsNode):
  @classmethod
  def read(cls, stream):
    self = super(FakeSpotLights3Node, cls).read(stream)
    self.v3_directions = None
    self.v3_backside_flags = None
    self.v3_trailing_blob = b""
    self.v3_trailing_blob_offset = None

    # Most files appear layout-compatible with FakeSpotLightsNode.
    if _next_v10_token_looks_like_type(stream) is True:
      return self

    # Some variants carry an additional directions array (+ optional flags).
    pos = stream.tell()
    try:
      count = stream.read_uint()
      if 0 <= count <= 65535:
        dirs = [tuple(stream.read_floats(3)) for _ in range(count)]

        flags = None
        if _next_v10_token_looks_like_type(stream) is not True:
          fpos = stream.tell()
          try:
            flags = list(stream.read_uchars(count))
          except Exception:
            flags = None
          if _next_v10_token_looks_like_type(stream) is not True:
            stream.seek(fpos)
            flags = None

        if _next_v10_token_looks_like_type(stream) is True and count == len(getattr(self, "data", [])):
          self.v3_directions = dirs
          self.v3_backside_flags = flags
          for i, d in enumerate(dirs):
            self.data[i]["direction_v3"] = d
            # Prefer explicit v3 payload direction when available.
            self.data[i]["direction"] = d
          if flags is not None:
            for i, flag in enumerate(flags):
              self.data[i]["back_side"] = bool(flag)
          return self
    except Exception:
      pass

    # Restore and preserve any unknown trailing payload until the next model token.
    stream.seek(pos)
    skip = _scan_to_next_v10_type_token(stream)
    if skip is not None and skip > 0:
      self.v3_trailing_blob_offset = stream.tell()
      self.v3_trailing_blob = stream.read(skip)
      logger.warning(
        "FakeSpotLights3Node '%s': preserved %d trailing bytes",
        getattr(self, "name", ""),
        len(self.v3_trailing_blob),
      )
    return self

@reads_type("model::FakeOmniLightsNode")
class FakeOmniLightsNode(BaseNode):
  category = NodeCategory.render
  @classmethod
  def read(cls, stream):
    self = super(FakeOmniLightsNode, cls).read(stream)
    self.data_start = stream.read_uints(5)
    count = stream.read_uint()
    # Each FakeOmniLight is 6 doubles. The trailing 3 doubles are not a stable
    # "size + 2 packed UV pairs" tuple across exporter-facing representations,
    # so keep the raw tuple and let decode_fake_omni_entry() classify it.
    self.data = [stream.read_doubles(6) for _ in range(count)]
    self.decoded_data = [decode_fake_omni_entry(entry) for entry in self.data]
    stream.mark_type_read("model::FakeOmniLight", count)
    return self
  def prepare(self, nodes, materials):
    self.material = _resolve_material_from_candidates(
      materials,
      list(getattr(self, "data_start", ()) or ()),
      "fake_omni",
    )

@reads_type("model::AnimatedFakeOmniLightsNode")
class AnimatedFakeOmniLightsNode(FakeOmniLightsNode):
  pass

@reads_type("model::FakeALSNode")
class FakeALSNode(BaseNode):
  category = NodeCategory.render
  @classmethod
  def read(cls, stream):
    self = super(FakeALSNode, cls).read(stream)
    # batumi.edm 1138915 x 340
    self.als_header = stream.read_uints(3)
    count = stream.read_uint()
    self.raw_data = [stream.read(80) for _ in range(count)]
    stream.mark_type_read("model::FakeALSLight", count)

    # Parse raw 80-byte entries: 10 doubles
    # Layout: pos(3) + remaining(7) — positions are first 3 doubles
    self.data = []
    for raw in self.raw_data:
      doubles = struct.unpack_from('<10d', raw, 0)
      self.data.append({
        'position': doubles[0:3],
        'extra': doubles[3:10],
      })

    return self

  def prepare(self, nodes, materials):
    self.material = _resolve_material_from_candidates(
      materials,
      list(getattr(self, "als_header", ()) or ()),
      "fake_als",
    )

