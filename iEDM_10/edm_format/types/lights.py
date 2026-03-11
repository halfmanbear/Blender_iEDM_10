from .core import *  # noqa: F401,F403
from .render_shell import _read_index_data, _read_vertex_data  # noqa: F401


@reads_type("model::BillboardNode")
class BillboardNode(Node):
  @classmethod
  def read(cls, stream):
    self = super(BillboardNode, cls).read(stream)
    self.data = stream.read(154)
    return self

@reads_type("model::LightNode")
class LightNode(BaseNode):
  category = NodeCategory.light
  @classmethod
  def read(cls, stream):
    self = super(LightNode, cls).read(stream)
    self.parent = stream.read_uint()
    self.unknown = [stream.read_uchar()]
    # Preserve animation argument ids for light properties so importer can map
    # curves and EDMProps args exactly for official exporter round-trips.
    self.lightProps = PropertiesSet.read(stream, count=False, preserve_animated=True)
    self.unknown.append(stream.read_uchar())
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

    return self

  def prepare(self, nodes, materials):
    pass

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
    # Each FakeOmniLight: 6 doubles = [x, y, z, size, uv_lb_packed, uv_rt_packed]
    self.data = [stream.read_doubles(6) for _ in range(count)]
    stream.mark_type_read("model::FakeOmniLight", count)
    return self
  def prepare(self, nodes, materials):
    pass

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
    als_header = stream.read_uints(3)
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
    pass

