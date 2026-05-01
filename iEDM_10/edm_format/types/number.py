from .core import *  # noqa: F401,F403
from .render_shell import _read_index_data, _read_vertex_data  # noqa: F401
import struct


def _peek_lookup_token(stream):
  if not getattr(stream, "v10", False) or not getattr(stream, "strings", None):
    return None
  pos = stream.tell()
  try:
    token = stream.read_string()
  except Exception:
    token = None
  stream.seek(pos)
  return token


def _number_boundary_tokens(stream):
  if not getattr(stream, "strings", None):
    return set()
  return {
    "CONNECTORS",
    "LIGHT_NODES",
    "RENDER_NODES",
    "SHELL_NODES",
    "model::FakeOmniLightsNode",
    "model::AnimatedFakeOmniLightsNode",
    "model::FakeSpotLightsNode",
    "model::AnimatedFakeSpotLightsNode",
    "model::FakeSpotLights3Node",
    "model::NumberRoot",
    "model::TmpNumberRoot",
    "model::NumberNode",
    "model::RenderNode",
    "model::SegmentsNode",
    "model::ShellNode",
    "model::ShellSkinNode",
    "model::SkinNode",
    "model::TreeShellNode",
  }


def _scan_to_next_number_boundary(stream, max_bytes=1 << 20):
  if not getattr(stream, "v10", False) or not getattr(stream, "strings", None):
    return None, None

  boundary_tokens = _number_boundary_tokens(stream)
  boundary_indices = {
    i: token for i, token in enumerate(stream.strings)
    if token in boundary_tokens
  }
  if not boundary_indices:
    return None, None

  start = stream.tell()
  data = stream.read(max_bytes)
  stream.seek(start)
  if len(data) < 8:
    return None, None

  for off in range(1, len(data) - 8):
    idx = struct.unpack_from("<I", data, off)[0]
    token = boundary_indices.get(idx)
    if token is None:
      continue

    if token.startswith("model::"):
      if off + 16 > len(data):
        continue
      name_len = struct.unpack_from("<I", data, off + 4)[0]
      if name_len > 255:
        continue
      version_off = off + 8 + name_len
      props_off = version_off + 4
      if props_off + 4 > len(data):
        continue
      version = struct.unpack_from("<I", data, version_off)[0]
      props_len = struct.unpack_from("<I", data, props_off)[0]
      if version > 16 or props_len > 4096:
        continue
      return off, token

    if off + 8 > len(data):
      continue
    length = struct.unpack_from("<I", data, off + 4)[0]
    if length > 100000:
      continue
    return off, token

  return None, None


def _looks_like_next_number_boundary(stream):
  token = _peek_lookup_token(stream)
  if token in _number_boundary_tokens(stream):
    return True
  span, _boundary = _scan_to_next_number_boundary(stream, max_bytes=256)
  return span == 0


@reads_type("model::NumberNode")
class NumberNode(BaseNode):
  category = NodeCategory.render

  @classmethod
  def read(cls, stream):
    # Reads the standard BaseNode header (Name, Version, Props)
    self = super(NumberNode, cls).read(stream)
    # Placeholder record used by v10 files; geometry/payload can follow later.
    self.value = stream.read_float()
    self.parent = None
    self.material = None
    self.vertexData = []
    self.indexData = []
    self.unknown_indexPrefix = 5
    self.damage_argument = -1
    self.number_params_raw = None
    self.number_payload = {}
    self._post_payload_read = False
    self.inline_payload_raw = None

    inline_token = _peek_lookup_token(stream)
    if stream.v10 and inline_token and inline_token not in _number_boundary_tokens(stream):
      # Payload appears to be inline (next bytes are not a node-type boundary).
      # Attempt structured parsing first. read_v10_payload() validates the stream
      # layout via __gv_bytes/__gi_bytes assertions, so false positives (where u0
      # coincidentally matches a string index but the data is post-section format)
      # will raise and fall back to the boundary-scan skip path.
      pos = stream.tell()
      try:
        self.read_v10_payload(stream)
        if not _looks_like_next_number_boundary(stream):
          raise IOError("NumberNode inline payload did not end on a valid next-node boundary")
      except Exception as exc:
        stream.seek(pos)
        logger.warning(
          "NumberNode inline payload parse failed for '%s' (%s: %s); "
          "falling back to boundary scan skip.",
          getattr(self, "name", ""),
          type(exc).__name__,
          exc,
        )
        span, boundary_token = _scan_to_next_number_boundary(stream)
        if span:
          self.inline_payload_raw = stream.read(span)
          self.number_payload = {
            "mode": "inline_skipped",
            "bytes": int(span),
            "first_token": str(inline_token),
            "next_boundary": str(boundary_token),
          }
          self._post_payload_read = True
        else:
          logger.warning(
            "Detected inline NumberNode payload starting with '%s' for '%s' "
            "but could not find the next node boundary.",
            inline_token,
            getattr(self, "name", ""),
          )
    return self

  def read_v10_payload(self, stream):
    """Read post-object payload used by v10 NumberNode records."""
    # Different v10 files appear to swap these first two uints
    # (unknown_start/material). Pick the material candidate that best matches
    # known NumberNode material signatures when possible.
    u0 = stream.read_uint()
    u1 = stream.read_uint()
    self.unknown_start = u0
    self.material = u1

    mats = getattr(stream, "materials", None)
    if mats:
      valid0 = 0 <= u0 < len(mats)
      valid1 = 0 <= u1 < len(mats)

      if valid0 and not valid1:
        self.material = u0
        self.unknown_start = u1
      elif valid0 and valid1 and u0 != u1:
        def _score_material(idx):
          mat = mats[idx]
          textures = getattr(mat, "textures", None) or []
          score = len(textures)
          if any(getattr(tex, "index", -1) == 3 for tex in textures):
            score += 100
          if getattr(mat, "material_name", "") == "def_material":
            score += 5
          return score

        if _score_material(u0) > _score_material(u1):
          self.material = u0
          self.unknown_start = u1

    self.parent = stream.read_uint()
    self.damage_argument = stream.read_int()
    self.vertexData = _read_vertex_data(stream, "__gv_bytes")
    self.unknown_indexPrefix, self.indexData = _read_index_data(stream, classification="__gi_bytes")
    self.number_params_raw = (
      stream.read_int(),
      stream.read_int(),
      stream.read_int(),
      stream.read_int(),
      stream.read_float(),
    )
    self.number_payload = {
      "unknown_start": self.unknown_start,
      "material": self.material,
      "parent": self.parent,
      "damage_argument": self.damage_argument,
      "index_prefix": self.unknown_indexPrefix,
      "uv_params": {
        "unknown": int(self.number_params_raw[0]),
        "x_arg": int(self.number_params_raw[1]),
        "x_scale": float(self.number_params_raw[2]),
        "y_arg": int(self.number_params_raw[3]),
        "y_scale": float(self.number_params_raw[4]),
      },
    }
    self._post_payload_read = True
