from .core import *  # noqa: F401,F403
from .render_shell import _read_index_data, _read_vertex_data  # noqa: F401


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
