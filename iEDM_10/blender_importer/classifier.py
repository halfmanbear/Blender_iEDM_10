"""Best-effort legacy exporter-family classification.

This classifier is intentionally advisory. The importer now derives its primary
behavior from the parsed EDM graph rather than raw-byte exporter-family guesses.
"""

import struct


EDM_MAGIC = b"EDM"

CLASS_3DSMAX_BANO = "3DSMAX_BANO"
CLASS_3DSMAX_DEF = "3DSMAX_DEF"
CLASS_BLENDER_DEF = "BLENDER_DEF"
CLASS_COLLISION_MESH = "COLLISION_MESH"
CLASS_BLOB_COLLISION = "BLOB_COLLISION"
CLASS_BLOB_RENDER = "BLOB_RENDER"
CLASS_BLOB_SKELETAL = "BLOB_SKELETAL"
CLASS_UNKNOWN = "UNKNOWN"
CLASS_INVALID = "INVALID"


def _read_string_table(data):
  """Return non-empty null-terminated strings from the EDM string table."""
  if len(data) < 9:
    return []
  strtab_size = struct.unpack_from("<I", data, 5)[0]
  blob = data[9: 9 + strtab_size]
  strings = []
  pos = 0
  while pos < len(blob):
    null = blob.find(b"\x00", pos)
    if null == -1:
      break
    text = blob[pos:null].decode("latin-1")
    if text:
      strings.append(text)
    pos = null + 1
  return strings


def _get_root_node_name(data, strtab_size):
  """Read the root node name from the binary data section when present."""
  offset = 9 + strtab_size + 12
  if offset + 4 > len(data):
    return ""
  name_len = struct.unpack_from("<I", data, offset)[0]
  if name_len == 0 or name_len > 64 or offset + 4 + name_len > len(data):
    return ""
  raw = data[offset + 4: offset + 4 + name_len]
  if any(b < 0x20 for b in raw):
    return ""
  return raw.decode("latin-1", errors="replace")


def classify_edm_bytes(data):
  """Classify an EDM file payload by exporter family / content generation."""
  if not data.startswith(EDM_MAGIC):
    return CLASS_INVALID, "missing EDM magic"
  if len(data) < 9:
    return CLASS_INVALID, "file too short"

  strings = _read_string_table(data)
  strtab_size = struct.unpack_from("<I", data, 5)[0]
  if not strings:
    return CLASS_UNKNOWN, "empty string table"

  str0 = strings[0]
  str1 = strings[1] if len(strings) > 1 else ""
  str3 = strings[3] if len(strings) > 3 else ""
  str8 = strings[8] if len(strings) > 8 else ""

  has_gv = any(s in ("__gv_bytes", "__gi_bytes") for s in strings)
  if has_gv or str0 in ("model::Node", "model::RenderNode"):
    has_skeletal = "model::ArgAnimatedBone" in strings or "model::Bone" in strings
    if has_skeletal:
      detail = f"str[0]={str0!r}, has_gv_bytes={has_gv}, has_skeletal=True"
      return CLASS_BLOB_SKELETAL, detail
    return CLASS_BLOB_RENDER, f"str[0]={str0!r}, has_gv_bytes={has_gv}"

  if str0 != "model::RootNode":
    return CLASS_UNKNOWN, f"unexpected str[0]={str0!r}"

  has_cv = any(s in ("__cv_bytes", "__ci_bytes") for s in strings)
  if has_cv:
    return CLASS_BLOB_COLLISION, f"str[1]={str1!r}, has_cv_bytes=True"

  if str3 != "BLENDING":
    root_name = _get_root_node_name(data, strtab_size)
    return CLASS_COLLISION_MESH, (
      f"str[1]={str1!r}, str[3]={str3!r}, root_node={root_name!r}"
    )

  root_name = _get_root_node_name(data, strtab_size)
  has_bano = b"bano_material\x00" in data
  has_embedded_collision = "SHELL_NODES" in strings or "model::ShellNode" in strings
  detail = (
    f"str[8]={str8!r}, primary_mat={str8!r}, root_node={root_name!r}, "
    f"bano_in_file={has_bano}, embedded_collision={has_embedded_collision}"
  )

  if str8 == "bano_material":
    return CLASS_3DSMAX_BANO, detail
  if root_name == "Scene Root":
    return CLASS_3DSMAX_DEF, detail
  return CLASS_BLENDER_DEF, detail


def classify_edm_file(path):
  try:
    with open(path, "rb") as handle:
      data = handle.read()
  except OSError as exc:
    return CLASS_INVALID, f"read error: {exc}"
  return classify_edm_bytes(data)
