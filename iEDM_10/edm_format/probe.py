"""Lightweight model-file identification helpers.

The DCS UniModelDesc DLL selects ``ClassReader20`` when the source path ends
with ``'2'`` (normally ``.edm2``), and uses the older EDM header/version path
for standard ``.edm`` files.  This module mirrors that distinction so the
importer can fail early with a precise message instead of attempting to parse an
EDM2 container as EDM v10 data.
"""

from dataclasses import dataclass
import os
import struct


EDM_MAGIC = b"EDM"


@dataclass(frozen=True)
class ModelFormatInfo:
  family: str
  version: int | None
  detail: str


class UnsupportedModelFormatError(IOError):
  """Raised when a model container is recognized but not supported."""


def identify_model_file(path):
  """Identify whether *path* looks like standard EDM or EDM2/ClassReader20.

  Returns ``ModelFormatInfo`` with ``family`` values:
  - ``"EDM"`` for classic EDM files with an ``EDM`` header.
  - ``"EDM2"`` for the ClassReader20 path, typically ``.edm2``.
  - ``"UNKNOWN"`` when the file does not match either known container.
  """

  lower_name = os.path.basename(path).lower()
  with open(path, "rb") as handle:
    header = handle.read(8)

  if lower_name.endswith(".edm2"):
    return ModelFormatInfo(
      family="EDM2",
      version=20,
      detail="filename uses .edm2 / ClassReader20 container path",
    )

  if header.startswith(EDM_MAGIC) and len(header) >= 5:
    version = struct.unpack_from("<H", header, 3)[0]
    return ModelFormatInfo(
      family="EDM",
      version=version,
      detail="classic EDM header",
    )

  if lower_name.endswith("2"):
    return ModelFormatInfo(
      family="EDM2",
      version=20,
      detail="filename ends with '2' (matches UniModelDesc ClassReader20 heuristic)",
    )

  return ModelFormatInfo(
    family="UNKNOWN",
    version=None,
    detail="unrecognized model container",
  )


def require_supported_import_format(path):
  """Validate that *path* is an importer-supported model container."""

  info = identify_model_file(path)
  if info.family == "EDM2":
    raise UnsupportedModelFormatError(
      "Unsupported model format: EDM2 / ClassReader20 identified "
      f"({info.detail}). This importer currently supports classic EDM v10 only."
    )
  if info.family != "EDM":
    raise UnsupportedModelFormatError(
      f"Unsupported model format: {info.detail}."
    )
  return info
