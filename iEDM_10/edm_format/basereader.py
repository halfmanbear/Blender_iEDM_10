"""
BaseReader

A very simple extended stream reader, with capability to read single or
arrays of standard types.

It additionally has functions to read a uint-prefixed string, and a
uint-prefixed list of some item, defined by the function passed in
"""

from __future__ import annotations

import struct
from typing import BinaryIO, Callable, Literal, TypeVar

from .mathtypes import Matrix, Quaternion, Vector, sequence_to_matrix

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Upper bound on any length/count prefix read from an EDM file. Real DCS
# assets never approach this; it exists to fail fast on corrupt or hostile
# input instead of attempting a multi-gigabyte allocation or a loop that
# runs until end-of-file.
MAX_REASONABLE_COUNT = 10_000_000


class EDMFormatError(IOError):
    """Raised when EDM binary data fails a structural sanity check."""


class BaseReader(object):
    filename: str
    stream: BinaryIO
    version: int | None
    # Set externally on v10 streams once the string lookup table has been
    # read (see EDMFile._read); None until then and on v8 streams.
    strings: list[str] | None = None

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.stream = open(filename, "rb")
        self.version = None

    def __enter__(self) -> "BaseReader":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
        self.close()
        return False

    def tell(self) -> int:
        return self.stream.tell()

    def seek(self, offset: int, from_what: int = 0) -> None:
        self.stream.seek(offset, from_what)

    def close(self) -> None:
        self.stream.close()

    @property
    def v8(self) -> bool:
        return self.version == 8

    @property
    def v10(self) -> bool:
        return self.version == 10

    def read_constant(self, data: bytes) -> None:
        filedata = self.stream.read(len(data))
        if not data == filedata:
            raise EDMFormatError(
                "Expected constant not encountered; {!r} != {!r}".format(
                    filedata, data
                )
            )

    def read(self, length: int) -> bytes:
        return self.stream.read(length)

    def read_uchar(self) -> int:
        return int(struct.unpack("B", self.stream.read(1))[0])

    def read_uchars(self, count: int) -> tuple[int, ...]:
        return struct.unpack("{}B".format(count), self.stream.read(1 * count))

    def read_ushort(self) -> int:
        return int(struct.unpack("<H", self.stream.read(2))[0])

    def read_ushorts(self, count: int) -> tuple[int, ...]:
        return struct.unpack("<{}H".format(count), self.stream.read(2 * count))

    def read_uint(self) -> int:
        """Read an unsigned integer from the data"""
        return int(struct.unpack("<I", self.stream.read(4))[0])

    def read_uints(self, count: int) -> tuple[int, ...]:
        """Read an unsigned integer from the data"""
        return struct.unpack("<{}I".format(count), self.stream.read(4 * count))

    def read_uint_be(self) -> int:
        """Read a big-endian unsigned integer from the data"""
        return int(struct.unpack(">I", self.stream.read(4))[0])

    def read_int(self) -> int:
        """Read a signed integer from the data"""
        return int(struct.unpack("<i", self.stream.read(4))[0])

    def read_ints(self, count: int) -> tuple[int, ...]:
        """Read a signed integer from the data"""
        return struct.unpack("<{}i".format(count), self.stream.read(4 * count))

    def read_float(self) -> float:
        return float(struct.unpack("<f", self.stream.read(4))[0])

    def read_floats(self, count: int) -> tuple[float, ...]:
        return struct.unpack("<{}f".format(count), self.stream.read(4 * count))

    def read_double(self) -> float:
        return float(struct.unpack("<d", self.stream.read(8))[0])

    def read_doubles(self, count: int) -> tuple[float, ...]:
        return struct.unpack("<{}d".format(count), self.stream.read(8 * count))

    def read_format(self, format: str) -> tuple[float, ...]:
        """Read a struct format from the data (only float/double formats used)"""
        return struct.unpack(format, self.stream.read(struct.calcsize(format)))

    def read_count(self, label: str = "count") -> int:
        """Read a uint length/count prefix, rejecting unreasonable values.

        EDM count prefixes come straight from the file and drive allocation
        sizes and loop bounds; a corrupt or hostile file could otherwise
        request an effectively unbounded read.
        """
        prepos = self.stream.tell()
        count = self.read_uint()
        if count > MAX_REASONABLE_COUNT:
            raise EDMFormatError(
                "Implausible {} {} at position {} (max {})".format(
                    label, count, prepos, MAX_REASONABLE_COUNT
                )
            )
        return count

    def read_string(self, lookup: bool = True) -> str:
        """Read a length-prefixed string from the file.
        lookup: If v10, string will be read as lookup. Has no effect on v8"""

        prepos = self.stream.tell()
        if self.v10 and lookup:
            index = self.read_uint()
            strings = self.strings or []
            if index >= len(strings):
                raise EDMFormatError(
                    "Got index higher than lookup count; {} at {}".format(
                        index, prepos
                    )
                )
            return strings[index]
        else:
            length = self.read_uint()
            if length >= 200:
                raise EDMFormatError(
                    "Overly long string length found; {} at {}".format(
                        length, prepos
                    )
                )
            data = self.stream.read(length)
            try:
                return data.decode("windows-1251")
            except UnicodeDecodeError:
                # latin-1 accepts every byte value, so this cannot itself
                # raise; it exists only to normalize non-Cyrillic bytes.
                return data.decode("latin-1")

    def read_list(self, reader: Callable[["BaseReader"], T]) -> list[T]:
        """Reads a length-prefixed list of something"""
        length = self.read_count("list length")
        return [reader(self) for _ in range(length)]

    def read_vec2f(self) -> Vector:
        return Vector(self.read_format("<ff"))

    def read_vec3f(self) -> Vector:
        return Vector(self.read_format("<fff"))

    def read_vec4f(self) -> Vector:
        return Vector(self.read_format("<ffff"))

    def read_vec3d(self) -> Vector:
        return Vector(self.read_format("<ddd"))

    def read_matrixf(self) -> Matrix:
        md = self.read_floats(16)
        return sequence_to_matrix(md)

    def read_matrixd(self) -> Matrix:
        md = self.read_doubles(16)
        return sequence_to_matrix(md)

    def read_quaternion(self) -> Quaternion:
        qd = self.read_doubles(4)
        # Reorder as osg saves xyzw and we want wxyz
        return Quaternion([qd[3], qd[0], qd[1], qd[2]])
