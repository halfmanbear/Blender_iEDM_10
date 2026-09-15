from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Any

from .mathtypes import Vector

# Actually a types.core.TrackingReader (BaseReader plus read_named_type() /
# mark_type_read()); typed as Any here to avoid a circular import with
# edm_format.types.core, which imports this module.
StreamReader = Any


def _property_type_name(value: Any) -> str:
    """Return the EDM type string for a property value."""
    if isinstance(value, bool):
        return "model::Property<unsigned int>"
    if isinstance(value, float):
        return "model::Property<float>"
    if isinstance(value, int):
        return "model::Property<unsigned int>"
    if isinstance(value, str):
        return "model::Property<const char*>"
    if isinstance(value, Vector):
        return "model::Property<osg::Vec{}f>".format(len(value))
    raise IOError("Unknown property type {}/{}".format(value, type(value)))


class PropertiesSet(OrderedDict[str, Any]):
    @classmethod
    def read(
        cls, stream: StreamReader, count: bool = True, preserve_animated: bool = False
    ) -> "PropertiesSet":
        data = cls()
        length = stream.read_count("properties set length")
        for _ in range(length):
            prop = stream.read_named_type()
            # Handle regular and animated properties sets the same
            if hasattr(prop, "keys"):
                data[prop.name] = prop if preserve_animated else prop.keys
            else:
                data[prop.name] = prop.value
        # This only counts towards the general count if we had data
        if data and count:
            stream.mark_type_read("model::PropertiesSet")

        return data

    _SCALAR_WRITERS = {
        "model::Property<unsigned int>": lambda w, v: w.write_uint(int(v)),
        "model::Property<float>": lambda w, v: w.write_float(v),
        "model::Property<const char*>": lambda w, v: w.write_string(v),
    }

    def write(self, writer: Any) -> None:
        writer.write_uint(len(self))
        for key, value in self.items():
            type_name = _property_type_name(value)
            writer.write_string(type_name)
            writer.write_string(key)
            if isinstance(value, Vector):
                writer.write_vecf(value)
            else:
                self._SCALAR_WRITERS[type_name](writer, value)

    def audit(self) -> Counter[str]:
        c: Counter[str] = Counter()
        for value in self.values():
            c[_property_type_name(value)] += 1
        return c
