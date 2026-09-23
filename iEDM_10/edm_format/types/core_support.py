import logging
import struct
from abc import ABC
from collections import Counter, OrderedDict
from enum import Enum

from ..basereader import BaseReader
from ..typereader import get_type_reader as _tr_get_type_reader

logger = logging.getLogger(__name__)


class NodeCategory(Enum):
    """Enum to describe what file-category the node is in"""

    transform = "transform"
    connector = "CONNECTORS"
    render = "RENDER_NODES"
    shell = "SHELL_NODES"
    light = "LIGHT_NODES"


class AnimatingNode(ABC):
    """Abstract base class for all nodes that animate the object"""


_all_IndexA = {
    "model::TransformNode",
    "model::FakeOmniLightsNode",
    "model::AnimatedFakeOmniLightsNode",
    "model::SkinNode",
    "model::Connector",
    "model::ShellNode",
    "model::ShellSkinNode",
    "model::TreeShellNode",
    "model::SegmentsNode",
    "model::FakeSpotLightsNode",
    "model::AnimatedFakeSpotLightsNode",
    "model::FakeSpotLights3Node",
    "model::BillboardNode",
    "model::ArgAnimatedBone",
    "model::RootNode",
    "model::TmpNumberRoot",
    "model::NumberRoot",
    "model::Node",
    "model::ArgAnimationNode",
    "model::ArgRotationNode",
    "model::ArgPositionNode",
    "model::ArgScaleNode",
    "model::LightNode",
    "model::LodNode",
    "model::Bone",
    "model::RenderNode",
    "model::ArgVisibilityNode",
}


_all_IndexB = {
    "model::Key<key::ROTATION>",
    "model::Property<float>",
    "model::ArgAnimationNode::Position",
    "__pointers",
    "model::FakeOmniLight",
    "model::Key<key::SCALE>",
    "model::AnimatedProperty<osg::Vec3f>",
    "model::AnimatedProperty<osg::Vec4f>",
    "model::Key<key::VEC3F>",
    "model::Key<key::VEC4F>",
    "model::Property<osg::Vec2f>",
    "model::Property<osg::Vec3f>",
    "model::Property<osg::Vec4f>",
    "model::ArgAnimationNode::Rotation",
    "model::ArgVisibilityNode::Range",
    "model::Key<key::POSITION>",
    "model::AnimatedProperty<osg::Vec2f>",
    "model::Key<key::FLOAT>",
    "__ci_bytes",
    "__gv_bytes",
    "model::ArgVisibilityNode::Arg",
    "model::AnimatedProperty<float>",
    "model::RNControlNode",
    "model::SegmentsNode::Segments",
    "__gi_bytes",
    "__cv_bytes",
    "model::Property<unsigned int>",
    "model::Key<key::VEC2F>",
    "model::ArgAnimationNode::Scale",
    "model::LodNode::Level",
    "model::PropertiesSet",
    "model::FakeSpotLight",
}


def _is_root_box_sentinel_pair(
    min_vec, max_vec, sentinel=3.4028234663852886e38, eps=1.0e30
):
    try:
        mins = tuple(float(v) for v in min_vec)
        maxs = tuple(float(v) for v in max_vec)
    except Exception:
        return False
    return all(abs(v - sentinel) <= eps for v in mins) and all(
        abs(v + sentinel) <= eps for v in maxs
    )


def _same_vec3(a, b, eps=1.0e-6):
    try:
        return all(abs(float(x) - float(y)) <= eps for x, y in zip(a, b))
    except Exception:
        return False


def _next_v10_token_looks_like_type(stream):
    """Best-effort probe to see if next uint is a v10 type-name table index."""
    if not getattr(stream, "v10", False) or not getattr(stream, "strings", None):
        return None
    pos = stream.tell()
    try:
        idx = stream.read_uint()
    except Exception:
        stream.seek(pos)
        return None
    stream.seek(pos)
    if not (0 <= idx < len(stream.strings)):
        return False
    token = stream.strings[idx]
    return isinstance(token, str) and token.startswith("model::")


def _read_with_layout_fallback(stream, readers):
    """Try alternative reader layouts from the same stream offset."""
    start = stream.tell()
    best = None
    best_end = start
    errors = []
    for layout, reader in readers:
        stream.seek(start)
        try:
            node = reader(stream)
            looks_aligned = _next_v10_token_looks_like_type(stream)
            if looks_aligned is True:
                node._layout_variant = layout
                return node
            if best is None:
                best = node
                best_end = stream.tell()
                best._layout_variant = layout
        except Exception as exc:
            errors.append((layout, exc))
    if best is not None:
        stream.seek(best_end)
        return best
    stream.seek(start)
    error_summary = ", ".join(
        "{}: {}".format(name, type(exc).__name__) for name, exc in errors
    )
    raise IOError("All fallback readers failed ({})".format(error_summary))


_V10_CATEGORY_KEYS = frozenset(
    ("CONNECTORS", "LIGHT_NODES", "RENDER_NODES", "SHELL_NODES")
)


def _scan_to_next_v10_type_token(
    stream, max_bytes=16384, validate_node_header=False, stop_tokens=None
):
    """Find next likely model:: token index in v10 string table on 4-byte boundaries.

    validate_node_header: when True, also require that the 4 bytes immediately
    following the candidate type token decode to a plausible inline BaseNode name
    length (< 80).  This rejects false-positive token matches where an unrelated
    binary value happens to be a valid string-table index.

    stop_tokens: string-table tokens that also end the scan (e.g. the category key
    that follows the last node of a category).
    """
    if not getattr(stream, "v10", False) or not getattr(stream, "strings", None):
        return None
    pos = stream.tell()
    data = stream.read(max_bytes)
    stream.seek(pos)
    if not data:
        return None
    for off in range(0, len(data) - 3, 4):
        idx = struct.unpack_from("<I", data, off)[0]
        if not (0 <= idx < len(stream.strings)):
            continue
        token = stream.strings[idx]
        if stop_tokens and token in stop_tokens:
            return off
        if not (isinstance(token, str) and token.startswith("model::")):
            continue
        if validate_node_header:
            if off + 8 > len(data):
                continue
            name_len = struct.unpack_from("<I", data, off + 4)[0]
            if name_len >= 80:
                continue
        return off
    return None


def get_type_reader(name):
    _readfun = _tr_get_type_reader(name)

    def _reader(reader):
        reader.typecount[name] += 1
        return _readfun(reader)

    return _reader


class TrackingReader(BaseReader):
    def __init__(self, *args, **kwargs):
        self.typecount = Counter()
        self.autoTypeCount = Counter()
        super(TrackingReader, self).__init__(*args, **kwargs)

    def mark_type_read(self, name, amount=1):
        self.typecount[name] += amount

    def read_named_type(self, selfOrNone=None):
        assert selfOrNone is None or selfOrNone is self
        typeName = self.read_string()
        try:
            return get_type_reader(typeName)(self)
        except KeyError:
            print("Error at position {}".format(self.tell()))
            raise


def _read_index(stream):
    """Reads a dictionary of type String : uint"""
    length = stream.read_count("index length")
    data = OrderedDict()
    for _ in range(length):
        key = stream.read_string()
        value = stream.read_uint()
        data[key] = value
    return data


def _write_index(writer, data):
    writer.write_uint(len(data))
    keys = sorted(data.keys())
    for key in keys:
        writer.write_string(key)
        writer.write_uint(data[key])


def _read_main_object_dictionary(stream):
    count = stream.read_count("main object dictionary count")
    objects = {}
    for _ in range(count):
        name = stream.read_string()
        objects[name] = stream.read_list(stream.read_named_type)
    return objects
