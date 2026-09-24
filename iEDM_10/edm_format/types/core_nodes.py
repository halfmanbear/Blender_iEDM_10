import logging
import math
from collections import Counter

from ..material_types import Material
from ..mathtypes import Vector
from ..propertiesset import PropertiesSet
from ..typereader import reads_type
from .core_support import (
    NodeCategory,
    _is_root_box_sentinel_pair,
    _read_with_layout_fallback,
    _same_vec3,
)

logger = logging.getLogger(__name__)


class GraphNode(object):
    def __init__(self):
        self.parent = None
        self.children = []

    def set_parent(self, parent):
        if self.parent is parent:
            return
            # Unregister from a parent before reassigning an indexed parent value.
        if self.parent and not isinstance(self.parent, int):
            self.parent.children.remove(self)
        self.parent = parent
        if self.parent is not None:
            self.parent.children.append(self)

    def add_child(self, child):
        if child in self.children:
            return
        if child.parent:
            child.parent.children.remove(child)
        child.parent = self
        self.children.append(child)


@reads_type("model::BaseNode")
class BaseNode(GraphNode):
    def __init__(self, name=None):
        super(BaseNode, self).__init__()

        self.name = name or ""
        self.version = 0
        self.props = PropertiesSet()

    @classmethod
    def read(cls, stream):
        node = cls()
        # Node names are raw length-prefixed bytes in every version,
        # never string-table lookups, so lookup=False is required even in v10 files.
        name = stream.read_string(lookup=False)
        # v10: official exporter sometimes wraps inline node names in single quotes
        if name.startswith("'") and name.endswith("'") and len(name) >= 2:
            name = name[1:-1]
        node.name = name
        # This uint is read but discarded without storing or using it.
        # We preserve it for lossless round-trips.
        # This is NOT the __VERSION__ value — that comes from the PropertiesSet below.
        node.version = stream.read_uint()
        node.props = PropertiesSet.read(stream, count=False)
        return node

    def audit(self):
        c = Counter()
        if self.props:
            c += self.props.audit()
        return c

    def write(self, writer):
        writer.write_string(
            self.name, lookup=False
        )  # Node names are always inline, even in v10
        writer.write_uint(self.version)
        self.props.write(writer)

    def __repr__(self):
        if not self.name:
            return "<{}>".format(type(self).__name__)
        else:
            return '<{} "{}">'.format(type(self).__name__, self.name)


@reads_type("model::RootNode")
class RootNode(BaseNode):
    def __init__(self):
        super(RootNode, self).__init__()
        self.name = "Scene Root"
        self.props["__VERSION__"] = 3
        self.unknownB = []
        self.unknownC = 0
        self.maxArgPlusOne = 0
        self.scene_bounds_box = None
        self.user_box = None
        self.light_box = None
        self.root_boxes = {}

    def _refresh_box_metadata(self):
        self.scene_bounds_box = (self.boundingBoxMin, self.boundingBoxMax)
        self.user_box = None
        self.light_box = None
        self.root_boxes = {
            "bounding_box": self.scene_bounds_box,
        }

        payload = list(getattr(self, "unknownB", []) or [])
        if len(payload) >= 2:
            candidate_a = (payload[0], payload[1])
            if not _same_vec3(candidate_a[0], self.boundingBoxMin) or not _same_vec3(
                candidate_a[1], self.boundingBoxMax
            ):
                self.user_box = candidate_a
                self.root_boxes["user_box"] = candidate_a
        if len(payload) >= 4:
            candidate_b = (payload[2], payload[3])
            if not _is_root_box_sentinel_pair(candidate_b[0], candidate_b[1]):
                self.light_box = candidate_b
                self.root_boxes["light_box"] = candidate_b

    @classmethod
    def read(cls, stream):
        self = super(RootNode, cls).read(stream)

        if self.props.get("__VERSION__") == 2:
            self.unknownA = stream.read_uchar()

        self.boundingBoxMin = stream.read_vec3d()
        self.boundingBoxMax = stream.read_vec3d()
        self.unknownB = [stream.read_vec3d() for _ in range(4)]
        material_count = stream.read_count("material count")
        self.materials = [Material.read(stream) for i in range(material_count)]
        stream.materials = self.materials
        self.unknownC = stream.read_uint()
        self.maxArgPlusOne = stream.read_uint()
        self._refresh_box_metadata()
        return self

    def audit(self):
        c = super(RootNode, self).audit()
        for material in self.materials:
            c += material.audit()
        return c

    def write(self, writer):
        super(RootNode, self).write(writer)

        # Only write unknownA for version 2 (matches read logic)
        if self.props.get("__VERSION__") == 2:
            writer.write_uchar(getattr(self, "unknownA", 0))

        writer.write_vecd(self.boundingBoxMin)
        writer.write_vecd(self.boundingBoxMax)
        # v10 stores two additional box slots after the main AABB. In the common
        # case slot A duplicates the scene bounds and slot B is an FLT_MAX sentinel.
        # Preserve any decoded user/light boxes when present.
        extra_boxes = list(getattr(self, "unknownB", []) or [])
        if len(extra_boxes) < 4:
            extra_boxes = list(extra_boxes)
            if len(extra_boxes) < 2:
                user_box = self.user_box or self.scene_bounds_box
                extra_boxes.extend(user_box)
            if len(extra_boxes) < 4:
                light_box = self.light_box
                if light_box is not None:
                    extra_boxes.extend(light_box)
                else:
                    extra_boxes.extend(
                        [
                            Vector(
                                (
                                    3.4028234663852886e38,
                                    3.4028234663852886e38,
                                    3.4028234663852886e38,
                                )
                            ),
                            Vector(
                                (
                                    -3.4028234663852886e38,
                                    -3.4028234663852886e38,
                                    -3.4028234663852886e38,
                                )
                            ),
                        ]
                    )
        for vec in extra_boxes[:4]:
            writer.write_vecd(vec)

        writer.write_uint(len(self.materials))
        for mat in self.materials:
            mat.write(writer)
        writer.write_uint(0)
        writer.write_uint(0)


@reads_type("model::Node")
class Node(BaseNode):
    category = NodeCategory.transform


@reads_type("model::TmpNumberRoot")
class TmpNumberRoot(Node):
    """Observed in exporter metadata; layout appears transform-like."""

    @classmethod
    def read(cls, stream):
        return _read_with_layout_fallback(
            stream,
            [
                ("transform", TransformNode.read),
                ("node", Node.read),
            ],
        )


@reads_type("model::NumberRoot")
class NumberRoot(Node):
    """Treat NumberRoot as a transform-like control root when encountered."""

    @classmethod
    def read(cls, stream):
        return _read_with_layout_fallback(
            stream,
            [
                ("transform", TransformNode.read),
                ("node", Node.read),
            ],
        )


@reads_type("model::TransformNode")
class TransformNode(Node):
    @classmethod
    def read(cls, stream):
        self = super(TransformNode, cls).read(stream)
        # TransformNode serializes a single matrixd after the common node header.
        # The extra bone_matrix field belongs to Bone, not TransformNode.
        self.matrix = stream.read_matrixd()
        return self

    def write(self, stream):
        super(TransformNode, self).write(stream)
        stream.write_matrixd(self.matrix)


@reads_type("model::Bone")
class Bone(TransformNode):
    @classmethod
    def read(cls, reader):
        self = super(Bone, cls).read(reader)
        # Bone uses the TransformNode preamble, then serializes another matrixd for the
        # inverse bind / bone matrix.
        self.bone_matrix = reader.read_matrixd()
        return self

    def write(self, stream):
        super(Bone, self).write(stream)
        stream.write_matrixd(self.bone_matrix)


@reads_type("model::LodNode")
class LodNode(Node):
    @classmethod
    def read(cls, stream):
        self = super(LodNode, cls).read(stream)
        count = stream.read_count("lod level count")
        self.level = [
            tuple(math.sqrt(v) for v in stream.read_doubles(2)) for _ in range(count)
        ]
        stream.mark_type_read("model::LodNode::Level", count)
        return self

    def audit(self):
        c = super(LodNode, self).audit()
        c["model::LodNode::Level"] += len(self.level)
        return c

    def write(self, stream):
        super(LodNode, self).write(stream)
        stream.write_uint(len(self.level))
        for low, high in self.level:
            stream.write_double(low**2)
            stream.write_double(high**2)


@reads_type("model::Connector")
class Connector(BaseNode):
    category = NodeCategory.connector

    def __init__(self):
        super(Connector, self).__init__()
        self.data = 0

    @classmethod
    def read(cls, stream):
        self = super(Connector, cls).read(stream)
        self.parent = stream.read_uint()
        # After the control-node index a full Properties block is read.
        # In practice this block is always empty
        # (count = 0) so the count uint32 == 0 and no entries follow. If count > 0
        # the entries must be consumed to keep the stream aligned.
        self.extra_props = PropertiesSet.read(stream, count=False)
        self.data = len(self.extra_props)
        return self

    def write(self, stream):
        super(Connector, self).write(stream)
        stream.write_uint(self.parent.index)
        stream.write_uint(self.data)
