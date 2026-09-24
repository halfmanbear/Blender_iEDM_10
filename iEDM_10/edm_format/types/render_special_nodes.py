"""Registered shell, skin, morph, and segment node readers."""

from ..material_types import VertexFormat
from ..validation import resolve_node_indices
from .core import (
    _V10_CATEGORY_KEYS,
    BaseNode,
    NodeCategory,
    _read_with_layout_fallback,
    _scan_to_next_v10_type_token,
    logger,
    reads_type,
)
from .render_nodes import RenderNode
from .render_payload import (
    _read_index_data,
    _read_vertex_data,
    _render_audit,
    _tag_shell_family_node,
    _write_index_data,
    _write_vertex_data,
)


@reads_type("model::ShellNode")
class ShellNode(BaseNode):
    category = NodeCategory.shell

    @classmethod
    def read(cls, stream):
        self = super(ShellNode, cls).read(stream)
        self.parent = stream.read_uint()
        self.vertex_format = VertexFormat.read(stream)

        # Read the vertex and index data
        self.vertexData = _read_vertex_data(stream, "__cv_bytes")
        self.unknown_indexPrefix, self.indexData = _read_index_data(
            stream, classification="__ci_bytes"
        )

        return self

    def audit(self):
        return _render_audit(self, verts="__cv_bytes", inds="__ci_bytes")

    def write(self, writer):
        super(ShellNode, self).write(writer)
        # Write parent index, or 0 if no parent
        parent_index = self.parent.index if self.parent else 0
        writer.write_uint(parent_index)
        self.vertex_format.write(writer)
        _write_vertex_data(self.vertexData, writer)
        _write_index_data(self.indexData, len(self.vertexData), writer)


@reads_type("model::SkinNode")
class SkinNode(BaseNode):
    category = NodeCategory.render

    def __init__(self, name=None):
        super(SkinNode, self).__init__(name)
        self.unknown_start = 0
        self.material = None
        self.bones = []
        self.skin_binding_value = 0
        self.bbox = None
        self.vertexData = []
        self.unknown_indexPrefix = 5
        self.indexData = []
        self.unknown = 0  # Legacy exporter alias for unknown_start.
        self.post_bone = 0  # Legacy exporter alias for skin_binding_value.

    @classmethod
    def read(cls, stream):
        self = super(SkinNode, cls).read(stream)
        # SkinNode payload starts with the same reserved uint32 + properties-set
        # id pair that RenderNode uses. The local importer still stores the resolved
        # properties-set id in `material` for parity with the rest of the codebase.
        self.unknown_start = stream.read_uint()
        self.material = stream.read_uint()

        boneCount = stream.read_count("skin bone count")
        self.bones = stream.read_uints(boneCount)
        # SkinNode virtual getter/setter exposes this value; preserve it with
        # a semantic field while retaining the legacy alias used by older code.
        self.skin_binding_value = stream.read_uint()

        # When __VERSION__ > 0, readBoundingBoxf() follows —
        # osg::BoundingBoxImpl<Vec3f> = 6 floats (min_x min_y min_z max_x max_y max_z).
        # Not consuming these bytes desynchs all subsequent vertex/index reads.
        if self.props.get("__VERSION__", 0):
            self.bbox = stream.read_floats(6)

        # Read the vertex and index data
        self.vertexData = _read_vertex_data(stream, "__gv_bytes")
        self.unknown_indexPrefix, self.indexData = _read_index_data(
            stream, classification="__gi_bytes"
        )
        self.unknown = self.unknown_start
        self.post_bone = self.skin_binding_value

        return self

    def prepare(self, nodes, materials):
        self.bones = resolve_node_indices(nodes, self.bones)

    def audit(self):
        return _render_audit(self)

    def write(self, writer):
        super(SkinNode, self).write(writer)
        writer.write_uint(
            int(getattr(self, "unknown_start", getattr(self, "unknown", 0)) or 0)
        )
        material = (
            self.material.index if not isinstance(self.material, int) else self.material
        )
        writer.write_uint(material if material is not None else 0)
        writer.write_uint(len(self.bones))
        writer.write_uints(
            [
                bone.index if not isinstance(bone, int) else bone
                for bone in getattr(self, "bones", []) or []
            ]
        )
        writer.write_uint(
            int(getattr(self, "skin_binding_value", getattr(self, "post_bone", 0)) or 0)
        )
        if self.props.get("__VERSION__", 0):
            bbox = list(getattr(self, "bbox", None) or [])
            if len(bbox) != 6:
                bbox = [0.0] * 6
            writer.write_floats(bbox)
        _write_vertex_data(self.vertexData, writer)
        _write_index_data(self.indexData, len(self.vertexData), writer)


@reads_type("model::ShellSkinNode")
class ShellSkinNode(ShellNode):
    """Some files expose ShellSkinNode; accept shell/skin payload variants."""

    category = NodeCategory.shell

    @classmethod
    def read(cls, stream):
        node = _read_with_layout_fallback(
            stream,
            [
                ("shell", ShellNode.read),
                ("skin", SkinNode.read),
            ],
        )
        _tag_shell_family_node(node, "model::ShellSkinNode")
        if isinstance(node, SkinNode):
            logger.warning(
                "Parsed ShellSkinNode using SkinNode layout for '%s'",
                getattr(node, "name", ""),
            )
        return node


@reads_type("model::TreeShellNode")
class TreeShellNode(ShellNode):
    """Observed in exporter metadata; treat as shell-family node layout."""

    category = NodeCategory.shell

    @classmethod
    def read(cls, stream):
        node = _read_with_layout_fallback(
            stream,
            [
                ("shell", ShellNode.read),
                ("shell_skin", ShellSkinNode.read),
                ("skin", SkinNode.read),
            ],
        )
        return _tag_shell_family_node(node, "model::TreeShellNode")


@reads_type("model::MorphNode")
class MorphNode(RenderNode):
    """Morph-target renderable node.

    Shares the RenderNode binary layout for the base mesh (name, material,
    parentData, vertexData, indexData).  An additional morph-target payload
    follows the base data; its exact layout is not yet fully reversed, so we
    preserve those bytes as a blob until shape-key import is implemented.

    Because MorphNode subclasses RenderNode, the existing create_object() and
    graph-pipeline code handle it transparently as a regular render mesh.
    """

    category = NodeCategory.render

    @classmethod
    def read(cls, stream):
        self = super(MorphNode, cls).read(stream)
        # Preserve any morph-target data that follows the base mesh payload.
        # Use _scan_to_next_v10_type_token to stay stream-aligned.
        if getattr(stream, "v10", False):
            skip = _scan_to_next_v10_type_token(stream, stop_tokens=_V10_CATEGORY_KEYS)
            if skip is not None and skip > 0:
                self._morph_payload = stream.read(skip)
                logger.warning(
                    "MorphNode '%s': preserved %d morph-target bytes "
                    "(shape key import not yet implemented)",
                    self.name,
                    len(self._morph_payload),
                )
        return self


@reads_type("model::SegmentsNode")
class SegmentsNode(BaseNode):
    category = NodeCategory.shell

    @classmethod
    def read(cls, stream):
        self = super(SegmentsNode, cls).read(stream)
        # v10 SegmentsNode uses the same control-node link preamble as ShellNode.
        # The uint32 after the base node is the control-node index, not an
        # opaque field. Treating it as "unknown" loses the authored transform
        # chain and rotates collision lines into the wrong basis on import.
        self.parent = stream.read_uint()
        count = stream.read_count("segments count")
        self.data = [stream.read_floats(6) for x in range(count)]
        stream.mark_type_read("model::SegmentsNode::Segments", count)
        return self

    def audit(self):
        c = super(SegmentsNode, self).audit()
        c["model::SegmentsNode::Segments"] += len(self.data)
        return c

    def write(self, writer):
        super(SegmentsNode, self).write(writer)
        parent_index = self.parent.index if self.parent else 0
        writer.write_uint(parent_index)
        writer.write_uint(len(self.data))
        for segment in self.data:
            writer.write_floats(segment)
