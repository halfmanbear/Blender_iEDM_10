"""Registered render-node reader and owner-based mesh splitting."""

from .core import BaseNode, NodeCategory, logger, reads_type
from .render_payload import (
    _classify_render_parent_data,
    _owner_triangles,
    _read_index_data,
    _read_parent_data,
    _read_vertex_data,
    _render_audit,
    _write_index_data,
    _write_vertex_data,
)


@reads_type("model::RenderNode")
class RenderNode(BaseNode):
    category = NodeCategory.render

    def __init__(self, name=None):
        super(RenderNode, self).__init__(name)
        self.version = 1
        self.children = []
        self.unknown_start = 0
        self.material = None
        self.parentData = None
        self.parent = None
        self.vertexData = []
        self.unknown_indexPrefix = 5
        self.indexData = []
        # If e.g. split then we don't know the real name
        self.name_unknown = False

    def __repr__(self):
        return '<RenderNode "{}">'.format(self.name)

    @classmethod
    def read(cls, stream):
        self = super(RenderNode, cls).read(stream)
        self.unknown_start = stream.read_uint()
        self.material = stream.read_uint()

        self.parentData = _read_parent_data(stream)

        # Read the vertex and index data
        self.vertexData = _read_vertex_data(stream, "__gv_bytes")
        self.unknown_indexPrefix, self.indexData = _read_index_data(
            stream, classification="__gi_bytes"
        )

        return self

    def write(self, writer):
        super(RenderNode, self).write(writer)
        writer.write_uint(int(getattr(self, "unknown_start", 0) or 0))
        material = (
            self.material.index if not isinstance(self.material, int) else self.material
        )
        writer.write_uint(material)

        def _node_index(ref):
            if isinstance(ref, int):
                return ref
            if ref is None:
                return 0
            return getattr(ref, "index", 0)

        parent_data = getattr(self, "parentData", None)
        if parent_data:
            writer.write_uint(len(parent_data))
            if len(parent_data) == 1 and len(parent_data[0]) >= 2:
                writer.write_uint(_node_index(parent_data[0][0]))
                writer.write_int(int(parent_data[0][1]))
            else:
                for entry in parent_data:
                    parent_ref = entry[0] if len(entry) > 0 else 0
                    val1 = entry[1] if len(entry) > 1 else 0
                    val2 = entry[2] if len(entry) > 2 else -1
                    writer.write_uint(_node_index(parent_ref))
                    writer.write_int(int(val1))
                    writer.write_int(int(val2))
        else:
            writer.write_uint(1)
            parent_index = _node_index(getattr(self, "parent", None))
            writer.write_uint(parent_index)
            writer.write_int(int(getattr(self, "damage_argument", -1)))

        _write_vertex_data(self.vertexData, writer)
        _write_index_data(self.indexData, len(self.vertexData), writer)

    def audit(self):
        c = _render_audit(self)
        # We may, or may not, have any extra parent data at the moment.
        if self.parentData and len(self.parentData) > 1:
            c["model::RNControlNode"] += len(self.parentData) - 1
        return c

    def split(self):
        """Returns an array of renderNode objects. If there is no splitting to be
        done, it will just return [self]. Otherwise, each entry is to be counted
        as a separate renderNode object."""
        logger.debug("Splitting RenderNode %s", self.name)

        if self.parentData is None:
            raise RuntimeError(
                "Attempting to split RenderNode without parent data; "
                "it may already be split"
            )
        assert len(self.parentData) >= 1, (
            "Should never have a RenderNode without parent data"
        )

        # If one parent, no splitting to be done. Just assign our parent index.
        if len(self.parentData) == 1:
            logger.debug("Single parent for %s", self.name)
            self.parent = self.parentData[0][0]
            self.damage_argument = self.parentData[0][1]
            return [self]

        # We have more than one parent object. Do some splitting.
        total_indices = len(self.indexData)
        logger.debug(
            "Multiple parents (%d) for %s, total_indices=%d",
            len(self.parentData),
            self.name,
            total_indices,
        )

        parent_layout = _classify_render_parent_data(
            self.parentData, self.vertexData, self.indexData
        )
        self.parentData_layout = parent_layout
        owner_values = parent_layout.get("owner_values")

        if parent_layout["mode"] == "owner_encoded":
            logger.debug("V10 owner-encoded split detected for %s", self.name)
            shared_parent = parent_layout.get("shared_parent")
            owner_tris = _owner_triangles(self.indexData, owner_values)
            children = []
            for owner_idx, pd in enumerate(self.parentData):
                parent = pd[0]
                node = RenderNode()
                node.version = self.version
                node.name = "{}_{}".format(self.name, owner_idx)
                node.name_unknown = True
                node.props = self.props
                node.material = self.material
                node.parent = parent
                node.damage_argument = pd[2]
                node.vertexData = self.vertexData
                node.parentData_layout = dict(parent_layout)
                # V10 owner-encoded split metadata: geometry coordinates are shared
                # across all owners relative to the same control-space parent.
                node.shared_parent = shared_parent
                node.owner_index = owner_idx
                node.split_owner_encoded = True
                node.indexData = owner_tris.get(owner_idx, [])
                children.append(node)
            return children

        # V10 zero-coverage tables without usable owner variation should preserve
        # all parent attachments by reusing the full geometry on each child.
        if parent_layout["mode"] == "duplicate_geometry":
            return self._split_duplicate_geometry(parent_layout)

        start = 0
        children = []

        for i, (parent, val1, val2) in enumerate(self.parentData):
            node = RenderNode()
            node.version = self.version
            node.name = "{}_{}".format(self.name, i)
            node.name_unknown = True
            node.props = self.props
            node.material = self.material
            node.parent = parent
            node.parentData_layout = dict(parent_layout)

            # Apply logic determined above
            if parent_layout["force_fallback"]:
                # If falling back, give EVERYTHING to the first node, others get empty
                if i == 0:
                    idxTo = total_indices
                else:
                    idxTo = total_indices  # or start, effectively empty
                damageArg = val2
            elif parent_layout["swap_columns"]:
                idxTo = val2
                damageArg = val1
            else:
                idxTo = val1
                damageArg = val2

            node.indexData = self.indexData[start:idxTo]
            node.damage_argument = damageArg

            # Give them all the whole vertex subarray for now
            node.vertexData = self.vertexData
            start = idxTo
            children.append(node)

        return children

    def _split_duplicate_geometry(self, parent_layout):
        """Duplicate a zero-coverage mesh for each declared parent."""
        print(
            "Info: V10 zero-coverage split for {}; duplicating geometry "
            "across {} parents".format(self.name, len(self.parentData))
        )
        children = []
        for i, (parent, _val1, val2) in enumerate(self.parentData):
            node = RenderNode()
            node.version = self.version
            node.name = "{}_{}".format(self.name, i)
            node.name_unknown = True
            node.props = self.props
            node.material = self.material
            node.parent = parent
            node.indexData = self.indexData
            node.damage_argument = val2
            node.vertexData = self.vertexData
            node.parentData_layout = dict(parent_layout)
            children.append(node)
        return children
