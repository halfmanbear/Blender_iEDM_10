"""Registered billboard, texture, and real-light node readers."""

from ..propertiesset import PropertiesSet
from .core import BaseNode, Node, NodeCategory, reads_type


@reads_type("model::BillboardNode")
class BillboardNode(Node):
    category = NodeCategory.render

    @classmethod
    def read(cls, stream):
        self = super(BillboardNode, cls).read(stream)
        # v10 BillboardNode payload
        self.billboard_type = stream.read_uchar()
        self.billboard_axis = stream.read_uchar()

        # When __VERSION__ is non-zero in the node's PropertiesSet, a matrixd and
        # vec3d follow in the stream.
        # NOTE: props is an OrderedDict (PropertiesSet), NOT a list of property objects,
        # so the previous loop over self.properties was always a no-op and the matrix
        # was never consumed — causing stream desync for versioned BillboardNodes.
        if self.props.get("__VERSION__", 0):
            self.matrix = stream.read_matrixd()
            self.pivot = stream.read_vec3d()
        else:
            self.matrix = None
            self.pivot = None

        return self


class Texture2dProperties(object):
    """
    Verified layout of model::Texture2dProperties::load():
      uint32   index
      string   name        (string-table lookup in v10)
      uint32   wrap_s
      uint32   wrap_t
      uint32   mag_filter
      uint32   min_filter
      matrixf  uv_transform  (16 floats)
    """

    __slots__ = (
        "index",
        "name",
        "wrap_s",
        "wrap_t",
        "mag_filter",
        "min_filter",
        "uv_transform",
    )

    @classmethod
    def read(cls, stream):
        self = cls()
        self.index = stream.read_uint()
        self.name = stream.read_string()  # string-table lookup in v10
        self.wrap_s = stream.read_uint()
        self.wrap_t = stream.read_uint()
        self.mag_filter = stream.read_uint()
        self.min_filter = stream.read_uint()
        self.uv_transform = stream.read_matrixf()
        return self


@reads_type("model::LightNode")
class LightNode(BaseNode):
    category = NodeCategory.light

    @classmethod
    def read(cls, stream):
        self = super(LightNode, cls).read(stream)
        self.parent = stream.read_uint()
        self.unknown = [stream.read_uchar()]
        self.pre_props_flag = self.unknown[0]
        # Preserve animation argument ids for light properties so importer can map
        # curves and EDMProps args exactly for official exporter round-trips.
        self.lightProps = PropertiesSet.read(
            stream, count=False, preserve_animated=True
        )
        # This byte is has_texture. When nonzero Texture2dProperties::load() follows.
        # Failing to consume the payload causes stream desync for all subsequent nodes.
        self.unknown.append(stream.read_uchar())
        self.post_props_flag = self.unknown[1]
        self.texture = None
        if self.post_props_flag:
            self.texture = Texture2dProperties.read(stream)
        return self
