import logging

from ..mathtypes import Matrix, Quaternion, Vector
from ..typereader import reads_type
from .core_nodes import Node
from .core_support import AnimatingNode, get_type_reader

logger = logging.getLogger(__name__)


class ArgAnimationBase(object):
    def __init__(
        self, matrix=None, position=None, quat_1=None, quat_2=None, scale=None
    ):
        self.matrix = matrix or Matrix()
        self.position = position or Vector()
        self.quat_1 = quat_1 or Quaternion((1, 0, 0, 0))
        # quat_2 is the orientation basis for base.scale:
        #   S_base = R(quat_2) @ diag(scale) @ R(quat_2)^-1
        # Preserve it so the importer/exporter can reconstruct the authored scale
        # basis instead of treating base.scale as always axis-aligned.
        self.quat_2 = quat_2 or Quaternion((1, 0, 0, 0))
        self.scale = scale or Vector((1, 1, 1))

    @classmethod
    def read(cls, stream):
        self = cls()
        self.matrix = stream.read_matrixd()
        self.position = stream.read_vec3d()
        self.quat_1 = stream.read_quaternion()
        # Record the file offset of quat_2 so it can be overwritten by patchers.
        if hasattr(stream, "tell"):
            self._quat_2_file_offset = stream.tell()
        self.quat_2 = stream.read_quaternion()
        self.scale = stream.read_vec3d()
        return self

    def write(self, stream):
        stream.write_matrixd(self.matrix)
        stream.write_vec3d(self.position)
        stream.write_quaternion(self.quat_1)
        stream.write_quaternion(self.quat_2)
        stream.write_vec3d(self.scale)


@reads_type("model::ArgAnimationNode")
class ArgAnimationNode(Node, AnimatingNode):
    def __init__(self, *args, **kwargs):
        super(ArgAnimationNode, self).__init__(*args, **kwargs)
        self.base = ArgAnimationBase()
        self.posData = []
        self.rotData = []
        self.scaleData = []
        self.parent = None

    def __repr__(self):
        if self.posData and not self.rotData and not self.scaleData:
            nodeName = "ArgPositionNode"
        elif self.rotData and not self.posData and not self.scaleData:
            nodeName = "ArgRotationNode"
        elif self.scaleData and not self.posData and not self.rotData:
            nodeName = "ArgScaleNode"
        else:
            nodeName = "ArgAnimationNode"

        return "<{} {:}{:}{:}{}>".format(
            nodeName,
            len(self.posData),
            len(self.rotData),
            len(self.scaleData),
            " " + self.name if self.name else "",
        )

    @classmethod
    def read(cls, stream):
        self = super(ArgAnimationNode, cls).read(stream)
        self.base = ArgAnimationBase.read(stream)
        self.posData = stream.read_list(ArgPositionNode._read_AANPositionArg)
        self.rotData = stream.read_list(ArgRotationNode._read_AANRotationArg)
        self.scaleData = stream.read_list(ArgScaleNode._read_AANScaleArg)
        return self

    def write(self, stream):
        super(ArgAnimationNode, self).write(stream)
        self.base.write(stream)
        # Position args
        stream.write_uint(len(self.posData))
        for arg, keyframes in self.posData:
            stream.write_uint(arg)
            stream.write_uint(len(keyframes))
            for k in keyframes:
                stream.write_double(k.frame)
                stream.write_vec3d(k.value)
        # Rotation args
        stream.write_uint(len(self.rotData))
        for arg, keyframes in self.rotData:
            stream.write_uint(arg)
            stream.write_uint(len(keyframes))
            for k in keyframes:
                stream.write_double(k.frame)
                stream.write_quaternion(k.value)
        # Scale args (matches read: (arg, (keys4, keys3)))
        stream.write_uint(len(self.scaleData))
        for arg, (keys4, keys3) in self.scaleData:
            stream.write_uint(arg)
            # First set: 4-component scale keys
            stream.write_uint(len(keys4))
            for k in keys4:
                stream.write_double(k.frame)
                # ScaleKey.read(stream, 4) -> 4 doubles
                for v in k.value:
                    stream.write_double(v)
            # Second set: 3-component scale keys
            stream.write_uint(len(keys3))
            for k in keys3:
                stream.write_double(k.frame)
                # ScaleKey.read(stream, 3) -> 3 doubles
                for v in k.value:
                    stream.write_double(v)

    def audit(self):
        c = super(ArgAnimationNode, self).audit()
        if type(self) is not ArgAnimationNode:
            c["model::ArgAnimationNode"] += 1
        c["model::ArgAnimationNode::Rotation"] = len(self.rotData)
        c["model::ArgAnimationNode::Position"] = len(self.posData)
        c["model::ArgAnimationNode::Scale"] = len(self.scaleData)

        rotKeys = sum([len(x[1]) for x in self.rotData])
        posKeys = sum([len(x[1]) for x in self.posData])
        scaKeys = sum([len(x[1][0]) + len(x[1][1]) for x in self.scaleData])
        c["model::Key<key::ROTATION>"] = rotKeys
        c["model::Key<key::POSITION>"] = posKeys
        c["model::Key<key::SCALE>"] = scaKeys
        return c

    def get_all_args(self):
        "Return all represented arguments in stable source order."
        ordered = []
        seen = set()
        for dataset in [self.posData, self.rotData, self.scaleData]:
            for entry in dataset:
                arg = entry[0]
                if arg in seen:
                    continue
                seen.add(arg)
                ordered.append(arg)
        return ordered


@reads_type("model::ArgAnimatedBone")
class ArgAnimatedBone(ArgAnimationNode):
    @classmethod
    def read(cls, stream):
        self = super(ArgAnimatedBone, cls).read(stream)
        self.inv_base_bone_matrix = stream.read_matrixd()
        return self


@reads_type("model::ArgRotationNode")
class ArgRotationNode(ArgAnimationNode):
    """A special case of ArgAnimationNode with only rotational data.
    Despite this, it is written and read from disk in exactly the same way."""

    @classmethod
    def read(cls, stream):
        stream.mark_type_read("model::ArgAnimationNode")
        return super(ArgRotationNode, cls).read(stream)

    @classmethod
    def _read_AANRotationArg(cls, stream):
        stream.mark_type_read("model::ArgAnimationNode::Rotation")
        arg = stream.read_uint()
        count = stream.read_count("rotation key count")
        keys = [
            get_type_reader("model::Key<key::ROTATION>")(stream) for _ in range(count)
        ]
        return (arg, keys)


@reads_type("model::ArgPositionNode")
class ArgPositionNode(ArgAnimationNode):
    """A special case of ArgAnimationNode with only positional data.
    Despite this, it is written and read from disk in exactly the same way."""

    @classmethod
    def read(cls, stream):
        stream.mark_type_read("model::ArgAnimationNode")
        return super(ArgPositionNode, cls).read(stream)

    @classmethod
    def _read_AANPositionArg(cls, stream):
        stream.mark_type_read("model::ArgAnimationNode::Position")
        arg = stream.read_uint()
        count = stream.read_count("position key count")
        keys = [
            get_type_reader("model::Key<key::POSITION>")(stream) for _ in range(count)
        ]
        return (arg, keys)


@reads_type("model::ArgScaleNode")
class ArgScaleNode(ArgAnimationNode):
    @classmethod
    def read(cls, stream):
        stream.mark_type_read("model::ArgAnimationNode")
        return super(ArgScaleNode, cls).read(stream)

    @classmethod
    def _read_AANScaleArg(cls, stream):
        stream.mark_type_read("model::ArgAnimationNode::Scale")
        arg = stream.read_uint()
        count = stream.read_count("scale key count")
        # Set 1 (4-component): scale orientation quaternion Q.
        # Scale is applied as Q * diag(sx,sy,sz) * Q^-1.
        # When Q is identity this reduces to plain axis-aligned scale.
        keys = [ScaleKey.read(stream, 4) for _ in range(count)]
        count2 = stream.read_count("scale key2 count")
        # Set 2 (3-component): scale magnitudes (sx, sy, sz) along the Q axes.
        key2s = [ScaleKey.read(stream, 3) for _ in range(count2)]
        # Warn when oriented scale data is present. The importer can reconstruct it
        # for the single-action helper-chain path, but other paths may still fall
        # back to axis-aligned scale.
        for k in keys:
            v = k.value  # (x, y, z, w) in file order; identity = (0, 0, 0, ±1)
            # Treat both w=+1 and w=-1 (negative identity quaternion) as identity.
            # Some exporters write (0,0,0,-1) as the default orientation even for
            # pure rotation nodes; this represents the same rotation as (0,0,0,+1).
            if (
                abs(v[0]) > 1e-4
                or abs(v[1]) > 1e-4
                or abs(v[2]) > 1e-4
                or abs(abs(v[3]) - 1.0) > 1e-4
            ):
                logger.warning(
                    "Scale orientation quaternion is non-identity "
                    "(arg=%d, frame=%g, xyzw=%s). Importer will try helper-chain "
                    "reconstruction; unsupported paths may differ.",
                    arg,
                    k.frame,
                    tuple(round(float(x), 4) for x in v),
                )
                break
        return (arg, (keys, key2s))


@reads_type("model::Key<key::ROTATION>")
class RotationKey(object):
    def __init__(self, frame=None, value=None):
        self.frame = frame
        self.value = value

    @classmethod
    def read(cls, stream):
        self = cls()
        self.frame = stream.read_double()
        self.value = stream.read_quaternion()
        return self

    def __repr__(self):
        return "Key(frame={}, value={})".format(self.frame, repr(self.value))


@reads_type("model::Key<key::POSITION>")
class PositionKey(object):
    def __init__(self, frame=None, value=None):
        self.frame = frame
        self.value = value

    @classmethod
    def read(cls, stream):
        self = cls()
        self.frame = stream.read_double()
        self.value = Vector(stream.read_doubles(3))
        return self

    def __repr__(self):
        return "Key(frame={}, value={})".format(self.frame, repr(self.value))


@reads_type("model::Key<key::SCALE>")
class ScaleKey(object):
    @classmethod
    def read(cls, stream, entrylength):
        self = cls()
        self.frame = stream.read_double()
        self.value = Vector(stream.read_doubles(entrylength))
        return self

    def __repr__(self):
        return "Key(frame={}, value={})".format(self.frame, repr(self.value))


@reads_type("model::ArgVisibilityNode")
class ArgVisibilityNode(Node, AnimatingNode):
    @classmethod
    def read(cls, stream):
        self = super(ArgVisibilityNode, cls).read(stream)
        self.visData = stream.read_list(cls._read_AANVisibilityArg)
        # When the serialized __VERSION__ property is non-zero, the reader consumes
        # one trailing matrixf as a legacy compatibility payload, then normalizes
        # the node version back to 0. The writer does not emit that matrix on save.
        self.vis_matrix = None
        if self.props.get("__VERSION__", 0):
            self.vis_matrix = stream.read_matrixf()
            self.props["__VERSION__"] = 0
        return self

    @classmethod
    def _read_AANVisibilityArg(cls, stream):
        stream.mark_type_read("model::ArgVisibilityNode::Arg")
        arg = stream.read_uint()
        count = stream.read_count("visibility range count")
        data = [stream.read_doubles(2) for _ in range(count)]
        stream.mark_type_read("model::ArgVisibilityNode::Range", count)
        return (arg, data)

    def write(self, stream):
        super(ArgVisibilityNode, self).write(stream)
        stream.write_uint(len(self.visData))
        for arg, ranges in self.visData:
            stream.write_uint(arg)
            stream.write_uint(len(ranges))
            for low, high in ranges:
                stream.write_double(low)
                stream.write_double(high)

    def audit(self):
        c = super(ArgVisibilityNode, self).audit()
        c["model::ArgVisibilityNode::Arg"] += len(self.visData)
        c["model::ArgVisibilityNode::Range"] += sum(len(x[1]) for x in self.visData)
        return c
