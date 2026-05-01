from ..typereader import reads_type
from ..typereader import get_type_reader as _tr_get_type_reader
from ..basereader import BaseReader
from ..probe import require_supported_import_format

from ..material_types import VertexFormat, Material, Texture, ShadowSettings
from ..propertiesset import PropertiesSet

from collections import namedtuple, OrderedDict, Counter
import itertools
import struct

from ..mathtypes import Vector, sequence_to_matrix, Matrix, Quaternion

from abc import ABC
from enum import Enum
import math

import logging
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


# All possible entries for indexA and indexB
_all_IndexA = {'model::TransformNode', 'model::FakeOmniLightsNode', 'model::AnimatedFakeOmniLightsNode', 'model::SkinNode', 'model::Connector', 'model::ShellNode', 'model::ShellSkinNode', 'model::TreeShellNode', 'model::SegmentsNode', 'model::FakeSpotLightsNode', 'model::AnimatedFakeSpotLightsNode', 'model::FakeSpotLights3Node', 'model::BillboardNode', 'model::ArgAnimatedBone', 'model::RootNode', 'model::TmpNumberRoot', 'model::NumberRoot', 'model::Node', 'model::ArgAnimationNode', 'model::ArgRotationNode', 'model::ArgPositionNode', 'model::ArgScaleNode', 'model::LightNode', 'model::LodNode', 'model::Bone', 'model::RenderNode', 'model::ArgVisibilityNode'}
_all_IndexB = {'model::Key<key::ROTATION>', 'model::Property<float>', 'model::ArgAnimationNode::Position', '__pointers', 'model::FakeOmniLight', 'model::Key<key::SCALE>', 'model::AnimatedProperty<osg::Vec3f>', 'model::AnimatedProperty<osg::Vec4f>', 'model::Key<key::VEC3F>', 'model::Key<key::VEC4F>', 'model::Property<osg::Vec2f>', 'model::Property<osg::Vec3f>', 'model::Property<osg::Vec4f>', 'model::ArgAnimationNode::Rotation', 'model::ArgVisibilityNode::Range', 'model::Key<key::POSITION>', 'model::AnimatedProperty<osg::Vec2f>', 'model::Key<key::FLOAT>', '__ci_bytes', '__gv_bytes', 'model::ArgVisibilityNode::Arg', 'model::AnimatedProperty<float>', 'model::RNControlNode', 'model::SegmentsNode::Segments', '__gi_bytes', '__cv_bytes', 'model::Property<unsigned int>', 'model::Key<key::VEC2F>', 'model::ArgAnimationNode::Scale', 'model::LodNode::Level', 'model::PropertiesSet', 'model::FakeSpotLight'}


def _is_root_box_sentinel_pair(min_vec, max_vec, sentinel=3.4028234663852886e+38, eps=1.0e+30):
  try:
    mins = tuple(float(v) for v in min_vec)
    maxs = tuple(float(v) for v in max_vec)
  except Exception:
    return False
  return (
    all(abs(v - sentinel) <= eps for v in mins)
    and all(abs(v + sentinel) <= eps for v in maxs)
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
  errors = []
  for layout, reader in readers:
    stream.seek(start)
    try:
      node = reader(stream)
      looks_aligned = _next_v10_token_looks_like_type(stream)
      if looks_aligned is True:
        setattr(node, "_layout_variant", layout)
        return node
      if best is None:
        best = node
        setattr(best, "_layout_variant", layout)
    except Exception as exc:
      errors.append((layout, exc))
  if best is not None:
    return best
  stream.seek(start)
  error_summary = ", ".join("{}: {}".format(name, type(exc).__name__) for name, exc in errors)
  raise IOError("All fallback readers failed ({})".format(error_summary))


def _scan_to_next_v10_type_token(stream, max_bytes=16384, validate_node_header=False):
  """Find next likely model:: token index in v10 string table on 4-byte boundaries.

  validate_node_header: when True, also require that the 4 bytes immediately
  following the candidate type token decode to a plausible inline BaseNode name
  length (< 80).  This rejects false-positive token matches where an unrelated
  binary value happens to be a valid string-table index.
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
  length = stream.read_uint()
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
  count = stream.read_uint()
  objects = {}
  for _ in range(count):
    name = stream.read_string()
    objects[name] = stream.read_list(stream.read_named_type)
  return objects

class EDMFile(object):
  def __init__(self, filename=None, version=8):
    if filename:
      require_supported_import_format(filename)
      reader = TrackingReader(filename)
      try:
        self._read(reader)
      except Exception:
        err_pos = None
        try:
          err_pos = reader.tell()
        except Exception:
          err_pos = "unknown"
        print("ERROR at {}".format(err_pos))
        reader.close()
        raise
    else:
      self.version = version
      self.indexA = {}
      self.indexB = {}
      self.root = None
      self.nodes = []
      self.connectors = []
      self.renderNodes = []
      self.lightNodes = []
      self.shellNodes = []

  def _read(self, reader):
    reader.read_constant(b'EDM')
    self.version = reader.read_ushort()
    if self.version != 10:
      raise IOError("Unsupported EDM version {}; only v10 is supported".format(self.version))
    logger.info("Reading EDM version 10 file")
    reader.version = self.version

    if reader.v10:
      stringsize = reader.read_uint()
      sdata = reader.read(stringsize)
      # Split by null byte, keeping empty strings (they are valid table entries)
      parts = sdata.split(b'\x00')
      if parts and not parts[-1]:
        parts = parts[:-1]  # strip trailing empty from final null terminator
      reader.strings = [x.decode("windows-1251") for x in parts]
    else:
      reader.strings = None

    # Read the two indexes
    self.indexA = _read_index(reader)
    self.indexB = _read_index(reader)
    self.root = reader.read_named_type()

    self.nodes = reader.read_list(reader.read_named_type)
    if self.nodes:
        self.transformRoot = self.nodes[0]
    else:
        self.transformRoot = None
        # Handle empty nodes list gracefully if possible, or raise error if critical
        print("Warning: EDM file has no transform nodes.")

    # Read the node parenting data
    for (node, parent) in zip(self.nodes, reader.read_ints(len(self.nodes))):
      if parent == -1:
        node.parent = None
        continue
      if parent >= len(self.nodes):
        raise IOError("Invalid node parent data")

      node.set_parent(self.nodes[parent])

    # Read the renderable objects
    objects = _read_main_object_dictionary(reader)
    logger.debug("EDM object categories: %s", list(objects.keys()))
    self.connectors = objects.get("CONNECTORS", [])
    self.shellNodes = objects.get("SHELL_NODES", [])
    self.lightNodes = objects.get("LIGHT_NODES", [])
    for node_list in objects.values():
      for node in node_list:
        prepare = getattr(node, "prepare", None)
        if callable(prepare):
          try:
            prepare(self.nodes, self.root.materials)
          except Exception as exc:
            logger.warning(
              "Prepare failed for %s '%s' (%s: %s)",
              type(node).__name__,
              getattr(node, "name", ""),
              type(exc).__name__,
              exc,
            )
    self.renderNodes = []
    # Split any renderNodes as one may contain several objects
    for node in objects.get("RENDER_NODES", []):
      if isinstance(node, RenderNode):
        for splitNode in node.split():
          self.renderNodes.append(splitNode)
      else:
        self.renderNodes.append(node)

    # v10 stores NumberNode mesh payloads in a trailing post-section after the
    # main object dictionary. Bind payload blocks onto NumberNode placeholders.
    if reader.v10:
      number_nodes = [
        n for n in self.renderNodes
        if isinstance(n, NumberNode) and not getattr(n, "_post_payload_read", False)
      ]
      if number_nodes:
        payload_error = False
        for n in number_nodes:
          # Bail out safely if payload is not available/parseable.
          pos = reader.tell()
          try:
            n.read_v10_payload(reader)
          except Exception as exc:
            payload_error = True
            reader.seek(pos)
            logger.warning("NumberNode post-payload parse failed (%s: %s); "
                           "skipping remaining NumberNode payloads.",
                           type(exc).__name__, exc)
            break
        if payload_error:
          for n in number_nodes:
            if not getattr(n, "_post_payload_read", False):
              n.parent = None

    # ClassReader10_loadRoot ends cleanly after the sections loop — there is no
    # trailing data written by standard DCS World EDM tooling.
    # Any bytes remaining here come from a non-standard exporter.
    # Consume them so the end-of-file check below doesn't false-alarm, and warn
    # so unexpected payloads surface in logs rather than being silently ignored.
    if reader.v10:
      tail_start = reader.tell()
      tail_bytes = reader.stream.read()
      if tail_bytes:
        logger.warning(
          "Tail parse: %d unexpected bytes at offset %d "
          "(not present in standard DCS EDM files — non-standard exporter output?)",
          len(tail_bytes), tail_start,
        )
      reader.seek(tail_start + len(tail_bytes))

    # Verify we are at the end of the file without unconsumed data.
    endPos = reader.tell()
    if len(reader.read(1)) != 0:
      print("Warning: Ended parse at {} but still have data remaining".format(endPos))
    reader.close()

    # Set up parents and other links (e.g. material, bone...)
    for node in itertools.chain(self.connectors, self.shellNodes, self.lightNodes, self.renderNodes):
      if hasattr(node, "parent") and node.parent is not None:
        if isinstance(node.parent, int):
            if 0 <= node.parent < len(self.nodes):
                node.set_parent(self.nodes[node.parent])
            else:
                print(f"Warning: Node {node} has invalid parent index {node.parent}")
        else:
             # Already resolved or unexpected type
             pass
      # Owner-encoded split RenderNodes keep a shared control parent index.
      # Resolve it to the actual transform node so importer code can map
      # mesh-local coordinates from shared parent space.
      if hasattr(node, "shared_parent") and isinstance(node.shared_parent, int):
        if 0 <= node.shared_parent < len(self.nodes):
          node.shared_parent = self.nodes[node.shared_parent]
      if hasattr(node, "material"):
        if isinstance(node.material, int):
          if 0 <= node.material < len(self.root.materials):
              node.material = self.root.materials[node.material]
          else:
              print(f"Warning: Invalid material index {node.material} for node {node}")
              node.material = None
      if hasattr(node, "bones"):
        if node.bones and isinstance(node.bones[0], int):
          node.bones = [self.nodes[x] for x in node.bones]
        # If we have bones we have no single 'parent'. Stick it on the root.
        node.set_parent(self.nodes[0])

    # Validate against the index
    self.selfCount = self.audit()
    rems = Counter(self.indexA)
    rems.subtract(Counter({x: c for (x,c) in reader.typecount.items() if x in self.indexA}))
    for k in [x for x in rems.keys() if rems[x] == 0]:
      del rems[k]

    if rems:
      print("IndexA items remaining before RENDER_NODES/CONNECTORS: {}".format(rems))
    cB = Counter({x: c for (x,c) in reader.typecount.items() if x in self.indexB})
    remBs = Counter(self.indexB)
    remBs.subtract(cB)
    for k in [x for x in remBs.keys() if remBs[x] == 0]:
      del remBs[k]
    if remBs:
      print("IndexB items remaining before RENDER_NODES/CONNECTORS: {}".format(remBs))

  def audit(self):
    _index = Counter()
    _index[RootNode.forTypeName] += 1
    _index += self.root.audit()
    for node in self.nodes:
      _index[node.forTypeName] += 1
      _index += node.audit()
    for rn in itertools.chain(self.renderNodes, self.shellNodes, self.lightNodes, self.connectors):
      _index[rn.forTypeName] += 1
      try:
        _index += rn.audit()
      except Exception:
        raise RuntimeError("Trouble reading audit from {}".format(type(rn)))

    return _index

  def _write_body(self, writer, indexA, indexB):
    """Writes the body of the EDM file (everything after version/string table)"""
    # For v10, write empty indices (they seem to not be used in v10 format)
    if self.version == 10:
      _write_index(writer, {})
      _write_index(writer, {})
    else:
      _write_index(writer, indexA)
      _write_index(writer, indexB)

    # Write the Root node
    writer.write_named_type(self.root)

    # For each parent node, set it's index
    for i, node in enumerate(self.nodes):
      node.index = i

    writer.write_uint(len(self.nodes))
    for node in self.nodes:
      writer.write_named_type(node)

    # Write the parent data for the nodes
    writer.write_int(-1)
    # Everything without a parent has 0 as it's parent
    for node in self.nodes[1:]:
      if node.parent:
        writer.write_uint(node.parent.index)
      else:
        writer.write_uint(0)

    # Now do the render objects dictionary
    objects = {}
    if self.renderNodes:
      objects["RENDER_NODES"] = self.renderNodes
    if self.connectors:
      objects["CONNECTORS"] = self.connectors
    if self.shellNodes:
      objects["SHELL_NODES"] = self.shellNodes
    if self.lightNodes:
      objects["LIGHT_NODES"] = self.lightNodes
    writer.write_uint(len(objects))
    for key, nodes in objects.items():
      writer.write_string(key)
      writer.write_uint(len(nodes))
      for node in nodes:
        writer.write_named_type(node)

  def write(self, writer):
    # Generate the file index with an audit
    _allIndex = self.audit()
    indexA = {k: v for k, v in _allIndex.items() if k in _all_IndexA}
    indexB = {k: v for k, v in _allIndex.items() if k in _all_IndexB}

    # Write EDM header
    writer.write(b'EDM')
    writer.write_ushort(self.version)

    # For v10, do a collection pass to build string table
    if self.version == 10:
      import io
      print("Building v10 string table...")

      # Save the real stream and use a dummy stream for collection
      real_stream = writer.stream
      writer.stream = io.BytesIO()  # Dummy stream - data gets discarded
      writer.string_collect_mode = True
      self._write_body(writer, indexA, indexB)
      writer.string_collect_mode = False
      writer.stream = real_stream  # Restore real stream

      # Write the string table
      string_data = writer.get_string_table_data()
      writer.write_uint(len(string_data))
      writer.write(string_data)
      print(f"String table: {len(writer.string_table)} unique strings, {len(string_data)} bytes")

    # Now write the actual data
    self._write_body(writer, indexA, indexB)


class GraphNode(object):
  def __init__(self):
    self.parent = None
    self.children = []

  def set_parent(self, parent):
    if self.parent is parent:
      return
    # If we have a parent, unregister from it (special case: Reassigning an indexed parent value)
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
    # The DLL reads this uint but discards it without storing or using it.
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
    writer.write_string(self.name, lookup=False)  # Node names are always inline, even in v10
    writer.write_uint(self.version)
    self.props.write(writer)

  def __repr__(self):
    if not self.name:
      return "<{}>".format(type(self).__name__)
    else:
      return "<{} \"{}\">".format(type(self).__name__, self.name)

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
      if not _same_vec3(candidate_a[0], self.boundingBoxMin) or not _same_vec3(candidate_a[1], self.boundingBoxMax):
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
    material_count = stream.read_uint()
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
      writer.write_uchar(getattr(self, 'unknownA', 0))

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
          extra_boxes.extend([
            Vector((3.4028234663852886e+38, 3.4028234663852886e+38, 3.4028234663852886e+38)),
            Vector((-3.4028234663852886e+38, -3.4028234663852886e+38, -3.4028234663852886e+38)),
          ])
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
    return _read_with_layout_fallback(stream, [
      ("transform", TransformNode.read),
      ("node", Node.read),
    ])


@reads_type("model::NumberRoot")
class NumberRoot(Node):
  """Treat NumberRoot as a transform-like control root when encountered."""
  @classmethod
  def read(cls, stream):
    return _read_with_layout_fallback(stream, [
      ("transform", TransformNode.read),
      ("node", Node.read),
    ])

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
    # Bone uses the TransformNode preamble, then serializes an additional matrixd for the
    # inverse bind / bone matrix.
    self.bone_matrix = reader.read_matrixd()
    return self

  def write(self, stream):
    super(Bone, self).write(stream)
    stream.write_matrixd(self.bone_matrix)

class ArgAnimationBase(object):
  def __init__(self, matrix=None, position=None, quat_1=None, quat_2=None, scale=None):
    self.matrix = matrix or Matrix()
    self.position = position or Vector()
    self.quat_1 = quat_1 or Quaternion((1,0,0,0))
    # quat_2 is the orientation basis for base.scale:
    #   S_base = R(quat_2) @ diag(scale) @ R(quat_2)^-1
    # Preserve it so the importer/exporter can reconstruct the authored scale
    # basis instead of treating base.scale as always axis-aligned.
    self.quat_2 = quat_2 or Quaternion((1,0,0,0))
    self.scale = scale or Vector((1,1,1))
  @classmethod
  def read(cls, stream):
    self = cls()
    self.matrix = stream.read_matrixd()
    self.position = stream.read_vec3d()
    self.quat_1 = stream.read_quaternion()
    # Record the file offset of quat_2 so binary patchers can overwrite it.
    if hasattr(stream, 'tell'):
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

    return "<{} {:}{:}{:}{}>".format(nodeName,
      len(self.posData), len(self.rotData), len(self.scaleData),
      " "+self.name if self.name else "")

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
    if not type(self) is ArgAnimationNode:
      c["model::ArgAnimationNode"] += 1
    c["model::ArgAnimationNode::Rotation"] = len(self.rotData)
    c["model::ArgAnimationNode::Position"] = len(self.posData)
    c["model::ArgAnimationNode::Scale"] = len(self.scaleData)

    rotKeys = sum([len(x[1]) for x in self.rotData])
    posKeys = sum([len(x[1]) for x in self.posData])
    scaKeys = sum([len(x[1][0])+len(x[1][1]) for x in self.scaleData])
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
    count = stream.read_uint()
    keys = [get_type_reader("model::Key<key::ROTATION>")(stream) for _ in range(count)]
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
    count = stream.read_uint()
    keys = [get_type_reader("model::Key<key::POSITION>")(stream) for _ in range(count)]
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
    count = stream.read_uint()
    # Set 1 (4-component): scale orientation quaternion Q.
    # The DLL applies scale as Q * diag(sx,sy,sz) * Q^-1.
    # When Q is identity this reduces to plain axis-aligned scale.
    keys = [ScaleKey.read(stream, 4) for _ in range(count)]
    count2 = stream.read_uint()
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
      if abs(v[0]) > 1e-4 or abs(v[1]) > 1e-4 or abs(v[2]) > 1e-4 or abs(abs(v[3]) - 1.0) > 1e-4:
        logger.warning(
          "Scale orientation quaternion is non-identity (arg=%d, frame=%g, xyzw=%s). "
          "Importer will try helper-chain reconstruction; unsupported paths may still differ.",
          arg, k.frame, tuple(round(float(x), 4) for x in v),
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
    count = stream.read_uint()
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

@reads_type("model::LodNode")
class LodNode(Node):
  @classmethod
  def read(cls, stream):
    self = super(LodNode, cls).read(stream)
    count = stream.read_uint()
    self.level = [tuple(math.sqrt(v) for v in stream.read_doubles(2)) for _ in range(count)]
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
