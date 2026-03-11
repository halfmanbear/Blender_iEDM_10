from .core import *  # noqa: F401,F403


def _read_index_data(stream, classification=None):
  "Performs the common index-reading operation"
  dtPos = stream.tell()
  dataType = stream.read_uchar()
  entries = stream.read_uint()
  unknown = stream.read_uint()

  if dataType == 0:
    data = stream.read_uchars(entries)
    _bytes = entries
  elif dataType == 1:
    data = stream.read_ushorts(entries)
    _bytes = entries * 2
  elif dataType == 2:
    data = stream.read_uints(entries)
    _bytes = entries * 4
  else:
    raise IOError("Don't know how to read index data type {} @ {}".format(int(dataType), dtPos))

  if classification:
    stream.mark_type_read(classification, _bytes)

  return (unknown, data)

def _write_index_data(indexData, vertexDataLength, writer):
  # Index data
  if vertexDataLength < 256:
    writer.write_uchar(0)
    iWriter = writer.write_uchars
  elif vertexDataLength < 2**16:
    writer.write_uchar(1)
    iWriter = writer.write_ushorts
  elif vertexDataLength < 2**32:
    writer.write_uchar(2)
    iWriter = writer.write_uints
  else:
    raise IOError("Do not know how to write index arrays with {} members".format(vertexDataLength))

  writer.write_uint(len(indexData))
  writer.write_uint(5)
  iWriter(indexData)

def _read_vertex_data(stream, classification=None):
  count = stream.read_uint()
  stride = stream.read_uint()
  vtxData = stream.read_floats(count*stride)

  # If given a classification, mark it off
  if classification:
    stream.mark_type_read(classification, count*stride*4)

  # Group the vertex data according to stride
  vtxData = [vtxData[i:i+stride] for i in range(0, len(vtxData), stride)]
  return vtxData

def _write_vertex_data(data, writer):
  writer.write_uint(len(data))
  writer.write_uint(len(data[0]))
  flat_data = list(itertools.chain(*data))
  writer.write_floats(flat_data)

def _read_parent_data(stream):
    # Read the parent section
  parentCount = stream.read_uint()
  stream.mark_type_read("model::RNControlNode", parentCount-1)

  if parentCount == 1:
    return [[stream.read_uint(), stream.read_int()]]
  else:
    parentData = []
    for _ in range(parentCount):
      node = stream.read_uint()
      ranges = list(stream.read_ints(2))
      parentData.append((node, ranges[0], ranges[1]))
    return parentData

def _render_audit(self, verts="__gv_bytes", inds="__gi_bytes"):
  c = Counter()
  c[verts] += 4 * len(self.vertexData) * len(self.vertexData[0])
  # c["__gi_bytes"] += 
  if len(self.vertexData) < 256:
    c[inds] += len(self.indexData)
  elif len(self.vertexData) < 2**16:
    c[inds] += len(self.indexData) * 2
  elif len(self.vertexData) < 2**32:
    c[inds] += len(self.indexData) * 4
  else:
    raise IOError("Do not know how to write index arrays with {} members".format(len(self.indexData)))
  return c

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
    return "<RenderNode \"{}\">".format(self.name)

  @classmethod
  def read(cls, stream):
    self = super(RenderNode, cls).read(stream)
    self.unknown_start = stream.read_uint()
    self.material = stream.read_uint()

    self.parentData = _read_parent_data(stream)

    # Read the vertex and index data
    self.vertexData = _read_vertex_data(stream, "__gv_bytes")
    self.unknown_indexPrefix, self.indexData = _read_index_data(stream, classification="__gi_bytes")

    return self

  def write(self, writer):
    super(RenderNode, self).write(writer)
    writer.write_uint(0)
    if not isinstance(self.material, int):
      self.material = self.material.index
    writer.write_uint(self.material)

    # Rebuild the parentdata
    writer.write_uint(1)
    parent_index = self.parent.index if self.parent else 0
    writer.write_uint(parent_index)
    writer.write_int(-1)

    _write_vertex_data(self.vertexData, writer)
    _write_index_data(self.indexData, len(self.vertexData), writer)

  def audit(self):
    c = _render_audit(self)
    # We may, or may not, have any extra parent data at the moment.
    if self.parentData and len(self.parentData) > 1:
      c["model::RNControlNode"] += len(self.parentData)-1
    return c

  def split(self):
      """Returns an array of renderNode objects. If there is no splitting to be
      done, it will just return [self]. Otherwise, each entry is to be counted
      as a separate renderNode object."""
      logger.debug("Splitting RenderNode %s", self.name)

      if self.parentData is None:
        raise RuntimeError("Attempting to split renderNode without parent data - has it already been split?")
      assert len(self.parentData) >= 1, "Should never have a RenderNode without parent data"
      
      # If one parent, no splitting to be done. Just assign our parent index.
      if len(self.parentData) == 1:
        logger.debug("Single parent for %s", self.name)
        self.parent = self.parentData[0][0]
        self.damage_argument = self.parentData[0][1]
        return [self]

      # We have more than one parent object. Do some splitting.
      total_indices = len(self.indexData)
      logger.debug("Multiple parents (%d) for %s, total_indices=%d", len(self.parentData), self.name, total_indices)

      # V10 variants often encode parent attachments with idxTo == 0 for every
      # entry (i.e. not a V8-style coverage table). Some files then encode an
      # owner index per vertex; others intend the geometry to be reused for
      # each parent entry (typically different damage arguments / control nodes).
      all_zero_idx_to = bool(self.parentData) and all(len(pd) == 3 and pd[1] == 0 for pd in self.parentData)
      all_damage_neg1 = bool(self.parentData) and all(len(pd) == 3 and pd[2] == -1 for pd in self.parentData)
      owner_values = None
      if all_zero_idx_to and self.vertexData:
        try:
          raw_owner_values = [int(round(v[3])) for v in self.vertexData]
          nOwners = len(self.parentData)
          out_of_range = sum(1 for o in raw_owner_values if not (0 <= o < nOwners))
          if out_of_range:
            print(f"Info: {self.name} has {out_of_range} vertices with out-of-range owner, clamping to [0, {nOwners - 1}]")
          owner_values = [max(0, min(nOwners - 1, o)) for o in raw_owner_values]
        except Exception:
          owner_values = None

      # Require either the strong legacy signal (all damageArg == -1) or actual
      # owner variation across vertices before treating this as owner-encoded.
      if all_zero_idx_to and owner_values is not None and (all_damage_neg1 or len(set(owner_values)) > 1):
        logger.debug("V10 owner-encoded split detected for %s", self.name)
        shared_parent = self.parentData[0][0]
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
          # V10 owner-encoded split metadata: geometry coordinates are shared
          # across all owners relative to the same control-space parent.
          node.shared_parent = shared_parent
          node.owner_index = owner_idx
          node.split_owner_encoded = True
          # Preserve original triangle boundaries. Filtering the flat index
          # stream and then regrouping in triples mixes vertices from
          # different source triangles and corrupts geometry/pivots.
          tri_indices = []
          mixed_owner_tris = 0
          for i in range(0, len(self.indexData), 3):
            tri = self.indexData[i:i+3]
            if len(tri) != 3:
              continue
            tri_owners = [owner_values[ix] for ix in tri]
            if tri_owners[0] == tri_owners[1] == tri_owners[2] == owner_idx:
              tri_indices.extend(tri)
              continue
            # Fallback for malformed data: assign mixed-owner triangles to
            # the majority owner to avoid holes.
            if tri_owners.count(owner_idx) >= 2:
              mixed_owner_tris += 1
              tri_indices.extend(tri)
          # if mixed_owner_tris:
          #   print(f"Info: {self.name} owner {owner_idx} absorbed {mixed_owner_tris} mixed-owner triangles")
          node.indexData = tri_indices
          children.append(node)
        return children

      # V10 zero-coverage tables without usable owner variation should preserve
      # all parent attachments by reusing the full geometry on each child.
      if all_zero_idx_to:
        print(f"Info: V10 zero-coverage attachment split for {self.name}; duplicating geometry across {len(self.parentData)} parents")
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
          children.append(node)
        return children
      
      # --- FIX START: V10/Mod Compatibility Logic ---
      # Check if the last entry covers the whole range (Standard V8 behavior)
      last_val1 = self.parentData[-1][1] # Usually idxTo (End Index)
      last_val2 = self.parentData[-1][2] # Usually damageArg

      swap_columns = False
      force_fallback = False

      # Scenario A: Standard mismatch (The warning you saw)
      if last_val1 != total_indices:
          # Scenario B: Columns are swapped? (val2 is the count, val1 is damage/0)
          if last_val2 == total_indices:
              print(f"Info: Detected V10 data swap for {self.name}. Swapping interpretation.")
              swap_columns = True
          # Scenario C: Both are wrong or 0? Force render.
          elif last_val1 == 0:
               print(f"Warning: {self.name} has 0 coverage. Forcing geometry to first node to prevent invisible mesh.")
               force_fallback = True
          else:
               print(f"Warning: Split mismatch {self.name}: covered {last_val1} vs total {total_indices}")
      # --- FIX END ---

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
        
        # Apply logic determined above
        if force_fallback:
            # If falling back, give EVERYTHING to the first node, others get empty
            if i == 0:
                idxTo = total_indices
            else:
                idxTo = total_indices # or start, effectively empty
            damageArg = val2
        elif swap_columns:
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
    self.unknown_indexPrefix, self.indexData = _read_index_data(stream, classification="__ci_bytes")

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
  @classmethod
  def read(cls, stream):
    self = super(SkinNode, cls).read(stream)
    self.unknown = stream.read_uint()
    self.material = stream.read_uint()

    boneCount = stream.read_uint()
    self.bones = stream.read_uints(boneCount)
    self.post_bone = stream.read_uint()

    # Read the vertex and index data
    self.vertexData = _read_vertex_data(stream, "__gv_bytes")
    self.unknown_indexPrefix, self.indexData = _read_index_data(stream, classification="__gi_bytes")

    return self

  def prepare(self, nodes, materials):
    self.bones = [nodes[x] for x in self.bones]

  def audit(self):
    return _render_audit(self)

@reads_type("model::ShellSkinNode")
class ShellSkinNode(ShellNode):
  """Some files expose ShellSkinNode; accept shell/skin payload variants."""
  category = NodeCategory.shell

  @classmethod
  def read(cls, stream):
    node = _read_with_layout_fallback(stream, [
      ("shell", ShellNode.read),
      ("skin", SkinNode.read),
    ])
    if isinstance(node, SkinNode):
      logger.warning("Parsed ShellSkinNode using SkinNode layout for '%s'", getattr(node, "name", ""))
    return node

@reads_type("model::TreeShellNode")
class TreeShellNode(ShellNode):
  """Observed in exporter metadata; treat as shell-family node layout."""
  category = NodeCategory.shell

  @classmethod
  def read(cls, stream):
    return _read_with_layout_fallback(stream, [
      ("shell", ShellNode.read),
      ("shell_skin", ShellSkinNode.read),
      ("skin", SkinNode.read),
    ])

@reads_type("model::SegmentsNode")
class SegmentsNode(BaseNode):
  category = NodeCategory.shell
  @classmethod
  def read(cls, stream):
    self = super(SegmentsNode, cls).read(stream)
    self.unknown = stream.read_uint()
    count = stream.read_uint()
    self.data = [stream.read_floats(6) for x in range(count)]
    stream.mark_type_read("model::SegmentsNode::Segments", count)
    return self

  def audit(self):
    c = super(SegmentsNode, self).audit()
    c["model::SegmentsNode::Segments"] += len(self.data)
    return c

  def write(self, writer):
    super(SegmentsNode, self).write(writer)
    writer.write_uint(self.unknown)
    writer.write_uint(len(self.data))
    for segment in self.data:
      writer.write_floats(segment)

