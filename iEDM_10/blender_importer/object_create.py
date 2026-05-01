def create_connector(connector):
  """Create an empty object representing a connector"""
  ob = bpy.data.objects.new(connector.name, None)
  ob.empty_display_type = "CUBE"
  ob.edm.is_connector = True
  _set_official_special_type(ob, 'CONNECTOR')

  # Preserve connector properties for round-trip export
  if hasattr(connector, 'props') and connector.props:
    lines = []
    for key, value in connector.props.items():
      if key.startswith("__"):
        continue
      if isinstance(value, str):
        lines.append("{} = '{}'".format(key, value.replace("'", "\\'")))
      elif isinstance(value, (int, float)):
        lines.append("{} = {}".format(key, repr(value)))
    if lines:
      _set_edmprop(ob, "CONNECTOR_EXT", "\n".join(lines))

  bpy.context.collection.objects.link(ob)
  return ob

def create_segments(segments_node):
  """Create a mesh with edges from a SegmentsNode (collision lines)"""
  # Create a new mesh
  mesh = bpy.data.meshes.new(segments_node.name)

  # Extract vertices and edges from segment data
  verts = []
  edges = []

  for i, segment in enumerate(segments_node.data):
    # Each segment is 6 floats: [x1, y1, z1, x2, y2, z2]
    v1 = Vector((segment[0], segment[1], segment[2]))
    v2 = Vector((segment[3], segment[4], segment[5]))

    v1 = Vector(v1)
    v2 = Vector(v2)

    # Add vertices
    verts.extend([v1, v2])
    # Add edge connecting these two vertices
    edges.append([i*2, i*2+1])

  # Create mesh from vertex and edge data
  mesh.from_pydata(verts, edges, [])
  mesh.update()

  # Create object and set properties
  ob = bpy.data.objects.new(segments_node.name, mesh)
  ob.edm.is_collision_line = True
  ob.edm.is_renderable = False
  _set_official_special_type(ob, 'COLLISION_LINE')
  bpy.context.collection.objects.link(ob)

  return ob
