def _create_mesh(vertexData, indexData, vertexFormat, compact=True, join_triangles=False, transform=None):
  """Creates a blender mesh object from vertex, index and format data
  
  Args:
    transform: Optional 4x4 matrix to transform vertex positions (for Y-up to Z-up conversion)
  """

  if compact:
    used_indices = sorted(set(indexData))
    source_vertices = [vertexData[i] for i in used_indices]
    index_lookup = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
    new_indices = [index_lookup[i] for i in indexData]
  else:
    used_indices = list(range(len(vertexData)))
    source_vertices = vertexData
    new_indices = indexData

  if transform is not None:
    transform_mat = Matrix(transform) if not hasattr(transform, 'to_4x4') else transform
    pos_idx = vertexFormat.position_indices
    norm_idx = vertexFormat.normal_indices
    # Normals transform by the inverse-transpose of the linear part.
    # _ROOT_BASIS_FIX is orthonormal so this equals the matrix itself, but
    # the general form is correct for any future caller.
    normal_mat = transform_mat.to_3x3().inverted_safe().transposed()
    new_vertices = []
    for vtx in source_vertices:
      vtx = list(vtx)
      pos = Vector(vtx[i] for i in pos_idx)
      tp = transform_mat @ pos.to_4d()
      vtx[0], vtx[1], vtx[2] = tp[0], tp[1], tp[2]
      if norm_idx and len(norm_idx) >= 3:
        norm = Vector(vtx[i] for i in norm_idx[:3])
        tn = normal_mat @ norm
        if tn.length_squared > 1e-12:
          tn.normalize()
        for i, ni in enumerate(norm_idx[:3]):
          vtx[ni] = tn[i]
      new_vertices.append(vtx)
  else:
    new_vertices = source_vertices

  primitive_mode = "triangles"
  if len(new_indices) % 3 != 0:
    if len(new_indices) == 1:
      primitive_mode = "points"
    elif len(new_indices) % 2 == 0:
      primitive_mode = "lines"
    else:
      usable = len(new_indices) - (len(new_indices) % 3)
      print(
        "Warning: Non-triangle index count {} encountered; truncating to {} indices".format(
          len(new_indices), usable
        )
      )
      new_indices = new_indices[:usable]

  bm = bmesh.new()
  import_custom_normals = True
  vertex_normals = [] if (import_custom_normals and vertexFormat.normal_indices) else None

  posIndex = vertexFormat.position_indices
  normIndex = vertexFormat.normal_indices
  uvIndexSets = []
  uv_offset = sum(vertexFormat.data[:4]) if hasattr(vertexFormat, "data") else 0
  if hasattr(vertexFormat, "data"):
    for channel in (4, 5, 6):
      if channel >= len(vertexFormat.data):
        continue
      count = int(vertexFormat.data[channel])
      if count >= 2:
        uvIndexSets.append(list(range(uv_offset, uv_offset + 2)))
      uv_offset += max(0, count)

  for i, vtx in enumerate(new_vertices):
    pos = Vector(vtx[x] for x in posIndex)
    vert = bm.verts.new(pos)
    if normIndex and vertex_normals is not None:
      norm = Vector(vtx[x] for x in normIndex)
      if norm.length_squared > 1e-12:
        norm.normalize()
      vert.normal = norm
      vertex_normals.append(norm.copy())

  bm.verts.ensure_lookup_table()

  # Prepare for texture information
  uv_layers = []
  for uv_set_idx, _ in enumerate(uvIndexSets):
    layer_name = "UVMap" if uv_set_idx == 0 else "UVMap.{:03d}".format(uv_set_idx)
    uv_layer = bm.loops.layers.uv.get(layer_name)
    if uv_layer is None:
      uv_layer = bm.loops.layers.uv.new(layer_name)
    uv_layers.append(uv_layer)

  # Generate geometry, with texture coordinate information on triangle faces.
  skipped_degenerate = 0
  skipped_duplicate = 0
  if primitive_mode == "triangles":
    # Some EDM files include degenerate or duplicate triangles that bmesh rejects.
    for face in [new_indices[i:i+3] for i in range(0, len(new_indices), 3)]:
      if len(set(face)) < 3:
        skipped_degenerate += 1
        continue
      try:
        f = bm.faces.new([bm.verts[i] for i in face])
        # Add UV data if we have any
        if uvIndexSets:
          for loop, v_idx in zip(f.loops, face):
            vtx = new_vertices[v_idx]
            for uv_layer, uv_idx in zip(uv_layers, uvIndexSets):
              if len(uv_idx) < 2:
                continue
              if uv_idx[1] >= len(vtx):
                continue
              loop[uv_layer].uv = (vtx[uv_idx[0]], 1 - vtx[uv_idx[1]])
      except ValueError:
        # BMesh rejects exact duplicate faces; keep track and continue.
        skipped_duplicate += 1
  elif primitive_mode == "lines":
    for edge in [new_indices[i:i+2] for i in range(0, len(new_indices), 2)]:
      if len(edge) < 2 or edge[0] == edge[1]:
        skipped_degenerate += 1
        continue
      try:
        bm.edges.new([bm.verts[edge[0]], bm.verts[edge[1]]])
      except ValueError:
        skipped_duplicate += 1
  else:
    # Point-list payloads need no extra topology; keep the referenced vertices.
    pass

  if skipped_degenerate or skipped_duplicate:
    print(
      "Info: Skipped {} degenerate and {} duplicate {} while building mesh".format(
        skipped_degenerate,
        skipped_duplicate,
        "faces" if primitive_mode == "triangles" else "edges",
      )
    )

  if join_triangles and bm.faces:
    try:
      bmesh.ops.join_triangles(
        bm,
        faces=list(bm.faces),
        cmp_seam=False,
        cmp_sharp=False,
        cmp_uvs=False,
        cmp_vcols=False,
        cmp_materials=False,
        angle_face_threshold=3.14159,
        angle_shape_threshold=3.14159,
      )
    except Exception as e:
      print("Warning: Failed to join shell triangles: {}".format(e))

  # Create the mesh object
  mesh = bpy.data.meshes.new("Mesh")
  bm.to_mesh(mesh)
  bm.free()

  # EDM meshes are authored for smooth shading in most cases; keep curved
  # surfaces visually faithful by enabling smooth polygons and importing
  # provided normals as custom split normals.
  for poly in mesh.polygons:
    poly.use_smooth = True

  if vertex_normals and len(vertex_normals) == len(mesh.vertices):
    try:
      normals = [tuple(n) for n in vertex_normals]
      if hasattr(mesh, "normals_split_custom_set_from_vertices"):
        mesh.normals_split_custom_set_from_vertices(normals)
      else:
        loop_normals = [normals[loop.vertex_index] for loop in mesh.loops]
        mesh.normals_split_custom_set(loop_normals)
    except Exception as e:
      print("Warning: Failed to assign custom normals: {}".format(e))

  if hasattr(mesh, "use_auto_smooth"):
    mesh.use_auto_smooth = True

  mesh.update()

  return mesh
