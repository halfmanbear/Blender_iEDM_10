def _recenter_mesh_object_to_geometry(obj):
  """Move mesh object origin to local bounds center, preserving world mesh."""
  if obj.type != "MESH" or not obj.data or not obj.data.vertices:
    return
  min_v = Vector((float('inf'), float('inf'), float('inf')))
  max_v = Vector((float('-inf'), float('-inf'), float('-inf')))
  for v in obj.data.vertices:
    min_v.x = min(min_v.x, v.co.x)
    min_v.y = min(min_v.y, v.co.y)
    min_v.z = min(min_v.z, v.co.z)
    max_v.x = max(max_v.x, v.co.x)
    max_v.y = max(max_v.y, v.co.y)
    max_v.z = max(max_v.z, v.co.z)
  center = (min_v + max_v) * 0.5
  if center.length < 1e-8:
    return
  for v in obj.data.vertices:
    v.co -= center
  # Mesh vertex coordinates are already in Blender local space here.
  obj.location += center
  obj.data.update()


def _transform_mesh_data(obj, matrix):
  """Apply a mesh-space transform to object data in-place."""
  if obj.type != "MESH" or not obj.data:
    return
  obj.data.transform(matrix)
  obj.data.update()


def _offset_mesh_world(obj, delta_world):
  """Translate mesh vertices by a world-space vector, preserving object transforms."""
  if delta_world.length < 1e-9:
    return
  if obj.type != "MESH" or not obj.data:
    return
  local_delta = obj.matrix_world.inverted().to_3x3() @ delta_world
  _transform_mesh_data(obj, Matrix.Translation(local_delta))


def _is_identity_matrix_approx(mat, eps=1e-6):
  try:
    ident = Matrix.Identity(4)
    for r in range(4):
      for c in range(4):
        if abs(float(mat[r][c]) - float(ident[r][c])) > eps:
          return False
    return True
  except Exception:
    return False


def _is_fake_light_render_node(render):
  if render is None:
    return False
  return type(render).__name__ in ("FakeOmniLightsNode", "FakeSpotLightsNode", "FakeALSNode")


