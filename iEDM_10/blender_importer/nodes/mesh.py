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
  obj.data.transform(Matrix.Translation(-center))
  # Include object rotation and scale when compensating in mesh-local space.
  obj.matrix_basis = obj.matrix_basis @ Matrix.Translation(center)
  obj.data.update()


def _transform_mesh_data(obj, matrix):
  """Apply a mesh-space transform to object data in-place."""
  if obj.type != "MESH" or not obj.data:
    return
  obj.data.transform(matrix)
  obj.data.update()


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


