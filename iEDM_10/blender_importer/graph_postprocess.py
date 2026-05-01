def _reparent_preserve_world(child, new_parent):
  if child is None:
    return
  world = child.matrix_world.copy()
  child.parent = new_parent
  child.matrix_parent_inverse = Matrix.Identity(4)
  child.matrix_world = world
