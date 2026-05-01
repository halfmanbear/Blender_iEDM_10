def _arg_anim_vector_to_blender(node, value):
  # v10 top-level pylon positions are already in Z-up (Blender) space —
  # no basis conversion needed; _anim_vector_to_blender handles the Y/Z swap.
  return _anim_vector_to_blender(value)


def _normalize_angle_radians(angle, eps=1e-6):
  try:
    a = float(angle)
  except Exception:
    return angle
  two_pi = math.tau
  while a > math.pi:
    a -= two_pi
  while a <= -math.pi:
    a += two_pi
  if abs(a) <= eps or abs(a - two_pi) <= eps or abs(a + two_pi) <= eps:
    return 0.0
  return a


def _normalize_euler_xyz(euler, eps=1e-6):
  try:
    return Euler((
      _normalize_angle_radians(euler.x, eps),
      _normalize_angle_radians(euler.y, eps),
      _normalize_angle_radians(euler.z, eps),
    ), 'XYZ')
  except Exception:
    return euler


def _normalize_euler_action_curves(action, eps=1e-6):
  if action is None:
    return
  curves = [action.fcurves.find('rotation_euler', index=i) for i in range(3)]
  if any(fc is None for fc in curves):
    return
  point_count = min(len(fc.keyframe_points) for fc in curves)
  for idx in range(point_count):
    try:
      euler = Euler((
        curves[0].keyframe_points[idx].co[1],
        curves[1].keyframe_points[idx].co[1],
        curves[2].keyframe_points[idx].co[1],
      ), 'XYZ')
      norm = _normalize_euler_xyz(euler, eps)
      curves[0].keyframe_points[idx].co[1] = norm.x
      curves[1].keyframe_points[idx].co[1] = norm.y
      curves[2].keyframe_points[idx].co[1] = norm.z
    except Exception:
      continue
  for fc in curves:
    try:
      fc.update()
    except Exception:
      pass


def _finalize_authored_transform_action(action):
  if action is None:
    return
  _normalize_euler_action_curves(action)
  for fc in action.fcurves:
    try:
      fc.extrapolation = 'CONSTANT'
    except Exception:
      pass


def add_position_fcurves(action, keys, transform_left, transform_right, node=None, frame_mapper=None):
  "Adds position fcurve data to an animation action"
  curves = []
  for i in range(3):
    curve = action.fcurves.find("location", index=i)
    if curve is None:
      curve = action.fcurves.new(data_path="location", index=i)
    curves.append(curve)

  frame_mapper = frame_mapper or _anim_frame_to_scene_frame

  for framedata in keys:
    frame = frame_mapper(framedata.frame)
    anim_pos = _arg_anim_vector_to_blender(node, framedata.value) if node is not None else _anim_vector_to_blender(framedata.value)
    newPosMat = transform_left @ Matrix.Translation(anim_pos) @ transform_right
    newPos = newPosMat.decompose()[0]

    for curve, component in zip(curves, newPos):
      curve.keyframe_points.add(1)
      curve.keyframe_points[-1].co = (frame, component)
      curve.keyframe_points[-1].interpolation = 'LINEAR'

  for curve in curves:
    try:
      curve.update()
    except Exception:
      pass


def add_rotation_fcurves(action, keys, transform_left, transform_right, quat_to_blender=None, frame_mapper=None, use_euler=False):
  "Adds rotation fcurve action to an animation action"
  if quat_to_blender is None:
    quat_to_blender = _anim_quaternion_to_blender
  frame_mapper = frame_mapper or _anim_frame_to_scene_frame
  curves = []
  data_path = "rotation_euler" if use_euler else "rotation_quaternion"
  curve_count = 3 if use_euler else 4
  for i in range(curve_count):
    curve = action.fcurves.find(data_path, index=i)
    if curve is None:
      curve = action.fcurves.new(data_path=data_path, index=i)
    curves.append(curve)

  previous_quat = None
  previous_euler = None
  for framedata in keys:
    frame = frame_mapper(framedata.frame)
    newRotQuat = transform_left @ quat_to_blender(framedata.value) @ transform_right
    if previous_quat is not None and previous_quat.dot(newRotQuat) < 0.0:
      newRotQuat = -newRotQuat
    previous_quat = newRotQuat.copy()

    if use_euler:
      euler = newRotQuat.to_euler('XYZ')
      if previous_euler is not None:
        euler.make_compatible(previous_euler)
      previous_euler = euler.copy()
      components = euler
    else:
      components = newRotQuat

    for curve, component in zip(curves, components):
      curve.keyframe_points.add(1)
      curve.keyframe_points[-1].co = (frame, component)
      curve.keyframe_points[-1].interpolation = 'LINEAR'

  for curve in curves:
    try:
      curve.update()
    except Exception:
      pass


def add_scale_fcurves(action, keys, frame_mapper=None, base_scale=None):
  "Adds scale fcurve action to an animation action"
  if not keys:
    return
  frame_mapper = frame_mapper or _anim_frame_to_scene_frame
  base_components = (1.0, 1.0, 1.0)
  if base_scale is not None:
    try:
      base_components = tuple(float(base_scale[i]) for i in range(3))
    except Exception:
      base_components = (1.0, 1.0, 1.0)
  curves = []
  for i in range(3):
    curve = action.fcurves.find("scale", index=i)
    if curve is None:
      curve = action.fcurves.new(data_path="scale", index=i)
    curves.append(curve)

  for framedata in keys:
    frame = frame_mapper(framedata.frame)
    value = framedata.value
    comps = tuple(
      float(component) * base_components[idx]
      for idx, component in enumerate(_anim_scale_components(value))
    )

    for curve, component in zip(curves, comps):
      curve.keyframe_points.add(1)
      curve.keyframe_points[-1].co = (frame, component)
      curve.keyframe_points[-1].interpolation = 'LINEAR'

  for curve in curves:
    try:
      curve.update()
    except Exception:
      pass


def _is_pos90_x_basis_matrix(mat, eps=1e-3):
  try:
    target = (
      (1.0, 0.0, 0.0),
      (0.0, 0.0, -1.0),
      (0.0, 1.0, 0.0),
    )
    for r in range(3):
      for c in range(3):
        if abs(float(mat[r][c]) - target[r][c]) > eps:
          return False
    return True
  except Exception:
    return False


def _quat_is_identity(q, eps=1e-4):
  try:
    qv = q if hasattr(q, "to_matrix") else Quaternion(q)
    # Both q=(1,0,0,0) and q=(-1,0,0,0) represent the same identity rotation.
    # Some exporters (e.g. io_scene_edm) write quat_2 as (-1,0,0,0) for
    # scale-orientation fields even on pure rotation nodes.  Treat both forms
    # as identity so oriented-scale rewrites are not triggered spuriously.
    return (
      abs(abs(float(qv.w)) - 1.0) <= eps
      and abs(float(qv.x)) <= eps
      and abs(float(qv.y)) <= eps
      and abs(float(qv.z)) <= eps
    )
  except Exception:
    return False


def _quat_is_halfturn_x(q, eps=1e-4):
  try:
    qv = q if hasattr(q, "to_matrix") else Quaternion(q)
    return (
      abs(float(qv.w)) <= eps
      and abs(abs(float(qv.x)) - 1.0) <= eps
      and abs(float(qv.y)) <= eps
      and abs(float(qv.z)) <= eps
    )
  except Exception:
    return False


def _quat_is_halfturn_z(q, eps=1e-4):
  try:
    qv = q if hasattr(q, "to_matrix") else Quaternion(q)
    return (
      abs(float(qv.w)) <= eps
      and abs(float(qv.x)) <= eps
      and abs(float(qv.y)) <= eps
      and abs(abs(float(qv.z)) - 1.0) <= eps
    )
  except Exception:
    return False


def _all_negative_base_scale(node, eps=1e-6):
  try:
    scale = getattr(getattr(node, "base", None), "scale", None)
    if scale is None:
      return False
    return all(float(v) < -eps for v in scale[:3])
  except Exception:
    return False
