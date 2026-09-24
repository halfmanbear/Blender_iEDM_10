import bpy
from mathutils import Vector


def _create_fake_light_mesh(name, positions):
    """Create a mesh with one vertex per light position (already in Blender coords)."""
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(positions))
    for i, pos in enumerate(positions):
        mesh.vertices[i].co = pos
    mesh.update()
    return mesh


def _fake_light_world_from_edm(pos_edm):
    """
    Fake-light point payloads are exported through the official addon's
    ROOT_TRANSFORM_MATRIX. Invert that here so importer-created Blender objects
    match the reference .blend topology and object placement.
    """
    return Vector((float(pos_edm[0]), -float(pos_edm[2]), float(pos_edm[1])))


def _create_fake_omni_mesh(name, world_positions):
    if not world_positions:
        return _create_fake_light_mesh(name, []), Vector((0.0, 0.0, 0.0))

    center = Vector((0.0, 0.0, 0.0))
    for pos in world_positions:
        center += pos
    center /= float(len(world_positions))

    local_positions = [pos - center for pos in world_positions]
    unique_positions = {
        tuple(round(float(c), 6) for c in pos) for pos in local_positions
    }
    if len(unique_positions) == 8:
        xs = sorted({round(pos[0], 6) for pos in unique_positions})
        ys = sorted({round(pos[1], 6) for pos in unique_positions})
        zs = sorted({round(pos[2], 6) for pos in unique_positions})
        if len(xs) == len(ys) == len(zs) == 2:
            verts = [
                (xs[0], ys[0], zs[0]),
                (xs[0], ys[0], zs[1]),
                (xs[0], ys[1], zs[0]),
                (xs[0], ys[1], zs[1]),
                (xs[1], ys[0], zs[0]),
                (xs[1], ys[0], zs[1]),
                (xs[1], ys[1], zs[0]),
                (xs[1], ys[1], zs[1]),
            ]
            faces = [
                (0, 1, 3, 2),
                (4, 6, 7, 5),
                (0, 4, 5, 1),
                (2, 3, 7, 6),
                (0, 2, 6, 4),
                (1, 5, 7, 3),
            ]
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata(verts, [], faces)
            mesh.update()
            return mesh, center

    return _create_fake_light_mesh(name, local_positions), center


def _axis_box_layout(positions):
    unique_positions = {tuple(round(float(c), 6) for c in pos) for pos in positions}
    if len(unique_positions) != 8:
        return None, None
    xs = sorted({round(pos[0], 6) for pos in unique_positions})
    ys = sorted({round(pos[1], 6) for pos in unique_positions})
    zs = sorted({round(pos[2], 6) for pos in unique_positions})
    if len(xs) != 2 or len(ys) != 2 or len(zs) != 2:
        return None, None
    verts = [
        (xs[0], ys[0], zs[0]),
        (xs[0], ys[0], zs[1]),
        (xs[0], ys[1], zs[0]),
        (xs[0], ys[1], zs[1]),
        (xs[1], ys[0], zs[0]),
        (xs[1], ys[0], zs[1]),
        (xs[1], ys[1], zs[0]),
        (xs[1], ys[1], zs[1]),
    ]
    faces = [
        (0, 1, 3, 2),
        (2, 3, 7, 6),
        (6, 7, 5, 4),
        (4, 5, 1, 0),
        (2, 6, 4, 0),
        (7, 3, 1, 5),
    ]
    return verts, faces


def _create_axis_box_mesh(name, positions):
    verts, faces = _axis_box_layout(positions)
    if verts is None:
        return None
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def _classify_fake_spot_mode(positions, dirs_bl):
    if not positions:
        return "surface"
    if not dirs_bl:
        return "surface"
    ref = None
    uniform = True
    for dvec in dirs_bl:
        if dvec.length <= 1.0e-8:
            continue
        cur = dvec.normalized()
        if ref is None:
            ref = cur
            continue
        if ref.dot(cur) < 0.999:
            uniform = False
            break
    if not uniform:
        return "surface"
    verts, _faces = _axis_box_layout(positions)
    if verts is not None:
        return "non_surface_box"
    return "non_surface_points"


def _default_fake_spot_uvs(two_sided):
    front_lb = (0.0, 0.5)
    front_rt = (0.25, 1.0)
    back_lb = (0.75, 0.5) if two_sided else (0.0, 0.0)
    back_rt = (1.0, 1.0)
    return front_lb, front_rt, back_lb, back_rt


def _add_fake_spot_direction_child(parent_obj, direction, distance=1.0):
    if parent_obj is None:
        return None
    dvec = Vector(direction)
    if dvec.length <= 1.0e-8:
        dvec = Vector((1.0, 0.0, 0.0))
    else:
        dvec.normalize()
    child = bpy.data.objects.new("Light_Dir", None)
    child.empty_display_type = "PLAIN_AXES"
    child.empty_display_size = 0.1
    child.parent = parent_obj
    child.location = tuple(dvec * float(distance))
    quat = Vector((1.0, 0.0, 0.0)).rotation_difference(dvec)
    child.rotation_mode = "QUATERNION"
    child.rotation_quaternion = quat
    bpy.context.collection.objects.link(child)
    return child
