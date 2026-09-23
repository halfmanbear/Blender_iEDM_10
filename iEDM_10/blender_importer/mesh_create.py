import bmesh
import bpy

from ..edm_format.mathtypes import (
    Matrix,
    Vector,
)


def _transform_mesh_vertices(source_vertices, vertex_format, transform):
    if transform is None:
        return source_vertices
    transform_mat = Matrix(transform) if not hasattr(transform, "to_4x4") else transform
    pos_idx = vertex_format.position_indices
    norm_idx = vertex_format.normal_indices
    normal_mat = transform_mat.to_3x3().inverted_safe().transposed()
    transformed = []
    for source in source_vertices:
        vertex = list(source)
        position = Vector(vertex[i] for i in pos_idx)
        transformed_position = transform_mat @ position.to_4d()
        vertex[0], vertex[1], vertex[2] = (
            transformed_position[0],
            transformed_position[1],
            transformed_position[2],
        )
        if norm_idx and len(norm_idx) >= 3:
            normal = Vector(vertex[i] for i in norm_idx[:3])
            transformed_normal = normal_mat @ normal
            if transformed_normal.length_squared > 1e-12:
                transformed_normal.normalize()
            for index, component in enumerate(norm_idx[:3]):
                vertex[component] = transformed_normal[index]
        transformed.append(vertex)
    return transformed


def _select_mesh_indices(vertex_data, index_data, compact):
    if compact:
        used_indices = sorted(set(index_data))
        source_vertices = [vertex_data[i] for i in used_indices]
        index_lookup = {
            old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)
        }
        return source_vertices, [index_lookup[i] for i in index_data]
    return vertex_data, index_data


def _mesh_primitive_mode(indices):
    if len(indices) % 3 == 0:
        return "triangles", indices
    if len(indices) == 1:
        return "points", indices
    if len(indices) % 2 == 0:
        return "lines", indices
    usable = len(indices) - (len(indices) % 3)
    print(
        "Warning: Non-triangle index count {}; truncating to {} indices".format(
            len(indices), usable
        )
    )
    return "triangles", indices[:usable]


def _create_mesh(
    vertexData,
    indexData,
    vertexFormat,
    compact=True,
    join_triangles=False,
    transform=None,
):
    """Creates a blender mesh object from vertex, index and format data

    Args:
    transform: Optional 4x4 matrix for converting vertex positions, e.g. Y-up to Z-up.
    """

    source_vertices, new_indices = _select_mesh_indices(vertexData, indexData, compact)

    new_vertices = _transform_mesh_vertices(source_vertices, vertexFormat, transform)
    primitive_mode, new_indices = _mesh_primitive_mode(new_indices)

    bm = bmesh.new()
    import_custom_normals = True
    vertex_normals = (
        [] if (import_custom_normals and vertexFormat.normal_indices) else None
    )

    position_indices = vertexFormat.position_indices
    normal_indices = vertexFormat.normal_indices
    uv_indices = _mesh_uv_index_sets(vertexFormat)
    _add_mesh_vertices(
        bm, new_vertices, position_indices, normal_indices, vertex_normals
    )
    uv_layers = _mesh_uv_layers(bm, len(uv_indices))
    skipped_degenerate, skipped_duplicate = _build_mesh_topology(
        bm, primitive_mode, new_indices, new_vertices, uv_layers, uv_indices
    )

    if skipped_degenerate or skipped_duplicate:
        print(
            "Info: Skipped {} degenerate and {} duplicate {} building mesh".format(
                skipped_degenerate,
                skipped_duplicate,
                "faces" if primitive_mode == "triangles" else "edges",
            )
        )

    _join_mesh_triangles(bm, join_triangles)
    return _finalize_mesh(bm, vertex_normals)


def _mesh_uv_index_sets(vertex_format):
    indices = []
    data = getattr(vertex_format, "data", None)
    if data is None:
        return indices
    offset = sum(data[:4])
    for channel in (4, 5, 6):
        if channel < len(data):
            count = int(data[channel])
            if count >= 2:
                indices.append(list(range(offset, offset + 2)))
            offset += max(0, count)
    return indices


def _add_mesh_vertices(bm, vertices, position_indices, normal_indices, vertex_normals):
    for values in vertices:
        position = Vector(values[index] for index in position_indices)
        vertex = bm.verts.new(position)
        if normal_indices and vertex_normals is not None:
            normal = Vector(values[index] for index in normal_indices)
            if normal.length_squared > 1e-12:
                normal.normalize()
            vertex.normal = normal
            vertex_normals.append(normal.copy())
    bm.verts.ensure_lookup_table()


def _mesh_uv_layers(bm, uv_set_count):
    layers = []
    for index in range(uv_set_count):
        name = "UVMap" if index == 0 else "UVMap.{:03d}".format(index)
        layer = bm.loops.layers.uv.get(name)
        if layer is None:
            layer = bm.loops.layers.uv.new(name)
        layers.append(layer)
    return layers


def _build_mesh_topology(bm, mode, indices, vertices, uv_layers, uv_index_sets):
    if mode == "triangles":
        return _build_triangle_faces(bm, indices, vertices, uv_layers, uv_index_sets)
    if mode == "lines":
        return _build_line_edges(bm, indices)
    # Point-list payloads need no extra topology; keep the referenced vertices.
    return 0, 0


def _build_triangle_faces(bm, indices, vertices, uv_layers, uv_index_sets):
    degenerate = duplicate = 0
    faces = (indices[i : i + 3] for i in range(0, len(indices), 3))
    for face in faces:
        if len(set(face)) < 3:
            degenerate += 1
            continue
        try:
            result = bm.faces.new([bm.verts[index] for index in face])
        except ValueError:
            duplicate += 1
            continue
        _set_face_uvs(result, face, vertices, uv_layers, uv_index_sets)
    return degenerate, duplicate


def _set_face_uvs(face, indices, vertices, uv_layers, uv_index_sets):
    for loop, vertex_index in zip(face.loops, indices, strict=False):
        vertex = vertices[vertex_index]
        for layer, uv_indices in zip(uv_layers, uv_index_sets, strict=False):
            if len(uv_indices) >= 2 and uv_indices[1] < len(vertex):
                loop[layer].uv = (vertex[uv_indices[0]], 1 - vertex[uv_indices[1]])


def _build_line_edges(bm, indices):
    degenerate = duplicate = 0
    edges = (indices[i : i + 2] for i in range(0, len(indices), 2))
    for edge in edges:
        if len(edge) < 2 or edge[0] == edge[1]:
            degenerate += 1
            continue
        try:
            bm.edges.new([bm.verts[edge[0]], bm.verts[edge[1]]])
        except ValueError:
            duplicate += 1
    return degenerate, duplicate


def _join_mesh_triangles(bm, join_triangles):
    if not join_triangles or not bm.faces:
        return
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
    except Exception as exc:
        print("Warning: Failed to join shell triangles: {}".format(exc))


def _finalize_mesh(bm, vertex_normals):
    mesh = bpy.data.meshes.new("Mesh")
    bm.to_mesh(mesh)
    bm.free()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    if vertex_normals and len(vertex_normals) == len(mesh.vertices):
        try:
            normals = [tuple(normal) for normal in vertex_normals]
            if hasattr(mesh, "normals_split_custom_set_from_vertices"):
                mesh.normals_split_custom_set_from_vertices(normals)
            else:
                mesh.normals_split_custom_set(
                    [normals[loop.vertex_index] for loop in mesh.loops]
                )
        except Exception as exc:
            print("Warning: Failed to assign custom normals: {}".format(exc))
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    mesh.update()
    return mesh
