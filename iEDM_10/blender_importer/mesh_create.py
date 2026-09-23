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

    for _i, vtx in enumerate(new_vertices):
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
        for face in [new_indices[i : i + 3] for i in range(0, len(new_indices), 3)]:
            if len(set(face)) < 3:
                skipped_degenerate += 1
                continue
            try:
                f = bm.faces.new([bm.verts[i] for i in face])
                # Add UV data if we have any
                if uvIndexSets:
                    for loop, v_idx in zip(f.loops, face, strict=False):
                        vtx = new_vertices[v_idx]
                        for uv_layer, uv_idx in zip(
                            uv_layers, uvIndexSets, strict=False
                        ):
                            if len(uv_idx) < 2:
                                continue
                            if uv_idx[1] >= len(vtx):
                                continue
                            loop[uv_layer].uv = (vtx[uv_idx[0]], 1 - vtx[uv_idx[1]])
            except ValueError:
                # BMesh rejects exact duplicate faces; keep track and continue.
                skipped_duplicate += 1
    elif primitive_mode == "lines":
        for edge in [new_indices[i : i + 2] for i in range(0, len(new_indices), 2)]:
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
            "Info: Skipped {} degenerate and {} duplicate {} building mesh".format(
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
