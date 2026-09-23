import bpy
from mathutils import Matrix, Vector

from .materials_bridge import (
    _create_official_group_node,
)
from .prelude import (
    _ensure_official_material_bridge,
    _set_official_special_type,
)


def _create_official_render_material(obj_name, kind="default"):
    bridge = _ensure_official_material_bridge()
    if not bridge.get("available"):
        return None

    names = bridge.get("names", {})
    official_name = names.get(kind, names.get("default", "EDM_Default_Material"))

    mat = bpy.data.materials.new(name=obj_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    group_node = _create_official_group_node(nodes, bridge, kind, official_name)
    if group_node is None:
        return None

    group_node.location = (0, 0)
    if group_node.outputs:
        try:
            links.new(group_node.outputs[0], output_node.inputs["Surface"])
        except Exception as e:
            print(f"Warning in blender_importer/lights.py: {e}")
    return mat


def _billboard_plane_axes(axis_name):
    axis = str(axis_name or "all")
    if axis in {"x", "along_x"}:
        return Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0))
    if axis in {"y", "along_y"}:
        return Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))
    return Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0))


def _create_billboard_surrogate_mesh(name, axis_name, matrix_value, pivot_value):
    pivot = Vector((0.0, 0.0, 0.0))
    if pivot_value is not None:
        try:
            pivot = Vector(
                (float(pivot_value[0]), float(pivot_value[1]), float(pivot_value[2]))
            )
        except Exception:
            pivot = Vector((0.0, 0.0, 0.0))

    axis_u, axis_v = _billboard_plane_axes(axis_name)
    half_size = 0.5
    local_verts = [
        pivot + ((-half_size) * axis_u) + ((-half_size) * axis_v),
        pivot + ((half_size) * axis_u) + ((-half_size) * axis_v),
        pivot + ((half_size) * axis_u) + ((half_size) * axis_v),
        pivot + ((-half_size) * axis_u) + ((half_size) * axis_v),
    ]

    transform = Matrix.Identity(4)
    if matrix_value is not None:
        try:
            transform = Matrix(matrix_value)
        except Exception:
            transform = Matrix.Identity(4)

    verts = []
    for vert in local_verts:
        try:
            verts.append(tuple((transform @ vert.to_4d()).xyz))
        except Exception:
            verts.append(tuple(vert))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()

    try:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        uv_values = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        for loop_index, uv in enumerate(uv_values):
            uv_layer.data[loop_index].uv = uv
    except Exception as e:
        print(f"Warning in blender_importer/lights.py: {e}")

    return mesh


def create_billboard(node):
    """Create an exporter-compatible approximate mesh surrogate for BillboardNode."""

    # Type mapping based on edm_plugin_ref.html
    type_map = {0: "direction", 1: "point"}
    axis_map = {
        0: "all",
        1: "x",
        2: "y",
        3: "z",
        4: "along_x",
        5: "along_y",
        6: "along_z",
    }

    billboard_type = type_map.get(getattr(node, "billboard_type", 0), "point")
    billboard_axis = axis_map.get(getattr(node, "billboard_axis", 0), "all")
    mesh = _create_billboard_surrogate_mesh(
        node.name or "Billboard",
        billboard_axis,
        getattr(node, "matrix", None),
        getattr(node, "pivot", None),
    )
    obj = bpy.data.objects.new(name=node.name or "Billboard", object_data=mesh)
    _set_official_special_type(obj, "UNKNOWN_TYPE")
    obj.edm.billboard_type = billboard_type
    obj.edm.billboard_axis = billboard_axis
    obj["_iedm_translation_status"] = "approximate"
    obj["_iedm_translation_source"] = "BillboardNode"
    obj["_iedm_billboard_surrogate"] = True

    if getattr(node, "matrix", None) is not None:
        obj["_iedm_billboard_matrix"] = [float(v) for row in node.matrix for v in row]
    if getattr(node, "pivot", None) is not None:
        obj["_iedm_billboard_pivot"] = [float(v) for v in node.pivot]

    if mesh is not None and not mesh.materials:
        fallback_mat = _create_official_render_material(
            f"{obj.name}_Billboard", kind="default"
        )
        if fallback_mat is not None:
            mesh.materials.append(fallback_mat)

    bpy.context.collection.objects.link(obj)
    return obj
