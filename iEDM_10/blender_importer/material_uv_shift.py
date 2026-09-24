"""UV-shift animation wiring the official exporter reads back."""

# UV-shift uniform -> texture slot whose Mapping node the exporter reads it
# from (base, emissive, decal and AO blocks). DCS only samples diffuse, decal
# and AO shifts; emissiveShift is written by the exporter but never read.
_UV_SHIFT_SLOTS = {
    "diffuseShift": 0,
    "emissiveShift": 8,
    "decalShift": 3,
    "ambientOcclusionShift": 9,
    "lightMapShift": 9,
}


def _uv_shift_mapping_node(nodes, links, tex_node):
    """Mapping node feeding tex_node's Vector, inserted if missing.

    The exporter reads the Mapping node on the texture's first input and the
    UV Map node on the Mapping node's first input.
    """
    vector = tex_node.inputs["Vector"]
    source = vector.links[0].from_socket if vector.links else None
    if source is not None and source.node.type == "MAPPING":
        return source.node
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (tex_node.location[0] - 220, tex_node.location[1])
    mapping.vector_type = "POINT"
    if source is None:
        tex_coord = nodes.new("ShaderNodeTexCoord")
        tex_coord.location = (mapping.location[0] - 220, mapping.location[1])
        source = tex_coord.outputs["UV"]
    links.new(source, mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], vector)
    return mapping


def _keyframe_material_uv_location(mat, location, anim_path, framedata):
    """Apply one animated UV location and insert its keyframe."""
    value = getattr(framedata, "value", None)
    if value is None:
        return
    blender_frame = (float(getattr(framedata, "frame", 0.0)) + 1.0) * 100.0
    value = [float(component) for component in value][:3]
    value += [0.0] * (3 - len(value))
    # The exporter writes Mapping locations as (x, 1 - y).
    value[1] = 1.0 - value[1]
    location.default_value = value
    mat.node_tree.keyframe_insert(data_path=anim_path, frame=blender_frame)
