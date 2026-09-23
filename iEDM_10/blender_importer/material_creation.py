"""Node-tree setup helpers for imported EDM materials."""

import bpy


def _create_material_node_tree(material):
    mat = bpy.data.materials.new(name=material.name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    principled_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    principled_bsdf.location = (0, 0)
    material_output = nodes.new("ShaderNodeOutputMaterial")
    material_output.location = (400, 0)
    links.new(principled_bsdf.outputs["BSDF"], material_output.inputs["Surface"])
    return mat, nodes, links, principled_bsdf


def _create_material_textures(
    material, nodes, links, find_texture_file, ensure_placeholder, wire_uv_transform
):
    texture_nodes = {}
    for tex_def in material.textures:
        filename = find_texture_file(tex_def.name)
        tex_image = nodes.new("ShaderNodeTexImage")
        tex_image.image = (
            bpy.data.images.load(filename, check_existing=True)
            if filename
            else ensure_placeholder(tex_def.name)
        )
        tex_image.label = str(getattr(tex_def, "name", "") or "")
        tex_image.location = (-400, tex_def.index * -300)
        texture_nodes[tex_def.index] = tex_image
        wire_uv_transform(nodes, links, tex_def, tex_image)
    return texture_nodes
