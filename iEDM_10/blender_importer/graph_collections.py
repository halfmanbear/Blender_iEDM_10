"""Categorize imported graph objects into Blender collections."""

import bpy as bpy
from mathutils import Matrix, Vector

from ..edm_format.types import ArgVisibilityNode
from .import_context import _log

_SEMANTIC_NAME_MAP = {}


def _get_or_create_child_col(parent, name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if parent is not None and col.name not in parent.children.keys():
        parent.children.link(col)
    return col


def _classify_graph_nodes(graph):
    obj_category = {}
    for n in graph.nodes:
        if not getattr(n, "_is_primary", False) or not n.blender:
            continue
        if n.render is None:
            obj_category[n.blender] = None
            continue
        rtype = type(n.render).__name__
        if rtype in ("ShellNode", "SegmentsNode"):
            obj_category[n.blender] = "collision"
        elif rtype == "Connector":
            obj_category[n.blender] = "vehicle"
        elif rtype == "RenderNode":
            if n.blender.parent is None:
                render_local = Matrix.Identity(4)
                try:
                    if getattr(n, "_local_bl", None) is not None:
                        render_local = Matrix(n._local_bl)
                except Exception:
                    render_local = Matrix.Identity(4)
                try:
                    if hasattr(n.render, "matrix"):
                        render_local = render_local @ Matrix(n.render.matrix)
                    elif hasattr(n.render, "pos"):
                        render_local = render_local @ Matrix.Translation(
                            Vector(n.render.pos[:3])
                        )
                except Exception as exc:
                    _log.debug("Optional operation failed: {}".format(exc), level=2)
                render_name = str(getattr(n.render, "name", "") or "")
                obj_category[n.blender] = (
                    "texture_anim"
                    if not render_name and render_local.is_identity
                    else None
                )
            else:
                obj_category[n.blender] = (
                    None
                    if isinstance(getattr(n, "transform", None), ArgVisibilityNode)
                    else "vehicle"
                )
        else:
            obj_category[n.blender] = None
    return obj_category


def _inherit_graph_collection_categories(graph, obj_category):
    changed = True
    while changed:
        changed = False
        for n in graph.nodes:
            if not getattr(n, "_is_primary", False) or not n.blender:
                continue
            if obj_category.get(n.blender) is not None:
                continue
            child_cats = {obj_category.get(child) for child in n.blender.children}
            if "collision" in child_cats and "vehicle" not in child_cats:
                obj_category[n.blender] = "collision"
                changed = True
            elif "vehicle" in child_cats or "texture_anim" in child_cats:
                obj_category[n.blender] = "vehicle"
                changed = True

    changed = True
    while changed:
        changed = False
        for obj, cat in list(obj_category.items()):
            if cat is None:
                continue
            for child in list(obj.children):
                if obj_category.get(child) is not None:
                    continue
                is_helper = child.get("_iedm_identity_passthrough") or child.get(
                    "_iedm_vis_passthrough"
                )
                if is_helper or getattr(child, "type", "") == "EMPTY":
                    obj_category[child] = cat
                    changed = True
    return obj_category


def _assign_collections(graph):
    """Create named import collections and assign Blender objects by category."""
    scene = bpy.context.scene

    col_vehicle = _get_or_create_child_col(scene.collection, "Vehicle")
    col_collision = _get_or_create_child_col(col_vehicle, "Collision")
    col_tex_anim = _get_or_create_child_col(scene.collection, "Texture_Animation")

    obj_category = _inherit_graph_collection_categories(
        graph, _classify_graph_nodes(graph)
    )

    _col_map = {
        "collision": col_collision,
        "vehicle": col_vehicle,
        "texture_anim": col_tex_anim,
    }

    _categorize_file_root(graph, obj_category)
    _move_objects_to_categories(obj_category, _col_map)

    def _exclude_layer_collection(layer_collection, name):
        if layer_collection.collection.name == name:
            layer_collection.exclude = True
            return True
        for child in layer_collection.children:
            if _exclude_layer_collection(child, name):
                return True
        return False

    _exclude_layer_collection(
        bpy.context.view_layer.layer_collection, "Texture_Animation"
    )

    _create_lod_collections(graph, col_vehicle)


def _categorize_file_root(graph, obj_category):
    """Place an unparented file root with its vehicle children."""
    root = getattr(getattr(graph, "root", None), "blender", None)
    if (
        not root
        or getattr(root, "name", "") != "_EDMFileRoot"
        or root.parent is not None
    ):
        return
    child_categories = {obj_category.get(child) for child in root.children}
    if "vehicle" in child_categories and "collision" not in child_categories:
        obj_category[root] = "vehicle"


def _move_objects_to_categories(obj_category, collection_map):
    """Move categorized Blender objects into their target collections."""
    for obj, category in obj_category.items():
        target = collection_map.get(category)
        if target is None:
            continue
        for current_collection in list(obj.users_collection):
            try:
                current_collection.objects.unlink(obj)
            except Exception as exc:
                _log.warn("collection unlink '{}': {}".format(obj.name, exc), exc=exc)
        try:
            target.objects.link(obj)
        except RuntimeError:
            pass


def _create_lod_collections(graph, vehicle_collection):
    """Create named collections for post-processed LOD levels."""
    # Names retain exporter-compatible LOD distances during round-trips.
    for n in graph.nodes:
        if not getattr(n, "_lod_post_children", False):
            continue
        levels = getattr(getattr(n, "transform", None), "level", [])
        for i, ((_start, end), child) in enumerate(
            zip(levels, n.children, strict=False)
        ):
            if not getattr(child, "blender", None):
                continue
            col_name = "LOD_{}_{}".format(i, int(end))
            col_lod = _get_or_create_child_col(vehicle_collection, col_name)
            objs_to_move = [child.blender] + list(child.blender.children_recursive)
            for obj in objs_to_move:
                if obj.name in vehicle_collection.objects:
                    try:
                        vehicle_collection.objects.unlink(obj)
                    except Exception as exc:
                        _log.debug("Optional operation failed: {}".format(exc), level=2)
                if obj.name not in col_lod.objects:
                    try:
                        col_lod.objects.link(obj)
                    except Exception as exc:
                        _log.debug("Optional operation failed: {}".format(exc), level=2)
