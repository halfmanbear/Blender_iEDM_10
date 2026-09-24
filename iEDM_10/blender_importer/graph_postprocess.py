from mathutils import Matrix

from ..edm_format.types import ArgScaleNode
from .prelude import (
    _ROOT_BASIS_FIX,
    _import_ctx,
    _is_root_visibility_pair_child,
    _is_static_root_visibility_wrapper,
    _log,
)


def _reparent_preserve_world(child, new_parent):
    if child is None:
        return
    world = child.matrix_world.copy()
    child.parent = new_parent
    child.matrix_parent_inverse = Matrix.Identity(4)
    child.matrix_world = world


def _zero_render_child_mesh_locals_under_transform(graph):
    """Leave render-only mesh children at identity under explicit transform wrappers."""
    for node in getattr(graph, "nodes", []) or []:
        render = getattr(node, "render", None)
        ob = getattr(node, "blender", None)
        if render is None or ob is None or getattr(ob, "type", "") != "MESH":
            continue
        if getattr(node, "transform", None) is not None:
            continue
        parent = getattr(node, "parent", None)
        parent_tf = getattr(parent, "transform", None) if parent is not None else None
        if parent_tf is None or isinstance(parent_tf, ArgScaleNode):
            continue
        shared_parent = getattr(render, "shared_parent", None)
        if shared_parent is not None and shared_parent is not parent_tf:
            continue
        try:
            _loc, _rot, _scale = ob.matrix_basis.decompose()
            if abs(_rot.angle) > 1e-3:
                continue
            ob.matrix_basis = Matrix.Identity(4)
        except Exception as e:
            _log.warn("_zero_render_child_mesh_locals_under_transform", exc=e)


def _apply_root_visibility_pair_wrapper_basis_fix(graph):
    seen = set()
    for node in getattr(graph, "nodes", []) or []:
        if not _is_root_visibility_pair_child(node):
            continue
        parent = getattr(node, "parent", None)
        ob = getattr(parent, "blender", None)
        if ob is None:
            continue
        key = ob.name_full if hasattr(ob, "name_full") else id(ob)
        if key in seen:
            continue
        seen.add(key)
        try:
            ob.matrix_basis = _ROOT_BASIS_FIX.inverted() @ ob.matrix_basis
        except Exception as e:
            _log.warn("_apply_root_visibility_pair_wrapper_basis_fix", exc=e)


def _apply_static_root_visibility_wrapper_basis_fix(graph):
    if getattr(_import_ctx, "use_scene_root_basis_object", True):
        return

    seen_objects = set()
    for node in getattr(graph, "nodes", []) or []:
        if not _is_static_root_visibility_wrapper(node):
            continue
        ob = getattr(node, "blender", None)
        if ob is None:
            continue
        ob_key = ob.name_full if hasattr(ob, "name_full") else id(ob)
        if ob_key in seen_objects:
            continue
        seen_objects.add(ob_key)
        try:
            ob.matrix_basis = _ROOT_BASIS_FIX @ ob.matrix_basis
        except Exception as e:
            _log.warn("_apply_static_root_visibility_wrapper_basis_fix", exc=e)
