"""Place visibility where the official exporter can write it.

The exporter reads one VISIBLE argument per object from its own action. Two
imported layouts lose it on export:

* Bone-parented objects: the exporter re-parents only the object's innermost
  node under the bone, orphaning the visibility node and, for animated
  objects, the static transform nodes above it.
* Skins: EDM skin visibility lives on the ArgVisibilityNodes above the skin's
  control bone (palette[0]); the skinned mesh itself carries none.

Both get a chain of identity helper empties, one per visibility argument,
between the parent and the object. A bone-parented chain starts with a static
empty on the bone so the whole chain moves under the bone together.
"""

from types import SimpleNamespace

import bpy
from mathutils import Matrix

from ..edm_format.types import ArgAnimatedBone, ArgVisibilityNode, Bone, SkinNode
from ..utils import action_fcurves
from .anim_actions import create_visibility_actions
from .prelude import _assign_action, _import_ctx


def _helper(name, parent, collection):
    helper = bpy.data.objects.new(name, None)
    helper.empty_display_size = 0.05
    collection.objects.link(helper)
    helper.parent = parent
    helper.matrix_parent_inverse = Matrix.Identity(4)
    helper.matrix_basis = Matrix.Identity(4)
    helper["_iedm_vis_passthrough"] = True
    return helper


def _vis_chain(obj, parent, entries):
    """Create one helper per (arg, ranges) entry below ``parent``."""
    collection = obj.users_collection[0]
    base = str(obj.get("_iedm_vis_export_name", "") or obj.name)
    for arg, ranges in entries:
        source = SimpleNamespace(name=base, visData=[(arg, ranges)])
        helper = _helper(f"v_{base}_{arg}", parent, collection)
        _assign_action(helper, create_visibility_actions(source)[0])
        parent = helper
    return parent


def _reparent(obj, parent, parent_inverse):
    """Swap parents without touching the basis.

    The helper chain reproduces the old parent frame exactly, so no world
    matrix is needed; objects hidden at the import frame have stale ones.
    """
    obj.parent = parent
    obj.parent_type = "OBJECT"
    # The exporter tests parent_bone, not parent_type.
    obj.parent_bone = ""
    obj.matrix_parent_inverse = parent_inverse


def _ancestor_vis_entries(node):
    """Visibility entries on ``node`` and its ancestors, outermost first."""
    chain = []
    while node is not None and not isinstance(node, (int, list)):
        if isinstance(node, ArgVisibilityNode):
            chain.append(node)
        node = getattr(node, "parent", None)
    return [(arg, ranges) for vis in reversed(chain) for arg, ranges in vis.visData]


def _bone_vis_entries():
    """Map armature bone name -> visibility above its source bone node."""
    ctx = _import_ctx.bone_import_ctx or {}
    mapping = ctx.get("bone_name_by_node", {}) or {}
    return {
        name: _ancestor_vis_entries(getattr(node, "transform", None))
        for node, name in mapping.items()
    }


def _source_bone_vis_entries(graph):
    """Map object -> visibility above the bone in its own source chain.

    Bone names are not unique enough to key on: two source bones can share a
    name while only one sits under a visibility node.
    """
    entries = {}
    for node in getattr(graph, "nodes", []) or []:
        obj = getattr(node, "blender", None)
        if obj is None or obj in entries:
            continue
        source = getattr(node, "transform", None)
        if source is None:
            source = getattr(getattr(node, "render", None), "parent", None)
        while source is not None and not isinstance(source, (int, list)):
            if isinstance(source, (Bone, ArgAnimatedBone)):
                entries[obj] = _ancestor_vis_entries(source)
                break
            source = getattr(source, "parent", None)
    return entries


def _needs_bone_wrapper(obj, entries):
    # The exporter moves only the innermost node of a bone-parented object
    # under the bone, so any visibility or animation node chain is broken.
    if entries:
        return True
    return bool(obj.animation_data and obj.animation_data.action)


def _wrap_bone_parented(obj, bone_entries, source_entries):
    """Put a static anchor on the bone and hang the object below it.

    With parent_bone cleared the exporter writes the object's own visibility
    and transform animation normally; the bone's ancestor visibility, which an
    armature cannot carry, goes on helpers between anchor and object.
    """
    if obj.parent_type != "BONE" or obj.parent is None:
        return False
    entries = source_entries.get(obj)
    if entries is None:
        entries = bone_entries.get(obj.parent_bone, [])
    if not _needs_bone_wrapper(obj, entries):
        return False
    anchor = _helper(obj.name + "_bone", obj.parent, obj.users_collection[0])
    anchor.parent_type = "BONE"
    anchor.parent_bone = obj.parent_bone
    anchor.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
    anchor.matrix_basis = Matrix.Identity(4)
    _reparent(obj, _vis_chain(obj, anchor, entries), Matrix.Identity(4))
    return True


def _skin_vis_entries(skin_node):
    """Visibility entries above the control bone, outermost first."""
    bones = [b for b in getattr(skin_node, "bones", []) or [] if not isinstance(b, int)]
    return _ancestor_vis_entries(bones[0] if bones else skin_node.parent)


def _wrap_skin(obj, skin_node):
    entries = _skin_vis_entries(skin_node)
    if not entries or obj.parent is None:
        return False
    action = obj.animation_data.action if obj.animation_data else None
    if action is not None and action_fcurves(action).find("VISIBLE") is not None:
        return False
    _reparent(
        obj, _vis_chain(obj, obj.parent, entries), obj.matrix_parent_inverse.copy()
    )
    return True


def place_export_visibility(graph):
    """Move visibility the exporter would drop onto helper chains."""
    counts = {"bone": 0, "skin": 0}
    bone_entries = _bone_vis_entries()
    source_entries = _source_bone_vis_entries(graph)
    for obj in list(bpy.data.objects):
        if _wrap_bone_parented(obj, bone_entries, source_entries):
            counts["bone"] += 1
    for node in getattr(graph, "nodes", []) or []:
        obj = getattr(node, "blender", None)
        if isinstance(getattr(node, "render", None), SkinNode) and obj is not None:
            if _wrap_skin(obj, node.render):
                counts["skin"] += 1
    bpy.context.view_layer.update()
    return counts
