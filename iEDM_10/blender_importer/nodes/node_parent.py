# Fragment: core node processing — creates Blender objects from the EDM graph.
import json
import logging

from ...edm_format.mathtypes import (
    Matrix,
)
from ...edm_format.types import (
    ArgAnimatedBone,
    ArgVisibilityNode,
    Bone,
    SkinNode,
)
from ..prelude import (
    _ROOT_BASIS_FIX,
)

_logger = logging.getLogger(__name__)


def _parent_node_object(node, ctx):
    """Establish the Blender parent chain for node.blender."""
    _stamp_skin_parent_diagnostics(node)

    if not node.blender.get("_iedm_skin_parent_override"):
        sibling_skin_helper = _find_sibling_skin_helper(node)
        bone_parent_node, bone_parent_name = _direct_bone_parent(node, ctx)
        arm_obj = (ctx or {}).get("armature")
        if (
            sibling_skin_helper is None
            and arm_obj is not None
            and bone_parent_name
            and node is not bone_parent_node
            and node.blender is not arm_obj
            and not isinstance(
                getattr(node, "transform", None), (Bone, ArgAnimatedBone)
            )
        ):
            try:
                bone = getattr(getattr(arm_obj, "data", None), "bones", {}).get(
                    bone_parent_name
                )
                tail_len = 0.0
                if bone is not None:
                    tail_len = float(getattr(bone, "length", 0.0))

                node.blender.parent = arm_obj
                node.blender.parent_type = "BONE"
                node.blender.parent_bone = bone_parent_name
                # Bone parenting uses the tip and Blender normalizes edit-bone
                # scale. Convert back to the EDM bone frame before authored locals.
                source = bone_parent_node.transform
                inv_bind = Matrix(
                    getattr(
                        source,
                        "bone_matrix",
                        getattr(source, "inv_base_bone_matrix", Matrix.Identity(4)),
                    )
                )
                rest_world = arm_obj.matrix_world @ bone.matrix_local
                node.blender.matrix_parent_inverse = (
                    Matrix.Translation((0.0, -tail_len, 0.0))
                    @ rest_world.inverted()
                    @ _ROOT_BASIS_FIX
                    @ inv_bind.inverted()
                )
                node.blender["_iedm_parented_to_bone"] = bone_parent_name
                node.blender["_iedm_bone_attachment_corrected"] = True
            except Exception as e:
                print(f"Warning in blender_importer/nodes/core.py: {e}")
        elif sibling_skin_helper is not None:
            node.blender.parent = sibling_skin_helper
            node.blender.matrix_parent_inverse = Matrix.Identity(4)
        elif (
            node.parent and node.parent.blender and node.parent.blender != node.blender
        ):
            node.blender.parent = node.parent.blender
            node.blender.matrix_parent_inverse = Matrix.Identity(4)

    _reparent_render_only_duplicate(node)


def _direct_bone_parent(node, ctx):
    bone_nodes = (ctx or {}).get("bone_name_by_node", {}) or {}
    parent = getattr(node, "parent", None)
    return (parent, bone_nodes[parent]) if parent in bone_nodes else (None, None)


def _stamp_skin_parent_diagnostics(node):
    """Store graph-level skin parent details for post-processing diagnostics."""
    try:
        if not isinstance(getattr(node, "render", None), SkinNode):
            return
        parent_tf = getattr(getattr(node, "parent", None), "transform", None)
        node.blender["_iedm_dbg_skin_parent_tf_cls"] = type(parent_tf).__name__
        node.blender["_iedm_dbg_skin_parent_tf_name"] = str(
            getattr(parent_tf, "name", "") or ""
        )
        node.blender["_iedm_dbg_skin_render_name"] = str(
            getattr(node.render, "name", "") or ""
        )
        siblings = []
        for sibling in list(
            getattr(getattr(node, "parent", None), "children", []) or []
        ):
            transform = getattr(sibling, "transform", None)
            render = getattr(sibling, "render", None)
            siblings.append(
                "{}:{}:{}".format(
                    type(transform).__name__ if transform is not None else "",
                    getattr(transform, "name", "") if transform is not None else "",
                    getattr(render, "name", "") if render is not None else "",
                )
            )
        if siblings:
            node.blender["_iedm_dbg_skin_siblings"] = json.dumps(
                siblings, separators=(",", ":")
            )
    except Exception as exc:
        print(f"Warning in blender_importer/nodes/core.py: {exc}")


def _find_sibling_skin_helper(node):
    """Find the visibility sibling transform corresponding to a skin render."""
    try:
        if not (
            isinstance(getattr(node, "render", None), SkinNode)
            and getattr(node, "transform", None) is None
            and node.parent is not None
            and isinstance(getattr(node.parent, "transform", None), ArgVisibilityNode)
        ):
            return None
        render_name = str(getattr(node.render, "name", "") or "")
        found = None
        for sibling in list(getattr(node.parent, "children", []) or []):
            transform = getattr(sibling, "transform", None)
            blender_obj = getattr(sibling, "blender", None)
            if sibling is node or transform is None or blender_obj is None:
                continue
            if blender_obj == node.blender or isinstance(transform, ArgVisibilityNode):
                continue
            if str(getattr(transform, "name", "") or "") == render_name:
                found = blender_obj
                break
        try:
            node.blender["_iedm_dbg_skin_helper_name"] = (
                getattr(found, "name", "") if found is not None else ""
            )
        except Exception as exc:
            _logger.debug("Could not stamp skin helper metadata: {}".format(exc))
        return found
    except Exception as exc:
        print(f"Warning in blender_importer/nodes/core.py: {exc}")


def _reparent_render_only_duplicate(node):
    """Move duplicate render chunks away from their render-mesh parents."""
    try:
        name = getattr(node.blender, "name", "") or ""
        parent = getattr(getattr(node, "parent", None), "blender", None)
        grandparent = getattr(
            getattr(getattr(node, "parent", None), "parent", None), "blender", None
        )
        is_duplicate = (
            node.render is not None
            and node.transform is None
            and len(name) > 4
            and name[-4] == "."
            and name[-3:].isdigit()
        )
        parent_is_mesh = (
            parent is not None
            and getattr(parent, "type", "") == "MESH"
            and getattr(getattr(node, "parent", None), "render", None) is not None
        )
        if not (is_duplicate and parent_is_mesh and grandparent is not None):
            return
        try:
            world_inverse = grandparent.matrix_world.inverted_safe()
            node.blender.parent = grandparent
            node.blender.matrix_parent_inverse = Matrix.Identity(4)
            node.blender.matrix_local = world_inverse @ parent.matrix_world
        except Exception:
            local = parent.matrix_local.copy()
            location, rotation, _scale = local.decompose()
            node.blender.parent = grandparent
            node.blender.matrix_parent_inverse = Matrix.Identity(4)
            node.blender.matrix_local = (
                Matrix.Translation(location) @ rotation.to_matrix().to_4x4()
            )
    except Exception as exc:
        print(f"Warning in blender_importer/nodes/core.py: {exc}")
