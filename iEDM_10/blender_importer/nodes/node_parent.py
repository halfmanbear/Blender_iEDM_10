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

    def _direct_bone_parent():
        bone_nodes = (ctx or {}).get("bone_name_by_node", {}) or {}
        cur = getattr(node, "parent", None)
        if cur in bone_nodes:
            return cur, bone_nodes[cur]
        return None, None

    # Skin: look for a same-name sibling transform to use as parent
    try:
        if isinstance(getattr(node, "render", None), SkinNode):
            parent_tf = getattr(getattr(node, "parent", None), "transform", None)
            node.blender["_iedm_dbg_skin_parent_tf_cls"] = str(
                type(parent_tf).__name__ or ""
            )
            node.blender["_iedm_dbg_skin_parent_tf_name"] = str(
                getattr(parent_tf, "name", "") or ""
            )
            node.blender["_iedm_dbg_skin_render_name"] = str(
                getattr(node.render, "name", "") or ""
            )
            sib_names = []
            for sib in list(
                getattr(getattr(node, "parent", None), "children", []) or []
            ):
                sib_tf = getattr(sib, "transform", None)
                sib_r = getattr(sib, "render", None)
                sib_names.append(
                    "{}:{}:{}".format(
                        type(sib_tf).__name__ if sib_tf is not None else "",
                        getattr(sib_tf, "name", "") if sib_tf is not None else "",
                        getattr(sib_r, "name", "") if sib_r is not None else "",
                    )
                )
            if sib_names:
                node.blender["_iedm_dbg_skin_siblings"] = json.dumps(
                    sib_names, separators=(",", ":")
                )
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")

    if not node.blender.get("_iedm_skin_parent_override"):
        sibling_skin_helper = None
        try:
            if (
                isinstance(getattr(node, "render", None), SkinNode)
                and getattr(node, "transform", None) is None
                and node.parent is not None
                and isinstance(
                    getattr(node.parent, "transform", None), ArgVisibilityNode
                )
            ):
                render_name = str(getattr(node.render, "name", "") or "")
                for sib in list(getattr(node.parent, "children", []) or []):
                    if sib is node:
                        continue
                    sib_tf = getattr(sib, "transform", None)
                    sib_bl = getattr(sib, "blender", None)
                    if sib_tf is None or sib_bl is None or sib_bl == node.blender:
                        continue
                    if isinstance(sib_tf, ArgVisibilityNode):
                        continue
                    sib_name = str(getattr(sib_tf, "name", "") or "")
                    if sib_name and sib_name == render_name:
                        sibling_skin_helper = sib_bl
                        break
                try:
                    node.blender["_iedm_dbg_skin_helper_name"] = (
                        getattr(sibling_skin_helper, "name", "")
                        if sibling_skin_helper is not None
                        else ""
                    )
                except Exception:
                    _logger.debug("Ignoring optional operation failure", exc_info=True)
        except Exception as e:
            print(f"Warning in blender_importer/nodes/core.py: {e}")

        bone_parent_node, bone_parent_name = _direct_bone_parent()
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

    # Reparent render-only duplicate chunks away from mesh parents
    try:
        ob_name = getattr(node.blender, "name", "") or ""
        parent_bl = getattr(getattr(node, "parent", None), "blender", None)
        grandparent_bl = getattr(
            getattr(getattr(node, "parent", None), "parent", None), "blender", None
        )
        is_render_only_dup = (
            node.render is not None
            and node.transform is None
            and len(ob_name) > 4
            and ob_name[-4] == "."
            and ob_name[-3:].isdigit()
        )
        parent_is_render_mesh = (
            parent_bl is not None
            and getattr(parent_bl, "type", "") == "MESH"
            and getattr(getattr(node, "parent", None), "render", None) is not None
        )
        if is_render_only_dup and parent_is_render_mesh and grandparent_bl is not None:
            try:
                gp_world_inv = grandparent_bl.matrix_world.inverted_safe()
                node.blender.parent = grandparent_bl
                node.blender.matrix_parent_inverse = Matrix.Identity(4)
                node.blender.matrix_local = gp_world_inv @ parent_bl.matrix_world
            except Exception:
                preserved_local = parent_bl.matrix_local.copy()
                _loc, _rot, _ = preserved_local.decompose()
                node.blender.parent = grandparent_bl
                node.blender.matrix_parent_inverse = Matrix.Identity(4)
                node.blender.matrix_local = (
                    Matrix.Translation(_loc) @ _rot.to_matrix().to_4x4()
                )
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
