# Fragment: core node processing — creates Blender objects from the EDM graph.


import json

from ...edm_format.mathtypes import (
    Matrix,
    Quaternion,
)
from ...edm_format.types import (
    AnimatingNode,
    ArgAnimatedBone,
    ArgAnimationNode,
    ArgVisibilityNode,
    Bone,
)
from ..prelude import (
    _debug_log_event,
    _import_ctx,
    _node_visibility_chain_args,
    _ob_local_is_identity,
)
from .node_helpers import _is_narrow_safe_identity_helper_name


def _stamp_transform_metadata(node):
    if not (node.transform and node.blender):
        return
    try:
        node.transform._blender_obj = node.blender
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
    if isinstance(node.transform, ArgVisibilityNode):
        _stamp_visibility_transform_metadata(node)
    else:
        _stamp_static_transform_metadata(node)


def _stamp_visibility_transform_metadata(node):
    try:
        vis_alias = getattr(node.transform, "name", "") or ""
        for _pfx in ("ar_", "al_", "as_"):
            if vis_alias.startswith(_pfx):
                vis_alias = vis_alias[len(_pfx) :]
                break
        if vis_alias:
            node.blender["_iedm_vis_export_name"] = vis_alias
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
    try:
        vis_payload = []
        for entry in getattr(node.transform, "visData", []) or []:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            try:
                varg = int(entry[0])
            except Exception:
                continue
            vranges = []
            src_ranges = entry[1] if isinstance(entry[1], (list, tuple)) else []
            for rng in src_ranges:
                if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                    continue
                try:
                    vranges.append([float(rng[0]), float(rng[1])])
                except Exception:
                    continue
            vis_payload.append([varg, vranges])
        if vis_payload:
            node.blender["_iedm_vis_raw_args"] = json.dumps(
                vis_payload, separators=(",", ":")
            )
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")
    try:
        if node.render is None and getattr(node.blender, "type", None) == "EMPTY":
            node.blender["_iedm_vis_passthrough"] = True
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")


def _stamp_static_transform_metadata(node):
    try:
        ob = node.blender
        tf_name = getattr(node.transform, "name", "") or ""
        ob_name = getattr(ob, "name", "") or ""
        is_static_tf = not isinstance(node.transform, AnimatingNode)
        is_renderless_empty = (
            node.render is None and getattr(ob, "type", None) == "EMPTY"
        )
        has_one_child = len(getattr(node, "children", []) or []) == 1
        has_blender_dup_suffix = (
            len(ob_name) > 4 and ob_name[-4] == "." and ob_name[-3:].isdigit()
        )
        not_bone_related = not isinstance(node.transform, (Bone, ArgAnimatedBone))
        if (
            is_static_tf
            and is_renderless_empty
            and has_one_child
            and has_blender_dup_suffix
            and not_bone_related
            and _ob_local_is_identity(ob)
        ):
            ob["_iedm_identity_passthrough"] = True
            if _is_narrow_safe_identity_helper_name(
                ob_name
            ) or _is_narrow_safe_identity_helper_name(tf_name):
                ob["_iedm_narrow_identity_passthrough"] = True
        elif (
            is_static_tf
            and is_renderless_empty
            and has_one_child
            and not_bone_related
            and isinstance(getattr(node.parent, "transform", None), ArgVisibilityNode)
            and _ob_local_is_identity(ob)
        ):
            try:
                child0 = (getattr(node, "children", []) or [None])[0]
                child_render_cls = (
                    type(getattr(child0, "render", None)).__name__
                    if child0 is not None
                    else None
                )
                child_tf_name = str(
                    getattr(getattr(child0, "transform", None), "name", "") or ""
                )
            except Exception:
                child_render_cls = None
                child_tf_name = ""
            if (
                child_render_cls == "LightNode"
                or child_tf_name == "Fake Light Transform"
            ):
                ob["_iedm_identity_passthrough"] = True
                ob["_iedm_narrow_identity_passthrough"] = True
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")


def _stamp_render_metadata(node):
    if node.render is not None and node.blender is not None:
        vis_chain = []
        try:
            vis_chain = _node_visibility_chain_args(node)
            if vis_chain:
                node.blender["_iedm_src_vis_chain"] = json.dumps(
                    vis_chain, separators=(",", ":")
                )
                node.blender["_iedm_src_vis_nearest"] = int(vis_chain[0])
        except Exception as e:
            print(f"Warning in blender_importer/nodes/core.py: {e}")
        try:
            render_cls_name = type(node.render).__name__
            node.blender["_iedm_src_render_cls"] = render_cls_name
            render_name = str(getattr(node.render, "name", "") or "")
            if render_name:
                node.blender["_iedm_src_render_name"] = render_name
        except Exception as e:
            print(f"Warning in blender_importer/nodes/core.py: {e}")
        try:
            dbg = getattr(_import_ctx, "visibility_debug", {}) or {}
            if dbg.get("enabled"):
                tracked_args = {38, 147, 183, 187, 188, 214, 215, 224, 225}
                hit_args = [arg for arg in vis_chain if arg in tracked_args]
                if hit_args:
                    _debug_log_event(
                        "[iEDM][VISOBJ] obj={!r} render_cls={} render_name={!r} "
                        "vis_chain={} parent={!r}".format(
                            getattr(node.blender, "name", None),
                            type(node.render).__name__,
                            getattr(node.render, "name", "") or "",
                            hit_args,
                            getattr(
                                getattr(node.blender, "parent", None), "name", None
                            ),
                        )
                    )
        except Exception as e:
            print(f"Warning in blender_importer/nodes/core.py: {e}")


def _stamp_animation_name(node):
    if (
        node.transform is not None
        and isinstance(node.transform, AnimatingNode)
        and not isinstance(node.transform, (Bone, ArgAnimatedBone))
    ):
        try:
            orig_anim_name = getattr(node.transform, "name", "") or ""
            if (
                orig_anim_name
                and not orig_anim_name.startswith("al_")
                and not orig_anim_name.startswith("ar_")
                and not orig_anim_name.startswith("as_")
            ):
                node.blender["_iedm_orig_anim_name"] = orig_anim_name
        except Exception as e:
            print(f"Warning in blender_importer/nodes/core.py: {e}")


def _stamp_raw_animation_payload(node):
    if not isinstance(node.transform, ArgAnimationNode) or node.blender is None:
        return
    try:
        raw = {}
        _add_arganim_base_payload(node.transform, raw)
        _add_arganim_key_payload(node.transform, raw)
        _add_arganim_matrix_payload(node.transform, raw)
        if raw:
            node.blender["_iedm_raw_arganim_payload"] = json.dumps(
                raw, separators=(",", ":")
            )
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")


def _add_arganim_base_payload(transform, payload):
    base = getattr(transform, "base", None)
    if base is None:
        return
    try:
        matrix = Matrix(base.matrix)
        payload["base_matrix"] = [float(value) for row in matrix for value in row]
    except Exception as exc:
        print(f"Warning in blender_importer/nodes/core.py: {exc}")
    for field, attribute, count in (
        ("base_position", "position", 3),
        ("base_scale", "scale", 3),
    ):
        try:
            payload[field] = [
                float(value) for value in getattr(base, attribute)[:count]
            ]
        except Exception as exc:
            print(f"Warning in blender_importer/nodes/core.py: {exc}")
    for field in ("quat_1", "quat_2"):
        try:
            value = getattr(base, field)
            quaternion = value if hasattr(value, "__iter__") else Quaternion(value)
            payload["base_{}".format(field)] = [
                float(component) for component in quaternion
            ]
        except Exception as exc:
            print(f"Warning in blender_importer/nodes/core.py: {exc}")


def _add_arganim_key_payload(transform, payload):
    position_data = _serialize_arganim_keys(getattr(transform, "posData", None), 3)
    rotation_data = _serialize_arganim_keys(getattr(transform, "rotData", None), 4)
    if position_data:
        payload["pos_data"] = position_data
    if rotation_data:
        payload["rot_data"] = rotation_data


def _serialize_arganim_keys(entries, component_count):
    serialized = []
    for entry in entries or []:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        try:
            argument = int(entry[0])
        except Exception:
            continue
        keys = []
        for key in entry[1] or []:
            try:
                value = key.value
                if component_count == 4 and not hasattr(value, "__iter__"):
                    value = Quaternion(value)
                row = [float(key.frame)] + [
                    float(value[index]) for index in range(component_count)
                ]
                keys.append(row)
            except Exception:
                continue
        serialized.append([argument, keys])
    return serialized


def _add_arganim_matrix_payload(transform, payload):
    for field, matrix in (
        ("bmat_inv", getattr(transform, "bmat_inv", None)),
        (
            "zero_transform_local_matrix",
            getattr(transform, "zero_transform_local_matrix", None),
        ),
    ):
        if field == "bmat_inv" and matrix is None:
            matrix = getattr(getattr(transform, "base", None), "bmat_inv", None)
        if matrix is None:
            continue
        try:
            payload[field] = [float(value) for row in Matrix(matrix) for value in row]
        except Exception as exc:
            print(f"Warning in blender_importer/nodes/core.py: {exc}")


def _stamp_debug_metadata(node):
    try:
        if node.blender:
            node.blender["_iedm_dbg_tf_cls"] = (
                type(node.transform).__name__ if node.transform is not None else ""
            )
            node.blender["_iedm_dbg_r_cls"] = (
                type(node.render).__name__ if node.render is not None else ""
            )
            if node.transform is not None:
                node.blender["_iedm_dbg_tf_name"] = str(
                    getattr(node.transform, "name", "") or ""
                )
            if node.render is not None:
                node.blender["_iedm_dbg_r_name"] = str(
                    getattr(node.render, "name", "") or ""
                )
            try:
                src_m = None
                tf = node.transform
                if tf is not None:
                    if hasattr(tf, "matrix"):
                        src_m = Matrix(tf.matrix)
                    elif hasattr(getattr(tf, "base", None), "matrix"):
                        src_m = Matrix(tf.base.matrix)
                    elif hasattr(getattr(tf, "base", None), "transform"):
                        src_m = Matrix(tf.base.transform)
                if src_m is not None:
                    src_rows = []
                    for r in range(3):
                        src_rows.append(
                            [round(float(src_m[r][c]), 4) for c in range(3)]
                        )
                    node.blender["_iedm_dbg_src_rows"] = str(src_rows)
            except Exception as e:
                print(f"Warning in blender_importer/nodes/core.py: {e}")
    except Exception as e:
        print(f"Warning in blender_importer/nodes/core.py: {e}")


def _stamp_node_properties(node, ctx):
    """Write EDM metadata and debug IDprops to node.blender."""
    _stamp_transform_metadata(node)
    _stamp_render_metadata(node)
    _stamp_animation_name(node)
    _stamp_raw_animation_payload(node)
    _stamp_debug_metadata(node)
