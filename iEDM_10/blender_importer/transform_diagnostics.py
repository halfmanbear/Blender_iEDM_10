"""Optional diagnostics for assigned transforms and parent chains."""

from .graph_pipeline import (
    _debug_filter_terms,
    _debug_fmt_rot_deg,
    _debug_fmt_vec3,
)
from .prelude import _import_ctx, _log


def _debug_set_trace(tfnode, obj, source_label, local_mat):
    if not _import_ctx.transform_debug.get("enabled"):
        return
    node_name = getattr(tfnode, "name", "") or type(tfnode).__name__
    filter_terms = _debug_filter_terms()
    if filter_terms:
        if not any(
            term in node_name.lower() or term in obj.name.lower()
            for term in filter_terms
        ):
            return
    loc, rot, scale = local_mat.decompose()
    print(
        "[iEDM][TFSET] {} src={} obj={} loc={} rot_deg={} scale={}".format(
            node_name,
            source_label,
            obj.name,
            _debug_fmt_vec3(loc),
            _debug_fmt_rot_deg(rot),
            _debug_fmt_vec3(scale),
        )
    )


def _debug_dump_parent_chain(tnode, tfnode, obj, local_mat):
    if not _import_ctx.transform_debug.get("enabled"):
        return
    filter_terms = _debug_filter_terms()
    if not filter_terms:
        return
    node_name = getattr(tfnode, "name", "") or type(tfnode).__name__
    if not any(
        term in node_name.lower() or term in obj.name.lower() for term in filter_terms
    ):
        return
    dumped = _import_ctx.transform_debug.setdefault("chain_dumped", set())
    dump_key = "{}::{}".format(node_name, obj.name)
    if dump_key in dumped:
        return
    dumped.add(dump_key)

    print(
        "[iEDM][CHAIN] begin node={} obj={} graph_parent={} blender_parent={}".format(
            node_name,
            obj.name,
            getattr(
                getattr(getattr(tnode, "parent", None), "transform", None),
                "name",
                "<ROOT>",
            )
            if tnode is not None
            else "<ROOT>",
            getattr(getattr(obj, "parent", None), "name", None),
        )
    )
    if tnode is not None:
        cur = tnode
        graph_parts = []
        while cur is not None:
            tf_cur = getattr(cur, "transform", None)
            rn_cur = getattr(cur, "render", None)
            if tf_cur is not None:
                label = "{}<{}>".format(
                    getattr(tf_cur, "name", "") or type(tf_cur).__name__,
                    type(tf_cur).__name__,
                )
            elif rn_cur is not None:
                label = "{}<{}>".format(
                    getattr(rn_cur, "name", "") or type(rn_cur).__name__,
                    type(rn_cur).__name__,
                )
            else:
                label = "<ROOT>"
            graph_parts.append(label)
            cur = getattr(cur, "parent", None)
        print("[iEDM][CHAIN] graph_path={}".format(" / ".join(reversed(graph_parts))))
    try:
        loc, rot, scale = local_mat.decompose()
        print(
            "[iEDM][CHAIN] assigned_local loc={} rot_deg={} scale={}".format(
                _debug_fmt_vec3(loc),
                _debug_fmt_rot_deg(rot),
                _debug_fmt_vec3(scale),
            )
        )
    except Exception as exc:
        _log.debug("Optional operation failed: {}".format(exc), level=2)

    current = obj
    level = 0
    while current is not None:
        try:
            basis_loc, basis_rot, basis_scale = current.matrix_basis.decompose()
            world_loc, world_rot, world_scale = current.matrix_world.decompose()
            print(
                "[iEDM][CHAIN] level={} obj={} type={} parent={} basis_loc={} "
                "basis_rot_deg={} basis_scale={} world_loc={} world_rot_deg={} "
                "world_scale={}".format(
                    level,
                    current.name,
                    getattr(current, "type", ""),
                    getattr(getattr(current, "parent", None), "name", None),
                    _debug_fmt_vec3(basis_loc),
                    _debug_fmt_rot_deg(basis_rot),
                    _debug_fmt_vec3(basis_scale),
                    _debug_fmt_vec3(world_loc),
                    _debug_fmt_rot_deg(world_rot),
                    _debug_fmt_vec3(world_scale),
                )
            )
        except Exception as e:
            print(
                "[iEDM][CHAIN] level={} obj={} error={}".format(
                    level, getattr(current, "name", "<unknown>"), e
                )
            )
        current = getattr(current, "parent", None)
        level += 1
    print("[iEDM][CHAIN] end node={} obj={}".format(node_name, obj.name))


