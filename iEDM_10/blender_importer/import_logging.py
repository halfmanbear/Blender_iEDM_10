"""Write bounded visibility, transform, and bone diagnostic records."""

import json as json

from ..edm_format.mathtypes import Matrix
from ..edm_format.types import ArgVisibilityNode
from .import_context import _import_ctx


def _append_debug_log_line(line):
    dbg = getattr(_import_ctx, "visibility_debug", {}) or {}
    if not dbg.get("enabled"):
        return
    log_path = dbg.get("log_path")
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(str(line))
            handle.write("\n")
    except Exception:  # noqa: S110 - logging must not recurse when its own writer fails
        pass


def _debug_log_event(line):
    dbg = getattr(_import_ctx, "visibility_debug", {}) or {}
    if not dbg.get("enabled"):
        return
    limit = int(dbg.get("event_limit", 200) or 200)
    emitted = int(dbg.get("events", 0) or 0)
    if emitted >= limit:
        return
    dbg["events"] = emitted + 1
    _append_debug_log_line(line)


def _append_transform_log_line(line):
    dbg = getattr(_import_ctx, "transform_debug", {}) or {}
    log_path = dbg.get("log_path")
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(str(line))
            handle.write("\n")
    except Exception:  # noqa: S110 - logging must not recurse when its own writer fails
        pass


def _transform_debug_matches(*names):
    dbg = getattr(_import_ctx, "transform_debug", {}) or {}
    if not dbg.get("enabled"):
        return False
    needle = dbg.get("filter")
    if not needle:
        return True
    try:
        needle_lc = str(needle).lower()
    except Exception:
        return False
    for name in names:
        if name is None:
            continue
        try:
            if needle_lc in str(name).lower():
                return True
        except Exception:
            continue
    return False


def _log_bone_debug_event(stage, payload, *names):
    if not _transform_debug_matches(*names):
        return
    try:
        line = "[iEDM][BONEDBG] {} {}".format(
            stage, json.dumps(payload, separators=(",", ":"), sort_keys=True)
        )
    except Exception:
        return
    print(line)
    _append_transform_log_line(line)


def _matrix_trs_summary(mat):
    try:
        loc, rot, scale = Matrix(mat).decompose()
        return {
            "loc": [
                round(float(loc.x), 6),
                round(float(loc.y), 6),
                round(float(loc.z), 6),
            ],
            "rot_quat": [
                round(float(rot.w), 6),
                round(float(rot.x), 6),
                round(float(rot.y), 6),
                round(float(rot.z), 6),
            ],
            "scale": [
                round(float(scale.x), 6),
                round(float(scale.y), 6),
                round(float(scale.z), 6),
            ],
        }
    except Exception:
        return None


def _node_visibility_chain_args(node):
    """Return visibility arg numbers from nearest parent to root for a graph node."""
    args = []
    curr = getattr(node, "parent", None)
    while curr is not None:
        tf = getattr(curr, "transform", None)
        if isinstance(tf, ArgVisibilityNode):
            try:
                vis_data = getattr(tf, "visData", []) or []
                if vis_data:
                    arg = int(vis_data[0][0])
                    args.append(arg)
            except Exception:  # noqa: S110 - logging must not recurse when its own writer fails
                pass
        curr = getattr(curr, "parent", None)
    return args
