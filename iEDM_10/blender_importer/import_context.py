"""Own the existing thread-local import context, settings, and logger."""

import re as re
import threading as threading

from ..edm_format.mathtypes import Matrix
from .import_capabilities import DEFAULT_CAPABILITIES


class _ProfileStub:
    name = "NONE"
    description = ""


DEFAULT_PROFILE = _ProfileStub()


_SUFFIX_RE = re.compile(r"\.\d{3}$")  # matches ".000" to ".999" at end


FRAME_SCALE = 200


_ROOT_BASIS_FIX = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def _default_transform_debug_state():
    return {
        "enabled": False,
        "filter": None,
        "limit": 200,
        "emitted": 0,
        "log_path": None,
    }


def _default_render_split_debug_state():
    return {"enabled": False}


def _default_visibility_debug_state():
    return {
        "enabled": False,
        "target": None,
        "log_path": None,
        "event_limit": 200,
        "events": 0,
    }


class ImportContext(threading.local):
    """Mutable importer runtime state shared across reader modules."""

    def __init__(self):
        self.edm_version = 10
        self.import_profile = DEFAULT_PROFILE
        self.import_capabilities = DEFAULT_CAPABILITIES
        self.transform_debug = _default_transform_debug_state()
        self.render_split_debug = _default_render_split_debug_state()
        self.visibility_debug = _default_visibility_debug_state()
        self.mesh_origin_mode = "APPROX"
        self.source_dir = None
        self.texture_search_cache = {}
        self.official_material_bridge = None
        self.bone_import_ctx = None
        self.bonetransform_prefix_matrix = (
            None  # compound M1@M2@... for files with Bonetransform prefix chain
        )
        self.legacy_v10_parent_compose = False
        self.file_has_bones = False
        self.use_scene_root_basis_object = True
        self.collision_geometry_basis_fix = False
        self.verbosity = 0  # 0=info+warn, 1=debug, 2=verbose debug


def _import_capability_name():
    caps = getattr(_import_ctx, "import_capabilities", None)
    if caps is not None:
        name = getattr(caps, "name", None)
        if name:
            return name
    return DEFAULT_CAPABILITIES.name


def _import_capability_detail():
    caps = getattr(_import_ctx, "import_capabilities", None)
    if caps is not None:
        detail = getattr(caps, "detail", "")
        if detail:
            return detail
    return ""


def _import_profile_name():
    profile = getattr(_import_ctx, "import_profile", DEFAULT_PROFILE)
    return getattr(profile, "name", DEFAULT_PROFILE.name)


def _import_profile_detail():
    profile = getattr(_import_ctx, "import_profile", DEFAULT_PROFILE)
    detail = str(getattr(profile, "description", "") or "")
    if detail:
        return detail
    return ""


def _import_profile_flag(flag_name, default=False):
    caps = getattr(_import_ctx, "import_capabilities", None)
    if caps is not None:
        flags = getattr(caps, "flags", None) or {}
        if flag_name in flags:
            return bool(flags[flag_name])
    profile = getattr(_import_ctx, "import_profile", DEFAULT_PROFILE)
    return bool(getattr(profile, flag_name, default))


_import_ctx = ImportContext()


class _IEDMLogger:
    """Minimal leveled logger for the iEDM importer.

    Verbosity levels (controlled via _import_ctx.verbosity):
      0  — info + warn always shown  (default)
      1  — also shows debug level 1  (node transforms, flag dumps)
      2  — also shows debug level 2  (per-keyframe data)
    """

    PREFIX = "[iEDM]"

    def info(self, msg):
        print("{} Info: {}".format(self.PREFIX, msg))

    def warn(self, msg, exc=None, node=None):
        parts = ["{} Warning: {}".format(self.PREFIX, msg)]
        if node is not None:
            parts.append("(node={})".format(node))
        if exc is not None:
            parts.append("({}: {})".format(type(exc).__name__, exc))
        print(" ".join(parts))

    def debug(self, msg, level=1):
        v = getattr(_import_ctx, "verbosity", 0)
        if v >= level:
            print("{} Debug[{}]: {}".format(self.PREFIX, level, msg))


_log = _IEDMLogger()


_PBR_WATTS_TO_LUMENS = 683.0


_BLENDER_LAMP_ENERGY_COEFFICIENT = 0.0000305


_BLENDER_LAMP_WEAK_COEFFICIENT = 1.0 / 2.9
