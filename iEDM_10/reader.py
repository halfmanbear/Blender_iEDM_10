"""Compatibility entry point for EDM -> Blender conversion.

Each blender_importer module now carries its own explicit imports (see
docs/ARCHITECTURE.md); this module only re-exports the public surface for
external consumers (io_operators.py, tests/) that do `from .reader import X`
or `from iEDM_10 import reader; reader.read_file(...)`.
"""

from .blender_importer.prelude import *  # noqa: F401,F403
from .blender_importer.import_capabilities import *  # noqa: F401,F403
from .blender_importer.materials_bridge import *  # noqa: F401,F403
from .blender_importer.graph_build import *  # noqa: F401,F403
from .blender_importer.graph_pipeline import *  # noqa: F401,F403
from .blender_importer.mesh_create import *  # noqa: F401,F403
from .blender_importer.object_create import *  # noqa: F401,F403
from .blender_importer.bbox_utils import *  # noqa: F401,F403
from .blender_importer.material_setup import *  # noqa: F401,F403
from .blender_importer.animation import *  # noqa: F401,F403
from .blender_importer.graph_postprocess import *  # noqa: F401,F403
from .blender_importer.anim_actions import *  # noqa: F401,F403
from .blender_importer.orient_scale import *  # noqa: F401,F403
from .blender_importer.vis_rewrites import *  # noqa: F401,F403
from .blender_importer.skin_rewrites import *  # noqa: F401,F403
from .blender_importer.orient_fixes import *  # noqa: F401,F403
from .blender_importer.ctrl_splits import *  # noqa: F401,F403
from .blender_importer.node_transform import *  # noqa: F401,F403
from .blender_importer.nodes.mesh import *  # noqa: F401,F403
from .blender_importer.nodes.visibility import *  # noqa: F401,F403
from .blender_importer.nodes.armature import *  # noqa: F401,F403
from .blender_importer.nodes.core import *  # noqa: F401,F403
from .blender_importer.nodes.diagnostics import *  # noqa: F401,F403
from .blender_importer.session import *  # noqa: F401,F403
from .blender_importer.import_pipeline import *  # noqa: F401,F403
from .blender_importer.lights import *  # noqa: F401,F403
