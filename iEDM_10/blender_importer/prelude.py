"""Compatibility exports for shared importer context and focused helpers."""

import json as json
import re as re
import threading as threading

from ..edm_format.mathtypes import *
from ..edm_format.mathtypes import Matrix as Matrix
from ..edm_format.mathtypes import MatrixScale as MatrixScale
from ..edm_format.mathtypes import Quaternion as Quaternion
from ..edm_format.mathtypes import Vector as Vector
from ..edm_format.types import *
from ..edm_format.types import AnimatingNode as AnimatingNode
from ..edm_format.types import ArgAnimatedBone as ArgAnimatedBone
from ..edm_format.types import ArgAnimationNode as ArgAnimationNode
from ..edm_format.types import ArgVisibilityNode as ArgVisibilityNode
from ..edm_format.types import Bone as Bone
from ..edm_format.types import Connector as Connector
from ..edm_format.types import TransformNode as TransformNode
from . import visibility_graph as _visibility_graph_compat
from .exporter_properties import (
    _ensure_official_material_bridge as _ensure_official_material_bridge,
)
from .exporter_properties import _set_edmprop as _set_edmprop
from .exporter_properties import (
    _set_official_special_type as _set_official_special_type,
)
from .import_capabilities import DEFAULT_CAPABILITIES as DEFAULT_CAPABILITIES
from .import_context import (
    _BLENDER_LAMP_ENERGY_COEFFICIENT as _BLENDER_LAMP_ENERGY_COEFFICIENT,
)
from .import_context import (
    _BLENDER_LAMP_WEAK_COEFFICIENT as _BLENDER_LAMP_WEAK_COEFFICIENT,
)
from .import_context import _PBR_WATTS_TO_LUMENS as _PBR_WATTS_TO_LUMENS
from .import_context import _ROOT_BASIS_FIX as _ROOT_BASIS_FIX
from .import_context import _SUFFIX_RE as _SUFFIX_RE
from .import_context import DEFAULT_PROFILE as DEFAULT_PROFILE
from .import_context import FRAME_SCALE as FRAME_SCALE
from .import_context import ImportContext as ImportContext
from .import_context import (
    _default_render_split_debug_state as _default_render_split_debug_state,
)
from .import_context import (
    _default_transform_debug_state as _default_transform_debug_state,
)
from .import_context import (
    _default_visibility_debug_state as _default_visibility_debug_state,
)
from .import_context import _IEDMLogger as _IEDMLogger
from .import_context import _import_capability_detail as _import_capability_detail
from .import_context import _import_capability_name as _import_capability_name
from .import_context import _import_ctx as _import_ctx
from .import_context import _import_profile_detail as _import_profile_detail
from .import_context import _import_profile_flag as _import_profile_flag
from .import_context import _import_profile_name as _import_profile_name
from .import_context import _log as _log
from .import_context import _ProfileStub as _ProfileStub
from .import_logging import _append_debug_log_line as _append_debug_log_line
from .import_logging import _append_transform_log_line as _append_transform_log_line
from .import_logging import _debug_log_event as _debug_log_event
from .import_logging import _log_bone_debug_event as _log_bone_debug_event
from .import_logging import _matrix_trs_summary as _matrix_trs_summary
from .import_logging import _node_visibility_chain_args as _node_visibility_chain_args
from .import_logging import _transform_debug_matches as _transform_debug_matches
from .node_identity import _assign_action as _assign_action
from .node_identity import _is_anim_node_name as _is_anim_node_name
from .node_identity import _is_bone_transform as _is_bone_transform
from .node_identity import _is_connector_object as _is_connector_object
from .node_identity import _is_connector_transform as _is_connector_transform
from .node_identity import _is_generic_render_name as _is_generic_render_name
from .node_identity import _is_skeleton_controller as _is_skeleton_controller
from .node_identity import _ob_local_is_identity as _ob_local_is_identity
from .node_identity import _strip_anim_prefix as _strip_anim_prefix
from .node_identity import _transform_display_name as _transform_display_name
from .node_identity import is_skeleton_node as is_skeleton_node
from .visibility_graph import _canonical_control_name as _canonical_control_name
from .visibility_graph import (
    _has_only_visibility_ancestors as _has_only_visibility_ancestors,
)
from .visibility_graph import (
    _is_authored_argvis_control_pair as _is_authored_argvis_control_pair,
)
from .visibility_graph import _is_child_of_file_root as _is_child_of_file_root
from .visibility_graph import (
    _is_nested_authored_argvis_control_pair as _is_nested_authored_argvis_control_pair,
)
from .visibility_graph import (
    _is_root_visibility_chain_authored_pair as _is_root_visibility_chain_authored_pair,
)
from .visibility_graph import (
    _is_root_visibility_pair_child as _is_root_visibility_pair_child,
)
from .visibility_graph import (
    _is_static_root_visibility_wrapper as _is_static_root_visibility_wrapper,
)
from .visibility_graph import (
    _is_top_level_visibility_authored_pair as _is_top_level_visibility_authored_pair,
)
from .visibility_graph import (
    _visibility_node_has_direct_anim_child as _visibility_node_has_direct_anim_child,
)
from .visibility_timeline import (
    _anim_frame_to_scene_frame as _anim_frame_to_scene_frame,
)
from .visibility_timeline import _visibility_scene_keys as _visibility_scene_keys
from .visibility_timeline import _visibility_scene_ranges as _visibility_scene_ranges

_nearest_visibility_ancestor_with_direct_anim_child = (
    _visibility_graph_compat._nearest_visibility_ancestor_with_direct_anim_child
)
