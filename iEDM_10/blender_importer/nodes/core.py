"""Compatibility imports for the focused implementation modules."""

from .node_helpers import (
    _is_narrow_safe_identity_helper_name as _is_narrow_safe_identity_helper_name,
)
from .node_helpers import _idprop_sequence_value as _idprop_sequence_value
from .node_helpers import _idprop_diag_value as _idprop_diag_value
from .node_helpers import _control_wrapper_prefix as _control_wrapper_prefix
from .node_helpers import (
    _preferred_control_wrapper_name as _preferred_control_wrapper_name,
)
from .node_helpers import _skin_bbox_local_matrix as _skin_bbox_local_matrix
from .node_helpers import (
    _wrap_skin_object_with_skin_box as _wrap_skin_object_with_skin_box,
)
from .node_create import _create_node_object as _create_node_object
from .node_parent import _parent_node_object as _parent_node_object
from .node_properties import _stamp_node_properties as _stamp_node_properties
from .node_position import _apply_render_positioning as _apply_render_positioning
from .node_animation import _hookup_node_animations as _hookup_node_animations
from .node_process import _dump_node_diagnostics as _dump_node_diagnostics
from .node_process import process_node as process_node
from .node_process import _process_lod_post_children as _process_lod_post_children
from .node_process import _apply_shadeless as _apply_shadeless

# Public names formerly imported by this module remain available.
import bpy as bpy
import json as json
import math as math
from ...edm_format.mathtypes import Matrix as Matrix
from ...edm_format.mathtypes import Quaternion as Quaternion
from ...edm_format.mathtypes import Vector as Vector
from ...edm_format.mathtypes import vector_to_blender as vector_to_blender
from ...edm_format.types import AnimatingNode as AnimatingNode
from ...edm_format.types import ArgAnimatedBone as ArgAnimatedBone
from ...edm_format.types import ArgAnimationNode as ArgAnimationNode
from ...edm_format.types import ArgPositionNode as ArgPositionNode
from ...edm_format.types import ArgRotationNode as ArgRotationNode
from ...edm_format.types import ArgScaleNode as ArgScaleNode
from ...edm_format.types import ArgVisibilityNode as ArgVisibilityNode
from ...edm_format.types import BillboardNode as BillboardNode
from ...edm_format.types import Bone as Bone
from ...edm_format.types import Connector as Connector
from ...edm_format.types import FakeALSNode as FakeALSNode
from ...edm_format.types import FakeOmniLightsNode as FakeOmniLightsNode
from ...edm_format.types import FakeSpotLightsNode as FakeSpotLightsNode
from ...edm_format.types import LightNode as LightNode
from ...edm_format.types import LodNode as LodNode
from ...edm_format.types import NumberNode as NumberNode
from ...edm_format.types import RenderNode as RenderNode
from ...edm_format.types import SegmentsNode as SegmentsNode
from ...edm_format.types import ShellNode as ShellNode
from ...edm_format.types import SkinNode as SkinNode
from ...edm_format.types import TransformNode as TransformNode
from ..anim_actions import get_actions_for_node as get_actions_for_node
from ..lights import create_billboard as create_billboard
from ..lights import create_fake_als_lights as create_fake_als_lights
from ..lights import create_fake_omni_lights as create_fake_omni_lights
from ..lights import create_fake_spot_lights as create_fake_spot_lights
from ..lights import create_lamp as create_lamp
from ..node_transform import apply_node_transform as apply_node_transform
from ..object_create import create_connector as create_connector
from ..object_create import create_object as create_object
from ..object_create import create_segments as create_segments
