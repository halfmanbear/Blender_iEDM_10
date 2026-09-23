"""Compatibility imports for the focused implementation modules."""

import itertools as itertools
import logging as logging
import math as math
import struct as struct
from abc import ABC as ABC
from collections import Counter as Counter
from collections import OrderedDict as OrderedDict
from collections import namedtuple as namedtuple
from enum import Enum as Enum

# Public names formerly imported by this module remain available.
from ..basereader import BaseReader as BaseReader
from ..material_types import Material as Material
from ..material_types import ShadowSettings as ShadowSettings
from ..material_types import Texture as Texture
from ..material_types import VertexFormat as VertexFormat
from ..mathtypes import Matrix as Matrix
from ..mathtypes import Quaternion as Quaternion
from ..mathtypes import Vector as Vector
from ..mathtypes import sequence_to_matrix as sequence_to_matrix
from ..probe import require_supported_import_format as require_supported_import_format
from ..propertiesset import PropertiesSet as PropertiesSet
from ..typereader import reads_type as reads_type
from .core_animation import ArgAnimatedBone as ArgAnimatedBone
from .core_animation import ArgAnimationBase as ArgAnimationBase
from .core_animation import ArgAnimationNode as ArgAnimationNode
from .core_animation import ArgPositionNode as ArgPositionNode
from .core_animation import ArgRotationNode as ArgRotationNode
from .core_animation import ArgScaleNode as ArgScaleNode
from .core_animation import ArgVisibilityNode as ArgVisibilityNode
from .core_animation import PositionKey as PositionKey
from .core_animation import RotationKey as RotationKey
from .core_animation import ScaleKey as ScaleKey
from .core_file import EDMFile as EDMFile
from .core_nodes import BaseNode as BaseNode
from .core_nodes import Bone as Bone
from .core_nodes import Connector as Connector
from .core_nodes import GraphNode as GraphNode
from .core_nodes import LodNode as LodNode
from .core_nodes import Node as Node
from .core_nodes import NumberRoot as NumberRoot
from .core_nodes import RootNode as RootNode
from .core_nodes import TmpNumberRoot as TmpNumberRoot
from .core_nodes import TransformNode as TransformNode
from .core_support import _V10_CATEGORY_KEYS as _V10_CATEGORY_KEYS
from .core_support import AnimatingNode as AnimatingNode
from .core_support import NodeCategory as NodeCategory
from .core_support import TrackingReader as TrackingReader
from .core_support import _all_IndexA as _all_IndexA
from .core_support import _all_IndexB as _all_IndexB
from .core_support import _is_root_box_sentinel_pair as _is_root_box_sentinel_pair
from .core_support import (
    _next_v10_token_looks_like_type as _next_v10_token_looks_like_type,
)
from .core_support import _read_index as _read_index
from .core_support import _read_main_object_dictionary as _read_main_object_dictionary
from .core_support import _read_with_layout_fallback as _read_with_layout_fallback
from .core_support import _same_vec3 as _same_vec3
from .core_support import _scan_to_next_v10_type_token as _scan_to_next_v10_type_token
from .core_support import _write_index as _write_index
from .core_support import get_type_reader as get_type_reader
from .core_support import logger as logger
