"""Compatibility exports for registered light node types and light helpers."""

import struct as struct

from ..propertiesset import PropertiesSet as PropertiesSet
from .core import _V10_CATEGORY_KEYS as _V10_CATEGORY_KEYS
from .core import BaseNode as BaseNode
from .core import Node as Node
from .core import NodeCategory as NodeCategory
from .core import _next_v10_token_looks_like_type as _next_v10_token_looks_like_type
from .core import _scan_to_next_v10_type_token as _scan_to_next_v10_type_token
from .core import logger as logger
from .core import reads_type as reads_type
from .fake_omni_nodes import AnimatedFakeOmniLightsNode as AnimatedFakeOmniLightsNode
from .fake_omni_nodes import FakeALSNode as FakeALSNode
from .fake_omni_nodes import FakeOmniLightsNode as FakeOmniLightsNode
from .fake_spot_nodes import AnimatedFakeSpotLightsNode as AnimatedFakeSpotLightsNode
from .fake_spot_nodes import FakeSpotLights3Node as FakeSpotLights3Node
from .fake_spot_nodes import FakeSpotLightsNode as FakeSpotLightsNode
from .light_nodes import BillboardNode as BillboardNode
from .light_nodes import LightNode as LightNode
from .light_nodes import Texture2dProperties as Texture2dProperties
from .light_parsing import _material_name_matches_kind as _material_name_matches_kind
from .light_parsing import (
    _read_animated_fake_lights_payload as _read_animated_fake_lights_payload,
)
from .light_parsing import _read_fake_omni_light as _read_fake_omni_light
from .light_parsing import (
    _resolve_material_from_candidates as _resolve_material_from_candidates,
)
from .light_parsing import _to_float as _to_float
from .light_records import (
    _unpack_float_pair_from_double as _unpack_float_pair_from_double,
)
from .light_records import decode_fake_omni_entry as decode_fake_omni_entry
