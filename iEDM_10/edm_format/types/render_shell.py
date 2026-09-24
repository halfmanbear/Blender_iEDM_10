"""Compatibility exports for registered render, shell, and skin node types."""

import itertools as itertools
from collections import Counter as Counter

from ..basereader import MAX_REASONABLE_COUNT as MAX_REASONABLE_COUNT
from ..basereader import EDMFormatError as EDMFormatError
from ..material_types import VertexFormat as VertexFormat
from ..validation import resolve_node_indices as resolve_node_indices
from .core import _V10_CATEGORY_KEYS as _V10_CATEGORY_KEYS
from .core import BaseNode as BaseNode
from .core import NodeCategory as NodeCategory
from .core import _read_with_layout_fallback as _read_with_layout_fallback
from .core import _scan_to_next_v10_type_token as _scan_to_next_v10_type_token
from .core import logger as logger
from .core import reads_type as reads_type
from .render_nodes import RenderNode as RenderNode
from .render_payload import _classify_render_parent_data as _classify_render_parent_data
from .render_payload import _owner_triangles as _owner_triangles
from .render_payload import _read_index_data as _read_index_data
from .render_payload import _read_parent_data as _read_parent_data
from .render_payload import _read_vertex_data as _read_vertex_data
from .render_payload import _render_audit as _render_audit
from .render_payload import _tag_shell_family_node as _tag_shell_family_node
from .render_payload import _write_index_data as _write_index_data
from .render_payload import _write_vertex_data as _write_vertex_data
from .render_special_nodes import MorphNode as MorphNode
from .render_special_nodes import SegmentsNode as SegmentsNode
from .render_special_nodes import ShellNode as ShellNode
from .render_special_nodes import ShellSkinNode as ShellSkinNode
from .render_special_nodes import SkinNode as SkinNode
from .render_special_nodes import TreeShellNode as TreeShellNode
