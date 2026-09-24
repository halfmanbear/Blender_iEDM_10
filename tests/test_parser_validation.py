"""Malformed node references must fail before graph construction."""

import io
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from iEDM_10.edm_format.basereader import (
    MAX_REASONABLE_COUNT,
    BaseReader,
    EDMFormatError,
)
from iEDM_10.edm_format.types import core_file
from iEDM_10.edm_format.types.core_nodes import BaseNode, GraphNode
from iEDM_10.edm_format.types.render_shell import SkinNode


class ParserValidationTests(unittest.TestCase):
    def test_skin_bone_count_rejected_before_reading_palette(self):
        reader = BaseReader.__new__(BaseReader)
        reader.stream = io.BytesIO(struct.pack("<3I", 0, 0, MAX_REASONABLE_COUNT + 1))
        with (
            patch.object(BaseNode, "read", return_value=SkinNode()),
            self.assertRaisesRegex(EDMFormatError, "skin bone count"),
        ):
            SkinNode.read(reader)

    def read_parents(self, parents):
        nodes = [GraphNode() for _ in parents]
        reader = Mock(v10=True)
        reader.read_ushort.return_value = 10
        reader.read_uint.return_value = 0
        reader.read.return_value = b""
        reader.stream.read.return_value = b""
        reader.tell.return_value = 0
        reader.read_list.return_value = nodes
        reader.read_ints.return_value = parents
        edm = core_file.EDMFile()
        with (
            patch.object(core_file, "_read_index", return_value={}),
            patch.object(core_file, "_read_main_object_dictionary", return_value={}),
            patch.object(edm, "_read_render_nodes"),
            patch.object(edm, "_validate_indexes"),
        ):
            edm._read(reader)
        return nodes

    def test_parent_bounds(self):
        for index in (-3, -2, 2, 100):
            with self.subTest(index=index), self.assertRaisesRegex(
                EDMFormatError, f"parent index {index}"
            ):
                self.read_parents([-1, index])

    def test_valid_parent_and_root_sentinel(self):
        nodes = self.read_parents([-1, 0])
        self.assertIsNone(nodes[0].parent)
        self.assertIs(nodes[1].parent, nodes[0])

    def test_skin_palette_validation_is_atomic(self):
        for index in (-1, 2, 100):
            skin = SkinNode()
            skin.bones = [0, index]
            with self.subTest(index=index), self.assertRaisesRegex(
                EDMFormatError, "Invalid bone index"
            ):
                skin.prepare([GraphNode(), GraphNode()], [])
            self.assertEqual(skin.bones, [0, index])

    def test_valid_skin_palette_preserves_order_and_duplicates(self):
        nodes = [GraphNode(), GraphNode()]
        skin = SkinNode()
        skin.bones = [1, 0, 1]
        skin.prepare(nodes, [])
        self.assertEqual(skin.bones, [nodes[1], nodes[0], nodes[1]])

    def test_prepare_format_error_is_not_swallowed(self):
        edm = core_file.EDMFile()
        edm.root = SimpleNamespace(materials=[])
        edm.nodes = [GraphNode()]
        skin = SkinNode()
        skin.bones = [5]
        with self.assertRaisesRegex(EDMFormatError, "Invalid bone index"):
            edm._read_render_nodes(Mock(), {"RENDER_NODES": [skin]}, SkinNode, SkinNode)

    def test_link_objects_validates_unprepared_palette(self):
        edm = core_file.EDMFile()
        edm.root = SimpleNamespace(materials=[])
        edm.nodes = [GraphNode()]
        skin = SkinNode()
        skin.bones = [3]
        edm.renderNodes = [skin]
        with self.assertRaisesRegex(EDMFormatError, "Invalid bone index"):
            edm._link_objects()

    def test_empty_transform_graph_rejects_skin(self):
        edm = core_file.EDMFile()
        edm.root = SimpleNamespace(materials=[])
        edm.renderNodes = [SkinNode()]
        with self.assertRaisesRegex(EDMFormatError, "no transform root"):
            edm._link_objects()
