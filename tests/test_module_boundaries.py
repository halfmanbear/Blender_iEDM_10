"""Keep historical imports, parser registration, and shared context intact."""

import importlib
import importlib.util
import unittest

from iEDM_10.edm_format import types
from iEDM_10.edm_format.typereader import get_type_reader


class ParserModuleTests(unittest.TestCase):
    def test_light_types_keep_registered_readers(self):
        self.check_types(
            "lights",
            (
                "BillboardNode",
                "LightNode",
                "FakeSpotLightsNode",
                "AnimatedFakeSpotLightsNode",
                "FakeSpotLights3Node",
                "FakeOmniLightsNode",
                "AnimatedFakeOmniLightsNode",
                "FakeALSNode",
            ),
        )

    def test_render_types_keep_registered_readers(self):
        self.check_types(
            "render_shell",
            (
                "RenderNode",
                "ShellNode",
                "SkinNode",
                "ShellSkinNode",
                "TreeShellNode",
                "MorphNode",
                "SegmentsNode",
            ),
        )

    def check_types(self, module_name, names):
        module = importlib.import_module(f"iEDM_10.edm_format.types.{module_name}")
        for name in names:
            with self.subTest(name=name):
                node_type = getattr(module, name)
                self.assertIs(node_type, getattr(types, name))
                self.assertEqual(get_type_reader(node_type.forTypeName), node_type.read)


@unittest.skipUnless(importlib.util.find_spec("bpy"), "Requires Blender")
class ImporterModuleTests(unittest.TestCase):
    def test_historical_helpers_reexport_the_same_implementations(self):
        cases = (
            ("anim_actions", "action_curves", "_build_arganimation_action"),
            ("graph_pipeline", "graph_diagnostics", "_debug_dump_node_transform"),
            ("graph_pipeline", "graph_collections", "_assign_collections"),
            ("material_setup", "material_animation", "_mat_set_linear_on_path"),
            ("material_setup", "material_payload", "_preserve_material_payload"),
            ("materials_bridge", "material_sockets", "_official_transparency_enum"),
            ("nodes.armature", "nodes.skin_binding", "_choose_skin_bind_target"),
            ("nodes.armature", "nodes.armature_build", "_build_edit_bones"),
            ("nodes.armature", "nodes.bone_actions", "_copy_bone_rotation_curves"),
            ("nodes.armature", "nodes.bone_rest", "_bone_rest_matrix_for_node"),
            ("object_create", "morph_objects", "_decode_morph_payload_to_shape_keys"),
            ("orient_scale", "orient_scale_geometry", "_insert_parent_wrapper_object"),
            ("prelude", "visibility_graph", "_is_child_of_file_root"),
            ("prelude", "node_identity", "is_skeleton_node"),
            ("prelude", "exporter_properties", "_ensure_official_material_bridge"),
            ("session", "session_setup", "ImportOptions"),
            ("session", "scene_root", "_create_graph_root_object"),
            ("session", "session_postprocess", "_run_import_postprocess"),
            (
                "session",
                "session_visibility",
                "_propagate_visibility_hide_to_render_nodes",
            ),
            (
                "vis_rewrites",
                "skin_visibility",
                "_fix_inverse_scaled_visibility_rest_offset",
            ),
        )
        for old, new, name in cases:
            with self.subTest(module=old, name=name):
                old_module = importlib.import_module(f"iEDM_10.blender_importer.{old}")
                new_module = importlib.import_module(f"iEDM_10.blender_importer.{new}")
                self.assertIs(getattr(old_module, name), getattr(new_module, name))

    def test_compatibility_modules_share_one_import_context(self):
        from iEDM_10 import reader
        from iEDM_10.blender_importer import (
            anim_actions,
            import_context,
            prelude,
            session,
        )

        for module in (anim_actions, prelude, session):
            self.assertIs(module._import_ctx, import_context._import_ctx)
        self.assertIs(reader.ImportContext, import_context.ImportContext)
        self.assertIs(reader.read_file, session.read_file)
