"""Run in Blender to verify optional light-value fallbacks."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("bpy"), "Requires Blender")
class LightValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from iEDM_10.blender_importer import light_values

        cls.values = light_values

    def test_valid_scalar_and_vector(self):
        self.assertEqual(self.values._to_float("2.5"), 2.5)
        self.assertEqual(self.values._to_vec3([1, "2", 3]), (1.0, 2.0, 3.0))

    def test_compatibility_imports(self):
        from iEDM_10.blender_importer.materials_bridge import (
            _SELF_ILLUM_MATERIALS,
            _map_edm_material_to_official_kind,
        )
        from iEDM_10.edm_format.types.lights import decode_fake_omni_entry

        self.assertIn("self_illum_material", _SELF_ILLUM_MATERIALS)
        self.assertEqual(
            _map_edm_material_to_official_kind("self_illum_material"), "default"
        )
        decoded = decode_fake_omni_entry({"position": (1, 2, 3), "size": 4})
        self.assertEqual(decoded["position"], (1.0, 2.0, 3.0))
        self.assertEqual(decoded["size"], 4.0)

    def test_invalid_scalar_logs_fallback(self):
        for value in (None, "invalid", object()):
            with self.subTest(value=value), self.assertLogs(
                self.values.logger, level="WARNING"
            ):
                self.assertEqual(self.values._to_float(value, 4), 4.0)

    def test_invalid_vector_logs_fallback(self):
        with self.assertLogs(self.values.logger, level="WARNING"):
            self.assertEqual(self.values._to_vec3([1]), (1.0, 1.0, 1.0))

    def test_unexpected_errors_propagate(self):
        class BrokenValue:
            def __float__(self):
                raise RuntimeError("broken conversion")

        with self.assertRaisesRegex(RuntimeError, "broken conversion"):
            self.values._to_float(BrokenValue())
