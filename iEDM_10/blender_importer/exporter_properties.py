"""Set exporter properties and resolve optional material integration."""

from .import_context import _import_ctx


def _set_official_special_type(obj, special_type):
    """Set official EDM exporter object type when that addon is present."""
    if hasattr(obj, "EDMProps"):
        obj.EDMProps.SPECIAL_TYPE = special_type


def _set_edmprop(obj, prop_name, value):
    if not hasattr(obj, "EDMProps"):
        return
    edm_props = obj.EDMProps
    if hasattr(edm_props, prop_name):
        try:
            # Clamp integers to 32-bit signed limit for Blender's IntProperty.
            if isinstance(value, int):
                value = min(2147483647, max(-2147483648, value))
            setattr(edm_props, prop_name, value)
        except (OverflowError, ValueError):
            pass
        except Exception as e:
            print(f"Warning in blender_importer/prelude.py: {e}")


def _ensure_official_material_bridge():
    """
    Resolve optional interop with the official EDM exporter material system.
    Returns a cached dict with:
      - available: bool
      - material_descs: dict[str, MatDesc]
      - names: dict[str, str]
      - error: str | None
    """
    if _import_ctx.official_material_bridge is not None:
        return _import_ctx.official_material_bridge

    bridge = {
        "available": False,
        "material_descs": {},
        "error": None,
        "names": {
            "default": "EDM_Default_Material",
            "deck": "EDM_Deck_Material",
            "fake_omni": "EDM_Fake_Omni_Material",
            "fake_spot": "EDM_Fake_Spot_Material",
            "glass": "EDM_Glass_Material",
            "mirror": "EDM_Mirror_Material",
        },
        "node_types": {
            "default": "EdmDefaultShaderNodeType",
            "deck": "EdmDeckShaderNodeType",
            "fake_omni": "EdmFakeOmniShaderNodeType",
            "fake_spot": "EdmFakeSpotShaderNodeType",
            "glass": "EdmGlassShaderNodeType",
            "mirror": "EdmMirrorShaderNodeType",
        },
    }

    try:
        # Available when official addon is loaded (io_scene_edm).
        from materials.materials import build_material_descriptions

        bridge["material_descs"] = build_material_descriptions() or {}
        bridge["available"] = True
    except Exception as exc:
        bridge["error"] = str(exc)

    _import_ctx.official_material_bridge = bridge
    return bridge
