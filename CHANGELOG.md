# Change Log

## Unreleased

**Fixed**
- Euler rotation normalization (`_normalize_euler_xyz`, `_normalize_euler_action_curves`) silently did nothing because `Euler` was never imported

**Changed**
- Minimum Blender version lowered to 4.5.0 (tested on 4.5.6)
- Release zips are built from git-tracked add-on files by `scripts/build_addon.py` and attached to GitHub releases on `v*` tags
- Test assets (ED `Learning_Demo` files) are no longer stored in the repository; see `tests/README.md`
- Removed the dev-only `replace_exceptions.py` from the add-on package; `index.md` moved to `docs/ARCHITECTURE.md`
- Imported connectors now have more sensible sizes

---

## 0.4.0 – 2026-01-19

**Added**
- Support for Blender 4.5 LTS

**Changed**
- Refactor to work with official ED `io_scene_edm` plugin types

**Removed**
- Deleted existing unofficial exporter code (in favor of using the official `io_scene_edm` exporter plugin)
- Removed Blender EDM Material Properties UI (Materials submenu code), in favor of using the official `io_scene_edm` plugin Sidebar menu ("EDM Export" tab in Object Properties)

---

## 0.2.0 – 2016-12-14

**Added**
- Writing of collision shells
- Reading of the new v10 `.EDM` files

**Changed**
- Fixed UV writing to be normally in the range 0–1
- Improved specularity conversion; not everything is super shiny now

---

## 0.1.0 – 2016-12-07

**Added**
- Initial release
