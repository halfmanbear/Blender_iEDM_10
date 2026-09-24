"""Run with Blender --background --factory-startup --python this_file.

Records transform changes made by orientation postprocessing. An existing
sentinel mesh detects whether importing an EDM modifies unrelated scene data.
Pass EDM paths after -- to audit other assets. Use --report to save details,
or --save-blend to save a single import.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import iEDM_10
from iEDM_10 import reader
from iEDM_10.blender_importer import session


def snapshot():
    bpy.context.view_layer.update()
    return {
        ob.name: {
            "world": [float(v) for row in ob.matrix_world for v in row],
            "action": getattr(getattr(ob.animation_data, "action", None), "name", None),
        }
        for ob in bpy.data.objects
    }


def main():
    iEDM_10.register()
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--save-blend")
    parser.add_argument("--report")
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    paths = args.paths or [
        str(p) for p in sorted((ROOT / "tests" / "assets").glob("*.edm"))
    ]
    if args.save_blend and len(paths) != 1:
        parser.error("--save-blend requires exactly one EDM")
    reports = []
    passes = ("_run_orientation_postprocess", "_run_skin_transform_postprocess")
    originals = {name: getattr(session, name) for name in passes}
    for path in paths:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.mesh.primitive_cube_add(location=(2, 3, 4))
        sentinel = bpy.context.object
        sentinel.name = "ExistingSceneMesh"
        before = snapshot()[sentinel.name]
        changes = {}

        def wrap(name, changes=changes):
            def run():
                old = snapshot()
                originals[name]()
                new = snapshot()
                changed = []
                for key in old.keys() & new.keys():
                    error = max(
                        abs(a - b)
                        for a, b in zip(
                            old[key]["world"], new[key]["world"], strict=False
                        )
                    )
                    if error > 1e-5 or old[key]["action"] != new[key]["action"]:
                        changed.append(
                            {
                                "object": key,
                                "matrix_delta": error,
                                "action_before": old[key]["action"],
                                "action_after": new[key]["action"],
                            }
                        )
                changes[name] = changed

            return run

        for name in passes:
            setattr(session, name, wrap(name))
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                reader.read_file(
                    str(Path(path).resolve()),
                    options={
                        "mesh_origin_mode": "RAW",
                        "preserve_scene_boxes": False,
                    },
                )
            after = snapshot()[sentinel.name]
            assert before == after, "Import modified the existing scene mesh"
            report = {
                "file": Path(path).name,
                "existing_mesh_changed": before != after,
                "passes": changes,
            }
            reports.append(report)
            print(
                "AUDIT "
                + json.dumps(
                    {**report, "passes": {k: len(v) for k, v in changes.items()}}
                )
            )
            if args.save_blend:
                bpy.data.objects.remove(sentinel, do_unlink=True)
                bpy.ops.wm.save_as_mainfile(
                    filepath=str(Path(args.save_blend).resolve())
                )
        except Exception:
            print(log.getvalue()[-4000:])
            raise
        finally:
            for name in passes:
                setattr(session, name, originals[name])
    if args.report:
        Path(args.report).write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
