"""Compare default and Raw origins at neutral and animated frames in Blender."""
import contextlib
import io
from pathlib import Path
import sys
import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import iEDM_10
from iEDM_10 import reader

iEDM_10.register()
for filename in sys.argv[sys.argv.index('--')+1:]:
    reference = {}
    path = Path(filename).resolve()
    max_error = 0.0
    for mode in ('RAW', 'APPROX'):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        with contextlib.redirect_stdout(io.StringIO()):
            reader.read_file(str(path), options={'mesh_origin_mode':mode, 'preserve_scene_boxes':True})
        for frame in (100, 0, 150, 200):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            meshes = {o.name:o for o in bpy.data.objects if o.type == 'MESH'}
            if mode == 'APPROX':
                assert set(meshes) == set(reference[frame]), (path.name, 'mesh names changed')
            else:
                reference[frame] = {}
            for name, obj in meshes.items():
                # Every vertex at neutral; sample across each mesh at other frames.
                vertices = obj.data.vertices
                indices = range(len(vertices)) if frame == 100 else range(0, len(vertices), max(1, len(vertices)//32))
                points = [obj.matrix_world @ vertices[i].co for i in indices]
                state = (obj.hide_viewport, obj.hide_render)
                if mode == 'RAW':
                    reference[frame][name] = (points, state)
                    continue
                expected, visibility = reference[frame][name]
                assert len(points) == len(expected) and state == visibility, (path.name, name, frame)
                error = max(((p-q).length for p,q in zip(points, expected)), default=0)
                max_error = max(max_error, error)
                assert error < 1e-4, (path.name, name, frame, 'origin mode moved geometry', error)
        print('PASS origin mode', path.name, mode, 'max_error', max_error, flush=True)
