import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, OperatorFileListElement
from bpy.props import ( StringProperty,
                        BoolProperty,
                        IntProperty,
                        CollectionProperty,
                        EnumProperty,
                        FloatProperty,
                      )
import os

import logging
logger = logging.getLogger(__name__)

from .reader import read_file

class ImportEDM(Operator, ImportHelper):
  bl_idname = "import_mesh.edm"
  bl_label = "Import EDM"
  filename_ext = ".edm"

  filter_glob: StringProperty(
          default="*.edm",
          options={'HIDDEN'},
          )
  files: CollectionProperty(
          name="File Path",
          type=OperatorFileListElement,
          )
  directory: StringProperty(
          subtype='DIR_PATH',
          )

  shadeless: BoolProperty(name="Shadeless",
      description="Import materials as shadeless (no lights required in blender)",
      default=False)
  import_bounding_box: BoolProperty(name="Import Bounding Box",
      description="Create an Empty that visualises the EDM bounding box",
      default=False)
  import_user_box: BoolProperty(name="Import User Box",
      description="Create an Empty that visualises the EDM user box",
      default=False)
  debug_transforms: BoolProperty(
      name="Debug Transforms",
      description="Print EDM-local vs Blender local/world transform details during import",
      default=False)
  debug_transform_filter: StringProperty(
      name="Debug Filter",
      description="Only print transform debug entries containing this text (name/path)",
      default="")
  debug_transform_limit: IntProperty(
      name="Debug Limit",
      description="Maximum number of transform debug entries to print",
      default=200,
      min=1,
      soft_max=5000)
  mesh_origin_mode: EnumProperty(
      name="Mesh Origins",
      description="How to place origins for render-only meshes that have no explicit EDM transform node",
      items=[
        ("APPROX", "Approximate", "Recover a practical origin from mesh bounds (matches current importer behavior)"),
        ("RAW", "Raw EDM", "Keep raw EDM-local mesh coordinates and do not recenter render-only mesh origins"),
      ],
      default="APPROX")

  def execute(self, context):
    path = self.filepath
    if not path:
        self.report({'ERROR'}, "No input file selected.")
        return {'CANCELLED'}

    # The importer only supports one file at a time.
    # We will only process the one in self.filepath, which avoids issues with the `files` collection.
    
    # Import the file
    logger.warning("Reading EDM file {}".format(path))
    
    try:
        read_file(path, options={
          "shadeless": self.shadeless,
          "debug_transforms": self.debug_transforms,
          "debug_transform_filter": self.debug_transform_filter,
          "debug_transform_limit": self.debug_transform_limit,
          "mesh_origin_mode": self.mesh_origin_mode,
          "import_bounding_box": self.import_bounding_box,
          "import_user_box": self.import_user_box,
        })
    except Exception as e:
        self.report({'ERROR'}, "Failed to import EDM: {}".format(e))
        import traceback
        traceback.print_exc()
        return {'CANCELLED'}
    
    return {'FINISHED'}

def menu_import(self, context):
  self.layout.operator(ImportEDM.bl_idname, text="DCS World (.edm)")

def register():
  bpy.utils.register_class(ImportEDM)
  bpy.types.TOPBAR_MT_file_import.append(menu_import)

def unregister():
  bpy.types.TOPBAR_MT_file_import.remove(menu_import)
  bpy.utils.unregister_class(ImportEDM)
