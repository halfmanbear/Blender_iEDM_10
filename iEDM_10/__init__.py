
bl_info = {
  'name': "EDM File Importer 10.0",
  'description': "Importing of .EDM model files",
  'author': "Nicholas Devenish - (Rework by HalfManBear)",
  'version': (0,4,0),
  'blender': (4, 5, 6),
  'location': "File > Import > .EDM Files",
  'category': 'Import',
}

try:
  import bpy

  def register():
    from .io_operators import register as importer_register
    from .rna import register as rna_register
    rna_register()
    importer_register()

  def unregister():
    from .io_operators import unregister as importer_unregister
    from .rna import unregister as rna_unregister
    importer_unregister()
    rna_unregister()

  if __name__ == "__main__":
    register()
except ImportError:
  # Allow for now, as we might just want to import the sub-package
  pass