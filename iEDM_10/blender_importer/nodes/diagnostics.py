def _print_import_diagnostics(edm, graph):
  """Print summary of EDM node types found vs Blender objects created."""
  from collections import Counter
  edm_counts = Counter()
  created_counts = Counter()
  skipped_empty = 0

  for node in iterate_all_objects(edm):
    type_name = type(node).__name__
    edm_counts[type_name] += 1

  def _count_created(tnode):
    nonlocal skipped_empty
    if tnode.render:
      type_name = type(tnode.render).__name__
      if tnode.blender and tnode.blender.type == 'MESH':
        created_counts[type_name] += 1
      elif tnode.blender and tnode.blender.type == 'EMPTY':
        created_counts[type_name + " (empty)"] += 1
      elif tnode.blender and tnode.blender.type == 'LIGHT':
        created_counts[type_name] += 1
      elif not tnode.blender or (tnode.parent and tnode.blender == tnode.parent.blender):
        skipped_empty += 1

  graph.walk_tree(_count_created)

  print("\n--- Import Diagnostics ---")
  print("EDM render nodes by type:")
  for name, count in sorted(edm_counts.items()):
    print(f"  {name}: {count}")
  print("Created Blender objects by type:")
  for name, count in sorted(created_counts.items()):
    print(f"  {name}: {count}")
  if skipped_empty:
    print(f"  Skipped (empty indexData / no handler): {skipped_empty}")
  print(f"Total Blender objects: {len([o for o in bpy.data.objects])}")
  print("--- End Diagnostics ---\n")


