def _print_import_diagnostics(edm, graph):
  """Print summary of EDM node types found vs Blender objects created."""
  from collections import Counter
  edm_counts = Counter()
  created_counts = Counter()
  skipped_empty = 0
  split_layout_counts = Counter()
  split_layout_examples = {}
  split_child_total = 0
  source_type_counts = Counter()
  shell_layout_counts = Counter()

  for node in iterate_all_objects(edm):
    type_name = type(node).__name__
    edm_counts[type_name] += 1
    source_type = getattr(node, "_source_type_name", None)
    if source_type:
      source_type_counts[source_type] += 1
    if type_name in {"ShellNode", "SkinNode"}:
      layout_variant = getattr(node, "_layout_variant", None)
      if layout_variant:
        shell_layout_counts["{}:{}".format(type_name, layout_variant)] += 1
    if type_name == "RenderNode":
      layout = getattr(node, "parentData_layout", None) or {}
      mode = layout.get("mode")
      if mode:
        split_layout_counts[mode] += 1
        split_child_total += 1
        split_layout_examples.setdefault(mode, node)

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
  if source_type_counts:
    print("Source EDM node types:")
    for name, count in sorted(source_type_counts.items()):
      print(f"  {name}: {count}")
  if shell_layout_counts:
    print("Shell-family fallback layouts:")
    for name, count in sorted(shell_layout_counts.items()):
      print(f"  {name}: {count}")
  root = getattr(edm, "root", None)
  if root is not None:
    print("Root boxes:")
    if getattr(root, "scene_bounds_box", None) is not None:
      print(f"  bounding_box: {root.scene_bounds_box}")
    if getattr(root, "user_box", None) is not None:
      print(f"  user_box: {root.user_box}")
    if getattr(root, "light_box", None) is not None:
      print(f"  light_box: {root.light_box}")
    print(f"  unknownC: {getattr(root, 'unknownC', None)}")
    print(f"  maxArgPlusOne: {getattr(root, 'maxArgPlusOne', None)}")
  billboard_nodes = [node for node in iterate_all_objects(edm) if type(node).__name__ == "BillboardNode"]
  if billboard_nodes:
    print("Billboard payloads:")
    for node in billboard_nodes[:5]:
      summary = getattr(node, "payload_summary", {}) or {}
      print(
        "  {}: bytes={} tail_u16={}".format(
          getattr(node, "name", "") or "<unnamed>",
          summary.get("payload_len"),
          summary.get("tail_u16"),
        )
      )
  if split_layout_counts:
    print("RenderNode split layouts:")
    for name, count in sorted(split_layout_counts.items()):
      print(f"  {name}: {count}")
  if skipped_empty:
    print(f"  Skipped (empty indexData / no handler): {skipped_empty}")
  if getattr(_import_ctx, "render_split_debug", {}).get("enabled") and split_layout_examples:
    print("RenderNode split details:")
    for mode in sorted(split_layout_examples):
      node = split_layout_examples[mode]
      layout = getattr(node, "parentData_layout", {}) or {}
      parent_data = getattr(node, "parentData", None) or []
      index_count = len(getattr(node, "indexData", []) or [])
      vertex_count = len(getattr(node, "vertexData", []) or [])
      extra = []
      if layout.get("swap_columns"):
        extra.append("swap_columns=True")
      if layout.get("force_fallback"):
        extra.append("force_fallback=True")
      if layout.get("shared_parent") is not None:
        extra.append(f"shared_parent={layout['shared_parent']}")
      owner_values = layout.get("owner_values")
      if owner_values:
        owner_hist = Counter(owner_values)
        extra.append("owner_hist={}".format(dict(sorted(owner_hist.items()))))
      print(
        "  {}: node='{}' parents={} verts={} indices={} {}".format(
          mode,
          getattr(node, "name", ""),
          len(parent_data),
          vertex_count,
          index_count,
          " ".join(extra).strip(),
        ).rstrip()
      )
      if parent_data:
        print(f"    parentData={parent_data}")
  print(f"Total Blender objects: {len([o for o in bpy.data.objects])}")
  print("--- End Diagnostics ---\n")


