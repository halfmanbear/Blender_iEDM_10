def read_file(filename, options=None):
  options = options or {}
  # Re-initialize the global import context for this new import session
  _import_ctx.__init__()
  _reset_transform_debug(options)
  _import_ctx.render_split_debug = {"enabled": bool(options.get("debug_render_splits", False))}
  _reset_mesh_origin_mode(options)
  _import_ctx.bone_import_ctx = None
  _import_ctx.source_dir = os.path.dirname(os.path.abspath(filename))

  # Parse the EDM file
  edm = EDMFile(filename)

  print("Raw file graph:")
  print_edm_graph(edm.transformRoot)

  # Store EDM version globally early so importer heuristics can use it before
  # addon-export patch toggles are configured.
  _import_ctx.edm_version = edm.version

  # Detect v10 owner-encoded split render graphs (shared-parent render chunks).
  # These small control-node assets (e.g. Geomerty+Animation+Collision.edm)
  # are sensitive to mesh recentering and plain-root basis suppression.
  has_bones = any(_is_bone_transform(n) for n in edm.nodes)
  has_owner_encoded_split_renders = False
  has_generic_render_chunks = False
  has_shell_nodes = False
  try:
    if _import_ctx.edm_version >= 10:
      for obj in iterate_all_objects(edm):
        if isinstance(obj, RenderNode) and _is_generic_render_name(getattr(obj, "name", "")):
          has_generic_render_chunks = True
        elif isinstance(obj, ShellNode):
          has_shell_nodes = True
        if getattr(obj, "shared_parent", None) is not None:
          has_owner_encoded_split_renders = True
        if has_owner_encoded_split_renders and has_generic_render_chunks and has_shell_nodes:
          break
  except Exception:
    has_owner_encoded_split_renders = False
    has_generic_render_chunks = False
    has_shell_nodes = False
  plain_root_v10 = bool(_import_ctx.edm_version >= 10 and type(getattr(edm, "transformRoot", None)) is Node)
  auto_geometry_safe_v10_split = bool(
    plain_root_v10
    and not has_bones
    and (
      has_owner_encoded_split_renders
      or (has_generic_render_chunks and has_shell_nodes)
    )
  )

  # Preserve authored pivots on v10 split control-node assets by disabling
  # mesh-origin recentering. Blender operator defaults may pass
  # mesh_origin_mode='APPROX' explicitly, so treat APPROX as auto-overridable.
  user_mesh_origin_mode = str(options.get("mesh_origin_mode", "") or "").upper()
  can_auto_override_mesh_origin = (user_mesh_origin_mode in {"", "APPROX"})
  if auto_geometry_safe_v10_split and can_auto_override_mesh_origin and _import_ctx.mesh_origin_mode != "RAW":
    _import_ctx.mesh_origin_mode = "RAW"
    print("Info: Auto-selected mesh origin mode RAW for v10 split control-node asset (plain root, no bones).")
  elif (
    can_auto_override_mesh_origin
    and _import_ctx.mesh_origin_mode != "RAW"
    and plain_root_v10
    and not has_bones
  ):
    # F-117-style plain-root non-skeletal v10 assets preserve authored object
    # transforms only when mesh origin recentering is disabled. APPROX mode adds
    # per-mesh local offsets (e.g. F-117_paint.012 +0.007624 X / +0.438309 Z).
    _import_ctx.mesh_origin_mode = "RAW"
    print("Info: Auto-selected mesh origin mode RAW for v10 plain-root non-skeletal asset (preserve authored transforms).")

  # Set the required blender preferences
  bpy.context.preferences.edit.use_negative_frames = False
  bpy.context.scene.use_preview_range = False
  bpy.context.scene.frame_start = 0
  bpy.context.scene.frame_end = FRAME_SCALE
  bpy.context.scene.frame_set(0)



  # Convert the materials (cwd matters for search)
  with chdir(os.path.dirname(os.path.abspath(filename))):
    for material in edm.root.materials:
      material.blender_material = create_material(material)
      if material.blender_material and options.get("shadeless", False):
        _apply_shadeless(material.blender_material)

  if _import_ctx.mesh_origin_mode == "RAW":
    print("Info: Mesh origin mode RAW (no geometry-center recenter on render-only nodes)")

  # Store version in scene for export
  bpy.context.scene.edm_version = edm.version

  _import_ctx.file_has_bones = bool(has_bones)

  # For ALL .edm files, we MUST apply _ROOT_BASIS_FIX (the inverse of the DCS 
  # Y-up coordinate swap) to the root children so that they import into Blender 
  # as Z-up. When re-exported, io_scene_edm will apply its ROOT_TRANSFORM_MATRIX 
  # (which is the forward DCS swap) and restore the file to its original state, 
  # whether it was originally exported from 3ds Max (with a root swap matrix) 
  # or from Blender (with optimized meshes in Y-up space).

  # Create explicit root box empties from decoded v10 root metadata.
  # The official exporter recomputes AABB from all scene geometry, including
  # extreme-position connectors (e.g. particle views at z=-280m), producing a
  # huge bounding box.  These special empties carry the original raw AABB values
  # so the export AABB passthrough patch can restore them exactly.
  _cache_root_aabb_payloads_on_scene(edm.root)
  if not _has_special_box("BOUNDING_BOX"):
    create_bounding_box_from_root(edm.root)
  _create_user_box_from_root(edm.root)
  _create_light_box_from_root(edm.root)

  # Build the translation graph with hierarchy simplification
  graph = build_graph(edm)
  graph.print_tree()

  # For plain model::Node roots, avoid forcing a Blender empty:
  # the official round-trip can preserve a non-transform root and forcing an
  # authored object turns it into `TransformNode '_'` on export.
  # Keep an explicit root object only when the file root actually carries
  # transform payload.
  root_tf = getattr(graph.root, "transform", None)
  has_root_transform_payload = isinstance(root_tf, (TransformNode, AnimatingNode))
  
  # For V10+, we ALWAYS create a root Blender object to carry the global 
  # coordinate basis fix. This ensures that ALL elements in the file (even 
  # those attached directly to the file root) are correctly rotated.
  force_root_obj = (_import_ctx.edm_version >= 10)

  if has_root_transform_payload or force_root_obj:
    root_name = getattr(root_tf, "name", "") or ""
    if not root_name.strip():
      root_name = "_EDMFileRoot"

    root_obj = bpy.data.objects.new(root_name, None)
    root_obj.empty_display_size = 0.1
    bpy.context.collection.objects.link(root_obj)
    graph.root.blender = root_obj

    # For V10, apply the global basis fix to the root object.
    # If the root carries its own transform (e.g. 3ds Max/Blender_ioEDM swap),
    # multiply them so they cancel out to Identity in Blender space.
    if _import_ctx.edm_version >= 10:
      m_root = Matrix.Identity(4)
      if hasattr(root_tf, "matrix"):
        m_root = Matrix(root_tf.matrix)
      elif hasattr(root_tf, "base") and hasattr(root_tf.base, "matrix"):
        m_root = Matrix(root_tf.base.matrix)
      
      # Apply the fix: Blender_World = _ROOT_BASIS_FIX * EDM_Root_Matrix
      root_obj.matrix_basis = _ROOT_BASIS_FIX @ m_root
    elif has_root_transform_payload:
      apply_node_transform(graph.root, graph.root.blender, used_shared_parent=False)
  else:
    graph.root.blender = None

  # Reconstruct armature data for Bone/ArgAnimatedBone transforms
  _prepare_bone_import(graph, graph.root.blender)

  # Walk through every node, and do the node processing
  graph.walk_tree(process_node)

  # Post-children pass: apply LodNode properties after children are created.
  graph.walk_tree(_process_lod_post_children)

  # Diagnostic: count EDM nodes vs created Blender objects
  _print_import_diagnostics(edm, graph)

  # Optional post-import collection organization (disabled by default for parity)
  if options.get("assign_collections", False):
    _assign_collections(graph)

  if _import_ctx.transform_debug.get("enabled"):
    if _import_ctx.transform_debug["emitted"] >= _import_ctx.transform_debug["limit"]:
      print("Info: Transform debug reached output limit of {}".format(_import_ctx.transform_debug["limit"]))
    print("Info: Transform debug emitted {} record(s)".format(_import_ctx.transform_debug["emitted"]))

  bpy.context.view_layer.update()

def create_visibility_actions(visNode):
  """Creates visibility actions from an ArgVisibilityNode"""
  def _vis_arg_to_frame(value):
    # EDM visibility ranges are authored on the same normalized timeline as
    # other v10 animation channels.
    return int(round((value + 1.0) * FRAME_SCALE / 2.0))

  actions = []
  for (arg, ranges) in visNode.visData:
    # Creates a visibility animation track
    vis_name = visNode.name or "node"
    # Strip visibility "v_" prefix and animation prefixes from name
    if vis_name.startswith("v_"):
      vis_name = vis_name[2:]
    vis_name = _strip_anim_prefix(vis_name)
    action = bpy.data.actions.new("{}_{}_Visib".format(arg, vis_name))
    actions.append(action)
    if hasattr(action, "argument"):
      action.argument = arg
    # Create f-curves for official exporter visibility property.
    # Do not key Blender hide flags here: exporter traversal can treat hidden
    # objects as non-exportable, which drops render children under ArgVisibility
    # wrappers (e.g. Dummy650/648/598) and breaks parity.
    curve_visible = action.fcurves.new(data_path="VISIBLE")

    def _add_constant_key(curve, frame, value):
      curve.keyframe_points.add(1)
      key = curve.keyframe_points[-1]
      key.co = (frame, float(value))
      key.interpolation = 'CONSTANT'

    # Create the keyframe data
    # Visibility ranges in EDM are [start, end) where end is the first OFF
    # frame boundary. Reconstruct official scene key style:
    # Emit:
    #   start -> 1
    #   end-1 -> 1
    #   end   -> 0
    for (start, end) in ranges:
      frameStart = max(0, min(FRAME_SCALE, _vis_arg_to_frame(start)))
      frameOff = FRAME_SCALE + 1 if end > 1.0 else _vis_arg_to_frame(end)
      if frameOff <= frameStart:
        frameOff = frameStart + 1
      frameEndVisible = frameOff - 1

      _add_constant_key(curve_visible, frameStart, 1.0)

      if frameOff <= FRAME_SCALE:
        _add_constant_key(curve_visible, frameEndVisible, 1.0)

        _add_constant_key(curve_visible, frameOff, 0.0)
  return actions

def _arg_anim_vector_to_blender(node, value):
  vec = _anim_vector_to_blender(value)
  try:
    if (
      _import_ctx.edm_version >= 10
      and _is_child_of_file_root(node)
      and hasattr(node, "base")
      and hasattr(node.base, "matrix")
      and _is_neg90_x_basis_matrix(Matrix(node.base.matrix))
    ):
      return _ROOT_BASIS_FIX.to_3x3() @ Vector(vec)
  except Exception as e:
    print(f"Warning in blender_importer\\import_pipeline.py: {e}")
  return vec


def add_position_fcurves(action, keys, transform_left, transform_right, node=None):
  "Adds position fcurve data to an animation action"
  # Create an fcurve for every component (reuse existing if present)
  curves = []
  for i in range(3):
    curve = action.fcurves.find("location", index=i)
    if curve is None:
      curve = action.fcurves.new(data_path="location", index=i)
    curves.append(curve)

  # Loop over every keyframe in this animation
  for framedata in keys:
    frame = _anim_frame_to_scene_frame(framedata.frame)

    # Calculate the position transformation (convert keyframe from EDM Y-up to Blender Z-up)
    anim_pos = _arg_anim_vector_to_blender(node, framedata.value) if node is not None else _anim_vector_to_blender(framedata.value)
    newPosMat = transform_left @ Matrix.Translation(anim_pos) @ transform_right
    newPos = newPosMat.decompose()[0]

    for curve, component in zip(curves, newPos):
      curve.keyframe_points.add(1)
      curve.keyframe_points[-1].co = (frame, component)
      curve.keyframe_points[-1].interpolation = 'LINEAR'

def add_rotation_fcurves(action, keys, transform_left, transform_right, quat_to_blender=None):
  "Adds rotation fcurve action to an animation action"
  # Keep rotation keys in quaternion form so official exporter can consume
  # them directly without Euler re-conversion drift.
  if quat_to_blender is None:
    quat_to_blender = _anim_quaternion_to_blender
  curves = []
  for i in range(4):
    curve = action.fcurves.find("rotation_quaternion", index=i)
    if curve is None:
      curve = action.fcurves.new(data_path="rotation_quaternion", index=i)
    curves.append(curve)

  # Loop over every keyframe in this animation
  previous_quat = None
  for framedata in keys:
    frame = _anim_frame_to_scene_frame(framedata.frame)
    
    # Calculate the rotation transformation (convert keyframe from EDM Y-up to Blender Z-up)
    newRotQuat = transform_left @ quat_to_blender(framedata.value) @ transform_right
    if previous_quat is not None and previous_quat.dot(newRotQuat) < 0.0:
      newRotQuat = -newRotQuat
    previous_quat = newRotQuat.copy()

    for curve, component in zip(curves, newRotQuat):
      curve.keyframe_points.add(1)
      curve.keyframe_points[-1].co = (frame, component)
      curve.keyframe_points[-1].interpolation = 'LINEAR'

def add_scale_fcurves(action, keys):
  "Adds scale fcurve action to an animation action"
  if not keys:
    return
  curves = []
  for i in range(3):
    curve = action.fcurves.find("scale", index=i)
    if curve is None:
      curve = action.fcurves.new(data_path="scale", index=i)
    curves.append(curve)

  for framedata in keys:
    frame = _anim_frame_to_scene_frame(framedata.frame)
    value = framedata.value
    comps = _anim_scale_components(value)

    for curve, component in zip(curves, comps):
      curve.keyframe_points.add(1)
      curve.keyframe_points[-1].co = (frame, component)
      curve.keyframe_points[-1].interpolation = 'LINEAR'


def _transform_uses_quaternion_rotation(tfnode, obj=None):
  try:
    if isinstance(tfnode, ArgAnimationNode) and getattr(tfnode, "rotData", None):
      return True
  except Exception:
    pass
  try:
    action = getattr(getattr(obj, "animation_data", None), "action", None)
    if action is not None:
      return any(fc.data_path == "rotation_quaternion" for fc in action.fcurves)
  except Exception:
    pass
  return False

def create_arganimation_actions(node):
  "Creates a set of actions to represent an ArgAnimationNode"
  actions = []

  # Work out the transformation chain for this object.
  # V10 arg transform payloads are already authored in Blender-local terms,
  # except top-level nodes which include the file root basis wrapper.
  aabS_raw = MatrixScale(Vector((node.base.scale[0], node.base.scale[1], node.base.scale[2])))
  aabT_raw = Matrix.Translation(Vector(node.base.position))
  q1_raw = node.base.quat_1 if hasattr(node.base.quat_1, "to_matrix") else Quaternion(node.base.quat_1)
  q1m_raw = q1_raw.to_matrix().to_4x4()
  base_mat = Matrix(node.base.matrix)
  
  # Parity Fix: ArgAnimatedBone has an extra boneTransform that must be 
  # included in the static part of the animation chain.
  if type(node).__name__ == "ArgAnimatedBone" and hasattr(node, "boneTransform"):
    base_mat = base_mat @ Matrix(node.boneTransform)

  mat_bl = base_mat
  aabS = aabS_raw
  aabT = Matrix.Translation(_arg_anim_vector_to_blender(node, node.base.position))
  q1m = _anim_base_quaternion_q1_to_blender(node, node.base.quat_1).to_matrix().to_4x4()
  key_quat_to_blender = lambda q: _anim_base_quaternion_q1_to_blender(node, q)
  q1 = q1m.to_quaternion()
  q1_rot = q1
  mat_anim = mat_bl
  matQuat = mat_bl.decompose()[1]

  # Some v10 ArgAnimationNode controllers embed a static pivot translation in
  # base.matrix and also carry a non-zero base.position. Treating both as
  # separate static translations pushes the object too far from the airframe.
  embedded_mat_translation = mat_bl.decompose()[0]
  base_position_vec = Vector(node.base.position)
  use_embedded_translation_only = (
    isinstance(node, ArgAnimationNode)
    and not isinstance(node, ArgPositionNode)
    and embedded_mat_translation.length > 1e-6
    and base_position_vec.length > 1e-6
  )

  # With:
  #   aabT = Matrix.Translation(argNode.base.position)
  #   aabS = argNode.base.scale as a Scale Matrix
  #   q1m  = argNode.base.quat_1 as a Matrix (e.g. fixQuat1)
  #   q2m  = argNode.base.quat_2 as a Matrix (e.g. fixQuat2)
  #   mat  = argNode.base.matrix
  #   m2b  = matrix_to_blender e.g. swapping Z -> -Y

  # Save the 'null' transform onto the node, because we will need to
  # apply it to any object using this node's animation.

  #               Geometry is in Blender-space in-file already, and the import
  #                  procedure over-rotated it into an undefined space. Rotate 
  #                                                    back into Blender-space
  #                                                                      |
  #                                Transforms SAVED in blender-space     |
  #                                                   |     |      |     |
  #             Rotates geometry into file-space      |     |      |     |
  #                                           |       |     |      |     |
  # Reconstruct the neutral local transform from the same chain used at export.
  # Positional arg animations drive object location curves directly, so keeping
  # base translation on the object would be overwritten by the F-curves at the
  # current frame. For those nodes, carry base translation in the action basis
  # instead and leave the static object transform without translation.
  has_position_keys = bool(getattr(node, "posData", None))
  if use_embedded_translation_only:
    zero_mat = (mat_bl @ q1m @ aabS)
  elif has_position_keys:
    zero_mat = (mat_bl @ q1m @ aabS)
  else:
    zero_mat = (mat_bl @ aabT @ q1m @ aabS)
  
  bmat_inv = getattr(node, "bmat_inv", None)
  if bmat_inv is not None:
    # bmat_inv is often a coordinate system correction (e.g. 3ds Max vs Blender)
    # for that specific animation node.
    try:
      zero_mat = zero_mat @ bmat_inv.inverted()
    except Exception as e:
      print(f"Warning in blender_importer\import_pipeline.py: {e}")

  node.zero_transform_local_matrix = zero_mat
  dcLoc, dcRot, dcScale = zero_mat.decompose()
  node.zero_transform_matrix = zero_mat
  node.zero_transform = dcLoc, dcRot, dcScale

  # Split a single node into separate actions based on the argument
  for arg in node.get_all_args():
    # Filter the animation data for this node to just this action
    posData = [x[1] for x in node.posData if x[0] == arg]
    rotData = [x[1] for x in node.rotData if x[0] == arg]
    scaleData = [x[1] for x in node.scaleData if x[0] == arg]
    # Create the action
    action_name = "{}_{}".format(arg, _strip_anim_prefix(node.name or "anim"))
    action = bpy.data.actions.new(action_name)
    actions.append(action)
    if hasattr(action, "argument"):
      action.argument = arg
    
    # Calculate the pre and post-animation-value transforms.
    # EDM data is stored in Y-up; convert into Blender (Z-up) by
    # reusing the same conversion applied to zero_transform.
    leftRotation = matQuat @ q1_rot
    rightRotation = Quaternion((1, 0, 0, 0))
    
    # Positional arg animations need base translation carried in their action
    # basis so keyframe evaluation preserves the authored rest location.
    if use_embedded_translation_only:
      leftPosition = mat_anim
    elif has_position_keys:
      leftPosition = (mat_anim @ aabT)
    else:
      # STRIP TRANSLATION from the Action Basis.
      leftPosition = (mat_anim @ aabT).to_3x3().to_4x4()
    
    # Factor in bmat_inv to the animation basis if present.
    rightPosition = q1m @ aabS
    if bmat_inv is not None:
      try:
        rightPosition = rightPosition @ bmat_inv.inverted()
        # Rotation basis also needs correction
        leftRotation = leftRotation @ bmat_inv.to_quaternion().inverted()
      except Exception as e:
        print(f"Warning in blender_importer\import_pipeline.py: {e}")

    # Build the f-curves for the action
    for pos in posData:
      add_position_fcurves(action, pos, leftPosition, rightPosition, node=node)
    for rot in rotData:
      add_rotation_fcurves(action, rot, leftRotation, rightRotation, quat_to_blender=key_quat_to_blender)
    for sca_pair in scaleData:
      # Scale entries are stored as (4-component keys, 3-component keys);
      # use the 3-component set for object XYZ scale curves.
      keys3 = sca_pair[1] if isinstance(sca_pair, tuple) and len(sca_pair) > 1 else []
      add_scale_fcurves(action, keys3)
  # Return these new actions
  return actions


def get_actions_for_node(node):
  """Accepts a node and gets or creates actions to apply their animations"""
  
  # Don't do this twice
  if hasattr(node, "actions") and node.actions:
    actions = node.actions
  else:
    actions = []
    if isinstance(node, ArgVisibilityNode):
      actions = create_visibility_actions(node)
    if isinstance(node, ArgAnimationNode):
      actions = create_arganimation_actions(node)
    # Save these actions on the node
    node.actions = actions

  return actions


def _find_texture_file(name):
  """
  Searches for a texture file given a basename without extension.

  The current working directory will be searched, as will any
  subdirectories called "textures/", for any file starting with the
  designated name '{name}.'
  """
  files = glob.glob(name+".*")
  if not files:
    matcher = re.compile(fnmatch.translate(name+".*"), re.IGNORECASE)
    files = [x for x in glob.glob("*.*") if matcher.match(x)]
    if not files:
      files = glob.glob("textures/"+name+".*")
      if not files:
        matcher = re.compile(fnmatch.translate("textures/"+name+".*"), re.IGNORECASE)
        files = [x for x in glob.glob("textures/*.*") if matcher.match(x)]
        if not files:
          print("Warning: Could not find texture named {}".format(name))
          return None
  # print("Found {} as: {}".format(name, files))
  if len(files) > 1:
    print("Warning: Found more than one possible match for texture named {}. Using {}".format(name, files[0]))
    files = [files[0]]
  textureFilename = files[0]
  return os.path.abspath(textureFilename)

def create_material(material):
  """Create a blender node-based PBR material from an EDM one"""
  # Create a new material
  mat = bpy.data.materials.new(name=material.name)
  mat.use_nodes = True
  nodes = mat.node_tree.nodes
  links = mat.node_tree.links

  # Clear existing nodes
  for node in nodes:
    nodes.remove(node)

  # Create Principled BSDF and Output nodes
  principled_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
  principled_bsdf.location = (0, 0)
  material_output = nodes.new('ShaderNodeOutputMaterial')
  material_output.location = (400, 0)
  links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

  # --- Handle Textures ---
  texture_nodes = {}
  for tex_def in material.textures:
    filename = _find_texture_file(tex_def.name)
    if not filename:
      continue

    # Create image texture node
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(filename)
    tex_image.location = (-400, tex_def.index * -300)
    texture_nodes[tex_def.index] = tex_image

  # Connect textures to Principled BSDF
  if 0 in texture_nodes: # Diffuse
    tex_image = texture_nodes[0]
    tex_image.image.colorspace_settings.name = 'sRGB'
    links.new(tex_image.outputs['Color'], principled_bsdf.inputs['Base Color'])

  if 1 in texture_nodes: # Normal
    tex_image = texture_nodes[1]
    tex_image.image.colorspace_settings.name = 'Non-Color'
    normal_map_node = nodes.new('ShaderNodeNormalMap')
    normal_map_node.location = (-200, -300)
    links.new(tex_image.outputs['Color'], normal_map_node.inputs['Color'])
    links.new(normal_map_node.outputs['Normal'], principled_bsdf.inputs['Normal'])

  if 2 in texture_nodes: # Specular
    tex_image = texture_nodes[2]
    tex_image.image.colorspace_settings.name = 'Non-Color'
    links.new(tex_image.outputs['Color'], principled_bsdf.inputs['Specular IOR Level'])

  # --- Handle Uniforms ---
  # Metallic
  if material.material_name == 'chrome_material':
    principled_bsdf.inputs['Metallic'].default_value = 1.0
  else:
    principled_bsdf.inputs['Metallic'].default_value = 0.0

  # Roughness from specPower
  specPower = material.uniforms.get("specPower", None)
  if specPower is not None:
    # This is a guess, might need tweaking. Assuming specPower is in a range of 0-1024 (common for older shaders)
    # Higher specPower means a smaller, more intense highlight, which means lower roughness.
    roughness = (1.0 - (specPower / 1024.0))**0.5 # Using sqrt for a more perceptually linear mapping
    principled_bsdf.inputs['Roughness'].default_value = max(0.0, min(1.0, roughness))
  else:
    principled_bsdf.inputs['Roughness'].default_value = 0.5

  # Specular from specFactor
  specFactor = material.uniforms.get("specFactor", None)
  if specFactor is not None:
    principled_bsdf.inputs['Specular IOR Level'].default_value = specFactor

  # --- Handle Blending ---
  if material.blending in (1, 2): # Blend or Alpha Test
    # Connect diffuse texture alpha to Principled BSDF Alpha input
    if 0 in texture_nodes:
      links.new(texture_nodes[0].outputs['Alpha'], principled_bsdf.inputs['Alpha'])
    mat.surface_render_method = 'DITHERED'

  # Add official exporter-compatible EDM group node when available.
  official_attached = _attach_official_material_bridge(mat, material, texture_nodes)
  if not official_attached:
    print(f"Warning: Material bridge not attached for '{material.material_name}' "
          f"(material '{material.name}') — re-export may not recognise this material")
  if official_attached:
    # Keep imported material node layout close to official reference scenes:
    # Output + official EDM group (+ texture nodes), without extra Principled.
    try:
      nodes.remove(principled_bsdf)
    except Exception as e:
      print(f"Warning in blender_importer\import_pipeline.py: {e}")

  # Set other material properties
  try:
    mat.edm_material = material.material_name
  except TypeError:
    print(f"Warning: Unknown material type '{material.material_name}', defaulting to 'def_material'")
    mat.edm_material = "def_material"
  mat.edm_blending = str(material.blending)

  return mat


def create_connector(connector):
  """Create an empty object representing a connector"""
  ob = bpy.data.objects.new(connector.name, None)
  ob.empty_display_type = "CUBE"
  ob.edm.is_connector = True
  _set_official_special_type(ob, 'CONNECTOR')

  # Preserve connector properties for round-trip export
  if hasattr(connector, 'props') and connector.props:
    lines = []
    for key, value in connector.props.items():
      if key.startswith("__"):
        continue
      if isinstance(value, str):
        lines.append("{} = '{}'".format(key, value.replace("'", "\\'")))
      elif isinstance(value, (int, float)):
        lines.append("{} = {}".format(key, repr(value)))
    if lines:
      _set_edmprop(ob, "CONNECTOR_EXT", "\n".join(lines))

  bpy.context.collection.objects.link(ob)
  return ob

def create_segments(segments_node):
  """Create a mesh with edges from a SegmentsNode (collision lines)"""
  # Create a new mesh
  mesh = bpy.data.meshes.new(segments_node.name)

  # Extract vertices and edges from segment data
  verts = []
  edges = []

  for i, segment in enumerate(segments_node.data):
    # Each segment is 6 floats: [x1, y1, z1, x2, y2, z2]
    v1 = Vector((segment[0], segment[1], segment[2]))
    v2 = Vector((segment[3], segment[4], segment[5]))

    v1 = Vector(v1)
    v2 = Vector(v2)

    # Add vertices
    verts.extend([v1, v2])
    # Add edge connecting these two vertices
    edges.append([i*2, i*2+1])

  # Create mesh from vertex and edge data
  mesh.from_pydata(verts, edges, [])
  mesh.update()

  # Create object and set properties
  ob = bpy.data.objects.new(segments_node.name, mesh)
  ob.edm.is_collision_line = True
  ob.edm.is_renderable = False
  _set_official_special_type(ob, 'COLLISION_LINE')
  bpy.context.collection.objects.link(ob)

  return ob

def apply_node_transform(node, obj, used_shared_parent=False):
  """Assigns the transform to a given node. node can be a TranslationNode or raw EDM node."""
  # Unpack TranslationNode if provided
  tnode = node if hasattr(node, "transform") else None
  tfnode = tnode.transform if tnode else node
  
  if tfnode is None:
    return

  wants_quaternion_rotation = _transform_uses_quaternion_rotation(tfnode, obj)
  obj.rotation_mode = "QUATERNION" if wants_quaternion_rotation else "XYZ"

  def _debug_set_trace(source_label, local_mat):
    if not _import_ctx.transform_debug.get("enabled"):
      return
    needle = _import_ctx.transform_debug.get("filter")
    node_name = getattr(tfnode, "name", "") or type(tfnode).__name__
    if needle:
      n = needle.lower()
      if n not in node_name.lower() and n not in obj.name.lower():
        return
    loc, rot, scale = local_mat.decompose()
    print("[iEDM][TFSET] {} src={} obj={} loc={} rot_deg={} scale={}".format(
      node_name, source_label, obj.name, _debug_fmt_vec3(loc), _debug_fmt_rot_deg(rot), _debug_fmt_vec3(scale)
    ))

  def _set_local_matrix(local_mat):
    try:
      obj.matrix_basis = local_mat
      return
    except Exception as e:
      print(f"Warning in blender_importer\import_pipeline.py: {e}")
    loc, rot, scale = local_mat.decompose()
    obj.location = loc
    if wants_quaternion_rotation:
      obj.rotation_mode = "QUATERNION"
      obj.rotation_quaternion = rot
    else:
      obj.rotation_mode = "XYZ"
      obj.rotation_euler = rot.to_euler('XYZ')
    obj.scale = scale

  def _compose_with_parent_if_enabled(local_mat):
    if _import_ctx.legacy_v10_parent_compose and obj.parent is not None and _import_ctx.edm_version >= 10:
      try:
        return obj.parent.matrix_basis @ local_mat
      except Exception:
        return local_mat
    return local_mat

  # 1. Base Transform from Node.transform
  final_local = Matrix.Identity(4)
  if isinstance(tfnode, TransformNode):
    is_connector_obj = _is_connector_transform(tfnode, obj)
    local_mat = Matrix(tfnode.matrix)
    if is_connector_obj:
      local_mat = local_mat @ Matrix.Rotation(math.radians(-90.0), 4, "X")
    final_local = _compose_with_parent_if_enabled(local_mat)
  elif isinstance(tfnode, AnimatingNode):
    if hasattr(tfnode, "zero_transform_local_matrix"):
      final_local = _compose_with_parent_if_enabled(tfnode.zero_transform_local_matrix)
    elif hasattr(tfnode, "zero_transform_matrix"):
      final_local = _compose_with_parent_if_enabled(tfnode.zero_transform_matrix)
    elif hasattr(tfnode, "zero_transform"):
      loc, rot, scale = tfnode.zero_transform
      final_local = _compose_with_parent_if_enabled(Matrix.LocRotScale(loc, rot, scale))

  # 2. Inherit RenderNode local offset if present
  # In EDM, RenderNodes can have their own pos/matrix separate from the transform node.
  render = tnode.render if tnode else None
  if render:
    m_rn = Matrix.Identity(4)
    if hasattr(render, "pos"):
      m_rn = Matrix.Translation(Vector(render.pos[:3]))
    elif hasattr(render, "matrix"):
      m_rn = Matrix(render.matrix)
    
    if not m_rn.is_identity:
      final_local = final_local @ m_rn

  _set_local_matrix(final_local)
  _debug_set_trace(type(tfnode).__name__, final_local)


def _create_mesh(vertexData, indexData, vertexFormat, compact=True):
  """Creates a blender mesh object from vertex, index and format data"""

  if compact:
    # Build compact per-part meshes from referenced vertices only.
    # Keeping full shared pools (e.g. 3k+ verts for every split chunk) causes
    # exporter merge/reindex behavior to diverge from official blend exports.
    used_indices = sorted(set(indexData))
    new_vertices = [vertexData[i] for i in used_indices]
    index_lookup = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
    new_indices = [index_lookup[i] for i in indexData]
  else:
    # For skinned meshes, we MUST use the full vertex pool to keep indices
    # in sync with the bone weights.
    new_vertices = vertexData
    new_indices = indexData

  # Make sure we have the right number of indices...
  assert len(new_indices) % 3 == 0

  bm = bmesh.new()
  # For strict official-export parity, rely on Blender-generated split normals.
  # Importing per-vertex custom normals here causes small vertex-split drift on
  # re-export (e.g. +2 verts on merged render chunks in this test asset).
  import_custom_normals = False
  vertex_normals = [] if (import_custom_normals and vertexFormat.normal_indices) else None

  # Extract where the indices are
  posIndex = vertexFormat.position_indices
  normIndex = vertexFormat.normal_indices
  uvIndexSets = []
  uv_offset = sum(vertexFormat.data[:4]) if hasattr(vertexFormat, "data") else 0
  if hasattr(vertexFormat, "data"):
    for channel in (4, 5, 6):
      if channel >= len(vertexFormat.data):
        continue
      count = int(vertexFormat.data[channel])
      if count >= 2:
        uvIndexSets.append(list(range(uv_offset, uv_offset + 2)))
      uv_offset += max(0, count)

  # Create the BMesh vertices, optionally with normals
  for i, vtx in enumerate(new_vertices):
    pos = Vector(vtx[x] for x in posIndex)
    vert = bm.verts.new(pos)
    if normIndex and vertex_normals is not None:
      norm = Vector(vtx[x] for x in normIndex)
      if norm.length_squared > 1e-12:
        norm.normalize()
      vert.normal = norm
      vertex_normals.append(norm.copy())

  bm.verts.ensure_lookup_table()

  # Prepare for texture information
  uv_layers = []
  for uv_set_idx, _ in enumerate(uvIndexSets):
    layer_name = "UVMap" if uv_set_idx == 0 else "UVMap.{:03d}".format(uv_set_idx)
    uv_layer = bm.loops.layers.uv.get(layer_name)
    if uv_layer is None:
      uv_layer = bm.loops.layers.uv.new(layer_name)
    uv_layers.append(uv_layer)
  
  # Generate faces, with texture coordinate information.
  # Some EDM files include degenerate or duplicate triangles that bmesh rejects.
  skipped_degenerate = 0
  skipped_duplicate = 0
  for face in [new_indices[i:i+3] for i in range(0, len(new_indices), 3)]:
    if len(set(face)) < 3:
      skipped_degenerate += 1
      continue
    try:
      f = bm.faces.new([bm.verts[i] for i in face])
      # Add UV data if we have any
      if uvIndexSets:
        for loop, v_idx in zip(f.loops, face):
          vtx = new_vertices[v_idx]
          for uv_layer, uv_idx in zip(uv_layers, uvIndexSets):
            if len(uv_idx) < 2:
              continue
            if uv_idx[1] >= len(vtx):
              continue
            loop[uv_layer].uv = (vtx[uv_idx[0]], 1 - vtx[uv_idx[1]])
    except ValueError as e:
      # BMesh rejects exact duplicate faces; keep track and continue.
      skipped_duplicate += 1

  if skipped_degenerate or skipped_duplicate:
    print("Info: Skipped {} degenerate and {} duplicate faces while building mesh".format(
      skipped_degenerate, skipped_duplicate
    ))

  # Create the mesh object
  mesh = bpy.data.meshes.new("Mesh")
  bm.to_mesh(mesh)
  bm.free()

  # EDM meshes are authored for smooth shading in most cases; keep curved
  # surfaces visually faithful by enabling smooth polygons and importing
  # provided normals as custom split normals.
  for poly in mesh.polygons:
    poly.use_smooth = True

  if vertex_normals and len(vertex_normals) == len(mesh.vertices):
    try:
      normals = [tuple(n) for n in vertex_normals]
      if hasattr(mesh, "normals_split_custom_set_from_vertices"):
        mesh.normals_split_custom_set_from_vertices(normals)
      else:
        loop_normals = [normals[loop.vertex_index] for loop in mesh.loops]
        mesh.normals_split_custom_set(loop_normals)
    except Exception as e:
      print("Warning: Failed to assign custom normals: {}".format(e))

  if hasattr(mesh, "use_auto_smooth"):
    mesh.use_auto_smooth = True

  mesh.update()

  return mesh


# Mapping from EDM animated-uniform names to EDMProps field names.
_ANIMATED_UNIFORM_TO_EDMPROPS = {
  "emissiveValue":  "EMISSIVE_ARG",
  "colorShift":     "COLOR_ARG",
  "opacityValue":   "OPACITY_VALUE_ARG",
}


def _map_animated_uniforms_to_edmprops(ob, node):
  """Extract animated uniform argument numbers and set EDMProps ARG fields."""
  if not hasattr(ob, "EDMProps"):
    return
  mat = getattr(node, "material", None)
  if mat is None:
    return
  anim_uniforms = getattr(mat, "animated_uniforms", None)
  if not anim_uniforms:
    return
  from ..edm_format.typereader import AnimatedProperty
  for name, prop in anim_uniforms.items():
    if not isinstance(prop, AnimatedProperty):
      continue
    edmprops_field = _ANIMATED_UNIFORM_TO_EDMPROPS.get(name)
    if edmprops_field and hasattr(ob.EDMProps, edmprops_field):
      if prop.argument is not None and prop.argument >= 0:
        try:
          # Clamp to 32-bit signed int limit for Blender's IntProperty
          val = min(2147483647, int(prop.argument))
          setattr(ob.EDMProps, edmprops_field, val)
        except (OverflowError, ValueError):
          pass


def create_object(node):
  """Accepts an edm renderable and returns a blender object"""

  if not isinstance(node, (RenderNode, ShellNode, NumberNode, SkinNode)):
    print("WARNING: Do not understand creating types {} yet".format(type(node)))
    return

  if isinstance(node, (RenderNode, NumberNode, SkinNode)):
    vertexFormat = node.material.vertex_format
  elif isinstance(node, ShellNode):
    vertexFormat = node.vertex_format

  # Skip empty split children (owner-encoded splits with no triangles)
  if not node.indexData:
    return None

  # Create the mesh. Skinned meshes must NOT use compaction as their vertex
  # weights rely on the original EDM vertex indices.
  compact = not isinstance(node, SkinNode)
  mesh = _create_mesh(node.vertexData, node.indexData, vertexFormat, compact=compact)
  mesh.name = node.name

  # Owner-channel extraction needs index remap from source vertexData to mesh points.
  # Keep import stable for now; recover owner metadata in a dedicated mesh-mapping pass.

  # Create and link the object
  ob = bpy.data.objects.new(node.name, mesh)
  ob.edm.is_collision_shell = isinstance(node, ShellNode)
  ob.edm.is_renderable      = isinstance(node, (RenderNode, NumberNode, SkinNode))
  ob.edm.damage_argument = -1 if not isinstance(node, (RenderNode, NumberNode)) else node.damage_argument
  try:
    source_type = getattr(node, "_source_type_name", None)
    if source_type:
      ob["_iedm_source_node_type"] = str(source_type)
    layout_variant = getattr(node, "_layout_variant", None)
    if layout_variant:
      ob["_iedm_layout_variant"] = str(layout_variant)
    shell_family = getattr(node, "_shell_family", None)
    if shell_family:
      ob["_iedm_shell_family"] = str(shell_family)
  except Exception as e:
    print(f"Warning in blender_importer\\import_pipeline.py: {e}")
  if isinstance(node, ShellNode):
    _set_official_special_type(ob, 'COLLISION_SHELL')
    # Match original collision-shell viewport style from official scenes.
    ob.display_type = 'BOUNDS'
  elif isinstance(node, NumberNode):
    _set_official_special_type(ob, 'NUMBER_TYPE')
    payload = getattr(node, "number_payload", None) or {}
    uv_params = payload.get("uv_params")
    raw = getattr(node, "number_params_raw", None)
    if uv_params and hasattr(ob, "EDMProps"):
      ob.EDMProps.NUMBER_UV_X_ARG = int(uv_params.get("x_arg", -1))
      ob.EDMProps.NUMBER_UV_X_SCALE = float(uv_params.get("x_scale", 0.0))
      ob.EDMProps.NUMBER_UV_Y_ARG = int(uv_params.get("y_arg", -1))
      ob.EDMProps.NUMBER_UV_Y_SCALE = float(uv_params.get("y_scale", 0.0))
    elif raw and hasattr(ob, "EDMProps"):
      # Legacy fallback for files parsed before structured number payloads
      # were attached.
      ob.EDMProps.NUMBER_UV_X_ARG = int(raw[1])
      ob.EDMProps.NUMBER_UV_X_SCALE = float(raw[2])
      ob.EDMProps.NUMBER_UV_Y_ARG = int(raw[3])
      ob.EDMProps.NUMBER_UV_Y_SCALE = float(raw[4])
  else:
    _set_official_special_type(ob, 'UNKNOWN_TYPE')

  # Map damage_argument → EDMProps.DAMAGE_ARG for official exporter.
  if hasattr(ob, "EDMProps") and isinstance(node, (RenderNode, NumberNode)):
    dmg = getattr(node, "damage_argument", -1)
    if dmg is not None and dmg >= 0:
      ob.EDMProps.DAMAGE_ARG = int(dmg)

  # Map animated uniforms from EDM material → EDMProps ARG fields.
  _map_animated_uniforms_to_edmprops(ob, node)

  bpy.context.collection.objects.link(ob)

  return ob

