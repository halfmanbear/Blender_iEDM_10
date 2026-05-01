# Mapping from EDM animated-uniform names to EDMProps field names.
_ANIMATED_UNIFORM_TO_EDMPROPS = {
  "emissiveValue":  "EMISSIVE_ARG",
  "colorShift":     "COLOR_ARG",
  "diffuseShift":   "COLOR_ARG",
  "selfIlluminationValue": "EMISSIVE_ARG",
  "selfIlluminationColor": "EMISSIVE_COLOR_ARG",
  "emissiveShift":  "EMISSIVE_COLOR_ARG",
  "aoValue":        "AO_ARG",
  "lightMapValue":  "AO_ARG",
  "lightMapShift":  "AO_ARG",
  "opacityValue":   "OPACITY_VALUE_ARG",
}


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


def _ensure_placeholder_texture_image(name):
  """Create a tiny generated image so the official exporter can still recover
  the EDM texture name from the image datablock when the source file is absent.
  """
  image_name = str(name or '__EMPTY__')
  existing = bpy.data.images.get(image_name)
  if existing is not None:
    return existing
  image = bpy.data.images.new(name=image_name, width=1, height=1, alpha=True)
  try:
    image.generated_color = (1.0, 1.0, 1.0, 1.0)
  except Exception:
    pass
  return image


def _wire_uv_transform(nodes, links, tex_def, tex_image):
  """Insert a TexCoord + Mapping node chain when the texture has a non-identity UV matrix."""
  matrix = getattr(tex_def, 'matrix', None)
  if matrix is None:
    return
  try:
    is_id = all(
      abs(float(matrix[r][c]) - (1.0 if r == c else 0.0)) < 1e-5
      for r in range(4) for c in range(4)
    )
    if is_id:
      return
  except Exception:
    return
  try:
    loc, rot, scale = matrix.decompose()
  except Exception as e:
    print(f"Warning: Could not decompose UV matrix for '{getattr(tex_def, 'name', '')}': {e}")
    return
  x = tex_image.location[0]
  y = tex_image.location[1]
  mapping = nodes.new('ShaderNodeMapping')
  mapping.location = (x - 200, y)
  mapping.vector_type = 'POINT'
  try:
    mapping.inputs['Location'].default_value = (loc.x, loc.y, loc.z)
    mapping.inputs['Rotation'].default_value = rot.to_euler('XYZ')
    mapping.inputs['Scale'].default_value = (scale.x, scale.y, scale.z)
  except Exception as e:
    print(f"Warning: Could not set mapping values: {e}")
  tex_coord = nodes.new('ShaderNodeTexCoord')
  tex_coord.location = (x - 400, y)
  links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
  links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])


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

    # Always create a texture node for preserved EDM slots. The official
    # exporter reads texture names from connected image nodes; if the source
    # file is unavailable locally, keep a generated placeholder image so the
    # texture binding name still round-trips.
    tex_image = nodes.new('ShaderNodeTexImage')
    if filename:
      tex_image.image = bpy.data.images.load(filename, check_existing=True)
    else:
      tex_image.image = _ensure_placeholder_texture_image(tex_def.name)
    tex_image.label = str(getattr(tex_def, 'name', '') or '')
    tex_image.location = (-400, tex_def.index * -300)
    texture_nodes[tex_def.index] = tex_image
    _wire_uv_transform(nodes, links, tex_def, tex_image)

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
  _metallic_materials = {'chrome_material', 'aluminium_material'}
  if (material.material_name or '').lower() in _metallic_materials:
    principled_bsdf.inputs['Metallic'].default_value = 1.0
    # reflectionBlurring: 0.0 = mirror-sharp, 1.0 = fully diffuse.
    # chrome uses ~0.01 (near-perfect mirror); aluminium uses ~0.8 (brushed).
    reflection_blurring = material.uniforms.get("reflectionBlurring", None)
    if reflection_blurring is not None:
      try:
        principled_bsdf.inputs['Roughness'].default_value = max(0.0, min(1.0, float(reflection_blurring)))
      except Exception:
        pass
  else:
    principled_bsdf.inputs['Metallic'].default_value = 0.0

  # Roughness from specPower (non-metallic materials only)
  specPower = material.uniforms.get("specPower", None)
  if specPower is not None and (material.material_name or '').lower() not in _metallic_materials:
    # This is a guess, might need tweaking. Assuming specPower is in a range of 0-1024 (common for older shaders)
    # Higher specPower means a smaller, more intense highlight, which means lower roughness.
    roughness = (1.0 - (specPower / 1024.0))**0.5 # Using sqrt for a more perceptually linear mapping
    principled_bsdf.inputs['Roughness'].default_value = max(0.0, min(1.0, roughness))
  elif (material.material_name or '').lower() not in _metallic_materials:
    principled_bsdf.inputs['Roughness'].default_value = 0.5

  # Specular from specFactor
  specFactor = material.uniforms.get("specFactor", None)
  if specFactor is not None:
    principled_bsdf.inputs['Specular IOR Level'].default_value = specFactor

  # Emissive from selfIlluminationValue/selfIlluminationColor (self-illum materials).
  # These are stored as animated_uniforms when the official exporter writes emissive blocks.
  _self_illum_names = {
    "self_illum_material", "transparent_self_illum_material",
    "additive_self_illum_material", "additive_self_illum_color_material",
    "additive_self_illum_tex_material",
  }
  if (material.material_name or "").lower() in _self_illum_names:
    # Connect diffuse texture to Emission socket so the material visibly glows.
    if 0 in texture_nodes:
      try:
        links.new(texture_nodes[0].outputs['Color'], principled_bsdf.inputs['Emission Color'])
      except Exception:
        pass
    # selfIlluminationValue drives emission strength.
    siv = material.uniforms.get("selfIlluminationValue", None)
    if siv is None:
      # Also check animated_uniforms (stored as key list when preserve_animated=False)
      anim_siv = material.animated_uniforms.get("selfIlluminationValue", None)
      if anim_siv and hasattr(anim_siv, "__iter__"):
        try:
          siv = float(next(iter(anim_siv)).value)
        except Exception:
          pass
    try:
      principled_bsdf.inputs['Emission Strength'].default_value = float(siv) if siv is not None else 1.0
    except Exception:
      pass

  # --- Handle Blending ---
  if material.blending in (1, 2): # Alpha blending or Alpha test
    # Connect diffuse texture alpha to Principled BSDF Alpha input
    if 0 in texture_nodes:
      links.new(texture_nodes[0].outputs['Alpha'], principled_bsdf.inputs['Alpha'])
    mat.surface_render_method = 'DITHERED'
  elif material.blending == 3: # SUM_BLENDING (additive) — used by self-illumination materials
    # Additive blending: drive Alpha by texture alpha if present; use BLENDED mode.
    if 0 in texture_nodes:
      links.new(texture_nodes[0].outputs['Alpha'], principled_bsdf.inputs['Alpha'])
    mat.surface_render_method = 'BLENDED'

  # --- Handle Culling ---
  # EDM culling=0 → standard backface culling enabled; non-zero → two-sided.
  try:
    mat.use_backface_culling = (getattr(material, "culling", 0) == 0)
  except Exception:
    pass

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
      print(f"Warning in blender_importer/material_setup.py: {e}")

  # Set other material properties
  try:
    mat.edm_material = material.material_name
  except TypeError:
    print(f"Warning: Unknown material type '{material.material_name}', defaulting to 'def_material'")
    mat.edm_material = "def_material"
  mat.edm_blending = str(material.blending)
  _preserve_material_payload(mat, material)

  # Create node-tree animation keyframes so the exporter can read them back.
  _create_material_socket_animations(mat, material)
  _create_material_uv_animations(mat, material, texture_nodes)

  return mat


# Animated-uniform name → EDM shader node group input socket name.
# These socket names match NodeSocketInDefaultEnum in the official exporter.
_ANIM_UNIFORM_TO_SOCKET = {
  "emissiveValue":         "Emissive Value",
  "selfIlluminationValue": "Emissive Value",
  "emissiveShift":         "Emissive",
  "selfIlluminationColor": "Emissive",
  "colorShift":            "Base Color",
  "diffuseShift":          "Base Color",
  "aoValue":               "AO Value*",
  "lightMapValue":         "AO Value*",
  "lightMapShift":         "AO Value*",
  "opacityValue":          "Opacity Value",
}

# Animated-uniform name substrings that signal UV-shift animation.
_UV_ANIM_KEYWORDS = ("uv", "UV", "scroll", "Scroll", "move", "Move", "shift", "Shift")

_HEURISTIC_ANIM_UNIFORM_TARGETS = {
  "emissive_value": {"socket": "Emissive Value", "edmprop": "EMISSIVE_ARG"},
  "emissive_color": {"socket": "Emissive", "edmprop": "EMISSIVE_COLOR_ARG"},
  "color": {"socket": "Base Color", "edmprop": "COLOR_ARG"},
  "ao": {"socket": "AO Value*", "edmprop": "AO_ARG"},
  "opacity": {"socket": "Opacity Value", "edmprop": "OPACITY_VALUE_ARG"},
  "uv": {"socket": None, "edmprop": None},
}


def _normalized_uniform_name(name):
  raw = str(name or "").strip().lower()
  if not raw:
    return ""
  return "".join(ch for ch in raw if ("a" <= ch <= "z") or ("0" <= ch <= "9"))


def _heuristic_uniform_category(name):
  norm = _normalized_uniform_name(name)
  if not norm:
    return None

  if any(token in norm for token in ("uv", "texcoord", "texcoords", "scroll", "offset", "pan", "translate")):
    return "uv"
  if any(token in norm for token in ("opacity", "alpha", "transparen")):
    return "opacity"
  if any(token in norm for token in ("lightmap", "ambientocclusion", "ao")):
    return "ao"
  if any(token in norm for token in ("emissive", "selfillum", "selfillumination", "glow", "incand")):
    if any(token in norm for token in ("color", "colour", "tint", "shift")):
      return "emissive_color"
    return "emissive_value"
  if any(token in norm for token in ("diffuse", "albedo", "basecolor", "basecolour", "color", "colour", "tint")):
    return "color"
  return None


def _resolve_animated_uniform_target(uniform_name):
  socket_name = _ANIM_UNIFORM_TO_SOCKET.get(uniform_name)
  edmprops_field = _ANIMATED_UNIFORM_TO_EDMPROPS.get(uniform_name)
  if socket_name is not None or edmprops_field is not None:
    return {
      "mode": "exact",
      "socket": socket_name,
      "edmprop": edmprops_field,
      "category": None,
    }

  category = _heuristic_uniform_category(uniform_name)
  if category is None:
    return None
  target = _HEURISTIC_ANIM_UNIFORM_TARGETS.get(category)
  if target is None:
    return None
  return {
    "mode": "heuristic",
    "socket": target.get("socket"),
    "edmprop": target.get("edmprop"),
    "category": category,
  }


def _looks_like_uv_anim_uniform(uniform_name):
  target = _resolve_animated_uniform_target(uniform_name)
  if target is not None:
    return target.get("category") == "uv"
  return any(kw in str(uniform_name or "") for kw in _UV_ANIM_KEYWORDS)


def _mat_ensure_node_tree_action(mat):
  if mat.node_tree.animation_data is None:
    mat.node_tree.animation_data_create()
  if mat.node_tree.animation_data.action is None:
    mat.node_tree.animation_data.action = bpy.data.actions.new(mat.name + "_mat_anim")
  return mat.node_tree.animation_data.action


def _mat_set_linear_on_path(action, anim_path):
  for fc in action.fcurves:
    if anim_path in fc.data_path:
      for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'


def _create_material_socket_animations(mat, material):
  """Create node-tree animation keyframes for animated material uniforms.

  The exporter checks bpy_material.node_tree.animation_data for animation on
  specific socket paths (via path_from_id). Without these keyframes the *_ARG
  int indices are set but the exporter treats every property as static.
  """
  if not (mat and mat.use_nodes and mat.node_tree):
    return
  anim_uniforms = getattr(material, "animated_uniforms", None)
  if not anim_uniforms:
    return

  try:
    from ..edm_format.typereader import AnimatedProperty
  except ImportError:
    return

  group_node = next((n for n in mat.node_tree.nodes if n.type == 'GROUP'), None)
  if group_node is None:
    return

  for uniform_name, prop in anim_uniforms.items():
    target = _resolve_animated_uniform_target(uniform_name)
    socket_name = target.get("socket") if target else None
    if socket_name is None:
      continue
    if not isinstance(prop, AnimatedProperty):
      continue
    keys = list(getattr(prop, "keys", []) or [])
    if not keys:
      continue

    socket = next((inp for inp in group_node.inputs if inp.name == socket_name), None)
    if socket is None:
      continue

    anim_path = socket.path_from_id('default_value')
    action = _mat_ensure_node_tree_action(mat)

    for framedata in keys:
      try:
        edm_time = float(getattr(framedata, "frame", 0.0))
        blender_frame = (edm_time + 1.0) * 100.0
        value = getattr(framedata, "value", None)
        if value is None:
          continue
        stype = socket.type
        if stype == 'VALUE':
          socket.default_value = float(value)
          mat.node_tree.keyframe_insert(data_path=anim_path, frame=blender_frame)
        elif stype in ('RGBA', 'VECTOR'):
          val_seq = tuple(float(v) for v in value)
          if stype == 'RGBA' and len(val_seq) == 3:
            val_seq = val_seq + (1.0,)
          socket.default_value = val_seq
          mat.node_tree.keyframe_insert(data_path=anim_path, frame=blender_frame)
      except Exception as e:
        print(f"Warning in material_setup._create_material_socket_animations: {e}")

    _mat_set_linear_on_path(action, anim_path)


def _create_material_uv_animations(mat, material, texture_nodes):
  """Label Mapping nodes with arg numbers and animate their Location inputs.

  The exporter finds UV-scroll animation by reading extract_arg_number(mapping_node.label)
  and checking bpy_material.node_tree.animation_data on the Location input path.
  Without the label and keyframes, UV scroll is exported as a static offset.
  """
  if not (mat and mat.use_nodes and mat.node_tree):
    return
  anim_uniforms = getattr(material, "animated_uniforms", None)
  if not anim_uniforms:
    return

  try:
    from ..edm_format.typereader import AnimatedProperty
  except ImportError:
    return

  nodes = mat.node_tree.nodes
  links = mat.node_tree.links

  for uniform_name, prop in anim_uniforms.items():
    target = _resolve_animated_uniform_target(uniform_name)
    if target and target.get("socket") is not None:
      continue  # handled by _create_material_socket_animations
    if not _looks_like_uv_anim_uniform(uniform_name):
      continue
    if not isinstance(prop, AnimatedProperty):
      continue
    keys = list(getattr(prop, "keys", []) or [])
    if not keys:
      continue
    arg = getattr(prop, "argument", None)
    if arg is None or int(arg) < 0:
      continue

    # Find an existing Mapping node; create one on the albedo texture if absent.
    mapping_node = next((n for n in nodes if n.type == 'MAPPING'), None)
    if mapping_node is None:
      tex_node = texture_nodes.get(0)
      if tex_node is None:
        continue
      mapping_node = nodes.new('ShaderNodeMapping')
      mapping_node.location = (tex_node.location[0] - 220, tex_node.location[1])
      mapping_node.vector_type = 'POINT'
      tex_coord = nodes.new('ShaderNodeTexCoord')
      tex_coord.location = (mapping_node.location[0] - 220, mapping_node.location[1])
      links.new(tex_coord.outputs['UV'], mapping_node.inputs['Vector'])
      links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])

    # Label encodes the arg number: exporter calls extract_arg_number(node.label).
    mapping_node.label = str(int(arg))

    loc_input = mapping_node.inputs.get('Location') or mapping_node.inputs[1]
    anim_path = loc_input.path_from_id('default_value')
    action = _mat_ensure_node_tree_action(mat)

    for framedata in keys:
      try:
        edm_time = float(getattr(framedata, "frame", 0.0))
        blender_frame = (edm_time + 1.0) * 100.0
        value = getattr(framedata, "value", None)
        if value is None:
          continue
        val_seq = tuple(float(v) for v in value)
        if len(val_seq) == 2:
          val_seq = val_seq + (0.0,)
        elif len(val_seq) > 3:
          val_seq = val_seq[:3]
        loc_input.default_value = val_seq
        mat.node_tree.keyframe_insert(data_path=anim_path, frame=blender_frame)
      except Exception as e:
        print(f"Warning in material_setup._create_material_uv_animations: {e}")

    _mat_set_linear_on_path(action, anim_path)


def _material_prop_scalar(value):
  if isinstance(value, (bool, int, float, str)):
    return value
  if isinstance(value, (bytes, bytearray)):
    return bytes(value).hex()
  return None


def _material_prop_value(value):
  scalar = _material_prop_scalar(value)
  if scalar is not None:
    return scalar
  if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray, dict)):
    try:
      return [_material_prop_value(item) for item in value]
    except Exception:
      return repr(value)
  return repr(value)


def _material_texture_payload(material):
  payload = []
  for tex in getattr(material, "textures", None) or []:
    entry = {
      "index": int(getattr(tex, "index", -1)),
      "name": str(getattr(tex, "name", "") or ""),
    }
    matrix = getattr(tex, "matrix", None)
    if matrix is not None:
      try:
        entry["matrix"] = [[float(v) for v in row] for row in matrix]
      except Exception:
        pass
    payload.append(entry)
  return payload


def _infer_material_texture_roles(material):
  textures = getattr(material, "textures", None) or []
  roles = {}
  material_name = str(getattr(material, "material_name", "") or "").lower()
  slot_names = {int(getattr(tex, "index", -1)): str(getattr(tex, "name", "") or "") for tex in textures}

  def _add(slot, role):
    if slot not in slot_names:
      return
    roles[role] = {"slot": int(slot), "name": slot_names[slot]}

  _add(0, "albedo")
  _add(1, "normal")
  _add(3, "decal")
  _add(8, "emissive")
  _add(9, "lightmap")
  if 10 in slot_names:
    _add(10, "damage_normal")
  elif 1 in slot_names:
    _add(1, "normal")
  if 13 in slot_names:
    _add(13, "roughmet")
  elif 2 in slot_names:
    _add(2, "roughmet")
  _add(14, "glass_filter")
  _add(18, "damage_mask")

  if material_name == "deck_material":
    _add(4, "decal_roughmet")
    _add(5, "wetmap")
    _add(7, "damage_albedo")
    if 9 in slot_names:
      roles.setdefault("damage_mask", {"slot": 9, "name": slot_names[9]})
  elif material_name == "glass_material":
    _add(5, "damage_albedo")
  else:
    _add(5, "damage_albedo")

  for slot, name in slot_names.items():
    lowered = name.lower()
    if "flir" in lowered and "flir" not in roles:
      roles["flir"] = {"slot": int(slot), "name": name}

  return roles


def _material_uniform_payload(props):
  return {str(key): _material_prop_value(value) for key, value in (props or {}).items()}


def _material_animated_uniform_payload(props):
  payload = {}
  for key, value in (props or {}).items():
    entry = {}
    arg = getattr(value, "argument", None)
    if arg is not None:
      try:
        entry["argument"] = int(arg)
      except Exception:
        entry["argument"] = _material_prop_value(arg)
    keys = []
    for framedata in getattr(value, "keys", []) or []:
      keys.append({
        "frame": _material_prop_value(getattr(framedata, "frame", None)),
        "value": _material_prop_value(getattr(framedata, "value", None)),
      })
    if keys:
      entry["keys"] = keys
    payload[str(key)] = entry or _material_prop_value(value)
  return payload


def _preserve_material_payload(mat, material):
  if mat is None or material is None:
    return
  try:
    mat["_iedm_material_name_raw"] = str(getattr(material, "material_name", "") or "")
    mat["_iedm_material_label_raw"] = str(getattr(material, "name", "") or "")
    mat["_iedm_texture_channels"] = json.dumps(
      [int(v) for v in (getattr(material, "texture_coordinates_channels", None) or [])],
      separators=(",", ":"),
    )
    mat["_iedm_texture_slots"] = json.dumps(_material_texture_payload(material), separators=(",", ":"))
    mat["_iedm_texture_roles"] = json.dumps(_infer_material_texture_roles(material), separators=(",", ":"), sort_keys=True)
    mat["_iedm_uniforms"] = json.dumps(
      _material_uniform_payload(getattr(material, "uniforms", None) or {}),
      separators=(",", ":"),
      sort_keys=True,
    )
    mat["_iedm_animated_uniforms"] = json.dumps(
      _material_animated_uniform_payload(getattr(material, "animated_uniforms", None) or {}),
      separators=(",", ":"),
      sort_keys=True,
    )
  except Exception as e:
    print(f"Warning in blender_importer/material_setup.py: {e}")


def _preserve_object_material_args(ob, node):
  mat = getattr(node, "material", None)
  anim_uniforms = getattr(mat, "animated_uniforms", None) or {}
  unknown_args = {}
  translated_args = {}
  for name, prop in anim_uniforms.items():
    arg = getattr(prop, "argument", None)
    if arg is None:
      continue
    target = _resolve_animated_uniform_target(name)
    if target is not None:
      translated_args[str(name)] = {
        "mode": str(target.get("mode", "")),
        "category": str(target.get("category", "") or ""),
        "socket": str(target.get("socket", "") or ""),
        "edmprop": str(target.get("edmprop", "") or ""),
      }
      if target.get("mode") == "exact":
        continue
    try:
      unknown_args[str(name)] = int(arg)
    except Exception:
      unknown_args[str(name)] = _material_prop_value(arg)
  if translated_args:
    try:
      ob["_iedm_anim_uniform_translation"] = json.dumps(translated_args, separators=(",", ":"), sort_keys=True)
    except Exception as e:
      print(f"Warning in blender_importer/material_setup.py: {e}")
  if unknown_args:
    try:
      ob["_iedm_anim_uniform_args"] = json.dumps(unknown_args, separators=(",", ":"), sort_keys=True)
    except Exception as e:
      print(f"Warning in blender_importer/material_setup.py: {e}")


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
    target = _resolve_animated_uniform_target(name)
    edmprops_field = target.get("edmprop") if target else None
    if edmprops_field and hasattr(ob.EDMProps, edmprops_field):
      if prop.argument is not None and prop.argument >= 0:
        try:
          # Clamp to 32-bit signed int limit for Blender's IntProperty
          val = min(2147483647, int(prop.argument))
          setattr(ob.EDMProps, edmprops_field, val)
        except (OverflowError, ValueError):
          pass
  _preserve_object_material_args(ob, node)
