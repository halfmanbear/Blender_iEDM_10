"""Registered static and animated fake-spot light node readers."""

import struct as struct

from .core import (
    _V10_CATEGORY_KEYS,
    BaseNode,
    NodeCategory,
    _next_v10_token_looks_like_type,
    _scan_to_next_v10_type_token,
    logger,
    reads_type,
)
from .light_parsing import (
    _read_animated_fake_lights_payload,
    _resolve_material_from_candidates,
)


@reads_type("model::FakeSpotLightsNode")
class FakeSpotLightsNode(BaseNode):
    category = NodeCategory.render

    @classmethod
    def read(cls, stream):
        self = super(FakeSpotLightsNode, cls).read(stream)

        # Seems to start relatively similar to renderNode
        self.unknown_start = stream.read_uint()
        self.material_ish = stream.read_uint()

        # We have parent-like blocks of two uints + three floats
        controlNodeCount = stream.read_count("control node count")

        self.parentData = []
        self.parentData_float_offsets = []
        for _ in range(controlNodeCount):
            _u0 = stream.read_uint()
            _u1 = stream.read_uint()
            _vec_pos = stream.tell()
            _vec = stream.read_floats(3)
            self.parentData_float_offsets.append(_vec_pos)
            self.parentData.append([_u0, _u1, _vec])
        # Control node seems to follow same rules as RenderNode
        if controlNodeCount:
            stream.mark_type_read("model::FSLNControlNode", controlNodeCount - 1)

        dataCount = stream.read_count("fake spot light data count")
        self.raw_data = [stream.read(65) for _ in range(dataCount)]
        stream.mark_type_read("model::FakeSpotLight", dataCount)

        # model::FakeSpotLight::load (UniModelDesc 0x180001270): position vec3d,
        # front uv (lb, rt) vec2f x2, size float, face arg uint32, back uv
        # (lb, rt) vec2f x2, back-side flag byte. Directions are not stored
        # per light (control links / trailing v3 payload carry them).
        self.data = []
        for raw in self.raw_data:
            self.data.append(
                {
                    "position": struct.unpack_from("<3d", raw, 0),
                    "uv_lb": struct.unpack_from("<2f", raw, 24),
                    "uv_rt": struct.unpack_from("<2f", raw, 32),
                    "size": struct.unpack_from("<f", raw, 40)[0],
                    "face_arg": struct.unpack_from("<I", raw, 44)[0],
                    "back_uv_lb": struct.unpack_from("<2f", raw, 48),
                    "back_uv_rt": struct.unpack_from("<2f", raw, 56),
                    "flag": raw[64],
                }
            )

        # Some v10 FakeSpotLightsNode records carry an extra trailing direction
        # vector (3 floats) after the light entries. If we leave it unread the
        # parser desynchronizes and the next v10 string-table lookup sees 1.0
        # (0x3f800000) as a bogus string index.
        self.trailing_direction = None
        self.trailing_direction_offset = None
        self.trailing_blob = b""
        self.trailing_blob_offset = None
        # AnimatedFakeSpotLightsNode bytes immediately following the light entries
        # are the animation payload, not trailing blob.  The subclass sets this flag
        # so we skip trailing detection and leave the stream positioned correctly for
        # _read_animated_fake_lights_payload.
        if getattr(stream, "v10", False) and not getattr(cls, "_skip_trailing", False):
            pos = stream.tell()
            try:
                next_u = stream.read_uint()
            except Exception:
                next_u = None
            finally:
                stream.seek(pos)

            if next_u is not None and getattr(stream, "strings", None):
                looks_like_next_type = (
                    0 <= next_u < len(stream.strings)
                    and isinstance(stream.strings[next_u], str)
                    and stream.strings[next_u].startswith("model::")
                )
                if not looks_like_next_type:
                    tpos = stream.tell()
                    try:
                        trailing = stream.read_floats(3)
                        next_after = stream.read_uint()
                        looks_aligned_after = (
                            0 <= next_after < len(stream.strings)
                            and isinstance(stream.strings[next_after], str)
                            and stream.strings[next_after].startswith("model::")
                        )
                        if looks_aligned_after:
                            self.trailing_direction = trailing
                            self.trailing_direction_offset = tpos
                            # Keep the next token unread; this was only a look-ahead
                            # probe.
                            stream.seek(tpos + 12)
                        else:
                            stream.seek(tpos)
                    except Exception:
                        stream.seek(tpos)

            preserve_from = stream.tell()
            skip = _scan_to_next_v10_type_token(
                stream, validate_node_header=True, stop_tokens=_V10_CATEGORY_KEYS
            )
            stream.seek(preserve_from)
            if skip is not None and skip > 0:
                self.trailing_blob_offset = preserve_from
                self.trailing_blob = stream.read(skip)

        return self

    def prepare(self, nodes, materials):
        self.material = _resolve_material_from_candidates(
            materials,
            [getattr(self, "material_ish", -1)],
            "fake_spot",
        )
        # v10 fake-spot records encode their control node in the first uint of the
        # first parentData entry. Recover the transform parent so build_graph() can
        # place the fake spot back under its original visibility/control wrapper
        # (same pattern as FakeOmniLightsNode).
        parent_data = getattr(self, "parentData", None)
        if parent_data and getattr(self, "parent", None) is None:
            try:
                raw = parent_data[0][0]
                if isinstance(raw, int) and 0 <= raw < len(nodes):
                    self.set_parent(nodes[raw])
                    self.control_node = nodes[raw]
                    self.control_node_index = raw
            except Exception:
                logger.debug("Ignoring optional operation failure", exc_info=True)


@reads_type("model::AnimatedFakeSpotLightsNode")
class AnimatedFakeSpotLightsNode(FakeSpotLightsNode):
    _skip_trailing = (
        True  # animation payload follows immediately; skip trailing-blob detection
    )

    @classmethod
    def read(cls, stream):
        self = super(AnimatedFakeSpotLightsNode, cls).read(stream)
        return _read_animated_fake_lights_payload(
            stream, self, "AnimatedFakeSpotLightsNode"
        )


@reads_type("model::FakeSpotLights3Node")
class FakeSpotLights3Node(FakeSpotLightsNode):
    @classmethod
    def read(cls, stream):
        self = super(FakeSpotLights3Node, cls).read(stream)
        self.v3_directions = None
        self.v3_backside_flags = None
        self.v3_trailing_blob = b""
        self.v3_trailing_blob_offset = None

        # Most files appear layout-compatible with FakeSpotLightsNode.
        if _next_v10_token_looks_like_type(stream) is True:
            return self

        # Some variants carry an additional directions array (+ optional flags).
        pos = stream.tell()
        try:
            count = stream.read_uint()
            if 0 <= count <= 65535:
                dirs = [tuple(stream.read_floats(3)) for _ in range(count)]

                flags = None
                if _next_v10_token_looks_like_type(stream) is not True:
                    fpos = stream.tell()
                    try:
                        flags = list(stream.read_uchars(count))
                    except Exception:
                        flags = None
                    if _next_v10_token_looks_like_type(stream) is not True:
                        stream.seek(fpos)
                        flags = None

                if _next_v10_token_looks_like_type(stream) is True and count == len(
                    getattr(self, "data", [])
                ):
                    self.v3_directions = dirs
                    self.v3_backside_flags = flags
                    for i, d in enumerate(dirs):
                        self.data[i]["direction_v3"] = d
                        # Prefer explicit v3 payload direction when available.
                        self.data[i]["direction"] = d
                    if flags is not None:
                        for i, flag in enumerate(flags):
                            self.data[i]["back_side"] = bool(flag)
                    return self
        except Exception:
            logger.debug("Ignoring optional operation failure", exc_info=True)

        # Restore and preserve any unknown trailing payload until the next model token.
        stream.seek(pos)
        skip = _scan_to_next_v10_type_token(stream, stop_tokens=_V10_CATEGORY_KEYS)
        if skip is not None and skip > 0:
            self.v3_trailing_blob_offset = stream.tell()
            self.v3_trailing_blob = stream.read(skip)
            logger.warning(
                "FakeSpotLights3Node '%s': preserved %d trailing bytes",
                getattr(self, "name", ""),
                len(self.v3_trailing_blob),
            )
        return self
