"""Registered fake-omni and ALS light node readers."""

import struct as struct

from .core import BaseNode, NodeCategory, reads_type
from .light_parsing import (
    _read_animated_fake_lights_payload,
    _read_fake_omni_light,
    _resolve_material_from_candidates,
)
from .light_records import decode_fake_omni_entry


@reads_type("model::FakeOmniLightsNode")
class FakeOmniLightsNode(BaseNode):
    category = NodeCategory.render

    @classmethod
    def read(cls, stream):
        self = super(FakeOmniLightsNode, cls).read(stream)
        # FakeOmniLightsNode starts with the same uint32 + properties-set id prefix as
        # RenderNode/SkinNode before its control-link vector payload.
        self.unknown_start = stream.read_uint()
        self.material_ish = stream.read_uint()
        # uint32 controlLinkCount, then node_index + discarded uint32 per link.
        # Hardcoding 5 uints only worked when controlLinkCount == 2 (1 + 2*2 = 5).
        control_link_count = stream.read_count("control link count")
        self.control_links = []
        for _ in range(control_link_count):
            node_idx = stream.read_uint()
            _discard = stream.read_uint()
            self.control_links.append(node_idx)
        count = stream.read_count("fake omni light count")
        # FakeOmniLight::load layout:
        #   Vec3d position, Vec2f uv0, Vec2f uv1, float size, uint32 face_arg.
        self.data = [_read_fake_omni_light(stream) for _ in range(count)]
        self.decoded_data = [decode_fake_omni_entry(entry) for entry in self.data]
        stream.mark_type_read("model::FakeOmniLight", count)
        return self

    def prepare(self, nodes, materials):
        self.material = _resolve_material_from_candidates(
            materials,
            list(getattr(self, "control_links", ()) or ()),
            "fake_omni",
        )
        # Use the last control link's node index as the transform parent — matches
        # the old data_start[3] behaviour (link1 for the common 2-link case) and
        # generalises cleanly to any link count.
        control_links = list(getattr(self, "control_links", ()) or ())
        if control_links and getattr(self, "parent", None) is None:
            control_idx = control_links[-1]
            if isinstance(control_idx, int) and 0 <= control_idx < len(nodes):
                try:
                    self.set_parent(nodes[control_idx])
                    self.control_node = nodes[control_idx]
                    self.control_node_index = control_idx
                except Exception:
                    self.control_node = None
                    self.control_node_index = control_idx


@reads_type("model::AnimatedFakeOmniLightsNode")
class AnimatedFakeOmniLightsNode(FakeOmniLightsNode):
    @classmethod
    def read(cls, stream):
        self = super(AnimatedFakeOmniLightsNode, cls).read(stream)
        return _read_animated_fake_lights_payload(
            stream, self, "AnimatedFakeOmniLightsNode"
        )


@reads_type("model::FakeALSNode")
class FakeALSNode(BaseNode):
    category = NodeCategory.render

    @classmethod
    def read(cls, stream):
        self = super(FakeALSNode, cls).read(stream)
        # batumi.edm 1138915 x 340
        self.als_header = stream.read_uints(3)
        count = stream.read_count("fake ALS light count")
        self.raw_data = [stream.read(80) for _ in range(count)]
        stream.mark_type_read("model::FakeALSLight", count)

        # Parse raw 80-byte entries: 10 doubles
        # Layout: pos(3) + remaining(7) — positions are first 3 doubles
        self.data = []
        for raw in self.raw_data:
            doubles = struct.unpack_from("<10d", raw, 0)
            self.data.append(
                {
                    "position": doubles[0:3],
                    "extra": doubles[3:10],
                }
            )

        return self

    def prepare(self, nodes, materials):
        self.material = _resolve_material_from_candidates(
            materials,
            list(getattr(self, "als_header", ()) or ()),
            "fake_als",
        )
