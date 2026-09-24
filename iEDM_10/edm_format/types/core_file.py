import itertools
import logging
from collections import Counter

from ..basereader import decode_edm_string
from ..probe import require_supported_import_format
from .core_nodes import RootNode
from .core_support import (
    TrackingReader,
    _all_IndexA,
    _all_IndexB,
    _read_index,
    _read_main_object_dictionary,
    _write_index,
)

logger = logging.getLogger(__name__)


class EDMFile(object):
    def __init__(self, filename=None, version=8):
        if filename:
            require_supported_import_format(filename)
            reader = TrackingReader(filename)
            try:
                self._read(reader)
            except Exception:
                err_pos = None
                try:
                    err_pos = reader.tell()
                except Exception:
                    err_pos = "unknown"
                print("ERROR at {}".format(err_pos))
                reader.close()
                raise
        else:
            self.version = version
            self.indexA = {}
            self.indexB = {}
            self.root = None
            self.nodes = []
            self.connectors = []
            self.renderNodes = []
            self.lightNodes = []
            self.shellNodes = []

    def _read(self, reader):
        # Deferred: render_shell/number import from core, so these can only
        # be imported here, not at module load time.
        from .number import NumberNode
        from .render_shell import RenderNode

        reader.read_constant(b"EDM")
        self.version = reader.read_ushort()
        if self.version != 10:
            raise IOError(
                "Unsupported EDM version {}; only v10 is supported".format(self.version)
            )
        logger.info("Reading EDM version 10 file")
        reader.version = self.version

        if reader.v10:
            stringsize = reader.read_uint()
            sdata = reader.read(stringsize)
            # Split by null byte, keeping empty strings (they are valid table entries)
            parts = sdata.split(b"\x00")
            if parts and not parts[-1]:
                parts = parts[:-1]  # strip trailing empty from final null terminator
            reader.strings = [decode_edm_string(x) for x in parts]
        else:
            reader.strings = None

        # Read the two indexes
        self.indexA = _read_index(reader)
        self.indexB = _read_index(reader)
        self.root = reader.read_named_type()

        self.nodes = reader.read_list(reader.read_named_type)
        if self.nodes:
            self.transformRoot = self.nodes[0]
        else:
            self.transformRoot = None
            # Handle empty nodes list gracefully if possible, or raise error if critical
            print("Warning: EDM file has no transform nodes.")

        # Read the node parenting data
        for node, parent in zip(
            self.nodes, reader.read_ints(len(self.nodes)), strict=False
        ):
            if parent == -1:
                node.parent = None
                continue
            if parent >= len(self.nodes):
                raise IOError("Invalid node parent data")

            node.set_parent(self.nodes[parent])

        objects = _read_main_object_dictionary(reader)
        self._read_render_nodes(reader, objects, NumberNode, RenderNode)

        # ClassReader10_loadRoot ends cleanly after the sections loop — there is no
        # trailing data written by standard DCS World EDM tooling.
        # Any bytes remaining here come from a non-standard exporter.
        # Consume them so the end-of-file check below doesn't false-alarm, and warn
        # so unexpected payloads surface in logs rather than being silently ignored.
        if reader.v10:
            tail_start = reader.tell()
            tail_bytes = reader.stream.read()
            if tail_bytes:
                logger.warning(
                    "Tail parse: %d unexpected bytes at offset %d "
                    "(not present in standard DCS EDM files; non-standard exporter?)",
                    len(tail_bytes),
                    tail_start,
                )
            reader.seek(tail_start + len(tail_bytes))

        # Verify we are at the end of the file without unconsumed data.
        endPos = reader.tell()
        if len(reader.read(1)) != 0:
            print(
                "Warning: Ended parse at {} but still have data remaining".format(
                    endPos
                )
            )
        reader.close()

        self._link_objects()

        self._validate_indexes(reader)

    def _read_render_nodes(self, reader, objects, number_node_type, render_node_type):
        """Prepare parsed objects, split render chunks and read trailing payloads."""
        logger.debug("EDM object categories: %s", list(objects.keys()))
        self.connectors = objects.get("CONNECTORS", [])
        self.shellNodes = objects.get("SHELL_NODES", [])
        self.lightNodes = objects.get("LIGHT_NODES", [])
        for node_list in objects.values():
            for node in node_list:
                prepare = getattr(node, "prepare", None)
                if callable(prepare):
                    try:
                        prepare(self.nodes, self.root.materials)
                    except Exception as exc:
                        logger.warning(
                            "Prepare failed for %s '%s' (%s: %s)",
                            type(node).__name__,
                            getattr(node, "name", ""),
                            type(exc).__name__,
                            exc,
                        )
        self.renderNodes = []
        for node in objects.get("RENDER_NODES", []):
            if isinstance(node, render_node_type):
                self.renderNodes.extend(node.split())
            else:
                self.renderNodes.append(node)

        if reader.v10:
            self._read_number_node_payloads(reader, number_node_type)

    def _read_number_node_payloads(self, reader, number_node_type):
        """Read trailing v10 mesh payloads for NumberNode placeholders."""
        number_nodes = [
            node
            for node in self.renderNodes
            if isinstance(node, number_node_type)
            and not getattr(node, "_post_payload_read", False)
        ]
        if not number_nodes:
            return
        payload_error = False
        for node in number_nodes:
            position = reader.tell()
            try:
                node.read_v10_payload(reader)
            except Exception as exc:
                payload_error = True
                reader.seek(position)
                logger.warning(
                    "NumberNode post-payload parse failed (%s: %s); "
                    "skipping remaining NumberNode payloads.",
                    type(exc).__name__,
                    exc,
                )
                break
        if payload_error:
            for node in number_nodes:
                if not getattr(node, "_post_payload_read", False):
                    node.parent = None

    def _link_objects(self):
        # Set up parents and other links (e.g. material, bone...)
        for node in itertools.chain(
            self.connectors, self.shellNodes, self.lightNodes, self.renderNodes
        ):
            if hasattr(node, "parent") and node.parent is not None:
                if isinstance(node.parent, int):
                    if 0 <= node.parent < len(self.nodes):
                        node.set_parent(self.nodes[node.parent])
                    else:
                        print(
                            "Warning: Node {} has invalid parent index {}".format(
                                node, node.parent
                            )
                        )
                else:
                    # Already resolved or unexpected type
                    pass
            # Owner-encoded split RenderNodes keep a shared control parent index.
            # Resolve it to the actual transform node so importer code can map
            # mesh-local coordinates from shared parent space.
            if hasattr(node, "shared_parent") and isinstance(node.shared_parent, int):
                if 0 <= node.shared_parent < len(self.nodes):
                    node.shared_parent = self.nodes[node.shared_parent]
            if hasattr(node, "material"):
                if isinstance(node.material, int):
                    if 0 <= node.material < len(self.root.materials):
                        node.material = self.root.materials[node.material]
                    else:
                        print(
                            "Warning: Invalid material index {} for node {}".format(
                                node.material, node
                            )
                        )
                        node.material = None
            if hasattr(node, "bones"):
                if node.bones and isinstance(node.bones[0], int):
                    node.bones = [self.nodes[x] for x in node.bones]
                # If we have bones we have no single 'parent'. Stick it on the root.
                node.set_parent(self.nodes[0])

    def _validate_indexes(self, reader):
        # Validate against the index
        self.selfCount = self.audit()
        rems = Counter(self.indexA)
        rems.subtract(
            Counter({x: c for (x, c) in reader.typecount.items() if x in self.indexA})
        )
        for k in [x for x in rems.keys() if rems[x] == 0]:
            del rems[k]

        if rems:
            print(
                "IndexA items remaining before RENDER_NODES/CONNECTORS: {}".format(rems)
            )
        cB = Counter({x: c for (x, c) in reader.typecount.items() if x in self.indexB})
        remBs = Counter(self.indexB)
        remBs.subtract(cB)
        for k in [x for x in remBs.keys() if remBs[x] == 0]:
            del remBs[k]
        if remBs:
            print(
                "IndexB items remaining before RENDER_NODES/CONNECTORS: {}".format(
                    remBs
                )
            )

    def audit(self):
        _index = Counter()
        _index[RootNode.forTypeName] += 1
        _index += self.root.audit()
        for node in self.nodes:
            _index[node.forTypeName] += 1
            _index += node.audit()
        for rn in itertools.chain(
            self.renderNodes, self.shellNodes, self.lightNodes, self.connectors
        ):
            _index[rn.forTypeName] += 1
            try:
                _index += rn.audit()
            except Exception as exc:
                raise RuntimeError(
                    "Trouble reading audit from {}".format(type(rn))
                ) from exc

        return _index

    def _write_body(self, writer, indexA, indexB):
        """Writes the body of the EDM file (everything after version/string table)"""
        # For v10, write empty indices (they seem to not be used in v10 format)
        if self.version == 10:
            _write_index(writer, {})
            _write_index(writer, {})
        else:
            _write_index(writer, indexA)
            _write_index(writer, indexB)

        # Write the Root node
        writer.write_named_type(self.root)

        # For each parent node, set it's index
        for i, node in enumerate(self.nodes):
            node.index = i

        writer.write_uint(len(self.nodes))
        for node in self.nodes:
            writer.write_named_type(node)

        # Write the parent data for the nodes
        writer.write_int(-1)
        # Everything without a parent has 0 as it's parent
        for node in self.nodes[1:]:
            if node.parent:
                writer.write_uint(node.parent.index)
            else:
                writer.write_uint(0)

        # Now do the render objects dictionary
        objects = {}
        if self.renderNodes:
            objects["RENDER_NODES"] = self.renderNodes
        if self.connectors:
            objects["CONNECTORS"] = self.connectors
        if self.shellNodes:
            objects["SHELL_NODES"] = self.shellNodes
        if self.lightNodes:
            objects["LIGHT_NODES"] = self.lightNodes
        writer.write_uint(len(objects))
        for key, nodes in objects.items():
            writer.write_string(key)
            writer.write_uint(len(nodes))
            for node in nodes:
                writer.write_named_type(node)

    def write(self, writer):
        # Generate the file index with an audit
        _allIndex = self.audit()
        indexA = {k: v for k, v in _allIndex.items() if k in _all_IndexA}
        indexB = {k: v for k, v in _allIndex.items() if k in _all_IndexB}

        # Write EDM header
        writer.write(b"EDM")
        writer.write_ushort(self.version)

        # For v10, do a collection pass to build string table
        if self.version == 10:
            import io

            print("Building v10 string table...")

            # Save the real stream and use a dummy stream for collection
            real_stream = writer.stream
            writer.stream = io.BytesIO()  # Dummy stream - data gets discarded
            writer.string_collect_mode = True
            self._write_body(writer, indexA, indexB)
            writer.string_collect_mode = False
            writer.stream = real_stream  # Restore real stream

            # Write the string table
            string_data = writer.get_string_table_data()
            writer.write_uint(len(string_data))
            writer.write(string_data)
            print(
                "String table: {} unique strings, {} bytes".format(
                    len(writer.string_table), len(string_data)
                )
            )

        # Now write the actual data
        self._write_body(writer, indexA, indexB)
