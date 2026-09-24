"""Read and write vertex, index, and parent payloads for render nodes."""

import itertools as itertools
from collections import Counter

from ..basereader import MAX_REASONABLE_COUNT, EDMFormatError


def _tag_shell_family_node(node, source_type):
    if node is None:
        return node
    node._source_type_name = source_type
    if source_type in {"model::ShellSkinNode", "model::TreeShellNode"}:
        node._shell_family = source_type.replace("model::", "").replace("Node", "")
    return node


def _read_index_data(stream, classification=None):
    "Performs the common index-reading operation"
    dtPos = stream.tell()
    dataType = stream.read_uchar()
    entries = stream.read_uint()
    unknown = stream.read_uint()

    if dataType == 0:
        data = stream.read_uchars(entries)
        _bytes = entries
    elif dataType == 1:
        data = stream.read_ushorts(entries)
        _bytes = entries * 2
    elif dataType == 2:
        data = stream.read_uints(entries)
        _bytes = entries * 4
    else:
        raise IOError(
            "Don't know how to read index data type {} @ {}".format(
                int(dataType), dtPos
            )
        )

    if classification:
        stream.mark_type_read(classification, _bytes)

    return (unknown, data)


def _write_index_data(indexData, vertexDataLength, writer):
    # Index data
    if vertexDataLength < 256:
        writer.write_uchar(0)
        iWriter = writer.write_uchars
    elif vertexDataLength < 2**16:
        writer.write_uchar(1)
        iWriter = writer.write_ushorts
    elif vertexDataLength < 2**32:
        writer.write_uchar(2)
        iWriter = writer.write_uints
    else:
        raise IOError(
            "Do not know how to write index arrays with {} members".format(
                vertexDataLength
            )
        )

    writer.write_uint(len(indexData))
    writer.write_uint(5)
    iWriter(indexData)


def _read_vertex_data(stream, classification=None):
    count = stream.read_count("vertex count")
    stride = stream.read_count("vertex stride")
    total = count * stride
    if total > MAX_REASONABLE_COUNT:
        raise EDMFormatError(
            "Implausible vertex data size {} (count={}, stride={})".format(
                total, count, stride
            )
        )
    vtxData = stream.read_floats(total)

    # If given a classification, mark it off
    if classification:
        stream.mark_type_read(classification, count * stride * 4)

    # Group the vertex data according to stride
    vtxData = [vtxData[i : i + stride] for i in range(0, len(vtxData), stride)]
    return vtxData


def _write_vertex_data(data, writer):
    writer.write_uint(len(data))
    writer.write_uint(len(data[0][0]))  # stride = floats per vertex
    flat_data = list(itertools.chain(*data))
    writer.write_floats(flat_data)


def _read_parent_data(stream):
    # Read the parent section
    parentCount = stream.read_count("parent count")
    stream.mark_type_read("model::RNControlNode", parentCount - 1)

    if parentCount == 1:
        return [[stream.read_uint(), stream.read_int()]]
    else:
        parentData = []
        for _ in range(parentCount):
            node = stream.read_uint()
            ranges = list(stream.read_ints(2))
            parentData.append((node, ranges[0], ranges[1]))
        return parentData


def _owner_triangles(indexData, owner_values):
    """Group whole triangles by owner in one pass.

    Filtering the flat index stream and regrouping in triples would mix
    vertices from different source triangles, so triangles stay intact. A
    mixed-owner triangle (malformed data) goes to its majority owner to avoid
    holes; one with three different owners is dropped.
    """
    count = len(owner_values)
    grouped = {}
    for i in range(0, len(indexData) - 2, 3):
        tri = indexData[i : i + 3]
        if tri[0] >= count or tri[1] >= count or tri[2] >= count:
            continue
        a, b, c = (owner_values[ix] for ix in tri)
        owner = a if a in (b, c) else (b if b == c else None)
        if owner is not None:
            grouped.setdefault(owner, []).extend(tri)
    return grouped


def _classify_render_parent_data(parentData, vertexData, indexData):
    """Best-effort classification of v10 RenderNode parent attachment layouts.

    Observed variants:
    - single-parent: (parent, damageArg)
    - coverage table: (parent, idxTo, damageArg)
    - swapped coverage/damage columns
    - zero-coverage attachment tables where geometry is duplicated per parent
    - owner-encoded split tables where vertex slot 3 identifies the owner
    """
    result = {
        "mode": "single_parent",
        "swap_columns": False,
        "force_fallback": False,
        "owner_values": None,
        "shared_parent": None,
        "coverage_total": len(indexData) if indexData else 0,
    }
    if not parentData:
        result["mode"] = "missing"
        return result
    if len(parentData) == 1:
        return result

    result["mode"] = "coverage_table"
    all_zero_idx_to = all(len(pd) == 3 and pd[1] == 0 for pd in parentData)
    all_damage_neg1 = all(len(pd) == 3 and pd[2] == -1 for pd in parentData)
    owner_values = None
    if all_zero_idx_to and vertexData:
        try:
            raw_owner_values = [int(round(v[3])) for v in vertexData]
            nOwners = len(parentData)
            owner_values = [max(0, min(nOwners - 1, o)) for o in raw_owner_values]
            result["owner_values"] = owner_values
        except Exception:
            owner_values = None

    if (
        all_zero_idx_to
        and owner_values is not None
        and (all_damage_neg1 or len(set(owner_values)) > 1)
    ):
        result["mode"] = "owner_encoded"
        result["shared_parent"] = parentData[0][0]
        return result

    if all_zero_idx_to:
        result["mode"] = "duplicate_geometry"
        return result

    total_indices = len(indexData)
    last_val1 = parentData[-1][1]
    last_val2 = parentData[-1][2]
    if last_val1 != total_indices:
        if last_val2 == total_indices:
            result["swap_columns"] = True
            result["mode"] = "coverage_table_swapped"
        elif last_val1 == 0:
            result["force_fallback"] = True
            result["mode"] = "fallback_first_parent"
        else:
            result["mode"] = "coverage_table_mismatch"
    return result


def _render_audit(self, verts="__gv_bytes", inds="__gi_bytes"):
    c = Counter()
    c[verts] += 4 * len(self.vertexData) * len(self.vertexData[0])
    # c["__gi_bytes"] +=
    if len(self.vertexData) < 256:
        c[inds] += len(self.indexData)
    elif len(self.vertexData) < 2**16:
        c[inds] += len(self.indexData) * 2
    elif len(self.vertexData) < 2**32:
        c[inds] += len(self.indexData) * 4
    else:
        raise IOError(
            "Do not know how to write index arrays with {} members".format(
                len(self.indexData)
            )
        )
    return c
