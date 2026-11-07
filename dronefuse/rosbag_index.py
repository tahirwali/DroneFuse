"""
Read the index of a ROS1 bag (format 2.0) from any byte-range reader, and decode sample
messages, without needing ROS or the whole file.

Bag layout (http://wiki.ros.org/Bags/Format/2.0):
  "#ROSBAG V2.0\\n" | bag-header record (gives index_pos) | chunk, index, chunk, index, ...
  | at index_pos: one connection record per topic connection, then one chunk-info record per chunk.
Every record = header_len (uint32) + header fields ("name=value", each prefixed by uint32 length)
             + data_len (uint32) + data.
The chunk-info records list, for every chunk, its file position, time range and the number of
messages per connection, so message counts and rates for every topic come from the index alone.
"""
import bz2
import struct
from dataclasses import dataclass, field

MAGIC = b"#ROSBAG V2.0\n"
OP_MESSAGE, OP_BAG_HEADER, OP_INDEX, OP_CHUNK, OP_CHUNK_INFO, OP_CONNECTION = 2, 3, 4, 5, 6, 7


def parse_header(buffer):
    fields, i = {}, 0
    while i + 4 <= len(buffer):
        (length,) = struct.unpack_from("<I", buffer, i)
        item = buffer[i + 4:i + 4 + length]
        i += 4 + length
        name, _, value = item.partition(b"=")
        fields[name.decode("latin-1")] = value
    return fields


def iter_records(buffer, start=0):
    """Yield (header fields, data, offset) for every complete record in `buffer`."""
    i = start
    while i + 4 <= len(buffer):
        (header_len,) = struct.unpack_from("<I", buffer, i)
        if i + 8 + header_len > len(buffer):
            return
        header = parse_header(buffer[i + 4:i + 4 + header_len])
        (data_len,) = struct.unpack_from("<I", buffer, i + 4 + header_len)
        data_start = i + 8 + header_len
        if data_start + data_len > len(buffer):
            return
        yield header, buffer[data_start:data_start + data_len], i
        i = data_start + data_len


def _u32(value):
    return struct.unpack("<I", value)[0]


def _u64(value):
    return struct.unpack("<Q", value)[0]


def _time(value):
    seconds, nanoseconds = struct.unpack("<II", value)
    return seconds + nanoseconds * 1e-9


def _op(header):
    return header["op"][0]


@dataclass
class Connection:
    id: int
    topic: str
    msgtype: str
    md5sum: str
    msgdef: str


@dataclass
class ChunkInfo:
    position: int
    start: float
    end: float
    counts: dict            # connection id -> number of messages in this chunk


@dataclass
class BagIndex:
    size: int
    index_pos: int
    connections: dict = field(default_factory=dict)   # id -> Connection
    chunks: list = field(default_factory=list)         # ChunkInfo, sorted by position

    def chunk_byte_sizes(self):
        """Approximate on-disk size of each chunk (+ its index records): distance to the next one."""
        positions = [c.position for c in self.chunks] + [self.index_pos]
        return [b - a for a, b in zip(positions, positions[1:])]


def read_index(reader):
    head = reader.read(0, len(MAGIC) + 8192)
    if not head.startswith(MAGIC):
        raise ValueError(f"not a ROS bag 2.0 file (starts with {head[:13]!r})")
    header, _, _ = next(iter_records(head, len(MAGIC)))
    if _op(header) != OP_BAG_HEADER:
        raise ValueError("first record is not the bag header")
    index_pos = _u64(header["index_pos"])
    if index_pos == 0:
        raise ValueError("bag is not indexed (was it closed properly?)")

    index = BagIndex(size=reader.size, index_pos=index_pos)
    tail = reader.read(index_pos, reader.size - index_pos)
    for header, data, _ in iter_records(tail):
        op = _op(header)
        if op == OP_CONNECTION:
            info = parse_header(data)
            conn_id = _u32(header["conn"])
            index.connections[conn_id] = Connection(
                id=conn_id,
                topic=header["topic"].decode(),
                msgtype=info.get("type", b"?").decode(),
                md5sum=info.get("md5sum", b"").decode(),
                msgdef=info.get("message_definition", b"").decode(errors="replace"))
        elif op == OP_CHUNK_INFO:
            n = _u32(header["count"])
            pairs = struct.unpack_from(f"<{2 * n}I", data)
            index.chunks.append(ChunkInfo(position=_u64(header["chunk_pos"]),
                                          start=_time(header["start_time"]), end=_time(header["end_time"]),
                                          counts=dict(zip(pairs[0::2], pairs[1::2]))))
    index.chunks.sort(key=lambda c: c.position)
    return index


def topic_table(index):
    """One row per topic: type, message count, time span and mean rate (from the index only)."""
    rows = {}
    for conn in index.connections.values():
        row = rows.setdefault(conn.topic, {"topic": conn.topic, "type": conn.msgtype, "messages": 0,
                                           "start": None, "end": None, "connections": 0})
        row["connections"] += 1
        for chunk in index.chunks:
            count = chunk.counts.get(conn.id, 0)
            if count:
                row["messages"] += count
                row["start"] = chunk.start if row["start"] is None else min(row["start"], chunk.start)
                row["end"] = chunk.end if row["end"] is None else max(row["end"], chunk.end)
    for row in rows.values():
        span = (row["end"] - row["start"]) if row["start"] is not None else 0.0
        row["seconds"] = round(span, 1)
        row["rate_hz"] = round((row["messages"] - 1) / span, 2) if span > 0 else None
    return sorted(rows.values(), key=lambda r: r["topic"])


def bag_time_span(index):
    if not index.chunks:
        return None, None
    return min(c.start for c in index.chunks), max(c.end for c in index.chunks)


def read_chunk(reader, position):
    """Return (compression, compressed bytes, list of (conn id, time, raw message bytes))."""
    head = reader.read(position, 4096)
    (header_len,) = struct.unpack_from("<I", head, 0)
    if 8 + header_len > len(head):
        head = reader.read(position, 8 + header_len)
    header = parse_header(head[4:4 + header_len])
    if _op(header) != OP_CHUNK:
        raise ValueError(f"no chunk record at byte {position}")
    (data_len,) = struct.unpack_from("<I", head, 4 + header_len)
    compression = header["compression"].decode()
    data = reader.read(position + 8 + header_len, data_len)
    return compression, data_len, _chunk_messages(compression, data)


def bz2_decoder(name="bz2"):
    """bz2 decompression function: the standard library, or the `indexed_bzip2` package if asked for."""
    if name == "indexed_bzip2":
        import io

        import indexed_bzip2
        def decompress(data):
            # Closed explicitly, so its decoder buffers are released after every chunk.
            file = indexed_bzip2.IndexedBzip2File(io.BytesIO(data), parallelization=1)
            try:
                return file.read()
            finally:
                file.close()
        return decompress
    return bz2.decompress


def _chunk_messages(compression, data, decompress_bz2=bz2.decompress):
    if compression == "bz2":
        plain = decompress_bz2(data)
    elif compression == "lz4":
        import lz4.frame
        plain = lz4.frame.decompress(data)
    elif compression == "none":
        plain = data
    else:
        raise ValueError(f"unknown chunk compression {compression!r}")
    return [(_u32(h["conn"]), _time(h["time"]), d) for h, d, _ in iter_records(plain) if _op(h) == OP_MESSAGE]


def read_chunk_range(reader, index, first, last):
    """
    Messages of chunks first..last (inclusive) fetched with ONE range request. Chunks lie one
    after another in the file (each followed by its index records), so the span is contiguous.
    """
    start = index.chunks[first].position
    stop = index.chunks[last + 1].position if last + 1 < len(index.chunks) else index.index_pos
    buffer = reader.read(start, stop - start)
    messages = []
    for header, data, _ in iter_records(buffer):
        if _op(header) == OP_CHUNK:
            messages += _chunk_messages(header["compression"].decode(), data)
    return messages


def choose_sample_chunks(index, n_spread=4, max_chunk_bytes=64_000_000):
    """
    Chunks to download: for every connection, the chunk nearest the middle of the bag that
    contains it, plus `n_spread` chunks spread evenly over the bag. Oversized chunks are skipped.
    """
    sizes = index.chunk_byte_sizes()
    usable = [i for i, s in enumerate(sizes) if s <= max_chunk_bytes]
    if not usable:
        return []
    middle = len(index.chunks) / 2
    chosen = set()
    for conn_id in index.connections:
        holding = [i for i in usable if index.chunks[i].counts.get(conn_id)]
        if holding:
            chosen.add(min(holding, key=lambda i: abs(i - middle)))
    step = max(len(usable) // (n_spread + 1), 1)
    chosen.update(usable[step * k] for k in range(1, n_spread + 1) if step * k < len(usable))
    return sorted(chosen)


def rosbags_type_name(msgtype):
    """ROS1 bags write 'pkg/Name'; the rosbags library names the same type 'pkg/msg/Name'."""
    package, _, name = msgtype.partition("/")
    return msgtype if "/msg/" in msgtype or not name else f"{package}/msg/{name}"


def deserialize(raw, msgtype, msgdef):
    """Decode one ROS1 message with the `rosbags` library, using the definition stored in the bag."""
    name = rosbags_type_name(msgtype)
    try:
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError:        # rosbags older than 0.9.20
        from rosbags.serde import deserialize_ros1
        from rosbags.typesys import get_types_from_msg, register_types
        register_types(get_types_from_msg(msgdef, name))
        return deserialize_ros1(raw, name)
    store = get_typestore(Stores.ROS1_NOETIC)
    try:
        store.register(get_types_from_msg(msgdef, name))
    except Exception:
        # A definition that clashes with a built-in type: decode with the built-in types only.
        pass
    return store.deserialize_ros1(raw, name)


def summarise_value(value, depth=0, max_depth=4):
    """Short, JSON-friendly description of a decoded message: field names, sizes, a few values."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(value, np.ndarray):
        out = {"array": list(value.shape), "dtype": str(value.dtype)}
        if value.size and value.dtype.kind in "iuf" and value.size <= 64:
            out["values"] = value.tolist()
        return out
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes": len(value)}
    if isinstance(value, str):
        return value if len(value) <= 80 else value[:77] + "..."
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        out = {"list": len(value)}
        if value and depth < max_depth:
            out["first"] = summarise_value(value[0], depth + 1, max_depth)
        return out
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        if depth >= max_depth:
            return {"type": type(value).__name__}
        return {name: summarise_value(getattr(value, name), depth + 1, max_depth)
                for name in fields if not name.startswith("__")}
    return repr(value)[:80]


def sample_messages(reader, index, chunk_ids, max_total_bytes=400_000_000):
    """
    Download the chosen chunks and return, per connection: the first raw message, how many
    messages were seen, their mean serialized size, plus the compression ratio of the chunks.
    """
    stats, fetched, plain_total = {}, 0, 0
    sizes = index.chunk_byte_sizes()
    for i in chunk_ids:
        if fetched + sizes[i] > max_total_bytes:
            print(f"  stopping chunk sampling at {fetched / 1e6:.0f} MB (limit {max_total_bytes / 1e6:.0f} MB)")
            break
        compression, compressed, messages = read_chunk(reader, index.chunks[i].position)
        fetched += compressed
        for conn_id, stamp, raw in messages:
            plain_total += len(raw)
            entry = stats.setdefault(conn_id, {"seen": 0, "bytes": 0, "first_raw": raw, "first_time": stamp})
            entry["seen"] += 1
            entry["bytes"] += len(raw)
    return {"connections": stats, "compressed_bytes": fetched, "plain_message_bytes": plain_total,
            "chunks_read": len(chunk_ids)}
