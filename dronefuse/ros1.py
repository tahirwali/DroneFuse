"""
A small, fast ROS1 message decoder driven by the message definition stored in the bag.

`rosbags` builds one Python object per array element, which is far too slow for LiDAR
messages with 24,000 points. Here an array of fixed-size elements (a CustomPoint, a Point32,
float32 values, image bytes) becomes ONE numpy array view on the raw bytes, so decoding a
LiDAR scan or a 7 MB image costs microseconds. Messages decode to plain dicts; `time` and
`duration` decode to float seconds.
"""
import struct

import numpy as np

PRIMITIVES = {"bool": "?", "int8": "b", "uint8": "B", "byte": "b", "char": "B", "int16": "h", "uint16": "H",
              "int32": "i", "uint32": "I", "int64": "q", "uint64": "Q", "float32": "f", "float64": "d"}
TIMES = {"time": "<II", "duration": "<ii"}


def _strip_comment(line):
    return line.split("#", 1)[0].strip()


def parse_definitions(msgtype, msgdef):
    """{type name: [(field type, field name, array length)]}; length None = scalar, -1 = variable."""
    sections, current, lines = {}, msgtype, []
    for line in msgdef.splitlines() + ["=" * 80]:
        if line.startswith("=" * 10):
            sections[current] = lines
            current, lines = None, []
            continue
        if line.startswith("MSG:"):
            current = line[4:].strip()
            continue
        lines.append(line)
    types = {}
    for name, body in sections.items():
        if name is None:
            continue
        fields = []
        for line in body:
            line = _strip_comment(line)
            if not line:
                continue
            if "=" in line:
                continue                       # a constant, not a field
            ftype, fname = line.split()[:2]
            length = None
            if ftype.endswith("]"):
                ftype, size = ftype[:-1].split("[")
                length = int(size) if size else -1
            fields.append((ftype, fname, length))
        types[name] = fields
    return types


class Schema:
    """Decoder for one message type (and the types it contains)."""

    def __init__(self, msgtype, msgdef):
        self.msgtype = msgtype
        self.types = parse_definitions(msgtype, msgdef)
        self._dtype_cache = {}

    def _resolve(self, ftype, owner):
        if ftype in PRIMITIVES or ftype in TIMES or ftype == "string":
            return ftype
        if ftype == "Header":
            return "std_msgs/Header"
        if "/" in ftype:
            return ftype
        local = f"{owner.split('/')[0]}/{ftype}"
        if local in self.types:
            return local
        matches = [t for t in self.types if t.split("/")[-1] == ftype]
        if not matches:
            raise KeyError(f"type {ftype!r} (used in {owner}) is not defined")
        return matches[0]

    def fixed_dtype(self, ftype, owner=None):
        """numpy dtype of a fixed-size type, or None if it contains strings or variable arrays."""
        name = self._resolve(ftype, owner or self.msgtype)
        if name in PRIMITIVES:
            return np.dtype("<" + PRIMITIVES[name])
        if name in TIMES:
            return np.dtype([("sec", "<" + TIMES[name][2]), ("nsec", "<" + TIMES[name][2])])
        if name == "string":
            return None
        if name in self._dtype_cache:
            return self._dtype_cache[name]
        parts = []
        for sub_type, sub_name, length in self.types[name]:
            sub = self.fixed_dtype(sub_type, name)
            if sub is None or length == -1:
                self._dtype_cache[name] = None
                return None
            parts.append((sub_name, sub) if length is None else (sub_name, sub, (length,)))
        dtype = np.dtype(parts)
        self._dtype_cache[name] = dtype
        return dtype

    def decode(self, raw):
        value, _ = self._read_message(self.msgtype, raw, 0)
        return value

    def _read_message(self, name, buf, pos):
        out = {}
        for ftype, fname, length in self.types[name]:
            resolved = self._resolve(ftype, name)
            if length is None:
                out[fname], pos = self._read_one(resolved, buf, pos)
            else:
                if length == -1:
                    (length,) = struct.unpack_from("<I", buf, pos)
                    pos += 4
                dtype = self.fixed_dtype(resolved, name)
                if dtype is not None:
                    out[fname] = np.frombuffer(buf, dtype=dtype, count=length, offset=pos)
                    pos += dtype.itemsize * length
                else:
                    items = []
                    for _ in range(length):
                        item, pos = self._read_one(resolved, buf, pos)
                        items.append(item)
                    out[fname] = items
        return out, pos

    def _read_one(self, name, buf, pos):
        if name in PRIMITIVES:
            fmt = "<" + PRIMITIVES[name]
            return struct.unpack_from(fmt, buf, pos)[0], pos + struct.calcsize(fmt)
        if name in TIMES:
            sec, nsec = struct.unpack_from(TIMES[name], buf, pos)
            return sec + nsec * 1e-9, pos + 8
        if name == "string":
            (n,) = struct.unpack_from("<I", buf, pos)
            return bytes(buf[pos + 4:pos + 4 + n]).decode("utf-8", errors="replace"), pos + 4 + n
        return self._read_message(name, buf, pos)
