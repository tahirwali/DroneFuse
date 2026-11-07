"""
Writes small ROS1 bags with MMAUD's real message types (definitions copied from the bags),
for tests only. Messages are serialised by hand so the decoder is checked against known bytes.
"""
import bz2
import struct
from pathlib import Path

import numpy as np

from dronefuse.rosbag_index import MAGIC

HEADER_DEF = "MSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n"
SEP = "=" * 80 + "\n"
DEFS = {
    "sensor_msgs/Image": "Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\n"
                         "uint32 step\nuint8[] data\n" + SEP + HEADER_DEF,
    "livox_ros_driver/CustomMsg": "Header header\nuint64 timebase\nuint32 point_num\nuint8  lidar_id\n"
                                  "uint8[3]  rsvd\nCustomPoint[] points\n" + SEP + HEADER_DEF + SEP +
                                  "MSG: livox_ros_driver/CustomPoint\nuint32 offset_time\nfloat32 x\nfloat32 y\n"
                                  "float32 z\nuint8 reflectivity # 0~255\nuint8 tag\nuint8 line\n",
    "sensor_msgs/PointCloud": "Header header\ngeometry_msgs/Point32[] points\nChannelFloat32[] channels\n" + SEP +
                              HEADER_DEF + SEP + "MSG: geometry_msgs/Point32\nfloat32 x\nfloat32 y\nfloat32 z\n" +
                              SEP + "MSG: sensor_msgs/ChannelFloat32\nstring name\nfloat32[] values\n",
    "audio_common_msgs/AudioData": "uint8[] data\n",
    "geometry_msgs/PointStamped": "Header header\nPoint point\n" + SEP + HEADER_DEF + SEP +
                                  "MSG: geometry_msgs/Point\n# a comment\nuint8 CONSTANT=3\nfloat64 x\nfloat64 y\nfloat64 z\n",
}

POINT_DTYPE = np.dtype([("offset_time", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("reflectivity", "u1"), ("tag", "u1"), ("line", "u1")])


def _string(text):
    data = text.encode()
    return struct.pack("<I", len(data)) + data


def _stamp(t):
    return struct.pack("<II", int(t), int(round((t - int(t)) * 1e9)))


def header(t, seq=0, frame="f"):
    return struct.pack("<I", seq) + _stamp(t) + _string(frame)


def image(t, pixels, encoding="rgb8"):
    h, w = pixels.shape[:2]
    data = pixels.astype(np.uint8).tobytes()
    return (header(t) + struct.pack("<II", h, w) + _string(encoding) + b"\x00" + struct.pack("<I", w * 3)
            + struct.pack("<I", len(data)) + data)


def livox(t, xyz, reflectivity):
    points = np.zeros(len(xyz), dtype=POINT_DTYPE)
    points["x"], points["y"], points["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    points["reflectivity"] = reflectivity
    return (header(t) + struct.pack("<QIB", 123, len(points), 0) + b"\x00\x00\x00"
            + struct.pack("<I", len(points)) + points.tobytes())


def pointcloud(t, xyz, channels):
    out = header(t) + struct.pack("<I", len(xyz)) + np.asarray(xyz, dtype="<f4").tobytes()
    out += struct.pack("<I", len(channels))
    for name, values in channels:
        out += _string(name) + struct.pack("<I", len(values)) + np.asarray(values, dtype="<f4").tobytes()
    return out


def audio(payload):
    return struct.pack("<I", len(payload)) + payload


def point_stamped(t, xyz):
    return header(t) + struct.pack("<ddd", *xyz)


def _record(fields, data):
    head = b"".join(struct.pack("<I", len(k) + 1 + len(v)) + k.encode() + b"=" + v for k, v in fields.items())
    return struct.pack("<I", len(head)) + head + struct.pack("<I", len(data)) + data


def _connection(conn_id, topic, msgtype):
    info = {"topic": topic.encode(), "type": msgtype.encode(), "md5sum": b"0" * 32,
            "message_definition": DEFS[msgtype].encode()}
    info_bytes = b"".join(struct.pack("<I", len(k) + 1 + len(v)) + k.encode() + b"=" + v for k, v in info.items())
    return _record({"op": b"\x07", "conn": struct.pack("<I", conn_id), "topic": topic.encode()}, info_bytes)


def write_bag(path, topics, chunks):
    """topics: {conn id: (topic, msgtype)}; chunks: list of lists of (conn id, bag time, serialised message)."""
    body, chunk_infos = bytearray(), []
    start = len(MAGIC) + 4096
    for messages in chunks:
        inner = b"".join(_connection(c, *topics[c]) for c in topics)
        inner += b"".join(_record({"op": b"\x02", "conn": struct.pack("<I", c), "time": _stamp(t)}, raw)
                          for c, t, raw in messages)
        position = start + len(body)
        body += _record({"op": b"\x05", "compression": b"bz2", "size": struct.pack("<I", len(inner))},
                        bz2.compress(inner))
        counts = {}
        for c, _, _ in messages:
            counts[c] = counts.get(c, 0) + 1
        times = [t for _, t, _ in messages]
        chunk_infos.append((position, min(times), max(times), counts))
    index_pos = start + len(body)
    tail = b"".join(_connection(c, *topics[c]) for c in topics)
    for position, t0, t1, counts in chunk_infos:
        pairs = b"".join(struct.pack("<II", c, n) for c, n in sorted(counts.items()))
        tail += _record({"op": b"\x06", "ver": struct.pack("<I", 1), "chunk_pos": struct.pack("<Q", position),
                         "start_time": _stamp(t0), "end_time": _stamp(t1), "count": struct.pack("<I", len(counts))},
                        pairs)
    fields = {"op": b"\x03", "index_pos": struct.pack("<Q", index_pos),
              "conn_count": struct.pack("<I", len(topics)), "chunk_count": struct.pack("<I", len(chunks))}
    head = _record(fields, b" " * (4096 - len(_record(fields, b""))))
    Path(path).write_bytes(MAGIC + head + bytes(body) + tail)
