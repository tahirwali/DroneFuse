"""
Quick checks for Step 0 (MMAUD probe) that run without a GPU, Kaggle or the internet.
Run with:  python -m pytest tests -q
"""
import bz2
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronefuse.hf_inventory import files_in, folder_totals, load_inventory, save_inventory
from dronefuse.probe import bag_sources
from dronefuse.remote import ConcatFile, LocalFile
from dronefuse.rosbag_index import (MAGIC, choose_sample_chunks, read_chunk, read_chunk_range, read_index,
                                    rosbags_type_name, summarise_value, topic_table)
from dronefuse.runner import StepRunner

STRING_DEF = "string data\n"


# A minimal ROS bag 2.0 writer, used only to build test files
def _record(fields, data):
    header = b"".join(struct.pack("<I", len(k) + 1 + len(v)) + k.encode() + b"=" + v for k, v in fields.items())
    return struct.pack("<I", len(header)) + header + struct.pack("<I", len(data)) + data


def _time(t):
    return struct.pack("<II", int(t), int(round((t - int(t)) * 1e9)))


def _connection(conn_id, topic, msgtype):
    info = {"topic": topic.encode(), "type": msgtype.encode(), "md5sum": b"992ce8a1687cec8c8bd883ec73ca41d1",
            "message_definition": STRING_DEF.encode()}
    info_bytes = b"".join(struct.pack("<I", len(k) + 1 + len(v)) + k.encode() + b"=" + v for k, v in info.items())
    return _record({"op": b"\x07", "conn": struct.pack("<I", conn_id), "topic": topic.encode()}, info_bytes)


def _string_msg(text):
    payload = text.encode()
    return struct.pack("<I", len(payload)) + payload


def write_bag(path, chunks, compression="bz2"):
    """chunks: list of lists of (conn_id, time, text). Connections: 0=/camera, 1=/audio."""
    topics = {0: ("/camera", "std_msgs/String"), 1: ("/audio", "std_msgs/String")}
    body, chunk_infos = bytearray(), []
    start = len(MAGIC) + 4096
    for messages in chunks:
        inner = b"".join(_connection(c, *topics[c]) for c in topics)
        inner += b"".join(_record({"op": b"\x02", "conn": struct.pack("<I", c), "time": _time(t)}, _string_msg(s))
                          for c, t, s in messages)
        data = bz2.compress(inner) if compression == "bz2" else inner
        position = start + len(body)
        body += _record({"op": b"\x05", "compression": compression.encode(), "size": struct.pack("<I", len(inner))}, data)
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
                         "start_time": _time(t0), "end_time": _time(t1), "count": struct.pack("<I", len(counts))},
                        pairs)
    fields = {"op": b"\x03", "index_pos": struct.pack("<Q", index_pos),
              "conn_count": struct.pack("<I", len(topics)), "chunk_count": struct.pack("<I", len(chunks))}
    header = _record(fields, b" " * (4096 - len(_record(fields, b""))))   # padded to exactly 4096 bytes
    Path(path).write_bytes(MAGIC + header + bytes(body) + tail)


@pytest.fixture
def bag(tmp_path):
    chunks = [
        [(0, 100.0, "img0"), (1, 100.02, "a0"), (1, 100.04, "a1")],
        [(0, 100.5, "img1"), (1, 100.52, "a2")],
        [(1, 101.0, "a3"), (0, 101.0, "img2")],
    ]
    path = tmp_path / "test.bag"
    write_bag(path, chunks)
    return path


# Tests
def test_index_gives_topics_counts_and_rates(bag):
    index = read_index(LocalFile(bag))
    assert {c.topic for c in index.connections.values()} == {"/camera", "/audio"}
    assert len(index.chunks) == 3
    rows = {r["topic"]: r for r in topic_table(index)}
    assert rows["/camera"]["messages"] == 3 and rows["/audio"]["messages"] == 4
    assert rows["/camera"]["type"] == "std_msgs/String"
    assert rows["/camera"]["seconds"] == pytest.approx(1.0, abs=0.01)
    assert rows["/camera"]["rate_hz"] == pytest.approx(2.0, abs=0.01)


def test_chunks_decompress_to_the_written_messages(bag):
    reader = LocalFile(bag)
    index = read_index(reader)
    compression, _, messages = read_chunk(reader, index.chunks[1].position)
    assert compression == "bz2"
    assert [(c, round(t, 2), raw[4:].decode()) for c, t, raw in messages] == [(0, 100.5, "img1"), (1, 100.52, "a2")]


def test_one_request_for_a_run_of_chunks_gives_the_same_messages(bag):
    reader = LocalFile(bag)
    index = read_index(reader)
    one_by_one = [m for c in index.chunks for m in read_chunk(reader, c.position)[2]]
    assert read_chunk_range(reader, index, 0, len(index.chunks) - 1) == one_by_one
    assert read_chunk_range(reader, index, 1, 1) == read_chunk(reader, index.chunks[1].position)[2]


def test_bag_split_into_byte_parts_reads_like_the_whole_file(bag, tmp_path):
    data = bag.read_bytes()
    cuts = [0, 5000, len(data) - 300, len(data)]
    parts = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        part = tmp_path / f"test.bag.part{i + 1:03d}"
        part.write_bytes(data[a:b])
        parts.append(LocalFile(part))
    joined = ConcatFile(parts)
    assert joined.size == len(data)
    assert joined.read(4990, 400) == data[4990:5390]
    assert topic_table(read_index(joined)) == topic_table(read_index(LocalFile(bag)))


def test_sample_chunks_cover_every_connection(bag):
    index = read_index(LocalFile(bag))
    chosen = choose_sample_chunks(index, n_spread=0)
    covered = {c for i in chosen for c in index.chunks[i].counts}
    assert covered == set(index.connections)


def test_decoding_with_rosbags_when_available(bag):
    pytest.importorskip("rosbags")
    from dronefuse.rosbag_index import deserialize
    reader = LocalFile(bag)
    index = read_index(reader)
    _, _, messages = read_chunk(reader, index.chunks[0].position)
    message = deserialize(messages[0][2], "std_msgs/String", STRING_DEF)
    assert message.data == "img0"


def test_type_names_are_converted_for_rosbags():
    assert rosbags_type_name("sensor_msgs/Image") == "sensor_msgs/msg/Image"
    assert rosbags_type_name("sensor_msgs/msg/Image") == "sensor_msgs/msg/Image"


def test_not_a_bag_is_rejected(tmp_path):
    path = tmp_path / "fake.bag"
    path.write_bytes(b"hello world" * 100)
    with pytest.raises(ValueError):
        read_index(LocalFile(path))


def test_summary_of_nested_values():
    np = pytest.importorskip("numpy")
    from dataclasses import dataclass

    @dataclass
    class Image:
        width: int
        encoding: str
        data: object

    summary = summarise_value(Image(2560, "rgb8", np.zeros(7_372_800, dtype=np.uint8)))
    assert summary == {"width": 2560, "encoding": "rgb8", "data": {"array": [7372800], "dtype": "uint8"}}
    assert summarise_value([1, 2, 3]) == {"list": 3, "first": 1}


def test_bag_sources_groups_byte_parts_in_order():
    files = [("V2_V3/rosbag/m3e.bag.part002", 20), ("V2_V3/rosbag/m3e.bag.part001", 20),
             ("V2_V3/rosbag/m3e.bag.part003", 14), ("V2_V3/rosbag/m3e.bag.parts.json", 1),
             ("V1/rosbag/Mavic3.bag", 11), ("V1/filesystem_format/Mavic3/image/1.png", 3)]
    bags = bag_sources(files)
    assert list(bags) == ["V1/rosbag/Mavic3.bag", "V2_V3/rosbag/m3e.bag"]
    assert [p for p, _ in bags["V2_V3/rosbag/m3e.bag"]] == [
        "V2_V3/rosbag/m3e.bag.part001", "V2_V3/rosbag/m3e.bag.part002", "V2_V3/rosbag/m3e.bag.part003"]


def test_inventory_totals_and_round_trip(tmp_path):
    files = [("V1/filesystem_format/Mavic3/image/2.png", 2_000_000_000),
             ("V1/filesystem_format/Mavic3/image/1.png", 1_000_000_000),
             ("V1/filesystem_format/Mavic3/lidar_360/1.npy", 500_000_000), ("README.md", 10)]
    totals = folder_totals(files)
    assert totals[0] == ("V1/filesystem_format/Mavic3/image", 2, 3.0)
    assert [p for p, _ in files_in(files, "V1/filesystem_format/Mavic3/image")] == [
        "V1/filesystem_format/Mavic3/image/2.png", "V1/filesystem_format/Mavic3/image/1.png"]
    save_inventory(files, tmp_path / "inv.csv.gz")
    assert load_inventory(tmp_path / "inv.csv.gz") == files


def test_runner_keeps_going_after_an_error(tmp_path):
    runner = StepRunner(tmp_path, time_budget_minutes=5)
    assert runner.run("good", lambda: 42) == 42
    assert runner.run("bad", lambda: 1 / 0) is None
    assert runner.run("after", lambda: "still runs") == "still runs"
    assert runner.failed() == ["bad"]
    assert (tmp_path / "steps.json").exists()


def test_runner_skips_steps_after_the_time_budget(tmp_path):
    runner = StepRunner(tmp_path, time_budget_minutes=0)
    assert runner.run("late", lambda: 1) is None
    assert runner.steps[0]["status"] == "skipped"
