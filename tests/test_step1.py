"""
Quick checks for Step 1 (extraction, windows, splits) that run without a GPU, Kaggle or the internet.
Run with:  python -m pytest tests -q
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bagwriter as bw
from dronefuse.extract import ExtractSettings, extract_recording, make_plan, make_tasks, read_truth, verify_recording
from dronefuse.recordings import DRONE_TYPES, RECORDINGS
from dronefuse.remote import LocalFile
from dronefuse.ros1 import Schema, parse_definitions
from dronefuse.rosbag_index import read_index
from dronefuse.store import StreamReader, save_meta
from dronefuse.windows import make_windows, protocol_sets

TOPICS = {0: ("/usb_cam/image_raw", "sensor_msgs/Image"), 1: ("/livox_avia/lidar", "livox_ros_driver/CustomMsg"),
          2: ("/radar_enhanced_pcl", "sensor_msgs/PointCloud"), 3: ("/audio1/audio", "audio_common_msgs/AudioData"),
          4: ("/rosout", "audio_common_msgs/AudioData")}       # 4: a topic that must be ignored
N_FRAMES = 12


def _stereo(k):
    """8x16 side-by-side pair: left half = k, right half = 200."""
    pixels = np.full((8, 16, 3), 200, dtype=np.uint8)
    pixels[:, :8] = k
    return pixels


def _messages():
    """One sensor 'second' per frame k: camera, a LiDAR scan with 2 zero points, a radar scan, 2 audio frames."""
    out = []
    for k in range(N_FRAMES):
        t = 100.0 + 0.1 * k
        xyz = np.array([[0, 0, 0], [k, 1, 2], [0, 0, 0], [3, k, 5]], dtype=np.float32)
        out.append((0, t, bw.image(t, _stereo(k * 10))))
        out.append((1, t + 0.01, bw.livox(t, xyz, [0, 10 + k, 0, 20])))
        out.append((2, t + 0.02, bw.pointcloud(t, [[1, 2, 3], [4, 5, k]],
                                               [("Doppler", [0.5, -0.5]), ("Power", [1, 2])])))
        out.append((3, t + 0.03, bw.audio(bytes([0xFF, 0xFB, k, 1]))))
        out.append((3, t + 0.06, bw.audio(bytes([0xFF, 0xFB, k, 2]))))
        out.append((4, t + 0.07, bw.audio(b"ignored")))
    return out


@pytest.fixture
def bag(tmp_path):
    messages = _messages()
    chunks = [messages[i:i + 7] for i in range(0, len(messages), 7)]      # chunk borders cut through frames
    path = tmp_path / "sensors.bag"
    bw.write_bag(path, TOPICS, chunks)
    return path


SETTINGS = ExtractSettings(camera_every=3, lidar_every=2, camera_size=(8, 8), jpeg_quality=95, chunks_per_task=2)


def _extract(bag, folder, workers=1, settings=SETTINGS):
    index = read_index(LocalFile(bag))
    summary = extract_recording(("local", str(bag)), index, folder, settings, workers=workers, log=lambda *a: None)
    save_meta(folder, summary)
    return summary


# Decoder
def test_definitions_skip_comments_and_constants():
    types = parse_definitions("geometry_msgs/PointStamped", bw.DEFS["geometry_msgs/PointStamped"])
    assert types["geometry_msgs/Point"] == [("float64", "x", None), ("float64", "y", None), ("float64", "z", None)]
    assert types["geometry_msgs/PointStamped"][0] == ("Header", "header", None)


def test_decoder_reads_every_message_type():
    image = Schema("sensor_msgs/Image", bw.DEFS["sensor_msgs/Image"]).decode(bw.image(5.25, _stereo(7)))
    assert image["header"]["stamp"] == pytest.approx(5.25)
    assert (image["height"], image["width"], image["encoding"]) == (8, 16, "rgb8")
    assert image["data"].reshape(8, 16, 3)[0, 0, 0] == 7
    scan = Schema("livox_ros_driver/CustomMsg", bw.DEFS["livox_ros_driver/CustomMsg"]).decode(
        bw.livox(1.0, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), [9, 8]))
    assert scan["point_num"] == 2 and scan["timebase"] == 123 and list(scan["rsvd"]) == [0, 0, 0]
    assert scan["points"]["y"].tolist() == [2, 5] and scan["points"]["reflectivity"].tolist() == [9, 8]
    cloud = Schema("sensor_msgs/PointCloud", bw.DEFS["sensor_msgs/PointCloud"]).decode(
        bw.pointcloud(1.0, [[1, 2, 3]], [("Doppler", [0.25])]))
    assert cloud["points"]["z"].tolist() == [3] and cloud["channels"][0]["name"] == "Doppler"
    assert cloud["channels"][0]["values"].tolist() == [0.25]
    point = Schema("geometry_msgs/PointStamped", bw.DEFS["geometry_msgs/PointStamped"]).decode(
        bw.point_stamped(2.5, (1.5, -2.0, 3.0)))
    assert point["point"] == {"x": 1.5, "y": -2.0, "z": 3.0}


def test_decoder_agrees_with_rosbags():
    pytest.importorskip("rosbags")
    from dronefuse.rosbag_index import deserialize
    raw = bw.livox(3.5, np.array([[1, 2, 3], [0, 0, 7]], dtype=np.float32), [4, 5])
    ours = Schema("livox_ros_driver/CustomMsg", bw.DEFS["livox_ros_driver/CustomMsg"]).decode(raw)
    theirs = deserialize(raw, "livox_ros_driver/CustomMsg", bw.DEFS["livox_ros_driver/CustomMsg"])
    assert ours["point_num"] == theirs.point_num
    assert [p.z for p in theirs.points] == ours["points"]["z"].tolist()


# Extraction
def test_tasks_carry_message_positions(bag):
    index = read_index(LocalFile(bag))
    plan = make_plan(index, SETTINGS)
    assert {e["stream"] for e in plan.values()} == {"camera", "livox_avia", "radar", "audio1"}
    tasks = make_tasks(index, plan, chunks_per_task=2)
    assert tasks[0]["ordinals"] == {0: 0, 1: 0, 2: 0, 3: 0}
    assert sum(t["last"] - t["first"] + 1 for t in tasks) == len(index.chunks)
    assert all(a["stop"] == b["start"] for a, b in zip(tasks, tasks[1:]))


def test_extraction_keeps_the_right_messages(bag, tmp_path):
    summary = _extract(bag, tmp_path / "out")
    assert summary["complete"] and not summary["chunks_failed"]
    folder = tmp_path / "out"
    camera = StreamReader(folder, "camera")
    assert len(camera) == 4                                  # frames 0, 3, 6, 9 of 12
    assert camera.time.tolist() == pytest.approx([100.0, 100.3, 100.6, 100.9])
    from PIL import Image
    frame = np.asarray(Image.open(io.BytesIO(camera.payload(1))))
    assert frame.shape == (8, 8, 3) and abs(int(frame.mean()) - 30) <= 2      # left half of frame 3
    lidar = StreamReader(folder, "livox_avia")
    assert len(lidar) == 6
    points = lidar.points(1)                                 # scan 2: the zero points are dropped
    assert points["x"].tolist() == [2, 3] and points["reflectivity"].tolist() == [12, 20]
    radar = StreamReader(folder, "radar")
    assert len(radar) == N_FRAMES and radar.meta["channel_names"] == ["Doppler", "Power"]
    assert radar.points(4)["z"].tolist() == [3, 4] and radar.points(4)["Doppler"].tolist() == [0.5, -0.5]
    audio = StreamReader(folder, "audio1")
    assert len(audio) == 2 * N_FRAMES and np.all(np.diff(audio.time) > 0)
    assert audio.payload(3) == bytes([0xFF, 0xFB, 1, 2])
    assert (folder / "camera_preview_stereo.jpg").exists()
    report = verify_recording(folder)
    assert report["camera"]["first"] == [8, 8] and report["audio1"]["first"]["mp3_sync"]


def test_result_does_not_depend_on_task_size_or_workers(bag, tmp_path):
    _extract(bag, tmp_path / "a")
    other = ExtractSettings(**{**SETTINGS.__dict__, "chunks_per_task": 1})
    _extract(bag, tmp_path / "b", workers=2, settings=other)
    for name in ("camera", "livox_avia", "radar", "audio1"):
        a, b = StreamReader(tmp_path / "a", name), StreamReader(tmp_path / "b", name)
        assert a.time.tolist() == b.time.tolist()
        assert [a.payload(i) for i in range(len(a))] == [b.payload(i) for i in range(len(b))]


def crash_once(spec, plan, task, settings):
    """Worker that dies abruptly (as if killed for lack of memory) the first time it gets chunk 2."""
    import os
    from dronefuse.extract import process_task
    marker = Path(os.environ["DRONEFUSE_TEST_CRASH_MARKER"])
    if task["first"] == 2 and not marker.exists():
        marker.write_text("crashed")
        os._exit(1)
    return process_task(spec, plan, task, settings)


def test_a_dying_worker_is_replaced_and_nothing_is_lost(bag, tmp_path, monkeypatch):
    monkeypatch.setenv("DRONEFUSE_TEST_CRASH_MARKER", str(tmp_path / "crashed.txt"))
    _extract(bag, tmp_path / "a")
    settings = ExtractSettings(**{**SETTINGS.__dict__, "chunks_per_task": 1, "tasks_per_worker": 2})
    index = read_index(LocalFile(bag))
    messages = []
    summary = extract_recording(("local", str(bag)), index, tmp_path / "b", settings, workers=2,
                                log=messages.append, worker=crash_once)
    save_meta(tmp_path / "b", summary)
    assert (tmp_path / "crashed.txt").exists()
    assert any("new pool with 1 workers" in m for m in messages)
    assert summary["complete"] and not summary["chunks_failed"]
    for name in ("camera", "livox_avia", "radar", "audio1"):
        a, b = StreamReader(tmp_path / "a", name), StreamReader(tmp_path / "b", name)
        assert a.time.tolist() == b.time.tolist()
        assert sorted(a.payload(i) for i in range(len(a))) == sorted(b.payload(i) for i in range(len(b)))


def test_deadline_stops_early_and_marks_incomplete(bag, tmp_path):
    index = read_index(LocalFile(bag))
    summary = extract_recording(("local", str(bag)), index, tmp_path / "c", SETTINGS, workers=1, deadline=0,
                                log=lambda *a: None)
    assert summary["stopped_by_time_budget"] and not summary["complete"] and summary["chunks_done"] == 0


def test_ground_truth_is_read_in_time_order(tmp_path):
    topics = {0: ("/leica/point/relative", "geometry_msgs/PointStamped"),
              1: ("/leica/point/absolute", "geometry_msgs/PointStamped")}
    chunks = [[(0, 2.0, bw.point_stamped(2.0, (2, 0, 0))), (1, 2.0, bw.point_stamped(2.0, (9, 9, 9)))],
              [(0, 1.0, bw.point_stamped(1.0, (1, 0, 0)))]]
    path = tmp_path / "gt.bag"
    bw.write_bag(path, topics, chunks)
    reader = LocalFile(path)
    truth = read_truth(reader, read_index(reader))
    assert truth["relative"]["time"].tolist() == [1.0, 2.0]
    assert truth["relative"]["xyz"][:, 0].tolist() == [1, 2] and len(truth["absolute"]["time"]) == 1


# Windows and splits
def test_windows_never_cross_or_touch_a_split_border():
    recording = RECORDINGS[0]
    spans = {"/a": (0.0, 205.0), "/b": (3.0, 210.0)}
    truth = np.arange(1.0, 200.0, 0.2)
    rows = make_windows(recording, spans, truth)
    assert rows[0]["t_start"] >= 3.0 and rows[-1]["t_end"] <= 199.8 + 1e-6
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    for a, b in zip(rows, rows[1:]):
        if a["split"] != b["split"]:
            assert b["t_start"] - a["t_end"] >= 2 * 1.0 - 1e-6        # GAP_S on both sides of the border
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    assert counts["train"] > counts["val"] > 0 and counts["test"] > 0


def test_windows_need_ground_truth():
    spans = {"/a": (0.0, 100.0)}
    truth = np.concatenate([np.arange(0, 40, 0.2), np.arange(60, 100, 0.2)])     # tracker lost from 40 to 60 s
    rows = make_windows(RECORDINGS[0], spans, truth)
    assert not any(40.0 < r["t_start"] and r["t_end"] < 60.0 for r in rows)


def _fake_rows():
    rows = []
    for r in RECORDINGS:
        for split in ("train", "val", "test"):
            rows.append({"recording": r.name, "version": r.version, "drone_type": r.drone_type, "split": split})
    return rows


def test_protocols_keep_unknown_types_out_of_training():
    rows = _fake_rows()
    for held_out in DRONE_TYPES:
        sets = protocol_sets(rows, f"loo_{held_out}")
        assert sets["unknown"] == [held_out]
        assert all(r["drone_type"] != held_out for r in sets["train"] + sets["val"])
    v1 = protocol_sets(rows, "v1_to_v2")
    assert v1["unknown"] == ["M30"] and all(r["version"] == "V1" for r in v1["train"] + v1["val"])
    v2 = protocol_sets(rows, "v2_to_v1")
    assert v2["unknown"] == ["Avata", "Mavic2"] and all(r["version"] == "V1" for r in v2["test"])
    closed = protocol_sets(rows, "closed")
    assert closed["unknown"] == [] and len(closed["test"]) == len(RECORDINGS)
