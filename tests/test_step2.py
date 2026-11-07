"""
Quick checks for Step 2 (per-window availability and features) without a GPU, Kaggle or the internet.
A synthetic recording: a static wall plus one moving "drone" point, a radar with Doppler, a camera,
and MP3 audio whose tone changes every second.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronefuse.extract import LIDAR_DTYPE
from dronefuse.features import (AUDIO_FRAME, FeatureSettings, background_voxels, cloud_descriptor,
                                descriptor_names, log_mel, recording_features, window_audio, decode_audio,
                                downsample3)
from dronefuse.step2 import find_recordings, linear_probe
from dronefuse.store import StreamReader, StreamWriter, dtype_to_json, save_meta

T0 = 1000.0
SECONDS = 6
RADAR_DTYPE = np.dtype([(n, "<f4") for n in ("x", "y", "z", "Doppler", "Range", "Power", "Alpha", "Beta")])
TONES = [300, 600, 900, 1200, 1500, 1800]         # Hz, one per second


def _wall():
    g = np.arange(0, 2, 0.25)
    return np.array([(10.0, y, z) for y in g for z in g], np.float32)


def _drone(t):
    """Moves +2 m/s along y, 5 m above the wall."""
    return np.array([[5.0, 2.0 * (t - T0), 5.0]], np.float32)


def _write_recording(folder, with_audio=True):
    folder.mkdir(parents=True, exist_ok=True)
    streams = {}
    lidar = StreamWriter(folder, "livox_mid360")
    for k in range(SECONDS * 5):
        t = T0 + 0.2 * k
        xyz = np.concatenate([_wall(), _drone(t)])
        p = np.zeros(len(xyz), LIDAR_DTYPE)
        p["x"], p["y"], p["z"] = xyz.T
        lidar.append(t, t, p.tobytes())
    lidar.close()
    streams["livox_mid360"] = {"kind": "livox", "payload": "points", "dtype": dtype_to_json(LIDAR_DTYPE)}
    radar = StreamWriter(folder, "radar")
    for k in range(SECONDS * 15):
        t = T0 + k / 15
        p = np.zeros(2, RADAR_DTYPE)
        p[0]["x"], p[0]["Doppler"], p[0]["Power"] = 10.0, 0.0, 20.0           # static
        p[1]["x"], p[1]["y"], p[1]["Doppler"], p[1]["Power"] = 5.0, 2.0 * (t - T0), 2.0, 10.0
        radar.append(t, t, p.tobytes())
    radar.close()
    streams["radar"] = {"kind": "pointcloud", "payload": "points", "dtype": dtype_to_json(RADAR_DTYPE),
                        "channel_names": ["Doppler", "Range", "Power", "Alpha", "Beta"]}
    camera = StreamWriter(folder, "camera")
    from PIL import Image
    for k in range(SECONDS * 10):
        buffer = io.BytesIO()
        Image.fromarray(np.full((8, 8, 3), k, np.uint8)).save(buffer, format="JPEG")
        camera.append(T0 + 0.1 * k, T0 + 0.1 * k, buffer.getvalue())
    camera.close()
    streams["camera"] = {"kind": "image", "payload": "jpeg"}
    if with_audio:
        _write_audio(folder)
        streams["audio1"] = {"kind": "audio", "payload": "mp3 frames"}
    save_meta(folder, {"recording": {"name": "V1_Mavic3"}, "complete": True, "streams": streams})
    return folder


def _mp3_frames(data):
    """Split an MP3 byte stream into frames (MPEG-1 Layer III, 48 kHz)."""
    rates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
    frames, i = [], data.find(b"\xff\xfb")
    while 0 <= i < len(data) - 4:
        bitrate = rates[data[i + 2] >> 4]
        length = 144 * bitrate * 1000 // 48000 + ((data[i + 2] >> 1) & 1)
        frames.append(data[i:i + length])
        i += length
    return frames


def _write_audio(folder):
    import soundfile
    t = np.arange(SECONDS * 48000) / 48000
    tone = np.concatenate([np.sin(2 * np.pi * f * t[:48000]) for f in TONES]) * 0.5
    buffer = io.BytesIO()
    soundfile.write(buffer, np.stack([tone, tone], 1), 48000, format="MP3")
    frames = _mp3_frames(buffer.getvalue())
    audio = StreamWriter(folder, "audio1")
    for k, frame in enumerate(frames):
        audio.append(T0 + k * AUDIO_FRAME / 48000, None, frame)
    audio.close()


def _windows(n=SECONDS - 1, extra_outside=True):
    rows = [{"window_id": f"V1_Mavic3_{k:05d}", "recording": "V1_Mavic3", "version": "V1", "drone_type": "Mavic3",
             "t_start": T0 + k, "t_end": T0 + k + 1.0, "block": 0, "split": "train", "gt_messages": 1}
            for k in range(n)]
    if extra_outside:
        rows.append({**rows[0], "window_id": "V1_Mavic3_99999", "t_start": T0 + 50, "t_end": T0 + 51})
    return rows


def test_static_background_is_removed_and_the_moving_point_kept(tmp_path):
    folder = _write_recording(tmp_path / "rec", with_audio=False)
    stream = StreamReader(folder, "livox_mid360")
    background = background_voxels(stream, 0.5, 0.1)
    features = recording_features(folder, _windows(extra_outside=False), FeatureSettings())
    counts = features["livox_mid360_counts"]
    assert len(background) == len(np.unique(np.floor(_wall() / 0.5), axis=0))
    assert (counts[:, 0] == 5 * (len(_wall()) + 1)).all()     # 5 scans per window
    assert (counts[:, 1] == 5).all()                            # only the drone is foreground
    names = list(features["livox_mid360_names"])
    velocity = features["livox_mid360_fg"][:, names.index("velocity_y")]
    assert np.allclose(velocity, 2.0, atol=0.05)


def test_radar_keeps_doppler_and_power(tmp_path):
    folder = _write_recording(tmp_path / "rec", with_audio=False)
    features = recording_features(folder, _windows(extra_outside=False))
    names = list(features["radar_names"])
    assert list(features["radar_point_channels"]) == ["x", "y", "z", "Doppler", "Power"]
    fg = features["radar_fg"]
    assert np.allclose(fg[:, names.index("Doppler_mean")], 2.0)
    assert np.allclose(fg[:, names.index("Power_mean")], 10.0)
    assert features["radar_mask"].sum(1).tolist() == [15] * (SECONDS - 1)   # 15 foreground points kept


def test_empty_window_gives_zero_descriptor():
    names = descriptor_names(["Doppler"])
    assert (cloud_descriptor([(0.0, np.zeros((0, 3), np.float32), np.zeros((0, 1), np.float32))], ["Doppler"])
            == 0).all()
    assert len(cloud_descriptor([], ["Doppler"])) == len(names)


def test_availability_and_coverage(tmp_path):
    folder = _write_recording(tmp_path / "rec", with_audio=False)
    features = recording_features(folder, _windows())
    names = list(features["availability_names"])
    avail = features["availability"]
    assert features["covered"].tolist() == [True] * (SECONDS - 1) + [False]
    assert avail[:-1, names.index("camera_frames")].tolist() == [10] * (SECONDS - 1)
    assert (avail[-1] == 0).all()                               # outside the extraction: nothing


def test_audio_windows_follow_message_times(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    if "MP3" not in soundfile.available_formats():
        pytest.skip("libsndfile without MP3")
    folder = _write_recording(tmp_path / "rec")
    stream = StreamReader(folder, "audio1", {"kind": "audio"})
    signal = decode_audio(folder, stream)
    for k, tone in enumerate(TONES[:-1]):
        samples, n = window_audio(signal, stream, T0 + k + 0.1, T0 + k + 0.9)
        assert n > 0
        spectrum = np.abs(np.fft.rfft(samples))
        assert abs(np.argmax(spectrum) * 48000 / len(samples) - tone) < 20
    mel = log_mel(downsample3(window_audio(signal, stream, T0, T0 + 1)[0]))
    assert mel.shape == (64, 101)


def test_find_recordings(tmp_path):
    _write_recording(tmp_path / "a" / "b" / "V1_Mavic3", with_audio=False)
    assert list(find_recordings(tmp_path)) == ["V1_Mavic3"]


def test_linear_probe_uses_train_and_val_only(tmp_path):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    windows = []
    for name, drone, k in (("V1_Mavic3", "Mavic3", 0), ("V1_Avata", "Avata", 1)):
        rows = [{"window_id": f"{name}_{i:05d}", "recording": name, "version": "V1", "drone_type": drone,
                 "split": ("train", "val", "test")[i % 3]} for i in range(60)]
        x = rng.normal(size=(60, 4)).astype(np.float32)
        x[:, 0] += 4 * k
        np.savez(tmp_path / f"{name}.npz", window_id=np.array([r["window_id"] for r in rows]),
                 covered=np.ones(60, bool), livox_avia_all=x)
        windows += rows
    result = linear_probe(tmp_path, windows, ["V1_Mavic3", "V1_Avata"], log=lambda *_: None)
    assert result["Livox Avia"]["val_accuracy"] > 0.95
    assert result["Livox Avia"]["train_windows"] == 40 and result["Livox Avia"]["val_windows"] == 40
    assert result["camera (avg pool)"] is None
