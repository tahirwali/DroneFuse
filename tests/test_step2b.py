"""
Checks for the redesign after Step 2: the cross-recording protocols, the drone locator of Step 2b on
synthetic frames, and the drone-crop camera variant in Step 3. No GPU, Kaggle or internet needed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")
from dronefuse import crops
from dronefuse.fusion import TrainSettings, load_step2, make_task
from dronefuse.step3 import resume_from, run_protocol, summarize, write_tables
from dronefuse.windows import FLIGHTS, PROTOCOLS, XREC_UNKNOWN, load_windows, protocol_sets

from test_step3 import step2_dir, windows   # noqa: F401  (pytest fixtures shared with the Step 3 tests)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("protocol", ["xrec_s1_to_s2", "xrec_s2_to_s1"])
def test_cross_recording_protocols_never_test_on_a_training_flight(protocol):
    rows = load_windows(ROOT / "splits" / "windows.csv")
    sets = protocol_sets(rows, protocol)
    train_recs = {r["recording"] for r in sets["train"] + sets["val"]}
    test_recs = {r["recording"] for r in sets["test"]}
    assert protocol in PROTOCOLS
    assert not train_recs & test_recs
    assert train_recs == set(FLIGHTS["s1" if protocol.endswith("s2") else "s2"])
    assert sets["known"] == ["M30", "M300", "Phantom4"] and sets["unknown"] == ["Mavic3"]
    unknown = [r for r in sets["test"] if r["drone_type"] == "Mavic3"]
    assert unknown and all(r["recording"] == XREC_UNKNOWN and r["split"] == "test" for r in unknown)


def _frames(n=12, h=240, w=320, seed=0):
    """A bright sky with a dark static 'tree' at the left edge and a small dark drone moving right."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n):
        f = np.full((h, w, 3), (150, 180, 230), np.uint8)
        f[:, :40] = (40, 60, 40)                              # tree / building
        f[100:110, 60:66] = (30, 30, 30)                      # a static dark spot in the sky (e.g. a mark)
        x = 120 + 8 * t
        f[150:154, x:x + 4] = (20, 20, 20)                    # the drone, 4 x 4 pixels
        f = np.clip(f.astype(np.int16) + rng.integers(-3, 4, f.shape), 0, 255).astype(np.uint8)
        frames.append(f)
    return frames


def test_locator_finds_the_moving_drone_not_the_static_spot(monkeypatch):
    monkeypatch.setattr(crops, "SKY_MARGIN", 5)
    frames = _frames()

    class Reader:
        def payload(self, i):
            import io
            from PIL import Image
            buffer = io.BytesIO()
            Image.fromarray(frames[i]).save(buffer, format="PNG")
            return buffer.getvalue()

    model = crops.background_model(Reader(), np.arange(len(frames)), "cpu")
    found = crops.locate(torch.from_numpy(np.stack(frames)), model).numpy()
    for t, (x, y, peak) in enumerate(found):
        assert abs(x - (121 + 8 * t)) <= 4 and abs(y - 151) <= 4, (t, x, y)
        assert peak > 50


def test_standardised_crops_hide_overall_brightness():
    rng = np.random.default_rng(1)
    crop = rng.integers(0, 120, (1, 64, 64, 3)).astype(np.uint8)
    brighter = np.clip(crop.astype(np.int16) + 100, 0, 255).astype(np.uint8)
    a, b = crops.standardise(torch.from_numpy(crop)), crops.standardise(torch.from_numpy(brighter))
    assert a.shape == (1, 3, crops.INPUT, crops.INPUT)
    assert torch.allclose(a, b, atol=1e-3)


def test_crop_variant_runs_into_its_own_folder(step2_dir, windows, tmp_path):   # noqa: F811
    rng = np.random.default_rng(2)
    for cpu in step2_dir.glob("*.npz"):
        if cpu.name.endswith(".deep.npz") or cpu.name.endswith(".crop.npz"):
            continue
        name = cpu.name[:-4]
        ids = np.load(cpu)["window_id"]
        n = len(ids)
        np.savez(step2_dir / f"{name}.crop.npz", window_id=ids,
                 camera_crop_avg=rng.normal(size=(n, 768)).astype(np.float16),
                 camera_crop_max=rng.normal(size=(n, 768)).astype(np.float16),
                 camera_crop_peak=rng.uniform(0, 150, n).astype(np.float32),
                 camera_crop_frame=np.arange(n), camera_crop_xy=np.zeros((n, 2), np.int32))
    data = load_step2(step2_dir, windows, camera="crop")
    assert data.x["camera"].shape[1] == 1537 and not data.skipped
    # the camera is present only where the drone was seen; elsewhere it carries nothing at all
    seen = data.counts[:, 0] > 0
    assert 0 < seen.mean() < 1
    peak = np.expm1(data.x["camera"][:, -1])
    assert (peak[seen] >= crops.DETECTED_PEAK - 1e-3).all() and not data.x["camera"][~seen].any()
    task = make_task(data, windows, "xrec_s1_to_s2")
    assert (task.y[task.train] >= 0).all() and (task.y[task.test] < 0).any()
    settings = TrainSettings(epochs=2, batch_size=128)
    run_protocol(data, windows, "xrec_s1_to_s2", [0], tmp_path, settings, "cpu", log=lambda *a: None,
                 methods=("single_camera", "late", "dronefuse"), camera="crop")
    assert (tmp_path / "runs_crop" / "xrec_s1_to_s2_s0_dronefuse.npz").exists()
    assert not (tmp_path / "runs").exists()
    summary = summarize(data, windows, tmp_path, protocols=("xrec_s1_to_s2",), seeds=(0,), camera="crop")
    assert summary["main"]["xrec_s1_to_s2"]["dronefuse"]["avail_energy_auroc"][0] is not None
    assert "tab_xrec.tex" in write_tables(summary, tmp_path / "tables")
    copied = tmp_path / "copy"
    assert resume_from(tmp_path, copied, log=lambda *a: None) > 0
    assert json.loads((copied / "runs_crop" / "xrec_s1_to_s2_threshold.json").read_text())["chosen"] in (1, 5, 25)


def test_check_gpu_stops_without_a_gpu_unless_allowed(monkeypatch, capsys):
    from dronefuse import environment
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU T4"):
        environment.check_gpu()
    assert "NO GPU FOUND" in capsys.readouterr().out
    assert environment.check_gpu(allow_cpu=True) is None


def test_drone_seen_blocks_keep_only_detected_windows(tmp_path):
    from dronefuse.step2 import load_block
    ids = np.array([f"w{i}" for i in range(6)])
    np.savez(tmp_path / "R.crop.npz", window_id=ids, camera_crop_avg=np.arange(6 * 768).reshape(6, 768).astype(np.float16),
             camera_crop_max=np.zeros((6, 768), np.float16),
             camera_crop_peak=np.array([5, 30, 19.9, 20, 150, 7], np.float32))
    x, kept = load_block(tmp_path, ["R"], "crop, drone seen (avg)", tmp_path)
    assert kept == ["w1", "w3", "w4"] and x.shape == (3, 768)
    x, kept = load_block(tmp_path, ["R"], "camera crop (avg pool)", tmp_path)
    assert len(kept) == 6
