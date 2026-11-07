"""
Quick checks for Step 3 (fusion, missing sensors, unknown rejection) on synthetic Step 2 output.
Every 6th window of the real splits gets fake features: the drone type is visible in the camera and
(weaker) in the audio; the radar is empty at the V2/V3 site, as in MMAUD.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")
from dronefuse.features import AVAILABILITY
from dronefuse.fusion import (SENSORS, TrainSettings, auroc, augment_mask, fpr_at_95, group_presence, load_step2,
                              make_task, presence)
from dronefuse.recordings import DRONE_TYPES
from dronefuse.step3 import format_step3_report, run_protocol, summarize, write_tables
from dronefuse.windows import load_windows

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def windows():
    return load_windows(ROOT / "splits" / "windows.csv")


@pytest.fixture(scope="module")
def step2_dir(tmp_path_factory, windows):
    out = tmp_path_factory.mktemp("step2")
    rng = np.random.default_rng(0)
    by_rec = {}
    for w in windows[::6]:
        by_rec.setdefault(w["recording"], []).append(w)
    for rec, rows in by_rec.items():
        n = len(rows)
        y = np.array([DRONE_TYPES.index(r["drone_type"]) for r in rows])
        v2 = rows[0]["version"] != "V1"
        camera = rng.normal(size=(n, 768)).astype(np.float32)
        for j in range(8):
            camera[np.arange(n), 8 * y + j] += 3.0
        ast = rng.normal(size=(n, 4, 768)).astype(np.float32)
        ast[np.arange(n), :, 10 + y] += 1.5
        avail = np.zeros((n, len(AVAILABILITY)), np.int64)
        avail[:, AVAILABILITY.index("camera_frames")] = 10
        avail[:, AVAILABILITY.index("livox_avia_points")] = rng.integers(0, 40, n)
        avail[:, AVAILABILITY.index("livox_mid360_fg_points")] = rng.integers(0, 300, n)
        avail[:, AVAILABILITY.index("radar_fg_points")] = 0 if v2 else rng.integers(0, 200, n)
        for k in range(4):
            avail[:, AVAILABILITY.index(f"audio{k + 1}_frames")] = 41
        covered = np.ones(n, bool)
        covered[-1] = False                                    # one uncovered window per recording
        ids = np.array([r["window_id"] for r in rows])
        np.savez(out / f"{rec}.npz", window_id=ids, covered=covered, availability=avail,
                 availability_names=np.array(AVAILABILITY),
                 livox_avia_all=rng.normal(size=(n, 39)).astype(np.float32),
                 livox_mid360_fg=rng.normal(size=(n, 39)).astype(np.float32),
                 radar_fg=rng.normal(size=(n, 49)).astype(np.float32),
                 audio_stats=rng.normal(size=(n, 4, 128)).astype(np.float32))
        np.savez(out / f"{rec}.deep.npz", window_id=ids, camera_avg=camera, camera_max=camera,
                 audio_ast=ast.astype(np.float16))
    return out


def test_load_keeps_covered_windows_and_all_sensors(step2_dir, windows):
    data = load_step2(step2_dir, windows)
    assert len(data) == len(windows[::6]) - 12                  # one uncovered window per recording
    assert set(data.x) == set(SENSORS) and data.x["camera"].shape[1] == 1536
    assert data.x["audio1"].shape[1] == 768 + 128
    v2 = data.version != "V1"
    assert (data.counts[v2, SENSORS.index("radar")] == 0).all()
    gm = group_presence(presence(data.counts, 1))
    assert not gm[v2, 4].any() and gm[:, 0].all()                # radar group absent at V2/V3, camera always


def test_unknown_types_never_reach_train_or_val(step2_dir, windows):
    data = load_step2(step2_dir, windows)
    for protocol in ("loo_Avata", "v2_to_v1"):
        task = make_task(data, windows, protocol)
        assert (task.y[task.train] >= 0).all() and (task.y[task.val] >= 0).all()
        unknown_test = task.y[task.test] < 0
        assert unknown_test.any()
        assert set(data.drone_type[task.test][unknown_test]) == set(task.unknown)


def test_augmentation_keeps_one_sensor():
    g = torch.Generator().manual_seed(0)
    mask = torch.rand(500, len(SENSORS), generator=g) < 0.3
    mask[:, 0] |= ~mask.any(1)
    pool = torch.zeros(50, len(SENSORS), dtype=torch.bool)          # patterns that would remove everything
    for kind in ("pattern", "iid", "mix"):
        new = augment_mask(mask, kind, 0.9, pool, g)
        assert new.any(1).all() and not (new & ~mask).any()      # never adds a sensor, never empties a window


def test_auroc_and_fpr():
    assert auroc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auroc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert auroc(np.array([1.0, 1.0]), np.array([1.0])) == 0.5
    assert fpr_at_95(np.arange(100.0), np.array([-1.0, 50.0])) == 0.5


def test_end_to_end_runs_resume_and_report(step2_dir, windows, tmp_path):
    data = load_step2(step2_dir, windows)
    settings = TrainSettings(epochs=4, batch_size=128)
    for protocol in ("closed", "loo_Avata"):
        run_protocol(data, windows, protocol, [0], tmp_path, settings, "cpu", log=lambda *a: None)
    runs = tmp_path / "runs"
    threshold = json.loads((runs / "closed_threshold.json").read_text())
    assert threshold["chosen"] in (1, 5, 25) and set(threshold["val_accuracy"]) == {"1", "5", "25"}
    stamp = (runs / "closed_s0_dronefuse.npz").stat().st_mtime
    run_protocol(data, windows, "closed", [0], tmp_path, settings, "cpu", log=lambda *a: None)
    assert (runs / "closed_s0_dronefuse.npz").stat().st_mtime == stamp     # finished runs are not repeated
    saved = np.load(runs / "closed_s0_dronefuse.npz")
    assert "subset_0" in saved.files and "perturb_pattern_drop" in saved.files
    summary = summarize(data, windows, tmp_path, protocols=("closed", "loo_Avata"), seeds=(0,))
    closed = summary["main"]["closed"]
    assert closed["dronefuse"]["known_accuracy"][0] > 0.6           # the camera carries the type
    assert closed["single_radar"]["known_accuracy"][0] < closed["single_camera"]["known_accuracy"][0]
    assert summary["main"]["loo_Avata"]["dronefuse"]["energy_auroc"][0] is not None
    assert summary["subsets"]["closed"]["dronefuse"]["camera"][0] is not None
    report = format_step3_report(summary, data)
    assert "DroneFuse" in report and "LMMUAVC-style" in report
    tables = write_tables(summary, tmp_path / "tables")
    assert "tab_main.tex" in tables and r"\toprule" in (tmp_path / "tables" / "tab_main.tex").read_text()
