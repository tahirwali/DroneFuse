"""
Checks for F1 (dronefuse.figures) on synthetic saved outputs: every figure is written as PDF, PNG and CSV and
listed in the manifest with its inputs; the DroneRFc-MM refit must reproduce the saved T3 result (and refuses
to draw when it does not); the figures read the Step 1, Step 2 and T3 formats.
"""
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
from dronefuse import extra, figures
from dronefuse.windows import save_windows

ROOT = Path(__file__).resolve().parents[1]
QUIET = lambda *a: None


def _written(out, name):
    m = json.loads((out / "figures_manifest.json").read_text())["figures"][name]
    assert all((out / f).exists() for f in m["files"]) and m["values_plotted"] > 0
    return m


def test_recording_timeline_lists_every_block_with_its_inputs(tmp_path):
    F = figures.Figures(tmp_path, log=QUIET)
    figures.fig_timeline(F, ROOT / "splits" / "windows.csv")
    m = _written(tmp_path, "fig_mmaud_recordings")
    assert m["inputs"][0]["sha256"] and m["values_plotted"] >= 12 * 9           # 12 recordings x ~10 blocks
    rows = (tmp_path / "fig_mmaud_recordings.csv").read_text().splitlines()
    assert rows[0].startswith("recording,drone_type,block,split")


def _rfcmm_like(seed=1, n=60):
    rng = np.random.default_rng(seed)
    sessions = ["A1", "A1-2", "B1", "C1", "E1", "E1-2", "F1", "G1", "G1-2"]
    rows = [(s, k) for s in sessions for k in range(n)]
    session = np.array([r[0] for r in rows])
    order = np.array([r[1] for r in rows], float)
    off = {s: rng.normal(size=8) * 2 for s in sessions}
    cen = {s.split("-")[0]: rng.normal(size=8) for s in sessions}
    x = {k: np.stack([cen[s.split("-")[0]] + off[s] + rng.normal(size=8) for s, _ in rows]).astype(np.float32)
         for k in ("ptz", "fisheye", "audio", "radar")}
    split, block = extra.time_blocks(session, order)
    return {"x": x, "type": np.array([s.split("-")[0] for s in session]), "session": session, "split": split,
            "block": block, "rows": len(rows), "segments": len(rows), "missing": {}}


def test_rfcmm_refit_reproduces_t3_and_draws_its_figures(tmp_path):
    d = _rfcmm_like()
    saved = extra.run_rfcmm(d, n_boot=10, log=QUIET)
    refit, checks = figures.rfcmm_refit(d, saved)
    assert len(checks) == 6 and all("refit = saved" in c for c in checks)
    F = figures.Figures(tmp_path, log=QUIET)
    figures.fig_confusion_rfcmm(F, refit, checks, ["cache"])
    figures.fig_scores_rfcmm(F, refit, checks, ["cache"])
    figures.fig_embedding_rfcmm(F, d, "cache")
    for name in ("fig_confusion_rfcmm", "fig_scores_rfcmm", "fig_embedding_rfcmm"):
        _written(tmp_path, name)
    m = _written(tmp_path, "fig_embedding_rfcmm")
    assert any("same session" in c for c in m["checks"])


def test_rfcmm_refit_refuses_a_result_it_cannot_reproduce():
    d = _rfcmm_like()
    saved = extra.run_rfcmm(d, n_boot=10, log=QUIET)
    saved["accuracy"]["cross_recording_1_to_2"]["concat"]["accuracy"] += 1.0
    with pytest.raises(ValueError, match="refit gives"):
        figures.rfcmm_refit(d, saved)


def test_tsms_embedding_reads_the_t3_feature_cache(tmp_path):
    cache = tmp_path / "t3" / "cache"
    (cache / "tsms").mkdir(parents=True)
    (cache / "t3_cache_marker.txt").write_text("x")
    rng = np.random.default_rng(0)
    for sensor in ("FMCW", "CW", "RF"):
        for dist in (2, 4, 6):
            targets = np.repeat(["Inspire 2", "Mavic 2 Pro", "Phantom 4 Pro"], 30)
            np.savez(cache / "tsms" / f"{sensor}_{dist:02d}m.npz", target=targets, distance=np.full(90, dist),
                     idx=np.tile(np.arange(30), 3), x=rng.normal(size=(90, 16)).astype(np.float16))
    F = figures.Figures(tmp_path / "out", log=QUIET)
    figures.fig_embedding_tsms(F, cache, per_target=50)
    m = _written(tmp_path / "out", "fig_embedding_tsms")
    assert len(m["inputs"]) == 9 and m["values_plotted"] == 3 * 3 * 50


def test_sensor_examples_read_a_step1_recording_and_tsms_packs(tmp_path):
    pytest.importorskip("soundfile")
    from PIL import Image
    from test_step2 import _windows, _write_recording
    folder = _write_recording(tmp_path / "V1_Mavic3")
    save_windows(_windows(), tmp_path / "windows.csv")
    F = figures.Figures(tmp_path / "out", log=QUIET)
    figures.fig_examples_mmaud(F, folder, tmp_path / "windows.csv", window_id="V1_Mavic3_00002")
    m = _written(tmp_path / "out", "fig_examples_mmaud")
    assert any(i["path"].endswith("camera.bin") for i in m["inputs"])
    for sensor in ("FMCW", "CW", "RF"):
        with tarfile.open(tmp_path / f"{sensor}_10m.tar", "w") as t:
            for target in ("Mavic 2 Pro", "Phantom 4 Pro"):
                buf = io.BytesIO()
                Image.fromarray(np.full((16, 16, 3), 90, np.uint8)).save(buf, format="PNG")
                info = tarfile.TarInfo(f"{target}/{target}_10m_000.png")
                info.size = buf.tell()
                buf.seek(0)
                t.addfile(info, buf)
    packs = extra.find_tsms([tmp_path])
    figures.fig_examples_more(F, None, packs)
    assert _written(tmp_path / "out", "fig_examples_more")["values_plotted"] == 6


def test_mmaud_probes_and_embedding_from_step2_features(tmp_path):
    pytest.importorskip("torch")
    from dronefuse.windows import load_windows
    from dronefuse.recordings import DRONE_TYPES
    from dronefuse.features import AVAILABILITY
    rows = load_windows(ROOT / "splits" / "windows.csv")
    out = tmp_path / "step2"
    out.mkdir()
    rng = np.random.default_rng(0)
    by_rec = {}
    for w in rows[::6]:
        by_rec.setdefault(w["recording"], []).append(w)
    for rec, rs in by_rec.items():
        n = len(rs)
        y = np.array([DRONE_TYPES.index(r["drone_type"]) for r in rs])
        flight = float(rec.endswith("s2"))
        camera = rng.normal(size=(n, 64)).astype(np.float32)
        camera[np.arange(n), y] += 3.0
        camera[:, 40] += 4 * flight                               # the flight shows in the camera
        ids = np.array([r["window_id"] for r in rs])
        avail = np.ones((n, len(AVAILABILITY)), np.int64)
        np.savez(out / f"{rec}.npz", window_id=ids, covered=np.ones(n, bool), availability=avail,
                 availability_names=np.array(AVAILABILITY),
                 livox_avia_all=rng.normal(size=(n, 39)).astype(np.float32),
                 livox_mid360_all=rng.normal(size=(n, 39)).astype(np.float32) + flight,
                 radar_all=rng.normal(size=(n, 49)).astype(np.float32),
                 audio_stats=rng.normal(size=(n, 4, 128)).astype(np.float32))
        np.savez(out / f"{rec}.deep.npz", window_id=ids, camera_avg=camera, camera_max=camera,
                 audio_ast=rng.normal(size=(n, 4, 32)).astype(np.float16))
    mm = figures.mmaud_probes(out, ROOT / "splits" / "windows.csv", log=QUIET)
    assert mm["recording"]["camera (avg pool)"]["balanced_accuracy"] > 0.9
    assert mm["type"]["camera (avg pool)"]["val_balanced_accuracy"] > 0.7
    F = figures.Figures(tmp_path / "out", log=QUIET)
    figures.fig_probes(F, mm, {}, [out])
    figures.fig_embedding_mmaud(F, out, ROOT / "splits" / "windows.csv")
    _written(tmp_path / "out", "fig_probes")
    m = _written(tmp_path / "out", "fig_embedding_mmaud")
    cam = next(c for c in m["checks"] if c.startswith("Camera"))
    assert "same session" in cam


def test_find_inputs_locates_each_saved_output(tmp_path):
    (tmp_path / "s3" / "runs").mkdir(parents=True)
    np.savez(tmp_path / "s3" / "runs" / "closed_s0_dronefuse.npz", test=np.zeros(1))
    (tmp_path / "s3" / "runs_crop").mkdir()
    (tmp_path / "t3" / "cache").mkdir(parents=True)
    (tmp_path / "t3" / "t3_results.json").write_text("{}")
    (tmp_path / "t3" / "cache" / "t3_cache_marker.txt").write_text("x")
    found = figures.find_inputs(tmp_path, ROOT)
    assert found["step3"].endswith("s3") and found["step3_crop"]
    assert found["t3_json"].endswith("t3_results.json") and found["t3_cache"].endswith("cache")
    assert found["windows_csv"].endswith("windows.csv") and found["step2"] is None and found["step1"] == {}
    assert any("NOT ATTACHED" in line for line in figures.describe_inputs(found))


def test_two_t3_runs_are_merged(tmp_path):
    """DroneRFc-MM from one T3 run and TSMS-Drone from another: both results and both caches are used."""
    import json
    a, b = tmp_path / "t3-old" / "t3", tmp_path / "t3-new" / "t3"
    (a / "cache").mkdir(parents=True)
    (b / "cache" / "tsms").mkdir(parents=True)
    (a / "cache" / "t3_cache_marker.txt").write_text("x")
    (b / "cache" / "t3_cache_marker.txt").write_text("x")
    (a / "cache" / "rfcmm_features.pkl").write_bytes(b"x")
    np.savez(b / "cache" / "tsms" / "RF_02m.npz", x=np.zeros(1))
    (a / "t3_results.json").write_text(json.dumps({"results": {"rfcmm": {"v": 1}}}))
    (b / "t3_results.json").write_text(json.dumps({"results": {"tsms": {"v": 2}, "rfcmm": {"v": 3}, "x": None}}))
    found = figures.find_inputs(tmp_path, ROOT)
    assert found["t3_cache"].startswith(str(a)) and found["t3_cache_tsms"].startswith(str(b))
    res, used = figures.load_t3(found)
    assert set(res) == {"rfcmm", "tsms"} and used == [str(b / "t3_results.json")]   # the larger file has both
    assert res["rfcmm"]["v"] == 3
    (b / "t3_results.json").write_text(json.dumps({"results": {"tsms": {"v": 2}}}))  # each run has one dataset
    res, used = figures.load_t3(figures.find_inputs(tmp_path, ROOT))
    assert res == {"rfcmm": {"v": 1}, "tsms": {"v": 2}} and len(used) == 2
