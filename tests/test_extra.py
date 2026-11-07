"""
Checks for dronefuse.extra on synthetic data: time blocks, the TSMS join and the full TSMS and DroneRFc-MM
experiment runs (probes, within vs cross-session accuracy, leave-one-out rejection).
"""
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronefuse import extra

pytest.importorskip("sklearn")


def test_time_blocks_follow_the_mmaud_pattern():
    session = np.array(["a"] * 20 + ["b"] * 10)
    split, block = extra.time_blocks(session, np.arange(30))
    assert list(split[:10]) == [s for s in extra.SPLIT_PATTERN for _ in range(2)][:10]
    assert len(set(block)) == 20


def _tsms_like(rng, session_effect):
    """Types differ a little; every session (target, distance) adds its own offset of size session_effect."""
    types = ["Corner Reflector", "Inspire 2", "Mavic 2 Pro", "Phantom 4 Pro"]
    rows = []
    for t in types:
        for d in range(2, 31, 2):
            for i in range(40):
                rows.append((t, d, i))
    target = np.array([r[0] for r in rows]); distance = np.array([r[1] for r in rows]); idx = np.array([r[2] for r in rows])
    centre = {t: rng.normal(size=16) for t in types}
    offset = {(t, d): rng.normal(size=16) * session_effect for t in types for d in range(2, 31, 2)}
    x = {s: np.stack([centre[t] * 0.3 + offset[(t, d)] + rng.normal(size=16) for t, d, _ in rows]).astype(np.float32)
         for s in ("FMCW", "CW", "RF")}
    session = np.array([f"{t}@{d}m" for t, d in zip(target, distance)])
    split, block = extra.time_blocks(session, idx)
    return {"x": x, "type": target, "session": session, "distance": distance, "split": split, "block": block,
            "counts": {s: len(rows) for s in x}, "rows": len(rows)}


def test_tsms_run_shows_the_session_effect():
    d = _tsms_like(np.random.default_rng(0), session_effect=2.0)
    r = extra.run_tsms(d, n_boot=20, log=lambda *a: None)
    within = r["accuracy"]["within"]["concat"]["accuracy"]
    cross = r["accuracy"]["cross_distance_a_to_b"]["concat"]["accuracy"]
    assert within > cross + 20                     # sessions are recognisable, types barely
    assert r["probes"]["FMCW"]["session"] > 50 and set(r["leave_one_out"]) == {"Inspire 2", "Mavic 2 Pro", "Phantom 4 Pro"}


def test_rfcmm_run_has_cross_recording_and_unknowns():
    rng = np.random.default_rng(1)
    sessions = ["A1", "A1-2", "B1", "C1", "E1", "E1-2", "F1", "G1", "G1-2"]
    rows = [(s, k) for s in sessions for k in range(60)]
    session = np.array([r[0] for r in rows]); order = np.array([r[1] for r in rows], float)
    off = {s: rng.normal(size=8) * 2 for s in sessions}
    cen = {s.split("-")[0]: rng.normal(size=8) for s in sessions}
    x = {k: np.stack([cen[s.split("-")[0]] + off[s] + rng.normal(size=8) for s, _ in rows]).astype(np.float32)
         for k in ("ptz", "fisheye", "audio", "radar")}
    split, block = extra.time_blocks(session, order)
    d = {"x": x, "type": np.array([s.split("-")[0] for s in session]), "session": session, "split": split,
         "block": block, "rows": len(rows), "segments": len(rows), "missing": {}}
    r = extra.run_rfcmm(d, n_boot=20, log=lambda *a: None)
    a = r["accuracy"]
    assert set(a) == {"within", "cross_recording_1_to_2", "cross_recording_2_to_1", "within_same_types"}
    assert a["cross_recording_1_to_2"]["_windows"]["test_unknown"] > 0 and "auroc_msp" in a["cross_recording_1_to_2"]["concat"]
    assert r["probes"]["audio"]["session_classes"] == 2 and len(r["leave_one_out"]) == 6


def test_tsms_features_reads_packed_tars(tmp_path):
    from PIL import Image
    for name in ("CW_02m.tar", "RF_02m.tar"):
        with tarfile.open(tmp_path / name, "w") as t:
            for i in range(3):
                buf = io.BytesIO(); Image.fromarray(np.full((8, 8, 3), i * 40, np.uint8)).save(buf, format="PNG")
                info = tarfile.TarInfo(f"Inspire 2/Inspire 2_2m_{i:03d}.png"); info.size = buf.tell(); buf.seek(0)
                t.addfile(info, buf)
    embed = lambda imgs: np.stack([np.full(768, im.mean(), np.float32) for im in imgs])
    per = extra.tsms_features([tmp_path], tmp_path / "cache", embed, log=lambda *a: None)
    d = extra.tsms_dataset(per)
    assert d["rows"] == 3 and set(d["x"]) == {"CW", "RF"} and d["x"]["CW"][2, 0] == 80
