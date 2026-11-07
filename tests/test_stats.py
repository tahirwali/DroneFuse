"""
Checks for dronefuse.stats on synthetic saved runs: the dropped-window search, presence read from the
single-group runs, accuracy where a group is present, and the block bootstrap.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronefuse import stats
from dronefuse.windows import load_windows, protocol_sets

ROOT = Path(__file__).resolve().parents[1]
ROWS = load_windows(ROOT / "splits" / "windows.csv")


def _write_runs(folder, protocol, test_rows, known, drop_index=None, present_radar=None):
    """Saved runs whose fusion models are right on every known window, as Step 3 would write them."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = [r for i, r in enumerate(test_rows) if i != drop_index]
    y = np.array([known.index(r["drone_type"]) if r["drone_type"] in known else -1 for r in rows])
    val = stats._ordered(protocol_sets(ROWS, protocol)["val"])
    yv = np.array([known.index(r["drone_type"]) for r in val])
    onehot = lambda labels: np.eye(len(known), dtype=np.float32)[np.clip(labels, 0, None)] * 5
    for method in stats.FUSION + list(stats.SINGLE.values()):
        for seed in stats.SEEDS:
            test, v = onehot(y), onehot(yv)
            if method == "single_radar":
                test[~present_radar] = 0
            np.savez(folder / f"{protocol}_s{seed}_{method}.npz", test=test, val=v)
            acc = float((stats._predict_class(test)[y >= 0] == y[y >= 0]).mean())
            (folder / f"{protocol}_s{seed}_{method}.json").write_text(json.dumps({"metrics": {"known_accuracy": acc}}))
    return y


def test_dropped_window_is_found_by_its_recording(tmp_path):
    p = "xrec_s1_to_s2"
    sets = protocol_sets(ROWS, p)
    test = stats._ordered(sets["test"])
    drop = next(i for i, r in enumerate(test) if r["recording"] == "V2_p4s2" and r["split"] == "train")
    n = len(test) - 1
    radar = np.zeros(n, bool)
    radar[::3] = True
    _write_runs(tmp_path, p, test, list(sets["known"]), drop, radar)
    # dropping a window of the wrong recording shifts the labels across a recording boundary, so only the
    # right recording reproduces the saved accuracies
    assert stats.find_dropped(ROWS, tmp_path, p).startswith("V2_p4s2_")


def test_presence_accuracy_and_bootstrap(tmp_path):
    p = "closed"
    sets = protocol_sets(ROWS, p)
    test = stats._ordered(sets["test"])
    radar = np.zeros(len(test), bool)
    radar[::4] = True
    _write_runs(tmp_path, p, test, list(sets["known"]), None, radar)
    d = stats.protocol_data(ROWS, tmp_path, p)
    assert d["present"]["radar"].sum() == radar.sum() and d["present"]["audio"].all()
    assert abs(stats.metric(d, "single_radar", "acc") - radar.mean()) < 1e-9      # missing counts as wrong
    assert stats.metric(d, "single_radar", "acc_present") == 1.0
    est, low, high = stats.bootstrap(d, lambda i: stats.metric(d, "single_radar", "acc", i), n=50)
    assert low <= est <= high and len(np.unique(d["block"])) == 24             # 12 recordings x 2 test blocks
