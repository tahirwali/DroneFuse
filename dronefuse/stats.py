"""
Statistics from the saved Step 3 logits, without retraining and without the Step 2 features.

- The test and validation windows of each protocol are rebuilt from splits/windows.csv in the order Step 3
  used (recording order of recordings.BY_NAME, then window id). Six training windows had no features in
  Step 2; they never reach the closed or leave-one-type-out test sets, and the one that reaches each
  cross-recording test set is found by `find_dropped`, which keeps the candidate that reproduces the saved
  accuracy of every run.
- Which sensor groups are present in a window is read from the single-group runs: a single-group model
  outputs all-zero logits where its group is missing.
- Confidence intervals: block bootstrap over the time blocks of the test recordings (each recording has
  ten blocks), so neighbouring, overlapping windows are resampled together. Each replicate averages the
  metric over the three seeds.
"""
import json
from pathlib import Path

import numpy as np

from .fusion import GROUP_NAMES, _predict_class, scores
from .recordings import BY_NAME
from .windows import load_windows, protocol_sets

FUSION = ["late", "concat", "lmmuavc", "dronefuse", "dronefuse_iid", "dronefuse_mix"]
SINGLE = {g: f"single_{g}" for g in GROUP_NAMES}
SEEDS = (0, 1, 2)


def _ordered(rows):
    order = {name: i for i, name in enumerate(BY_NAME)}
    return sorted(rows, key=lambda r: (order[r["recording"]], r["window_id"]))


def _load(runs, protocol, seed, method):
    z = np.load(Path(runs) / f"{protocol}_s{seed}_{method}.npz")
    meta = json.loads((Path(runs) / f"{protocol}_s{seed}_{method}.json").read_text())
    return z["test"].astype(np.float32), z["val"].astype(np.float32), meta


def auroc(known_scores, unknown_scores):
    """Same value as fusion.auroc (ties count half), computed with average ranks."""
    from scipy.stats import rankdata
    n1, n0 = len(known_scores), len(unknown_scores)
    ranks = rankdata(np.concatenate([known_scores, unknown_scores]))
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _accuracy(logits, y):
    known = y >= 0
    return float((_predict_class(logits)[known] == y[known]).mean())


def find_dropped(rows, runs, protocol, methods=("dronefuse", "concat", "lmmuavc", "late")):
    """The test window (if any) that Step 3 did not have: the single candidate whose removal reproduces the
    saved known accuracy of every run of `methods` and seeds. Candidates are training-split windows."""
    sets = protocol_sets(rows, protocol)
    test = _ordered(sets["test"])
    want = _load(runs, protocol, 0, "dronefuse")[0].shape[0]
    if want == len(test):
        return None
    if want != len(test) - 1:
        raise ValueError(f"{protocol}: {len(test) - want} windows missing, expected 0 or 1")
    known = list(sets["known"])
    y_all = np.array([known.index(r["drone_type"]) if r["drone_type"] in known else -1 for r in test])
    saved = [(_load(runs, protocol, s, m)[0], _load(runs, protocol, s, m)[2]["metrics"]["known_accuracy"])
             for m in methods for s in SEEDS]
    hits = []
    for i, r in enumerate(test):
        if r["split"] != "train":
            continue
        y = np.delete(y_all, i)
        if all(abs(_accuracy(t, y) - a) < 1e-9 for t, a in saved):
            hits.append(r["window_id"])
    # Labels are constant within a recording, so every training window of the right recording reproduces the
    # accuracies; the recording is identified, the window is not. One of them is dropped: at most one window
    # is then assigned to a neighbouring block, which does not matter for the block bootstrap.
    recordings = {h.rsplit("_", 1)[0] for h in hits}
    if len(recordings) != 1:
        raise ValueError(f"{protocol}: candidates in {len(recordings)} recordings reproduce the saved accuracies")
    return hits[0]


def protocol_data(rows, runs, protocol, dropped=()):
    """Labels (-1 unknown), recording, block and group presence of the test windows, labels and group counts
    of the validation windows, and the test/val logits of every method and seed."""
    sets = protocol_sets(rows, protocol)
    known = list(sets["known"])
    test = [r for r in _ordered(sets["test"]) if r["window_id"] not in set(dropped)]
    val = _ordered(sets["val"])
    lab = lambda rs: np.array([known.index(r["drone_type"]) if r["drone_type"] in known else -1 for r in rs])
    out = {"protocol": protocol, "known": known, "y": lab(test), "y_val": lab(val),
           "recording": np.array([r["recording"] for r in test]),
           "block": np.array([f"{r['recording']}/{r['block']}" for r in test]), "logits": {}}
    present, present_val = [], []
    for g, method in SINGLE.items():
        t, v, _ = _load(runs, protocol, 0, method)
        present.append(np.abs(t).sum(1) > 0)
        present_val.append(np.abs(v).sum(1) > 0)
    out["present"] = dict(zip(GROUP_NAMES, present))
    out["groups"] = np.sum(present, 0)
    out["groups_val"] = np.sum(present_val, 0)
    for method in FUSION + list(SINGLE.values()):
        out["logits"][method] = [_load(runs, protocol, s, method)[:2] for s in SEEDS]
        if out["logits"][method][0][0].shape[0] != len(test):
            raise ValueError(f"{protocol}/{method}: {out['logits'][method][0][0].shape[0]} test rows, "
                             f"rebuilt {len(test)}")
    return out


def metric(d, method, name, index=None):
    """Seed-averaged metric on the test windows `index` (all if None): 'acc' (known accuracy, a missing
    group counts as wrong), 'acc_present' (known windows where a single group is present) or an AUROC of
    'msp', 'energy' or 'avail_energy'."""
    index = np.arange(len(d["y"])) if index is None else index
    y = d["y"][index]
    values = []
    for t, v in d["logits"][method]:
        t = t[index]
        if name == "acc":
            values.append(_accuracy(t, y))
        elif name == "acc_present":
            group = method.replace("single_", "")
            sel = (y >= 0) & d["present"][group][index]
            values.append(float((_predict_class(t[sel]) == y[sel]).mean()) if sel.any() else np.nan)
        else:
            s = scores(t, d["groups"][index], v, d["groups_val"])[name]
            k = y >= 0
            values.append(auroc(s[k], s[~k]) if k.any() and (~k).any() else np.nan)
    return float(np.mean(values))


def bootstrap(d, fn, n=1000, seed=0):
    """(estimate, 2.5 %, 97.5 %) of fn(index) under a bootstrap of the test time blocks."""
    rng = np.random.default_rng(seed)
    blocks = np.unique(d["block"])
    members = {b: np.flatnonzero(d["block"] == b) for b in blocks}
    estimate = fn(np.arange(len(d["y"])))
    reps = []
    for _ in range(n):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        reps.append(fn(np.concatenate([members[b] for b in pick])))
    reps = np.array(reps, dtype=float)
    reps = reps[~np.isnan(reps)]
    return estimate, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def _rep_bootstrap(d, fn, n, seed):
    """The bootstrap replicates themselves (same block draws for the same seed), for paired comparisons."""
    rng = np.random.default_rng(seed)
    blocks = np.unique(d["block"])
    members = {b: np.flatnonzero(d["block"] == b) for b in blocks}
    return np.array([fn(np.concatenate([members[b] for b in rng.choice(blocks, size=len(blocks), replace=True)]))
                     for _ in range(n)], dtype=float)


def _ci(estimate, reps):
    reps = reps[~np.isnan(reps)]
    return {"estimate": round(100 * estimate, 1), "low": round(100 * float(np.percentile(reps, 2.5)), 1),
            "high": round(100 * float(np.percentile(reps, 97.5)), 1)}


def run_all(results_dir, windows_csv, n=1000, out=None, log=print):
    """Every interval used in the paper, from the saved Step 3 runs. Returns (and writes) a dict."""
    rows = load_windows(windows_csv)
    results_dir = Path(results_dir)
    res = {"n_bootstrap": n, "dropped": {}, "accuracy": {}, "present": {}, "paired": {}, "loo_auroc": {}}
    methods = FUSION + list(SINGLE.values())
    for variant, runs in (("full", results_dir / "runs"), ("crop", results_dir / "runs_crop")):
        data = {}
        for p in ("closed", "xrec_s1_to_s2", "xrec_s2_to_s1") + tuple(f"loo_{t}" for t in
                                                                      ("Avata", "M30", "M300", "Mavic2", "Mavic3", "Phantom4")):
            dropped = find_dropped(rows, runs, p) if p.startswith("xrec") else None
            res["dropped"][f"{variant}/{p}"] = dropped
            data[p] = protocol_data(rows, runs, p, [dropped] if dropped else [])
            log(f"{variant}/{p}: {len(data[p]['y'])} test windows")
        for p in ("closed", "xrec_s1_to_s2", "xrec_s2_to_s1"):
            d = data[p]
            reps = {m: _rep_bootstrap(d, lambda i, m=m: metric(d, m, "acc", i), n, 0) for m in methods}
            res["accuracy"][f"{variant}/{p}"] = {m: _ci(metric(d, m, "acc"), reps[m]) for m in methods}
            res["present"][f"{variant}/{p}"] = {
                m: {**_ci(metric(d, m, "acc_present"),
                          _rep_bootstrap(d, lambda i, m=m: metric(d, m, "acc_present", i), n, 0)),
                    "windows": int(((d["y"] >= 0) & d["present"][m.replace("single_", "")]).sum())}
                for m in SINGLE.values()}
            pairs = [("dronefuse", "lmmuavc"), ("dronefuse", "dronefuse_iid"), ("dronefuse", "concat"),
                     ("concat", "single_audio"), ("lmmuavc", "single_audio"), ("dronefuse", "single_audio")]
            res["paired"][f"{variant}/{p}"] = {
                f"{a} - {b}": _ci(metric(d, a, "acc") - metric(d, b, "acc"), reps[a] - reps[b]) for a, b in pairs}
        loo = [p for p in data if p.startswith("loo_")]
        for m in ("dronefuse", "lmmuavc", "concat", "single_audio", "single_camera"):
            per = {p: _rep_bootstrap(data[p], lambda i, p=p, m=m: metric(data[p], m, "avail_energy", i), n, 1)
                   for p in loo}
            est = {p: metric(data[p], m, "avail_energy") for p in loo}
            res["loo_auroc"][f"{variant}/{m}"] = {**{p: _ci(est[p], per[p]) for p in loo},
                                                  "mean": _ci(np.mean(list(est.values())),
                                                              np.nanmean(np.stack(list(per.values())), 0))}
            log(f"{variant} LOO {m}: mean {res['loo_auroc'][f'{variant}/{m}']['mean']}")
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import sys
    run_all(sys.argv[1], sys.argv[2], n=int(sys.argv[4]) if len(sys.argv) > 4 else 1000, out=sys.argv[3])


LABEL = {"late": "Late fusion", "concat": "Feature concatenation", "lmmuavc": "LMMUAVC-style",
         "dronefuse": "DroneFuse", "dronefuse_iid": "DroneFuse (i.i.d.)", "dronefuse_mix": "DroneFuse (mixed)",
         "single_camera": "Camera only", "single_audio": "Audio only", "single_avia": "Avia only",
         "single_mid360": "Mid-360 only", "single_radar": "Radar only"}


def _cell(c):
    return f"{c['estimate']:.1f} [{c['low']:.1f}, {c['high']:.1f}]"


def latex_tables(res, out_dir):
    """tab_xrec_ci.tex, tab_present.tex, tab_paired.tex and loo_means.json from the run_all result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [f"{v}/{p}" for v in ("full", "crop") for p in ("closed", "xrec_s1_to_s2", "xrec_s2_to_s1")]
    order = list(SINGLE.values()) + FUSION
    lines = []
    for m in order:
        if m == FUSION[0]:
            lines.append(r"\midrule")
        lines.append(f"{LABEL[m]} & " + " & ".join(_cell(res["accuracy"][c][m]) for c in cols) + r" \\")
    (out_dir / "tab_xrec_ci.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for m in SINGLE.values():
        cells = []
        for c in cols:
            x = res["present"][c][m]
            cells.append(f"{x['estimate']:.1f} ({x['windows']})")
        lines.append(f"{LABEL[m].replace(' only', '')} & " + " & ".join(cells) + r" \\")
    (out_dir / "tab_present.tex").write_text("\n".join(lines) + "\n")
    pairs = list(next(iter(res["paired"].values())))
    lines = []
    for pair in pairs:
        a, b = pair.split(" - ")
        lines.append(f"{LABEL[a]} $-$ {LABEL[b]} & " + " & ".join(_cell(res["paired"][c][pair]) for c in cols) + r" \\")
    (out_dir / "tab_paired.tex").write_text("\n".join(lines) + "\n")
    (out_dir / "loo_means.json").write_text(json.dumps({k: v["mean"] for k, v in res["loo_auroc"].items()}, indent=1))
