"""
F1: the figures of the paper, drawn only from the saved outputs of the experiments.

No number is typed in. Every figure reads the files that the Kaggle runs wrote (Step 1 extracts, Step 2/2b
features, Step 3 runs, the T3 results and feature cache, the T1/T2 data), and every figure is saved as
  <name>.pdf   vector, for the paper
  <name>.png   for a quick look
  <name>.csv   exactly the values that are plotted
and listed in figures_manifest.json with the input files it read (SHA-256 of each) and the checks it passed.

Where a figure needs per-window predictions that the runs did not keep (DroneRFc-MM), the model is refitted
from the cached features with the same code as T3, and the figure is drawn only if the refit reproduces the
saved result. MMAUD figures use the saved Step 3 logits, after checking that they give the saved accuracies.

Style: Elsevier widths (90 mm single column, 190 mm full width), serif text, one colour-blind-checked palette
(blue / orange / aqua for up to three series; train / validation / test as in Figure 1), and a diverging
blue-grey-red scale centred on random guessing for AUROC maps.
"""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

MM = 1 / 25.4
SINGLE, DOUBLE = 90 * MM, 190 * MM
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)
CATEGORICAL = (BLUE, ORANGE, AQUA, "#eda100", "#e87ba4")          # palette slots 1-5, with marker shapes past 3
SPLIT_COLOUR = {"train": BLUE, "val": "#c3c2b7", "test": ORANGE}
INK, MUTED, GRID = "#2b2b2b", "#8a8a85", "#e6e6e3"
RED, NEUTRAL = "#e34948", "#f0efec"
MARKERS = ("o", "s", "^", "D", "v")
METHOD_NAMES = {"single_camera": "Camera only", "single_audio": "Audio only", "single_avia": "Avia only",
                "single_mid360": "Mid-360 only", "single_radar": "Radar only", "late": "Late fusion",
                "concat": "Concatenation", "lmmuavc": "LMMUAVC-style", "dronefuse": "DroneFuse",
                "dronefuse_iid": "DroneFuse (i.i.d.)", "dronefuse_mix": "DroneFuse (mixed)"}
SENSOR_NAMES = {"FMCW": "FMCW radar", "CW": "CW radar", "RF": "Radio receiver", "ptz": "PTZ camera",
                "fisheye": "Fisheye camera", "audio": "Microphones", "radar": "mmWave radar",
                "concat": "Concatenation", "late": "Late fusion"}
SITE = {"V1": "rooftop", "V2_V3": "car park"}


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    pyplot.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 100})
    return pyplot


def _clean(ax, grid="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def sha256(path, limit=256 * 2 ** 20):
    """SHA-256 of a file; for files above `limit` bytes, of the first and last 8 MB plus the size (marked)."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= limit:
            for chunk in iter(lambda: f.read(2 ** 20), b""):
                h.update(chunk)
            return h.hexdigest()
        h.update(f.read(8 * 2 ** 20))
        f.seek(size - 8 * 2 ** 20)
        h.update(f.read())
        h.update(str(size).encode())
        return "partial:" + h.hexdigest()


class Figures:
    """Saves figures with their data and keeps the manifest."""

    def __init__(self, out_dir, code_version=None, log=print):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.hashes = {}
        self.manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code_version": code_version,
                         "figures": {}}

    def files(self, paths):
        out = []
        for p in paths:
            p = str(p)
            if p not in self.hashes and Path(p).is_file():
                self.hashes[p] = sha256(p)
            out.append({"path": p, "sha256": self.hashes.get(p)})
        return out

    def save(self, fig, name, rows, inputs, checks=(), note=""):
        pyplot = plt()
        fig.savefig(self.out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.out / f"{name}.png", dpi=200, bbox_inches="tight")
        pyplot.close(fig)
        rows = list(rows)
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(self.out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        self.manifest["figures"][name] = {"files": [f"{name}.pdf", f"{name}.png"] + ([f"{name}.csv"] if rows else []),
                                          "values_plotted": len(rows), "inputs": self.files(inputs),
                                          "checks": list(checks), "note": note}
        (self.out / "figures_manifest.json").write_text(json.dumps(self.manifest, indent=1))
        self.log(f"  wrote {name} ({len(rows)} values, {len(inputs)} input files, checks: {list(checks) or 'none'})")
        return name


# ============================================================================= finding the inputs
def find_inputs(root, code_root=None):
    """Where each saved output is attached (None if it is not)."""
    root = Path(root)
    found = {"root": str(root)}
    code_root = Path(code_root) if code_root else None
    csvs = ([code_root / "splits" / "windows.csv"] if code_root else []) + sorted(root.rglob("splits/windows.csv"))
    found["windows_csv"] = next((str(p) for p in csvs if p.exists()), None)
    runs = sorted({p.parent for p in root.rglob("closed_s0_dronefuse.npz") if p.parent.name == "runs"},
                  key=lambda p: -len(list(p.glob("*.npz"))))
    found["step3"] = str(runs[0].parent) if runs else None
    found["step3_crop"] = bool(runs) and (runs[0].parent / "runs_crop").is_dir()
    deep = sorted({p.parent for p in root.rglob("*.deep.npz")})
    found["step2"] = str(deep[0]) if deep else None
    crop = sorted({p.parent for p in root.rglob("*.crop.npz")})
    found["step2b"] = str(crop[0]) if crop else None
    from .step2 import find_recordings
    found["step1"] = {k: str(v) for k, v in find_recordings(root).items()}
    t3 = sorted(root.rglob("t3_results.json"), key=lambda p: -p.stat().st_size)
    found["t3_json"] = str(t3[0]) if t3 else None
    caches = sorted({p.parent for p in root.rglob("t3_cache_marker.txt")})
    found["t3_cache"] = str(caches[0]) if caches else None
    from .extra import find_tsms
    found["tsms_packs"] = {k: str(v) for k, v in find_tsms([root]).items()}
    rf = sorted({p.parent.parent.parent for p in root.rglob("PTZ_meta.yaml")})
    found["rfcmm_root"] = str(rf[0]) if rf else None
    return found


def describe_inputs(found):
    lines = []
    for key in ("windows_csv", "step3", "step2", "step2b", "t3_json", "t3_cache", "rfcmm_root"):
        lines.append(f"{key:12s} {found.get(key) or 'NOT ATTACHED'}")
    lines.append(f"{'step1':12s} {len(found['step1'])} recordings: {sorted(found['step1']) or 'NOT ATTACHED'}")
    lines.append(f"{'tsms':12s} {len(found['tsms_packs'])} packs")
    return lines


# ============================================================================= Figure: MMAUD recordings
def fig_timeline(F, windows_csv):
    """Every MMAUD recording on its own time axis, its ten time blocks coloured by split."""
    from .recordings import BY_NAME
    from .windows import load_windows
    rows = load_windows(windows_csv)
    pyplot = plt()
    names = [n for n in BY_NAME if any(r["recording"] == n for r in rows)]
    fig, ax = pyplot.subplots(figsize=(DOUBLE, 0.2 * len(names) + 0.9))
    data = []
    flight_no = {}
    for k, name in enumerate(names):
        mine = [r for r in rows if r["recording"] == name]
        t0 = min(float(r["t_start"]) for r in mine)
        y = len(names) - 1 - k
        for b in sorted({int(r["block"]) for r in mine}):
            part = [r for r in mine if int(r["block"]) == b]
            s, e = min(float(r["t_start"]) for r in part) - t0, max(float(r["t_end"]) for r in part) - t0
            ax.barh(y, e - s, left=s, height=0.62, color=SPLIT_COLOUR[part[0]["split"]], edgecolor="white", lw=0.8)
            data.append({"recording": name, "drone_type": part[0]["drone_type"], "block": b,
                         "split": part[0]["split"], "start_s": round(s, 2), "end_s": round(e, 2), "windows": len(part)})
        rec = BY_NAME[name]
        key = (rec.version, rec.drone_type)
        flight_no[key] = flight_no.get(key, 0) + 1
        ax.text(-4, y, f"{rec.drone_type}, {SITE[rec.version]}, flight {flight_no[key]}", ha="right", va="center",
                fontsize=7, color=INK)
        ax.text(max(float(r["t_end"]) for r in mine) - t0 + 3, y, f"{len(mine)}", ha="left", va="center",
                fontsize=6.5, color=MUTED)
    ax.set_yticks([])
    ax.set_xlabel("Time from the first analysis window of the recording (s)")
    ax.set_ylim(-0.6, len(names) - 0.4)
    _clean(ax)
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SPLIT_COLOUR[s], label=l) for s, l in
                       (("train", "Training block"), ("val", "Validation block"), ("test", "Test block"))],
              loc="lower right", frameon=False, ncol=3, bbox_to_anchor=(1, 1.0))
    ax.text(1.0, -0.17, "numbers: 1 s windows", transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)
    return F.save(fig, "fig_mmaud_recordings", data, [windows_csv],
                  checks=[f"{len(rows)} windows in {len(names)} recordings"])


# ============================================================================= Step 3 statistics
def mmaud_stats(step3_dir, windows_csv, n_boot, out_json, log=print):
    """The block-bootstrap intervals of the paper, recomputed from the saved Step 3 runs."""
    from . import stats
    return stats.run_all(step3_dir, windows_csv, n=n_boot, out=out_json, log=log)


def _dot_panel(ax, labels, series, chance, chance_label, xlabel):
    """Dot-and-interval rows: series = [(name, colour, marker, offset, [(est, low, high) per label])]."""
    n = len(labels)
    for name, colour, marker, dy, values in series:
        for j, (est, lo, hi) in enumerate(values):
            if est is None:
                continue
            y = n - 1 - j + dy
            ax.plot([lo, hi], [y, y], color=colour, lw=1.1, alpha=0.85, solid_capstyle="round")
            ax.plot(est, y, marker=marker, ms=5, color=colour, mec="white", mew=0.7, ls="none",
                    label=name if j == 0 else None)
    ax.axvline(chance, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(chance + 1.2, -0.75, chance_label, fontsize=6.5, color=MUTED, va="center")
    ax.set_ylim(-1.0, n - 0.5)
    ax.set_xlim(0, 100)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel(xlabel)
    _clean(ax)


def fig_within_vs_cross_mmaud(F, res, inputs):
    methods = ["single_camera", "single_audio", "late", "concat", "lmmuavc", "dronefuse"]
    pyplot = plt()
    fig, axes = pyplot.subplots(1, 2, figsize=(DOUBLE, 2.9), sharey=True)
    data = []
    prot = [("closed", "Same flights (time blocks)", BLUE, "o", 0.0),
            ("xrec_s1_to_s2", "New flight, 1 to 2", ORANGE, "^", 0.22),
            ("xrec_s2_to_s1", "New flight, 2 to 1", ORANGE, "v", -0.22)]
    for ax, variant, title in zip(axes, ("full", "crop"),
                                  ("Whole-frame camera", "Camera used only when it sees the drone")):
        series = []
        for p, name, colour, marker, dy in prot:
            vals = []
            for m in methods:
                c = res["accuracy"][f"{variant}/{p}"][m]
                vals.append((c["estimate"], c["low"], c["high"]))
                data.append({"camera": variant, "protocol": p, "method": m, "accuracy": c["estimate"],
                             "ci_low": c["low"], "ci_high": c["high"]})
            series.append((name, colour, marker, dy, vals))
        _dot_panel(ax, [METHOD_NAMES[m] for m in methods], series, 100 / 3, "random guessing, new flight",
                   "Accuracy on known drone types (%)")
        ax.set_title(title, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return F.save(fig, "fig_within_vs_cross", data, inputs,
                  checks=["intervals recomputed from the saved runs (block bootstrap)"])


def fig_within_vs_cross_more(F, t3, inputs):
    pyplot = plt()
    plan = [("rfcmm", "DroneRFc-MM (types A1, E1, G1)", "within_same_types", "cross_recording_1_to_2",
             "cross_recording_2_to_1", ("Same flights", "New flight, 1 to 2", "New flight, 2 to 1"), 100 / 3),
            ("tsms", "TSMS-Drone (five targets)", "within", "cross_distance_a_to_b", "cross_distance_b_to_a",
             ("Same distances", "New distances, even to odd", "New distances, odd to even"), 20.0)]
    plan = [p for p in plan if p[0] in t3]
    fig, axes = pyplot.subplots(1, len(plan), figsize=(DOUBLE, 2.7), squeeze=False)
    data = []
    for ax, (key, title, w, a, b, names, chance) in zip(axes[0], plan):
        r = t3[key]
        methods = sorted(r["probes"]) + ["concat", "late"]
        series = []
        for prot, name, colour, marker, dy in ((w, names[0], BLUE, "o", 0.0), (a, names[1], ORANGE, "^", 0.22),
                                               (b, names[2], ORANGE, "v", -0.22)):
            vals = []
            for m in methods:
                c = r["accuracy"][prot][m]
                vals.append((c["accuracy"], c["ci"][0], c["ci"][1]))
                data.append({"dataset": key, "protocol": prot, "method": m, "accuracy": c["accuracy"],
                             "ci_low": c["ci"][0], "ci_high": c["ci"][1]})
            series.append((name, colour, marker, dy, vals))
        _dot_panel(ax, [SENSOR_NAMES.get(m, m) for m in methods], series, chance, "random guessing",
                   "Accuracy on known types (%)")
        ax.set_title(title, color=INK)
        ax.legend(loc="upper center", frameon=False, bbox_to_anchor=(0.4, -0.2), ncol=3, handletextpad=0.2,
                  columnspacing=0.8)
    fig.tight_layout()
    return F.save(fig, "fig_within_vs_cross_more", data, inputs, checks=["values read from t3_results.json"])


# ============================================================================= Figure: probes
MMAUD_PROBE_BLOCKS = [("camera (avg pool)", "Camera, whole frame"), ("crop, drone seen (avg)", "Camera, gated crop"),
                      ("Livox Avia", "Livox Avia LiDAR"), ("Mid-360 (all)", "Mid-360 LiDAR"),
                      ("radar (all)", "mmWave radar"), ("audio log-mel stats", "Audio, log-mel"),
                      ("audio AST", "Audio, AST")]


def mmaud_probes(step2_dir, windows_csv, crop_dir=None, log=print):
    """Type probe (closed protocol, train -> val) and recording probe (flight 1 vs 2, train -> val) per sensor,
    with the Step 2 / 2b code."""
    from .recordings import BY_NAME
    from .step2 import linear_probe, recording_probe
    from .windows import load_windows
    rows = load_windows(windows_csv)
    names = [n for n in BY_NAME if (Path(step2_dir) / f"{n}.npz").exists()]
    typ = linear_probe(step2_dir, rows, names, log=log, crop_dir=crop_dir)
    rec = recording_probe(step2_dir, rows, names, crop_dir=crop_dir, log=log)
    return {"type": typ, "recording": rec, "recordings": names}


def fig_probes(F, mm, t3, inputs):
    """How well each sensor names the drone type and the session, on the three datasets."""
    pyplot = plt()
    panels = []
    if mm:
        rows = []
        for block, label in MMAUD_PROBE_BLOCKS:
            t, r = (mm["type"] or {}).get(block), (mm["recording"] or {}).get(block)
            if t and "val_balanced_accuracy" in t and r:
                rows.append((label, 100 * t["val_balanced_accuracy"], 100 / 6, 100 * r["balanced_accuracy"], 50.0))
        panels.append(("MMAUD", "recording: flight 1 vs 2", rows))
    for key, title, what in (("rfcmm", "DroneRFc-MM", "session: flight 1 vs 2"),
                             ("tsms", "TSMS-Drone", "session: 15 distances")):
        if key in t3:
            r = t3[key]["probes"]
            panels.append((title, what, [(SENSOR_NAMES.get(s, s), r[s]["type"], 100 / r[s]["type_classes"],
                                          r[s]["session"], 100 / r[s]["session_classes"])
                                         for s in sorted(r) if r[s]["session"] is not None]))
    sizes = [max(len(p[2]), 1) for p in panels]
    fig, axes = pyplot.subplots(len(panels), 1, figsize=(SINGLE * 1.45, 0.33 * sum(sizes) + 0.75 * len(panels)),
                                gridspec_kw={"height_ratios": sizes}, sharex=True, squeeze=False)
    data = []
    for ax, (title, what, rows) in zip(axes[:, 0], panels):
        n = len(rows)
        for j, (label, typ, typ0, ses, ses0) in enumerate(rows):
            y = n - 1 - j
            for val, base, dy, colour, name in ((typ, typ0, 0.19, BLUE, "Drone type"),
                                                (ses, ses0, -0.19, ORANGE, "Flight or session")):
                ax.barh(y + dy, val, height=0.36, color=colour, label=name if j == 0 else None)
                ax.plot([base, base], [y + dy - 0.2, y + dy + 0.2], color=INK, lw=1.0,
                        label="random guessing" if j == 0 and colour == BLUE else None)
                ax.text(val + 1, y + dy, f"{val:.1f}", va="center", fontsize=6, color=INK)
            data.append({"dataset": title, "sensor": label, "type_probe": round(typ, 1), "type_random": round(typ0, 1),
                         "session_probe": round(ses, 1), "session_random": round(ses0, 1)})
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[0] for r in rows][::-1])
        ax.set_xlim(0, 108)
        ax.set_title(f"{title} ({what})", loc="left", color=INK)
        _clean(ax)
    axes[-1, 0].set_xlabel("Balanced accuracy on validation windows (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return F.save(fig, "fig_probes", data, inputs,
                  checks=["MMAUD probes recomputed with step2.linear_probe / recording_probe" if mm else
                          "MMAUD probes skipped (Step 2 features not attached)"])


# ============================================================================= Figure: training curves
def fig_training_curves(F, step3_dir):
    """Validation accuracy per epoch (from training flights) next to the test accuracy of the same models."""
    runs = Path(step3_dir) / "runs"
    pyplot = plt()
    methods = ["single_camera", "concat", "dronefuse"]
    prots = [("closed", "Test on the same flights"), ("xrec_s1_to_s2", "Test on a new flight (1 to 2)"),
             ("xrec_s2_to_s1", "Test on a new flight (2 to 1)")]
    fig, axes = pyplot.subplots(1, 3, figsize=(DOUBLE, 2.5), sharey=True)
    data, used = [], []
    for ax, (p, title) in zip(axes, prots):
        for colour, marker, m in zip(SERIES, MARKERS, methods):
            curves, tests = [], []
            for seed in (0, 1, 2):
                path = runs / f"{p}_s{seed}_{m}.json"
                if not path.exists():
                    continue
                info = json.loads(path.read_text())
                used.append(path)
                if "history" in info:
                    curves.append(info["history"]["val_accuracy_per_epoch"])
                    tests.append(info["metrics"]["known_accuracy"])
            if not curves:
                continue
            n = min(map(len, curves))
            curve = 100 * np.mean([c[:n] for c in curves], 0)
            test = 100 * float(np.mean(tests))
            epochs = np.arange(1, n + 1)
            ax.plot(epochs, curve, color=colour, lw=1.4, label=f"{METHOD_NAMES[m]}, validation")
            ax.plot(n + 5, test, marker=marker, ms=6, color=colour, mec="white", mew=0.7, ls="none")
            ax.plot([n, n + 5], [curve[-1], test], color=colour, lw=0.7, ls=(0, (2, 2)))
            for e, v in zip(epochs, curve):
                data.append({"protocol": p, "method": m, "epoch": int(e), "val_accuracy": round(float(v), 2)})
            data.append({"protocol": p, "method": m, "epoch": "test", "test_known_accuracy": round(test, 2),
                         "seeds": len(tests)})
        ax.set_title(title, color=INK)
        ax.set_xlabel("Training epoch")
        ax.set_xticks([1, 10, 20, 30, 40, 45])
        ax.set_xticklabels(["1", "10", "20", "30", "40", "test"])
        ax.set_ylim(0, 102)
        _clean(ax, "y")
    axes[0].set_ylabel("Accuracy (%)")
    from matplotlib.lines import Line2D
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", color=MUTED, ls="none", ms=5))
    l.append("test accuracy of the same model (marker shape = method)")
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return F.save(fig, "fig_training_curves", data, used,
                  checks=["validation curves and test accuracies read from the run files (mean of 3 seeds)"])


# ============================================================================= Figure: confusion matrices
def _confusion(y, pred, k):
    c = np.zeros((k, k), int)
    for a, b in zip(y, pred):
        c[a, b] += 1
    return c


def _matrix(ax, counts, labels, title):
    share = 100 * counts / np.maximum(counts.sum(1, keepdims=True), 1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])
    ax.imshow(share, cmap=cmap, vmin=0, vmax=100)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{share[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if share[i, j] > 55 else INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title, color=INK)
    ax.set_xlabel("Predicted type")
    for s in ax.spines.values():
        s.set_visible(False)
    return share


def fig_confusion_mmaud(F, step3_dir, windows_csv, method="dronefuse"):
    """Known test windows of the closed and cross-recording protocols, saved logits of three seeds."""
    from . import stats
    from .fusion import _predict_class
    from .windows import load_windows
    rows = load_windows(windows_csv)
    runs = Path(step3_dir) / "runs"
    pyplot = plt()
    prots = [("closed", "Same flights"), ("xrec_s1_to_s2", "New flight (1 to 2)"),
             ("xrec_s2_to_s1", "New flight (2 to 1)")]
    fig, axes = pyplot.subplots(1, 3, figsize=(DOUBLE, 2.75), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    data, checks, used = [], [], []
    for ax, (p, title) in zip(axes, prots):
        dropped = stats.find_dropped(rows, runs, p) if p.startswith("xrec") else None
        d = stats.protocol_data(rows, runs, p, [dropped] if dropped else [])
        k = len(d["known"])
        known = d["y"] >= 0
        total = np.zeros((k, k), int)
        for seed, (t, _) in enumerate(d["logits"][method]):
            pred = _predict_class(t)
            acc = float((pred[known] == d["y"][known]).mean())
            info = json.loads((runs / f"{p}_s{seed}_{method}.json").read_text())
            used += [runs / f"{p}_s{seed}_{method}.json", runs / f"{p}_s{seed}_{method}.npz"]
            saved = info["metrics"]["known_accuracy"]
            if abs(acc - saved) > 1e-9:
                raise ValueError(f"{p} seed {seed}: rebuilt accuracy {acc} differs from the saved {saved}")
            total += _confusion(d["y"][known], pred[known], k)
        checks.append(f"{p}: every seed reproduces the saved known accuracy")
        share = _matrix(ax, total, d["known"], f"{title}: {100 * np.trace(total) / total.sum():.1f}%")
        for i, a in enumerate(d["known"]):
            for j, b in enumerate(d["known"]):
                data.append({"protocol": p, "method": method, "true": a, "predicted": b,
                             "windows_3_seeds": int(total[i, j]), "row_percent": round(float(share[i, j]), 1)})
    axes[0].set_ylabel("True type")
    fig.tight_layout()
    return F.save(fig, "fig_confusion_mmaud", data, used, checks=checks,
                  note=f"{METHOD_NAMES[method]}, whole-frame camera; counts summed over three seeds")


# ============================================================================= DroneRFc-MM refits
def load_rfcmm_cache(cache_dir):
    import pickle
    path = Path(cache_dir) / "rfcmm_features.pkl"
    return pickle.loads(path.read_bytes()), path


def rfcmm_protocols(d):
    """The (train, test, known) index sets of extra.run_rfcmm."""
    from .extra import RFCMM_REPEATS
    sp = d["split"]
    sessions = set(d["session"])
    types = sorted(str(t) for t in set(d["type"]))
    pairs = [list(p) for p in RFCMM_REPEATS if all(q in sessions for q in p)]
    first, second = [p[0] for p in pairs], [p[1] for p in pairs]
    known = sorted({s.split("-")[0] for s in first})
    others = np.isin(d["type"], [t for t in types if t not in known]) & (sp == "test")
    out = {"within_same_types": (np.flatnonzero(np.isin(d["type"], known) & (sp == "train")),
                                 np.flatnonzero((np.isin(d["type"], known) & (sp == "test")) | others), known)}
    for name, src, dst in (("cross_recording_1_to_2", first, second), ("cross_recording_2_to_1", second, first)):
        out[name] = (np.flatnonzero(np.isin(d["session"], src) & (sp == "train")),
                     np.flatnonzero(np.isin(d["session"], dst) | others), known)
    return out


def rfcmm_refit(d, t3_rfcmm, methods=("fisheye", "concat")):
    """Probabilities of the T3 models on each protocol, refitted from the cached features; raises if the
    refit does not reproduce the saved accuracy and AUROC (to the 0.1 % they were saved with)."""
    from .extra import _auroc, fit_predict
    out, checks = {}, []
    for name, (train, test, known) in rfcmm_protocols(d).items():
        probs = fit_predict(d, train, test, known)
        y = np.array([known.index(t) if t in known else -1 for t in d["type"][test]])
        for m in methods:
            p = probs[m]
            k = y >= 0
            acc = round(100 * float((p.argmax(1)[k] == y[k]).mean()), 1)
            au = round(100 * _auroc(p.max(1)[k], p.max(1)[~k]), 1)
            saved = t3_rfcmm["accuracy"][name][m]
            if abs(acc - saved["accuracy"]) > 0.051 or abs(au - saved["auroc_msp"]) > 0.051:
                raise ValueError(f"{name}/{m}: refit gives accuracy {acc}, AUROC {au}; saved "
                                 f"{saved['accuracy']}, {saved['auroc_msp']}")
            checks.append(f"{name}/{m}: refit = saved (accuracy {acc}, AUROC {au})")
        out[name] = {"probs": probs, "y": y, "known": known, "test": test}
    return out, checks


def fig_confusion_rfcmm(F, refit, checks, inputs, method="concat"):
    pyplot = plt()
    titles = {"within_same_types": "Same flights", "cross_recording_1_to_2": "New flight (1 to 2)",
              "cross_recording_2_to_1": "New flight (2 to 1)"}
    fig, axes = pyplot.subplots(1, 3, figsize=(DOUBLE * 0.8, 2.5))
    data = []
    for ax, (name, title) in zip(axes, titles.items()):
        r = refit[name]
        k = r["y"] >= 0
        counts = _confusion(r["y"][k], r["probs"][method].argmax(1)[k], len(r["known"]))
        share = _matrix(ax, counts, r["known"], f"{title}: {100 * np.trace(counts) / counts.sum():.1f}%")
        for i, a in enumerate(r["known"]):
            for j, b in enumerate(r["known"]):
                data.append({"protocol": name, "method": method, "true": a, "predicted": b,
                             "windows": int(counts[i, j]), "row_percent": round(float(share[i, j]), 1)})
    axes[0].set_ylabel("True type")
    fig.tight_layout()
    return F.save(fig, "fig_confusion_rfcmm", data, inputs, checks=checks,
                  note=f"{SENSOR_NAMES[method]} on DroneRFc-MM, refitted from the T3 feature cache")


def fig_scores_rfcmm(F, refit, checks, inputs, methods=("fisheye", "concat")):
    """Proposition 2 in a picture: maximum class probability of known and unknown test windows, when the known
    windows come from the training flights and when they come from new flights."""
    pyplot = plt()
    titles = {"within_same_types": "Known windows from the training flights",
              "cross_recording_1_to_2": "Known windows from new flights (1 to 2)",
              "cross_recording_2_to_1": "Known windows from new flights (2 to 1)"}
    from .extra import _auroc
    fig, axes = pyplot.subplots(len(methods), 3, figsize=(DOUBLE, 1.7 * len(methods) + 0.5), sharex=True,
                                squeeze=False)
    bins = np.linspace(0, 1, 26)
    data = []
    for i, m in enumerate(methods):
        for j, (name, title) in enumerate(titles.items()):
            ax = axes[i, j]
            r = refit[name]
            s = r["probs"][m].max(1)
            k = r["y"] >= 0
            for sel, colour, label in ((k, BLUE, "Known types (A1, E1, G1)"), (~k, ORANGE, "Unknown types (B1, C1, F1)")):
                h, _ = np.histogram(s[sel], bins=bins)
                share = 100 * h / max(sel.sum(), 1)
                ax.stairs(share, bins, color=colour, lw=1.3, label=label if i == 0 and j == 0 else None,
                          fill=False)
                ax.stairs(share, bins, color=colour, alpha=0.12, fill=True)
                for lo, hi, v in zip(bins[:-1], bins[1:], share):
                    data.append({"method": m, "protocol": name, "group": "known" if colour == BLUE else "unknown",
                                 "bin_low": round(lo, 2), "bin_high": round(hi, 2), "percent_of_windows": round(float(v), 2)})
            au = 100 * _auroc(s[k], s[~k])
            ax.text(0.03, 0.92, f"AUROC {au:.1f}%", transform=ax.transAxes, fontsize=7, color=INK, va="top")
            if i == 0:
                ax.set_title(title, color=INK)
            if j == 0:
                ax.set_ylabel(f"{SENSOR_NAMES[m]}\nwindows (%)")
            if i == len(methods) - 1:
                ax.set_xlabel("Maximum class probability")
            _clean(ax, "y")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return F.save(fig, "fig_scores_rfcmm", data, inputs, checks=checks,
                  note="refitted from the T3 feature cache; AUROC = rejection of the unknown types")


# ============================================================================= Figure: leave-one-out AUROC maps
def _diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", [RED, "#f2a3a2", NEUTRAL, "#86b6ef", BLUE])


def _heat(ax, grid, rows, cols, title):
    im = ax.imshow(grid, cmap=_diverging(), vmin=0, vmax=100, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(title, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_loo_auroc(F, res, t3, inputs):
    """AUROC of rejecting each left-out drone type, per method; 50 = no better than guessing."""
    pyplot = plt()
    panels, data = [], []
    if res:
        types = ["Avata", "M30", "M300", "Mavic2", "Mavic3", "Phantom4"]
        methods = ["single_camera", "single_audio", "concat", "lmmuavc", "dronefuse"]
        for variant, title in (("full", "MMAUD, whole-frame camera"), ("crop", "MMAUD, gated camera")):
            grid = np.full((len(methods), len(types) + 1), np.nan)
            for i, m in enumerate(methods):
                r = res["loo_auroc"].get(f"{variant}/{m}")
                if not r:
                    continue
                for j, t in enumerate(types):
                    grid[i, j] = r[f"loo_{t}"]["estimate"]
                    data.append({"dataset": "MMAUD", "variant": variant, "method": m, "left_out": t,
                                 "auroc": grid[i, j], "ci_low": r[f"loo_{t}"]["low"], "ci_high": r[f"loo_{t}"]["high"]})
                grid[i, -1] = r["mean"]["estimate"]
            panels.append((grid, [METHOD_NAMES[m] for m in methods], types + ["mean"], title))
    if "rfcmm" in t3:
        r = t3["rfcmm"]["leave_one_out"]
        held = sorted(r)
        methods = sorted(t3["rfcmm"]["probes"]) + ["concat", "late"]
        grid = np.array([[r[h].get(m, np.nan) if r[h].get(m) is not None else np.nan for h in held] for m in methods],
                        float)
        grid = np.concatenate([grid, np.nanmean(grid, 1, keepdims=True)], 1)
        for i, m in enumerate(methods):
            for j, h in enumerate(held):
                data.append({"dataset": "DroneRFc-MM", "variant": "within", "method": m, "left_out": h, "auroc": grid[i, j]})
        panels.append((grid, [SENSOR_NAMES.get(m, m) for m in methods], held + ["mean"],
                       "DroneRFc-MM (test windows from training flights)"))
    if "tsms" in t3:
        r = t3["tsms"]["leave_one_out"]
        held = sorted(r)
        methods = sorted(t3["tsms"]["probes"]) + ["concat", "late"]
        for part, title in (("within", "TSMS-Drone, same distances"), ("cross", "TSMS-Drone, new distances")):
            grid = np.array([[r[h][part].get(m) if r[h][part].get(m) is not None else np.nan for h in held]
                             for m in methods], float)
            grid = np.concatenate([grid, np.nanmean(grid, 1, keepdims=True)], 1)
            for i, m in enumerate(methods):
                for j, h in enumerate(held):
                    data.append({"dataset": "TSMS-Drone", "variant": part, "method": m, "left_out": h,
                                 "auroc": grid[i, j]})
            panels.append((grid, [SENSOR_NAMES.get(m, m) for m in methods],
                           [h.replace(" Pro", "").replace("Matrice ", "M") for h in held] + ["mean"], title))
    ncol = 2
    nrow = (len(panels) + 1) // 2
    fig, axes = pyplot.subplots(nrow, ncol, figsize=(DOUBLE, 2.35 * nrow + 0.3), squeeze=False,
                                layout="constrained")
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.1, wspace=0.12, hspace=0.08)
    im = None
    for ax, (grid, rows, cols, title) in zip(axes.flat, panels):
        im = _heat(ax, grid, rows, cols, title)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.01, aspect=30)
        cb.set_label("AUROC (%); 50 = random")
        cb.outline.set_visible(False)
    return F.save(fig, "fig_loo_auroc", data, inputs,
                  checks=["MMAUD from the recomputed bootstrap; DroneRFc-MM and TSMS from t3_results.json"])


# ============================================================================= Figure: embedding maps
def _tsne(x, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(np.nan_to_num(x.astype(np.float64)))
    if x.shape[1] > 50:
        x = PCA(n_components=50, random_state=seed).fit_transform(x)
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=seed, learning_rate="auto").fit_transform(x)


def _neighbour_share(x, types, sessions, k=10):
    """In standardised PCA-50 space, (share of each window's k nearest neighbours with the same type, share of
    the same-type neighbours that also come from the same session). The second is about 1 / (sessions per type)
    when the features ignore the session, and 1 when they cluster by session."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    z = StandardScaler().fit_transform(np.nan_to_num(x.astype(np.float64)))
    if z.shape[1] > 50:
        z = PCA(n_components=50, random_state=0).fit_transform(z)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z, return_distance=False)[:, 1:]
    same_type = types[idx] == types[:, None]
    same_session = (sessions[idx] == sessions[:, None]) & same_type
    return float(same_type.mean()), float(same_session.sum() / max(same_type.sum(), 1))


def _embedding_figure(F, name, blocks, types, sessions, inputs, session_label, note, colour_by_value=None,
                      flight=None):
    """Two rows of t-SNE maps per feature block: coloured by drone type (top) and by flight number or distance
    (bottom). `sessions` identifies the recording session of each window (for the neighbour shares)."""
    pyplot = plt()
    n = len(blocks)
    fig, axes = pyplot.subplots(2, n, figsize=(DOUBLE, 2 * 1.75 + 0.6), squeeze=False)
    data, checks = [], []
    utypes = sorted(set(types))
    for j, (label, x) in enumerate(blocks):
        xy = _tsne(x)
        tn, sn = _neighbour_share(x, types, sessions)
        checks.append(f"{label}: nearest neighbours with the same type {100 * tn:.1f}%; of those, same session "
                      f"{100 * sn:.1f}%")
        ax = axes[0, j]
        for k, t in enumerate(utypes):
            sel = types == t
            ax.scatter(xy[sel, 0], xy[sel, 1], s=2.5 if len(utypes) <= 3 else 4, lw=0, alpha=0.7,
                       color=CATEGORICAL[k % len(CATEGORICAL)],
                       marker=MARKERS[k % len(MARKERS)] if len(utypes) > 3 else "o", label=t)
        ax.set_title(label, color=INK)
        ax.text(0.02, 0.02, f"same type {100 * tn:.0f}%", transform=ax.transAxes, fontsize=6,
                color=INK)
        ax = axes[1, j]
        if colour_by_value is not None:
            from matplotlib.colors import LinearSegmentedColormap
            ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
            sc = ax.scatter(xy[:, 0], xy[:, 1], s=2.5, lw=0, alpha=0.8, c=colour_by_value, cmap=ramp,
                            vmin=float(np.min(colour_by_value)), vmax=float(np.max(colour_by_value)))
        else:
            for k, s in enumerate(("flight 1", "flight 2")):
                sel = flight == s
                ax.scatter(xy[sel, 0], xy[sel, 1], s=2.5, lw=0, alpha=0.7, color=(BLUE, ORANGE)[k], label=s)
        ax.text(0.02, 0.02, f"same {session_label} {100 * sn:.0f}%",
                transform=ax.transAxes, fontsize=6, color=INK)
        for a in axes[:, j]:
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color(GRID)
        for i in range(len(xy)):
            data.append({"features": label, "window": i, "tsne_1": round(float(xy[i, 0]), 3),
                         "tsne_2": round(float(xy[i, 1]), 3), "type": types[i], "session": sessions[i]})
    axes[0, 0].set_ylabel("coloured by drone type")
    axes[1, 0].set_ylabel("coloured by " + ("distance" if colour_by_value is not None else "flight"))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, markerscale=4, bbox_to_anchor=(0.5, 1.04))
    if colour_by_value is not None:
        cb = fig.colorbar(sc, ax=axes[1, :].tolist(), shrink=0.8, pad=0.01)
        cb.set_label("distance (m)")
        cb.outline.set_visible(False)
    else:
        h, l = axes[1, 0].get_legend_handles_labels()
        fig.legend(h[:2], ["first flight", "second flight"], loc="lower center", ncol=2, frameon=False, markerscale=4,
                   bbox_to_anchor=(0.5, -0.03))
    return F.save(fig, name, data, inputs, checks=checks, note=note)


def fig_embedding_mmaud(F, step2_dir, windows_csv):
    """Car-park flights of M300, M30 and Phantom 4 (two flights each)."""
    from .step2 import load_block
    from .windows import FLIGHTS, load_windows
    rows = {r["window_id"]: r for r in load_windows(windows_csv)}
    names = [n for pair in zip(FLIGHTS["s1"], FLIGHTS["s2"]) for n in pair]
    covered, used = {}, []
    for n in names:
        cpu = np.load(Path(step2_dir) / f"{n}.npz")
        covered.update(zip((str(i) for i in cpu["window_id"]), cpu["covered"]))
        used += [Path(step2_dir) / f"{n}.npz", Path(step2_dir) / f"{n}.deep.npz"]
    blocks, ref = [], None
    for block, label in (("camera (avg pool)", "Camera, whole frame"), ("Mid-360 (all)", "Mid-360 LiDAR"),
                         ("audio AST", "Audio, AST"), ("radar (all)", "mmWave radar")):
        x, ids = load_block(step2_dir, names, block)
        if x is None:
            continue
        keep = np.array([bool(covered.get(i, False)) for i in ids])
        ids = [i for i, k in zip(ids, keep) if k]
        if ref is None:
            ref = ids
        elif ids != ref:
            raise ValueError(f"{block}: window order differs between feature blocks")
        blocks.append((label, x[keep]))
    types = np.array([rows[i]["drone_type"] for i in ref])
    sessions = np.array([rows[i]["recording"] for i in ref])
    flight = np.array(["flight 1" if s in FLIGHTS["s1"] else "flight 2" for s in sessions])
    return _embedding_figure(F, "fig_embedding_mmaud", blocks, types, sessions, used, "flight",
                             "MMAUD car park, every covered window of the six flights (Step 2 features)",
                             flight=flight)


def fig_embedding_rfcmm(F, d, cache_path):
    from .extra import RFCMM_REPEATS
    pairs = [p for p in RFCMM_REPEATS if all(q in set(d["session"]) for q in p)]
    keep = np.isin(d["session"], [s for p in pairs for s in p])
    blocks = [(SENSOR_NAMES[s], d["x"][s][keep]) for s in ("ptz", "fisheye", "audio", "radar") if s in d["x"]]
    sessions = d["session"][keep].astype(str)
    flight = np.array(["flight 2" if "-" in s else "flight 1" for s in sessions])
    return _embedding_figure(F, "fig_embedding_rfcmm", blocks, d["type"][keep].astype(str), sessions, [cache_path],
                             "flight", "DroneRFc-MM types flown twice (A1, E1, G1), T3 feature cache", flight=flight)


def fig_embedding_tsms(F, cache_dir, per_target=1200, seed=0):
    """TSMS-Drone: a random sample of measurements per target, coloured by target and by distance."""
    from .extra import tsms_dataset
    cache = Path(cache_dir) / "tsms"
    files = sorted(cache.glob("*.npz"))
    per = {}
    for f in files:
        z = np.load(f)
        o = per.setdefault(f.name.split("_")[0], {"target": [], "distance": [], "idx": [], "x": []})
        for k in o:
            o[k].append(z[k])
    per = {s: {k: np.concatenate(v) for k, v in o.items()} for s, o in per.items()}
    d = tsms_dataset(per)
    rng = np.random.default_rng(seed)
    pick = np.concatenate([rng.choice(np.flatnonzero(d["type"] == t), min(per_target, int((d["type"] == t).sum())),
                                      replace=False) for t in sorted(set(d["type"]))])
    blocks = [(SENSOR_NAMES[s], d["x"][s][pick]) for s in ("FMCW", "CW", "RF") if s in d["x"]]
    types = np.array([t.replace(" Pro", "") for t in d["type"][pick]])
    return _embedding_figure(F, "fig_embedding_tsms", blocks, types, d["session"][pick].astype(str), files, "distance",
                             f"TSMS-Drone, {per_target} random measurements per target (seed {seed}), T3 feature cache",
                             colour_by_value=d["distance"][pick])


# ============================================================================= Figures: sensor examples
def _show(ax, image, title):
    ax.imshow(image)
    ax.set_title(title, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def _spectrogram(ax, signal, rate, title, t0=0.0, mark=None):
    from scipy.signal import spectrogram
    f, t, s = spectrogram(signal, fs=rate, nperseg=1024, noverlap=768)
    keep = f <= 8000
    ax.pcolormesh(t + t0, f[keep] / 1000, 10 * np.log10(s[keep] + 1e-12), cmap="Blues", shading="auto",
                  rasterized=True)
    if mark:
        for m in mark:
            ax.axvline(m, color=ORANGE, lw=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title, color=INK)


def _points(ax, xy, title, xlabel, ylabel, colour=BLUE, background=None):
    if background is not None and len(background):
        ax.scatter(background[:, 0], background[:, 1], s=0.3, lw=0, color="#c3c2b7", rasterized=True,
                   label="static background")
    ax.scatter(xy[:, 0], xy[:, 1], s=2 if len(xy) < 2000 else 0.6, lw=0, color=colour, rasterized=True,
               label="returns kept" if background is not None else None)
    ax.set_title(title, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    _clean(ax, None)


def pick_mmaud_window(step2_dir, recording):
    """The covered test window of `recording` with the most radar foreground points (needs Step 2 CPU features)."""
    cpu = np.load(Path(step2_dir) / f"{recording}.npz")
    names = [str(s) for s in cpu["availability_names"]]
    radar = cpu["availability"][:, names.index("radar_fg_points")]
    avia = cpu["availability"][:, names.index("livox_avia_points")]
    ok = cpu["covered"].astype(bool) & (avia > 0)
    score = np.where(ok, radar, -1)
    return str(cpu["window_id"][int(np.argmax(score))])


def fig_examples_mmaud(F, step1_folder, windows_csv, window_id=None):
    """One window of one recording: camera frame, Avia and Mid-360 returns, radar returns, sound."""
    import io as _io
    from PIL import Image
    from .features import (AUDIO_RATE, background_voxels, decode_audio, foreground, open_recording, xyz_of)
    from .windows import load_windows
    rows = {r["window_id"]: r for r in load_windows(windows_csv)}
    meta, streams = open_recording(step1_folder)
    rec = meta["recording"]["name"]
    mine = [r for r in rows.values() if r["recording"] == rec]
    w = rows[window_id] if window_id else mine[len(mine) // 2]
    t0, t1 = float(w["t_start"]), float(w["t_end"])
    pyplot = plt()
    fig = pyplot.figure(figsize=(DOUBLE, 3.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    used = [Path(step1_folder) / "meta.json"]
    data = []
    cam = streams["camera"]
    i = int(np.argmin(np.abs(cam.time - (t0 + t1) / 2)))
    frame = np.asarray(Image.open(_io.BytesIO(cam.payload(i))).convert("RGB"))
    used.append(Path(step1_folder) / "camera.bin")
    ax = fig.add_subplot(gs[0, :2])
    _show(ax, frame, f"(a) Camera frame ({frame.shape[1]} x {frame.shape[0]} px)")
    data.append({"panel": "camera", "message": i, "time": round(float(cam.time[i]), 3)})
    panels = [("livox_avia", "(b) Livox Avia, top view", None), ("livox_mid360", "(c) Mid-360, top view", 0.5),
              ("radar", "(d) mmWave radar, top view", None)]
    for col, (name, title, voxel) in zip((2, 3, 0), panels):
        ax = fig.add_subplot(gs[0, col] if col >= 2 else gs[1, 0])
        s = streams.get(name)
        if s is None or not len(s):
            ax.axis("off")
            continue
        idx = s.between(t0, t1)
        xyz = np.concatenate([xyz_of(s.points(int(k))) for k in idx]) if len(idx) else np.zeros((0, 3), np.float32)
        used.append(Path(step1_folder) / f"{name}.bin")
        bg = None
        if voxel and len(xyz):
            back = background_voxels(s, voxel, 0.1)
            fg = foreground(xyz, back, voxel)
            bg, xyz = xyz[~fg], xyz[fg]
        _points(ax, xyz[:, :2], title, "x (m)", "y (m)", background=bg[:, :2] if bg is not None else None)
        if bg is not None:
            ax.legend(loc="upper right", frameon=False, markerscale=6, fontsize=6)
        data.append({"panel": name, "scans": len(idx), "points_shown": len(xyz),
                     "background_points": 0 if bg is None else len(bg)})
    a = streams.get("audio1")
    ax = fig.add_subplot(gs[1, 1:])
    if a is not None and len(a):
        signal = decode_audio(step1_folder, a)
        used.append(Path(step1_folder) / "audio1.bin")
        starts = np.arange(len(a)) * 1152
        c = int(round(np.interp((t0 + t1) / 2, a.time, starts)))
        lo, hi = max(c - 3 * AUDIO_RATE, 0), c + 3 * AUDIO_RATE
        _spectrogram(ax, signal[lo:hi], AUDIO_RATE, "(e) Microphone node 1, 6 s around the window (window marked)",
                     t0=-(c - lo) / AUDIO_RATE, mark=[-(t1 - t0) / 2, (t1 - t0) / 2])
        data.append({"panel": "audio1", "samples": hi - lo, "rate": AUDIO_RATE})
    fig.tight_layout()
    return F.save(fig, "fig_examples_mmaud", data, used,
                  checks=[f"window {w['window_id']} ({w['drone_type']}, {w['split']} split)"],
                  note=f"MMAUD {rec}, window {w['window_id']}, read from the Step 1 extract")


def fig_examples_more(F, rfcmm_root=None, tsms_packs=None, session="G1", distance=10,
                      targets=("Mavic 2 Pro", "Phantom 4 Pro")):
    """DroneRFc-MM: PTZ and fisheye frames, sound and radar of one window. TSMS-Drone: the three sensor images
    of two targets at one distance."""
    from .extra import (_pack_images, _yaml_ranges, find_radar, read_audio, read_radar, rfcmm_segments,
                        video_frames)
    pyplot = plt()
    rows = int(bool(rfcmm_root)) + (len(targets) if tsms_packs else 0)
    fig = pyplot.figure(figsize=(DOUBLE, 1.75 * rows + 0.3))
    gs = fig.add_gridspec(rows, 4)
    used, data, r0 = [], [], 0
    if rfcmm_root:
        root = Path(rfcmm_root)
        segs = [s for s in rfcmm_segments(root) if s["session"] == session]
        seg = segs[len(segs) // 2]
        mid = (seg["start"] + seg["end"]) / 2
        for col, (cam, folder, meta) in enumerate((("PTZ camera", "PTZ_raw", "PTZ_meta.yaml"),
                                                   ("Fisheye camera", "fisheye_raw", "fisheye_meta.yaml"))):
            start = _yaml_ranges(root / "raw_data" / folder / meta)[session][0]
            video = root / "raw_data" / folder / f"{session}.mp4"
            frame = video_frames(video, start, 25.0, [mid], max_side=960).get(mid)
            used.append(video)
            ax = fig.add_subplot(gs[r0, col])
            if frame is not None:
                _show(ax, frame, cam)
                if col == 0:
                    ax.set_ylabel(f"DroneRFc-MM\nsession {session}")
            data.append({"panel": cam, "session": session, "time": round(mid, 3)})
        dat = root / "raw_data" / "audio_raw" / f"{session}.dat"
        if dat.exists():
            sr, t, sig = read_audio(dat)
            sel = (t >= mid - 3) & (t < mid + 3)
            _spectrogram(fig.add_subplot(gs[r0, 2]), sig[0, sel], sr, "Microphone channel 1", t0=-3,
                         mark=[-0.5, 0.5])
            used.append(dat)
            data.append({"panel": "audio", "samples": int(sel.sum()), "rate": sr})
        rz = find_radar(root, session)
        if rz is not None:
            times, frames = read_radar(rz)
            sel = [f for tt, f in zip(times, frames) if mid - 2.5 <= tt < mid + 2.5]
            pts = np.concatenate(sel) if sel else np.zeros((0, 6), np.float32)
            ax = fig.add_subplot(gs[r0, 3])
            az = np.deg2rad(pts[:, 1]) if len(pts) and np.abs(pts[:, 1]).max() > 3.2 else pts[:, 1]
            _points(ax, np.stack([pts[:, 0] * np.sin(az), pts[:, 0] * np.cos(az)], 1), "mmWave radar, 5 s",
                    "x (m)", "y (m)")
            used.append(rz)
            data.append({"panel": "radar", "frames": len(sel), "returns": len(pts)})
        r0 += 1
    if tsms_packs:
        for target in targets:
            for col, sensor in enumerate(("FMCW", "CW", "RF")):
                ax = fig.add_subplot(gs[r0, col])
                packs = [p for n, p in tsms_packs.items() if n.startswith(f"{sensor}_{distance:02d}m")]
                image = None
                for pack in packs:
                    names, images = _pack_images(pack)
                    hits = [k for k, n in enumerate(names) if n.startswith(f"{target}_{distance}m_")]
                    if hits:
                        image = images[hits[0]]
                        used.append(pack)
                        data.append({"panel": sensor, "target": target, "distance_m": distance, "file": names[hits[0]]})
                        break
                if image is not None:
                    _show(ax, image, SENSOR_NAMES[sensor] if target == targets[0] else "")
                    if col == 0:
                        ax.set_ylabel(f"TSMS-Drone\n{target}, {distance} m")
                else:
                    ax.axis("off")
            r0 += 1
    fig.tight_layout()
    return F.save(fig, "fig_examples_more", data, used, note="read from the raw DroneRFc-MM files and the T1 packs")


# ============================================================================= report
def format_report(found, figures, runner, notes):
    lines = ["F1 FIGURES", "", "INPUTS"] + ["    " + l for l in describe_inputs(found)]
    lines += ["", "NOTES"] + ["    " + n for n in notes]
    lines += ["", "FIGURES WRITTEN"]
    for name, info in figures.manifest["figures"].items():
        lines.append(f"    {name}: {info['values_plotted']} values, {len(info['inputs'])} input files")
        lines += [f"        check: {c}" for c in info["checks"]]
    failed = runner.failed()
    lines.append("\nSTEPS: " + ("all succeeded" if not failed else "not succeeded: " + ", ".join(failed)))
    lines += [f"    {s['step']}: {s['message']}" for s in runner.steps if s["status"] != "ok"]
    return "\n".join(lines)
