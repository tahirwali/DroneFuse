"""
Step 2 driver: features for every extracted recording that is attached, then a validation-only check.

Per recording two files are written to the output folder:
  <recording>.npz        CPU features (dronefuse.features): availability, descriptors, point sets, log-mel
  <recording>.deep.npz   GPU embeddings (dronefuse.deep): camera_avg, camera_max, audio_ast
They are separate so a failure of the GPU part never loses the CPU part.

The check (`linear_probe`) trains one logistic regression per sensor block on the closed-protocol
train split and reports accuracy on the validation split only. The test splits are not touched
before the method is fixed.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .features import AUDIO, FeatureSettings, audio_windows_16k, availability_summary, open_recording, \
    recording_features, save_features
from .recordings import BY_NAME, DRONE_TYPES
from .windows import protocol_sets


def find_recordings(root):
    """{recording name: folder} for every Step 1 folder (meta.json) below `root`."""
    found = {}
    for meta_path in sorted(Path(root).rglob("meta.json")):
        try:
            name = json.loads(meta_path.read_text())["recording"]["name"]
        except (KeyError, ValueError):
            continue
        if name in BY_NAME and name not in found:
            found[name] = meta_path.parent
    return found


def _cpu_job(name, folder, windows, out_dir, settings):
    began = time.time()
    lines = []
    features = recording_features(folder, windows, settings, log=lines.append)
    save_features(features, Path(out_dir) / f"{name}.npz")
    return {"recording": name, "seconds": round(time.time() - began, 1), "log": lines,
            "windows": len(windows), "covered": int(features["covered"].sum()),
            "complete_extraction": bool(features["complete_extraction"]),
            "availability": availability_summary(features)}


def cpu_features(recordings, windows, out_dir, settings=None, workers=4, log=print):
    """CPU features for all recordings, one process per recording. Returns {name: summary or error}."""
    settings = settings or FeatureSettings()
    rows_of = {name: [w for w in windows if w["recording"] == name] for name in recordings}
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_cpu_job, name, str(folder), rows_of[name], str(out_dir), settings): name
                   for name, folder in recordings.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                log(f"{name}: CPU features done in {results[name]['seconds']} s "
                    f"({results[name]['covered']}/{results[name]['windows']} windows covered)")
                for line in results[name]["log"]:
                    log(line)
            except Exception as error:      # one recording must not stop the others
                results[name] = {"recording": name, "error": f"{type(error).__name__}: {error}"}
                log(f"{name}: CPU features FAILED - {results[name]['error']}")
    return results


def deep_features(name, folder, windows, out_dir, log=print):
    """Camera and audio embeddings of one recording (needs <name>.npz from the CPU part).
    `windows` are the recording's rows of windows.csv, in file order (as for the CPU part)."""
    from . import deep
    cpu = np.load(Path(out_dir) / f"{name}.npz")
    out = {"window_id": cpu["window_id"]}
    began = time.time()
    if "camera_frames" in cpu.files:
        out.update(deep.camera_embeddings(folder, cpu["camera_frames"], log=log))
    log(f"  camera embeddings: {time.time() - began:.0f} s")
    began = time.time()
    _, streams = open_recording(folder)
    if [w["window_id"] for w in windows] != [str(i) for i in cpu["window_id"]]:
        raise ValueError(f"{name}: windows differ from those of the CPU features")
    signals = {node: audio_windows_16k(folder, streams, windows, node) for node in AUDIO}
    out.update(deep.audio_embeddings(signals, cpu["audio_frames"] > 0, log=log))
    log(f"  audio embeddings: {time.time() - began:.0f} s")
    np.savez_compressed(Path(out_dir) / f"{name}.deep.npz", **out)
    return {"camera_dim": int(out.get("camera_avg", np.zeros((0, 0))).shape[1]),
            "audio_dim": int(out["audio_ast"].shape[2])}


# The validation-only check
BLOCKS = {
    "camera (avg pool)": ("deep", "camera_avg"),
    "camera (max pool)": ("deep", "camera_max"),
    "Livox Avia": ("cpu", "livox_avia_all"),
    "Mid-360 (all)": ("cpu", "livox_mid360_all"),
    "Mid-360 (foreground)": ("cpu", "livox_mid360_fg"),
    "radar (all)": ("cpu", "radar_all"),
    "radar (foreground)": ("cpu", "radar_fg"),
    "audio log-mel stats": ("cpu", "audio_stats"),
    "audio AST": ("deep", "audio_ast"),
}


def load_block(out_dir, names, block):
    """X (rows of all recordings in `names`, window order) and window ids for one block."""
    kind, key = BLOCKS[block]
    xs, ids = [], []
    for name in names:
        path = Path(out_dir) / (f"{name}.npz" if kind == "cpu" else f"{name}.deep.npz")
        if not path.exists():
            return None, None
        data = np.load(path)
        if key not in data.files:
            return None, None
        x = data[key].astype(np.float32)
        xs.append(x.reshape(len(x), -1))
        ids += [str(i) for i in data["window_id"]]
    return np.concatenate(xs), ids


def linear_probe(out_dir, windows, names, log=print):
    """Closed protocol: train on train windows, accuracy on val windows, one block at a time."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    covered = {}
    for name in names:
        cpu = np.load(Path(out_dir) / f"{name}.npz")
        covered.update(zip((str(i) for i in cpu["window_id"]), cpu["covered"]))
    sets = protocol_sets([w for w in windows if w["recording"] in names], "closed")
    split_of = {w["window_id"]: s for s in ("train", "val") for w in sets[s]}
    label_of = {w["window_id"]: DRONE_TYPES.index(w["drone_type"]) for w in windows}
    results = {}
    for block in BLOCKS:
        x, ids = load_block(out_dir, names, block)
        if x is None:
            results[block] = None
            continue
        keep = np.array([covered.get(i, False) for i in ids])
        split = np.array([split_of.get(i, "") for i in ids])
        y = np.array([label_of[i] for i in ids])
        tr, va = keep & (split == "train"), keep & (split == "val")
        if len(np.unique(y[tr])) < 2 or not va.any():
            results[block] = {"note": "fewer than 2 drone types in train or no val windows"}
            continue
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit((x[tr] - mu) / sd, y[tr])
        pred = model.predict((x[va] - mu) / sd)
        results[block] = {"val_accuracy": round(float((pred == y[va]).mean()), 4),
                          "val_balanced_accuracy": round(float(balanced_accuracy_score(y[va], pred)), 4),
                          "train_windows": int(tr.sum()), "val_windows": int(va.sum()), "dim": int(x.shape[1])}
        log(f"  {block:22s} val acc {results[block]['val_accuracy']:.3f} "
            f"(balanced {results[block]['val_balanced_accuracy']:.3f}, dim {x.shape[1]})")
    return results


def format_step2_report(cpu_results, deep_results, probe, environment=None):
    lines = []
    if environment:
        lines.append(f"MACHINE  CPUs {environment.get('cpu_count')} | GPU {environment.get('gpu')} | "
                     f"run type {environment.get('kaggle_run_type')}")
    lines.append("\nAVAILABILITY (covered windows): fraction with no data | median count")
    names = [n for n in BY_NAME if n in cpu_results]
    for name in names:
        r = cpu_results[name]
        if "error" in r:
            lines.append(f"{name}: CPU FAILED {r['error']}")
            continue
        flag = "" if r["complete_extraction"] else "  (extraction incomplete)"
        lines.append(f"{name}: {r['covered']}/{r['windows']} windows covered, {r['seconds']} s{flag}")
        cells = [f"{k.replace('_points', '').replace('_frames', '')} {v['empty']:.2f}|{v['q25_50_75'][1]:.0f}"
                 for k, v in r["availability"].items() if v["empty"] is not None]
        lines.append("  " + "  ".join(cells))
    lines.append("\nGPU EMBEDDINGS")
    for name, r in deep_results.items():
        lines.append(f"{name}: {r}")
    lines.append("\nLINEAR PROBE, closed protocol, train -> VAL (test not used)")
    for block, r in (probe or {}).items():
        if r is None:
            lines.append(f"  {block:22s} not available")
        elif "note" in r:
            lines.append(f"  {block:22s} {r['note']}")
        else:
            lines.append(f"  {block:22s} acc {r['val_accuracy']:.3f}  balanced {r['val_balanced_accuracy']:.3f}  "
                         f"dim {r['dim']}  ({r['train_windows']} train / {r['val_windows']} val)")
    return "\n".join(lines)
