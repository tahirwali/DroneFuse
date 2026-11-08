"""
Step T3: the recording check on two further datasets.

TSMS-Drone (Han & Jung 2026): FMCW radar, CW radar and RF receiver, four drones and a corner reflector,
each hovering at 15 distances (2-30 m), 500 measurements per target and distance. A "session" is one target
at one distance; its 500 measurements were taken in one go.

DroneRFc-MM (Yu et al. 2026): PTZ camera, fisheye camera, microphone array, mmWave radar (and RF, LiDAR,
not used here), six DJI drones on one afternoon. A "session" is one flight; A1, E1 and G1 were flown twice.

For both datasets the same three questions are asked as on MMAUD:
1. Probes: how well does each sensor tell the drone type apart, and how well does it tell two sessions of
   the same drone apart?
2. Accuracy when training and test data come from the same sessions (time blocks) versus from different
   sessions.
3. Rejection of a drone type left out of training, when the known test data come from sessions seen in
   training versus from new sessions.

Models are deliberately simple and identical for every sensor: frozen pretrained features (ConvNeXt-T for
images, log-mel statistics for sound, hand-made statistics for radar returns), standardisation, PCA and a
logistic regression. Fusion: concatenation of all sensor features, and late fusion (average of the
single-sensor probabilities).
"""
import io
import json
import os
import re
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

SPLIT_PATTERN = ["train", "train", "val", "train", "test", "train", "train", "val", "train", "test"]


# ----------------------------------------------------------------------------- images
def make_embedder(device=None, batch=128):
    """ConvNeXt-T (ImageNet-1k) average-pooled 768-d embedding of RGB uint8 images (any size)."""
    import torch
    import torchvision
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = torchvision.models.convnext_tiny(weights=weights).eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(images):
        out = []
        for i in range(0, len(images), batch):
            x = torch.from_numpy(np.stack(images[i:i + batch])).to(device).permute(0, 3, 1, 2).float() / 255
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", antialias=True,
                                                align_corners=False)
            f = model.features((x - mean) / std)
            f = model.classifier[0](model.avgpool(f)).flatten(1)          # LayerNorm2d + flatten -> 768
            out.append(f.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    return embed


def _png(data):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


# ----------------------------------------------------------------------------- TSMS-Drone
TSMS_NAME = re.compile(r"^(.*)_(\d+)m_(\d+)\.png$")
TSMS_PACK = re.compile(r"^(FMCW|CW|RF)_\d+m.*")


def find_tsms(roots):
    """{pack name: path} of the TSMS packs under `roots`. A pack is a .tar written by T1 (FMCW_02m_Mavic3.tar,
    CW_02m.tar, ...) or the folder that Kaggle makes when it unpacks such a .tar in a dataset (with or without
    the .tar suffix). Packs are named without the .tar suffix, and the first copy found is kept."""
    packs = {}
    for root in map(Path, roots):
        for p in sorted(root.rglob("*")):
            if not TSMS_PACK.match(p.name):
                continue
            name = p.name[:-4] if p.name.endswith(".tar") else p.name
            if p.is_file() and p.suffix == ".tar" or p.is_dir() and any(p.rglob("*.png")):
                if not any(str(p).startswith(str(q) + os.sep) for q in packs.values()):
                    packs.setdefault(name, p)
    return dict(sorted(packs.items()))


def _pack_images(pack):
    """(file names, RGB images) of every .png in a TSMS pack (.tar or folder)."""
    names, images = [], []
    if Path(pack).is_dir():
        for f in sorted(Path(pack).rglob("*.png")):
            names.append(f.name)
            images.append(_png(f.read_bytes()))
    else:
        with tarfile.open(pack) as t:
            for m in t.getmembers():
                if m.isfile() and m.name.endswith(".png"):
                    names.append(Path(m.name).name)
                    images.append(_png(t.extractfile(m).read()))
    return names, images


def tsms_features(tar_dirs, cache_dir, embed, log=print):
    """{sensor: dict(target, distance, idx, x)} for FMCW, CW and RF, embedding every image of the packs
    once (cached per pack as .npz so a rerun resumes). Packs may be .tar files or unpacked folders."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    packs = find_tsms(tar_dirs)
    log(f"TSMS: {len(packs)} packs ({sum(Path(p).is_dir() for p in packs.values())} unpacked by Kaggle)")
    if not packs:
        raise ValueError(f"no TSMS packs (FMCW_*, CW_*, RF_*) under {[str(d) for d in tar_dirs]}")
    out = {}
    for name, pack in packs.items():
        sensor = name.split("_")[0]
        cache = cache_dir / (name + ".npz")
        if not cache.exists():
            names, images = _pack_images(pack)
            if not names:
                log(f"  {name}: no images, skipped")
                continue
            keys = [TSMS_NAME.match(n).groups() for n in names]
            np.savez(cache, target=np.array([k[0] for k in keys]), distance=np.array([int(k[1]) for k in keys]),
                     idx=np.array([int(k[2]) for k in keys]), x=embed(images).astype(np.float16))
            log(f"  {name}: {len(names)} images embedded")
        z = np.load(cache)
        o = out.setdefault(sensor, {"target": [], "distance": [], "idx": [], "x": []})
        for k in o:
            o[k].append(z[k])
    return {s: {k: np.concatenate(v) for k, v in o.items()} for s, o in out.items()}


def tsms_dataset(per_sensor):
    """Join the sensors on (target, distance, measurement index): one row per simultaneous measurement."""
    sensors = sorted(per_sensor)
    keys = None
    for s in sensors:
        k = set(zip(per_sensor[s]["target"], per_sensor[s]["distance"], per_sensor[s]["idx"]))
        keys = k if keys is None else keys & k
    keys = sorted(keys)
    x = {}
    for s in sensors:
        pos = {k: i for i, k in enumerate(zip(per_sensor[s]["target"], per_sensor[s]["distance"], per_sensor[s]["idx"]))}
        x[s] = per_sensor[s]["x"][[pos[k] for k in keys]].astype(np.float32)
    target = np.array([k[0] for k in keys])
    distance = np.array([k[1] for k in keys])
    idx = np.array([k[2] for k in keys])
    session = np.array([f"{t}@{d}m" for t, d in zip(target, distance)])
    split, block = time_blocks(session, idx)
    return {"x": x, "type": target, "session": session, "distance": distance, "split": split, "block": block,
            "counts": {s: len(per_sensor[s]["idx"]) for s in sensors}, "rows": len(keys)}


# ----------------------------------------------------------------------------- DroneRFc-MM
def _iso(s):
    return datetime.fromisoformat(str(s).strip().replace("+8:00", "+08:00")).timestamp()


def _yaml_ranges(path):
    """{session: (start, end)} from PTZ_meta.yaml / fisheye_meta.yaml (two-level 'name: / start_time: / end_time:')."""
    out, name = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(" ") and line.rstrip().endswith(":"):
            name = line.strip()[:-1]
            out[name] = [None, None]
        elif "start_time:" in line:
            out[name][0] = _iso(line.split("start_time:")[1])
        elif "end_time:" in line:
            out[name][1] = _iso(line.split("end_time:")[1])
    return {k: tuple(v) for k, v in out.items()}


def rfcmm_segments(root, length=1.0):
    """1 s windows over the time in which both cameras recorded each session (the cameras, the audio and the
    radar cover only the middle part of each flight log): [{group, session, start, end}]. The drone type is
    the session name without the repeat suffix (A1-2 -> A1)."""
    root = Path(root)
    ptz = _yaml_ranges(root / "raw_data" / "PTZ_raw" / "PTZ_meta.yaml")
    fish = _yaml_ranges(root / "raw_data" / "fisheye_raw" / "fisheye_meta.yaml")
    segments = []
    for sess in sorted(set(ptz) & set(fish)):
        t0, t1 = max(ptz[sess][0], fish[sess][0]), min(ptz[sess][1], fish[sess][1])
        k = 0
        while t0 + (k + 1) * length <= t1:
            segments.append({"group": sess.split("-")[0], "session": sess, "idx": k,
                             "start": t0 + k * length, "end": t0 + (k + 1) * length})
            k += 1
    return segments


def _mel_filters(sr, n_fft, n_mels=64, fmin=50.0, fmax=None):
    fmax = fmax or sr / 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


def read_audio(path):
    """(sample_rate, time of every sample, samples (channels, n_samples))."""
    import struct
    raw = Path(path).read_bytes()
    magic, sr, ch, spc, _ = struct.unpack("<4sIIII", raw[:20])
    if magic != b"ADAT":
        raise ValueError(f"{path}: not an ADAT file")
    step = 8 + ch * spc * 4
    n = (len(raw) - 20) // step
    buf = np.frombuffer(raw[20:20 + n * step], dtype=np.uint8).reshape(n, step)
    ts = buf[:, :8].copy().view("<f8")[:, 0]
    data = buf[:, 8:].copy().view("<f4").reshape(n, ch, spc).transpose(1, 0, 2).reshape(ch, n * spc)
    t = np.repeat(ts, spc) + np.tile(np.arange(spc) / sr, n)
    return sr, t, data


def audio_features(audio, start, end, n_fft=1024, hop=512, n_mels=64):
    """Mean and standard deviation of the log-mel spectrogram of each channel (4 x 128), or None if less than
    90 % of the segment is covered."""
    sr, t, data = audio
    sel = (t >= start) & (t < end)
    if sel.sum() < 0.9 * (end - start) * sr:
        return None
    x = data[:, sel]
    fb = _mel_filters(sr, n_fft, n_mels)
    win = np.hanning(n_fft)
    feats = []
    for c in x:
        frames = np.lib.stride_tricks.sliding_window_view(c, n_fft)[::hop] * win
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        logmel = np.log(power @ fb.T + 1e-10)
        feats += [logmel.mean(0), logmel.std(0)]
    return np.concatenate(feats).astype(np.float32)


RADAR_COLUMNS = ("range", "azimuth", "elevation", "doppler", "power", "snr")


def find_radar(root, sess):
    """The radar data of one session: mmRadar_raw/<session>.zip, or the folder that Kaggle makes when it
    unpacks that zip in a dataset (with or without the .zip suffix). None if there is neither."""
    base = Path(root) / "raw_data" / "mmRadar_raw"
    for name in (sess.lower(), sess):
        for p in (base / f"{name}.zip", base / name):
            if p.is_file() and zipfile.is_zipfile(p) or p.is_dir() and any(p.rglob("*.pcd")):
                return p
    return None


def _pcd_texts(path):
    """(file name, text) of every .pcd file in a zip or a folder, in name order."""
    if Path(path).is_dir():
        for f in sorted(Path(path).rglob("*.pcd"), key=lambda f: f.name):
            yield f.name, f.read_bytes().decode("ascii", "replace")
    else:
        with zipfile.ZipFile(path) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".pcd")):
                yield name, z.read(name).decode("ascii", "replace")


def read_radar(zip_path):
    """(frame times, list of (n, 6) arrays of range, azimuth, elevation, doppler, power, snr) from a radar zip
    or its unpacked folder."""
    times, frames = [], []
    for name, text in _pcd_texts(zip_path):
        text = text.splitlines()
        fields = next(l for l in text if l.startswith("FIELDS")).split()[1:]
        start = next(i for i, l in enumerate(text) if l.startswith("DATA")) + 1
        cols = [fields.index(c) for c in RADAR_COLUMNS]
        rows = [l.split() for l in text[start:] if l.strip()]
        arr = np.array([[float(r[c]) for c in cols] for r in rows], np.float32).reshape(-1, len(cols))
        stamp = Path(name).stem.split("_", 1)[1]
        times.append(float(stamp))
        frames.append(arr)
    order = np.argsort(times)
    return np.array(times)[order], [frames[i] for i in order]


def radar_features(radar, start, end):
    """Return statistics of one segment (34 values): frames, returns per frame, mean and spread of each column,
    histograms of range (0-80 m) and Doppler (-10..10 m/s)."""
    times, frames = radar
    sel = [f for t, f in zip(times, frames) if start <= t < end]
    allr = np.concatenate(sel) if sel else np.zeros((0, 6), np.float32)
    n = len(allr)
    stats = np.concatenate([allr.mean(0), allr.std(0)]) if n else np.zeros(12, np.float32)
    hr = np.histogram(allr[:, 0], bins=8, range=(0, 80))[0] / max(n, 1)
    hd = np.histogram(allr[:, 3], bins=8, range=(-10, 10))[0] / max(n, 1)
    return np.concatenate([[len(sel), np.log1p(n / max(len(sel), 1))], stats, hr, hd,
                           [np.log1p(n), float(n > 0)], np.zeros(2)]).astype(np.float32)


def video_frames(path, t0, fps_hint, wanted_times, max_side=640):
    """{time: RGB frame} for the wanted times (frame index = (time - start) * fps), read in one pass."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = {}
    for t in wanted_times:
        i = int(round((t - t0) * fps))
        if 0 <= i < n:
            want.setdefault(i, []).append(t)
    out, i = {}, 0
    last = max(want) if want else -1
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                s = max_side / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                for t in want[i]:
                    out[t] = np.ascontiguousarray(frame)
        i += 1
    cap.release()
    return out


def rfcmm_features(root, embed, log=print):
    """Rows = 1 s windows with every used sensor available: x = {ptz, fisheye, audio, radar}."""
    root = Path(root)
    segs = rfcmm_segments(root)
    log(f"DroneRFc-MM: {len(segs)} windows of 1 s in {len(set(s['session'] for s in segs))} sessions")
    sessions = sorted({s["session"] for s in segs})
    feats = {k: [None] * len(segs) for k in ("ptz", "fisheye", "audio", "radar")}
    for cam, folder, meta in (("ptz", "PTZ_raw", "PTZ_meta.yaml"), ("fisheye", "fisheye_raw", "fisheye_meta.yaml")):
        ranges = _yaml_ranges(root / "raw_data" / folder / meta)
        for sess in sessions:
            mine = [i for i, s in enumerate(segs) if s["session"] == sess]
            video = root / "raw_data" / folder / f"{sess}.mp4"
            if sess not in ranges or not video.exists():
                continue
            times = [(segs[i]["start"] + segs[i]["end"]) / 2 for i in mine]      # the middle frame of each window
            frames = video_frames(video, ranges[sess][0], 25.0, times)
            have = [i for i, t in zip(mine, times) if t in frames]
            emb = embed([frames[(segs[i]["start"] + segs[i]["end"]) / 2] for i in have])
            for i, e in zip(have, emb):
                feats[cam][i] = e
            log(f"  {cam} {sess}: {sum(feats[cam][i] is not None for i in mine)}/{len(mine)} segments")
    for sess in sessions:
        mine = [i for i, s in enumerate(segs) if s["session"] == sess]
        dat = root / "raw_data" / "audio_raw" / f"{sess}.dat"
        if dat.exists():
            audio = read_audio(dat)
            for i in mine:
                feats["audio"][i] = audio_features(audio, segs[i]["start"], segs[i]["end"])
        rz = find_radar(root, sess)
        if rz is not None:
            radar = read_radar(rz)
            for i in mine:
                feats["radar"][i] = radar_features(radar, segs[i]["start"], segs[i]["end"])
        log(f"  audio/radar {sess}: {sum(feats['audio'][i] is not None for i in mine)} / "
            f"{sum(feats['radar'][i] is not None for i in mine)} of {len(mine)} segments")
    keep = [i for i in range(len(segs)) if all(feats[k][i] is not None for k in feats)]
    if not keep:
        raise ValueError(f"no window has every sensor; missing per sensor: "
                         f"{ {k: sum(v is None for v in feats[k]) for k in feats} }")
    session = np.array([segs[i]["session"] for i in keep])
    order = np.array([segs[i]["start"] for i in keep])
    split, block = time_blocks(session, order)
    return {"x": {k: np.stack([feats[k][i] for i in keep]).astype(np.float32) for k in feats},
            "type": np.array([segs[i]["group"] for i in keep]), "session": session, "split": split,
            "block": block, "rows": len(keep), "segments": len(segs),
            "missing": {k: int(sum(v is None for v in feats[k])) for k in feats}}


# ----------------------------------------------------------------------------- shared evaluation
def time_blocks(session, order):
    """Ten equal consecutive blocks per session (by `order`), assigned train/val/test as on MMAUD."""
    split = np.empty(len(session), dtype=object)
    block = np.empty(len(session), dtype=object)
    for s in np.unique(session):
        idx = np.flatnonzero(session == s)
        idx = idx[np.argsort(order[idx], kind="stable")]
        for b, part in enumerate(np.array_split(idx, 10)):
            split[part] = SPLIT_PATTERN[b]
            block[part] = f"{s}/{b}"
    return split.astype(str), block.astype(str)


def _model(dim, n_samples=10 ** 9, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    steps = [StandardScaler()]
    k = min(128, dim, n_samples - 1)
    if dim > 128 and k >= 2:
        steps.append(PCA(n_components=k, random_state=seed))
    steps.append(LogisticRegression(max_iter=3000, C=0.5))
    return make_pipeline(*steps)


def fit_predict(d, train, test, known, sensors=None):
    """Class probabilities on `test` for every single sensor, feature concatenation and late fusion."""
    sensors = sensors or sorted(d["x"])
    y = np.array([known.index(t) for t in d["type"][train]])
    probs = {}
    for s in sensors:
        m = _model(d["x"][s].shape[1], len(train)).fit(d["x"][s][train], y)
        probs[s] = m.predict_proba(d["x"][s][test])
    cat = np.concatenate([d["x"][s] for s in sensors], 1)
    m = _model(cat.shape[1], len(train)).fit(cat[train], y)
    probs["concat"] = m.predict_proba(cat[test])
    probs["late"] = np.mean([probs[s] for s in sensors], 0)
    return probs


def _auroc(known_scores, unknown_scores):
    from scipy.stats import rankdata
    n1, n0 = len(known_scores), len(unknown_scores)
    r = rankdata(np.concatenate([known_scores, unknown_scores]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _boot(blocks, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    members = {b: np.flatnonzero(blocks == b) for b in ub}
    reps = []
    for _ in range(n):
        idx = np.concatenate([members[b] for b in rng.choice(ub, len(ub))])
        reps.append(fn(idx))
    reps = np.array(reps, float)
    reps = reps[~np.isnan(reps)]
    return [round(100 * float(np.percentile(reps, 2.5)), 1), round(100 * float(np.percentile(reps, 97.5)), 1)]


def accuracy_table(d, train, test, known, n_boot=1000):
    """Known-type accuracy (%) with block-bootstrap interval for every method."""
    probs = fit_predict(d, train, test, known)
    y = np.array([known.index(t) if t in known else -1 for t in d["type"][test]])
    k = y >= 0
    out = {}
    for m, p in probs.items():
        correct = (p.argmax(1) == y) & k
        acc = lambda i: correct[i][k[i]].mean() if k[i].any() else np.nan
        out[m] = {"accuracy": round(100 * float(acc(np.arange(len(y)))), 1),
                  "ci": _boot(d["block"][test], acc, n_boot)}
        if (~k).any():
            s = p.max(1)
            au = lambda i: _auroc(s[i][k[i]], s[i][~k[i]]) if k[i].any() and (~k[i]).any() else np.nan
            out[m]["auroc_msp"] = round(100 * float(au(np.arange(len(y)))), 1)
            out[m]["auroc_ci"] = _boot(d["block"][test], au, n_boot)
    out["_windows"] = {"train": int(len(train)), "test_known": int(k.sum()), "test_unknown": int((~k).sum())}
    return out


def probes(d, session_pairs=None, sessions_by_type=None):
    """Type probe (all types, train -> val) and session probe (balanced accuracy, train -> val) per sensor."""
    from sklearn.metrics import balanced_accuracy_score
    tr, va = np.flatnonzero(d["split"] == "train"), np.flatnonzero(d["split"] == "val")
    types = sorted(str(t) for t in set(d["type"]))
    out = {}
    for s in sorted(d["x"]):
        x = d["x"][s]
        m = _model(x.shape[1], len(tr)).fit(x[tr], d["type"][tr])
        res = {"type": round(100 * balanced_accuracy_score(d["type"][va], m.predict(x[va])), 1),
               "type_classes": len(types)}
        scores = []
        groups = session_pairs or [[s2 for s2 in np.unique(d["session"]) if s2.startswith(t + "@")]
                                   for t in (sessions_by_type or [])]
        for g in groups:
            a = np.flatnonzero(np.isin(d["session"], g) & (d["split"] == "train"))
            b = np.flatnonzero(np.isin(d["session"], g) & (d["split"] == "val"))
            if len(set(d["session"][a])) < 2 or len(set(d["session"][b])) < 2:
                continue
            m2 = _model(x.shape[1], len(a)).fit(x[a], d["session"][a])
            scores.append(balanced_accuracy_score(d["session"][b], m2.predict(x[b])))
        res["session"] = round(100 * float(np.mean(scores)), 1) if scores else None
        res["session_classes"] = len(groups[0]) if groups else None
        out[s] = res
    return out


def run_tsms(d, n_boot=1000, log=print):
    """Probes; within-session (time blocks) vs cross-distance accuracy; leave-one-drone-out rejection."""
    types = sorted(str(t) for t in set(d["type"]))
    drones = [t for t in types if "Corner" not in t]
    res = {"rows": d["rows"], "counts": d["counts"], "types": types}
    res["probes"] = probes(d, sessions_by_type=types)
    log("TSMS probes done")
    even = np.isin(d["distance"], [2, 6, 10, 14, 18, 22, 26, 30])
    sp = d["split"]
    prot = {"within": (sp == "train", sp == "test"),
            "cross_distance_a_to_b": (even & (sp == "train"), ~even),
            "cross_distance_b_to_a": (~even & (sp == "train"), even)}
    res["accuracy"] = {}
    for name, (tr, te) in prot.items():
        res["accuracy"][name] = accuracy_table(d, np.flatnonzero(tr), np.flatnonzero(te), types, n_boot)
        log(f"TSMS {name}: concat {res['accuracy'][name]['concat']['accuracy']}")
    res["leave_one_out"] = {}
    for held in drones:
        known = [t for t in types if t != held]
        kn = d["type"] != held
        within = accuracy_table(d, np.flatnonzero(kn & (sp == "train")), np.flatnonzero(sp == "test"), known, n_boot)
        cross = accuracy_table(d, np.flatnonzero(kn & even & (sp == "train")), np.flatnonzero(~even), known, n_boot)
        res["leave_one_out"][held] = {"within": {m: within[m].get("auroc_msp") for m in within if m != "_windows"},
                                      "cross": {m: cross[m].get("auroc_msp") for m in cross if m != "_windows"}}
        log(f"TSMS LOO {held}: concat within {res['leave_one_out'][held]['within']['concat']} "
            f"cross {res['leave_one_out'][held]['cross']['concat']}")
    return res


RFCMM_REPEATS = [("A1", "A1-2"), ("E1", "E1-2"), ("G1", "G1-2")]


def run_rfcmm(d, n_boot=1000, log=print):
    """Probes; within-session vs cross-recording accuracy (first vs second flights of A1, E1, G1, with the
    other types as unknown); leave-one-type-out rejection."""
    types = sorted(str(t) for t in set(d["type"]))
    res = {"rows": d["rows"], "segments": d["segments"], "missing": d["missing"], "types": types,
           "sessions": {s: int((d["session"] == s).sum()) for s in np.unique(d["session"])}}
    pairs = [list(p) for p in RFCMM_REPEATS if all(q in res["sessions"] for q in p)]
    res["probes"] = probes(d, session_pairs=pairs)
    log("DroneRFc-MM probes done")
    sp = d["split"]
    res["accuracy"] = {"within": accuracy_table(d, np.flatnonzero(sp == "train"), np.flatnonzero(sp == "test"),
                                                types, n_boot)}
    first, second = [p[0] for p in pairs], [p[1] for p in pairs]
    known = sorted({s.split("-")[0] for s in first})
    others = np.isin(d["type"], [t for t in types if t not in known]) & (sp == "test")
    for name, src, dst in (("cross_recording_1_to_2", first, second), ("cross_recording_2_to_1", second, first)):
        tr = np.flatnonzero(np.isin(d["session"], src) & (sp == "train"))
        te = np.flatnonzero(np.isin(d["session"], dst) | others)
        res["accuracy"][name] = accuracy_table(d, tr, te, known, n_boot)
        log(f"DroneRFc-MM {name}: concat {res['accuracy'][name]['concat']['accuracy']}")
    res["accuracy"]["within_same_types"] = accuracy_table(
        d, np.flatnonzero(np.isin(d["type"], known) & (sp == "train")),
        np.flatnonzero((np.isin(d["type"], known) & (sp == "test")) | others), known, n_boot)
    res["leave_one_out"] = {}
    for held in types:
        kn = [t for t in types if t != held]
        r = accuracy_table(d, np.flatnonzero((d["type"] != held) & (sp == "train")), np.flatnonzero(sp == "test"),
                           kn, n_boot)
        res["leave_one_out"][held] = {m: r[m].get("auroc_msp") for m in r if m != "_windows"}
    return res


SENSOR_NAMES = {"FMCW": "FMCW radar", "CW": "CW radar", "RF": "Radio receiver", "ptz": "PTZ camera",
                "fisheye": "Fisheye camera", "audio": "Microphones", "radar": "mmWave radar",
                "concat": "Concatenation", "late": "Late fusion"}


def latex_more(t3_json, out_dir):
    """tab_more_probes.tex and tab_more_acc.tex (rows only) from t3_results.json."""
    res = json.loads(Path(t3_json).read_text())["results"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probes, acc = [], []
    plan = {"tsms": ("TSMS-Drone", "within", "cross_distance_a_to_b", "cross_distance_b_to_a"),
            "rfcmm": ("DroneRFc-MM", "within_same_types", "cross_recording_1_to_2", "cross_recording_2_to_1")}
    for key, (name, w, a, b) in plan.items():
        r = res.get(key)
        if not r:
            continue
        sensors = sorted(r["probes"])
        for i, s in enumerate(sensors):
            p = r["probes"][s]
            probes.append(f"{name if i == 0 else ''} & {SENSOR_NAMES.get(s, s)} & {p['type']:.1f} & "
                          f"{p['session']:.1f} \\\\" if p["session"] is not None else
                          f"{name if i == 0 else ''} & {SENSOR_NAMES.get(s, s)} & {p['type']:.1f} & -- \\\\")
        for i, m in enumerate(sensors + ["concat", "late"]):
            cells = []
            for prot in (w, a, b):
                c = r["accuracy"][prot][m]
                cells.append(f"{c['accuracy']:.1f} [{c['ci'][0]:.1f}, {c['ci'][1]:.1f}]")
            acc.append(f"{name if i == 0 else ''} & {SENSOR_NAMES.get(m, m)} & " + " & ".join(cells) + " \\\\")
        probes.append(r"\midrule")
        acc.append(r"\midrule")
    (out_dir / "tab_more_probes.tex").write_text("\n".join(probes[:-1]) + "\n")
    (out_dir / "tab_more_acc.tex").write_text("\n".join(acc[:-1]) + "\n")
