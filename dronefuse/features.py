"""
Per-window features computed on the CPU (Step 2).

For every window in splits/windows.csv we store how much each sensor delivered (frames, points,
radar tracks), plus small fixed-length descriptors of the LiDAR and radar points, log-mel audio,
and the indices of the camera frames that deep.py will embed later. We don't threshold anything
here. Step 3 decides what "present" means.

The static background (walls, ground, trees) is removed per recording by voxel occupancy: a
voxel that is filled in most scans is background. It needs no labels, which is roughly what a
fixed counter-UAV site would do when it maps its surroundings before operating.
See docs/step2_features.md for the array layout.
"""
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .store import StreamReader, load_meta

CLOUDS = ["livox_avia", "livox_mid360", "radar"]
AUDIO = ["audio1", "audio2", "audio3", "audio4"]
AUDIO_FRAME = 1152          # samples per MP3 frame; every audio message is exactly one frame (Step 1 check)
AUDIO_RATE = 48000
MEL_RATE = 16000


@dataclass
class FeatureSettings:
    voxel_m: tuple = (("livox_avia", 0.5), ("livox_mid360", 0.5), ("radar", 1.0))
    min_fraction: float = 0.1           # occupied in >= 10 % of scans -> static background
    max_points: tuple = (("livox_avia", 512), ("livox_mid360", 512), ("radar", 256))
    fg_for_points: tuple = ("livox_mid360", "radar")   # these keep foreground points only
    n_mels: int = 64
    seed: int = 0

    def voxel(self, cloud):
        return dict(self.voxel_m)[cloud]

    def points_cap(self, cloud):
        return dict(self.max_points)[cloud]


# Reading a recording
def open_recording(folder):
    meta = load_meta(folder)
    streams = {name: StreamReader(folder, name, info) for name, info in meta["streams"].items()
               if (Path(folder) / f"{name}.index.npz").exists()}
    return meta, streams


def coverage(streams, windows):
    """True where the window lies inside the extracted range of every stream that has messages."""
    ok = np.ones(len(windows), dtype=bool)
    t0 = np.array([w["t_start"] for w in windows])
    t1 = np.array([w["t_end"] for w in windows])
    for stream in streams.values():
        if len(stream):
            ok &= (t0 >= stream.time[0]) & (t1 <= stream.time[-1])
    return ok


# Point clouds
def xyz_of(points):
    return np.stack([points["x"], points["y"], points["z"]], axis=1).astype(np.float32)


def voxel_keys(xyz, size):
    """One int64 key per point (21 bits per axis: +-1e6 voxels)."""
    v = np.floor(xyz / size).astype(np.int64) + (1 << 20)
    return (v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2]


def background_voxels(stream, size, min_fraction):
    """Sorted keys of voxels occupied in at least `min_fraction` of the scans of the stream."""
    per_scan = [np.unique(voxel_keys(xyz_of(stream.points(i)), size)) for i in range(len(stream))]
    per_scan = [k for k in per_scan if len(k)]
    if not per_scan:
        return np.zeros(0, dtype=np.int64)
    keys, counts = np.unique(np.concatenate(per_scan), return_counts=True)
    return keys[counts >= max(1, min_fraction * len(stream))]


def foreground(xyz, background, size):
    if not len(background) or not len(xyz):
        return np.ones(len(xyz), dtype=bool)
    keys = voxel_keys(xyz, size)
    pos = np.clip(np.searchsorted(background, keys), 0, len(background) - 1)
    return background[pos] != keys


RANGE_BINS = np.geomspace(0.5, 128.0, 9)
ELEV_BINS = np.linspace(-90.0, 90.0, 7)
DOPPLER_BINS = np.linspace(-8.0, 8.0, 9)


def descriptor_names(channels):
    names = ["log_count", "scans_with_points"]
    names += [f"mean_{a}" for a in "xyz"] + [f"std_{a}" for a in "xyz"]
    names += ["range_mean", "range_std", "range_min", "range_max", "elev_mean", "elev_std",
              "azim_sin", "azim_cos", "azim_concentration", "eig1", "eig2", "eig3",
              "velocity_x", "velocity_y", "velocity_z"]
    names += [f"range_hist{i}" for i in range(len(RANGE_BINS) - 1)]
    names += [f"elev_hist{i}" for i in range(len(ELEV_BINS) - 1)]
    for c in channels:
        names += [f"{c}_mean", f"{c}_std"]
        if c == "Doppler":
            names += [f"doppler_hist{i}" for i in range(len(DOPPLER_BINS) - 1)]
    return names


def cloud_descriptor(scans, channels):
    """
    Fixed-length description of all points of one window. `scans` is a list of (time, xyz, extra)
    with extra an (n, len(channels)) array. Velocity = centroid of the second half of the window
    minus centroid of the first half, divided by the time between them (0 if either half is empty).
    """
    size = len(descriptor_names(channels))
    out = np.zeros(size, dtype=np.float32)
    nonempty = [s for s in scans if len(s[1])]
    if not nonempty:
        return out
    xyz = np.concatenate([s[1] for s in nonempty])
    extra = np.concatenate([s[2] for s in nonempty]) if channels else np.zeros((len(xyz), 0), np.float32)
    r = np.linalg.norm(xyz, axis=1)
    elev = np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-6), -1, 1)))
    azim = np.arctan2(xyz[:, 1], xyz[:, 0])
    sin_m, cos_m = np.sin(azim).mean(), np.cos(azim).mean()
    eig = np.sqrt(np.clip(np.sort(np.linalg.eigvalsh(np.cov(xyz.T)) if len(xyz) > 1 else np.zeros(3))[::-1], 0, None))
    times = np.array([s[0] for s in nonempty])
    middle = 0.5 * (times.min() + times.max())
    first = [s[1] for s in nonempty if s[0] <= middle]
    second = [s[1] for s in nonempty if s[0] > middle]
    velocity = np.zeros(3)
    if first and second:
        t_first = np.mean([s[0] for s in nonempty if s[0] <= middle])
        t_second = np.mean([s[0] for s in nonempty if s[0] > middle])
        velocity = (np.concatenate(second).mean(0) - np.concatenate(first).mean(0)) / max(t_second - t_first, 1e-3)
    values = [np.log1p(len(xyz)), len(nonempty) / max(len(scans), 1), *xyz.mean(0), *xyz.std(0),
              r.mean(), r.std(), r.min(), r.max(), elev.mean(), elev.std(), sin_m, cos_m, np.hypot(sin_m, cos_m),
              *eig, *velocity]
    values += list(np.histogram(np.clip(r, RANGE_BINS[0], RANGE_BINS[-1]), RANGE_BINS)[0] / len(r))
    values += list(np.histogram(elev, ELEV_BINS)[0] / len(r))
    for j, c in enumerate(channels):
        values += [extra[:, j].mean(), extra[:, j].std()]
        if c == "Doppler":
            values += list(np.histogram(np.clip(extra[:, j], DOPPLER_BINS[0], DOPPLER_BINS[-1]),
                                        DOPPLER_BINS)[0] / len(r))
    out[:] = np.nan_to_num(np.asarray(values, dtype=np.float32))
    return out


def cloud_channels(stream):
    """Extra per-point channels kept in the descriptor (reflectivity for Livox, radar channels)."""
    names = [n for n in stream._dtype.names if n not in ("x", "y", "z")]
    if stream.meta.get("kind") == "pointcloud":
        names = [n for n in names if n in ("Doppler", "Power")]      # Range/Alpha/Beta repeat x, y, z
    return names


def cloud_features(stream, windows, cloud, settings):
    """Descriptors (all / foreground), counts and padded point sets for one point-cloud stream."""
    size = settings.voxel(cloud)
    background = background_voxels(stream, size, settings.min_fraction)
    channels = cloud_channels(stream)
    cap = settings.points_cap(cloud)
    use_fg = cloud in settings.fg_for_points
    n_feat = 3 + len(channels)
    rng = np.random.default_rng(settings.seed)
    cache = {}

    def scan(i):
        if i not in cache:
            p = stream.points(i)
            xyz = xyz_of(p)
            extra = np.stack([p[c].astype(np.float32) for c in channels], axis=1) if channels else \
                np.zeros((len(p), 0), np.float32)
            fg = foreground(xyz, background, size)
            cache[i] = (float(stream.time[i]), xyz, extra, fg)
        return cache[i]

    n = len(windows)
    d_all = np.zeros((n, len(descriptor_names(channels))), np.float32)
    d_fg = np.zeros_like(d_all)
    counts = np.zeros((n, 2), np.int64)                          # all points, foreground points
    points = np.zeros((n, cap, n_feat), np.float16)
    mask = np.zeros((n, cap), bool)
    for w, row in enumerate(windows):
        idx = stream.between(row["t_start"], row["t_end"])
        scans = [scan(int(i)) for i in idx]
        d_all[w] = cloud_descriptor([(t, x, e) for t, x, e, _ in scans], channels)
        d_fg[w] = cloud_descriptor([(t, x[f], e[f]) for t, x, e, f in scans], channels)
        counts[w] = (sum(len(x) for _, x, _, _ in scans), sum(int(f.sum()) for *_, f in scans))
        kept = [np.concatenate([x[f], e[f]], 1) if use_fg else np.concatenate([x, e], 1) for _, x, e, f in scans]
        kept = np.concatenate(kept) if kept else np.zeros((0, n_feat), np.float32)
        if len(kept) > cap:
            kept = kept[np.sort(rng.choice(len(kept), cap, replace=False))]
        points[w, :len(kept)] = kept
        mask[w, :len(kept)] = True
        for old in [k for k in cache if k < (idx[0] if len(idx) else 0)]:
            del cache[old]                                       # windows advance in time
    return {f"{cloud}_all": d_all, f"{cloud}_fg": d_fg, f"{cloud}_points": points, f"{cloud}_mask": mask,
            f"{cloud}_counts": counts, f"{cloud}_names": np.array(descriptor_names(channels)),
            f"{cloud}_point_channels": np.array(["x", "y", "z"] + channels),
            f"{cloud}_background_voxels": np.array(len(background))}


def radar_track_counts(stream, windows):
    return np.array([sum(len(stream.points(int(i))) for i in stream.between(r["t_start"], r["t_end"]))
                     for r in windows], np.int64)


# Audio
def decode_audio(folder, stream):
    """The whole node as one mono 48 kHz signal (the stereo channels of MMAUD nodes are identical)."""
    import soundfile
    raw = (Path(folder) / f"{stream.name}.bin").read_bytes()
    if not raw:
        return np.zeros(0, np.float32)
    signal, rate = soundfile.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if rate != AUDIO_RATE:
        raise ValueError(f"{stream.name}: expected {AUDIO_RATE} Hz, got {rate}")
    return signal.mean(axis=1)


def window_audio(signal, stream, t0, t1, rate=AUDIO_RATE):
    """
    Samples of [t0, t1): message i starts at sample i * 1152; times between messages are interpolated.
    Returns exactly round((t1 - t0) * rate) samples (zero-padded where the node has no data) and the
    number of audio messages inside the window.
    """
    n_out = int(round((t1 - t0) * rate))
    out = np.zeros(n_out, np.float32)
    idx = stream.between(t0, t1)
    if not len(idx) or not len(signal):
        return out, 0
    starts = np.arange(len(stream)) * AUDIO_FRAME
    s0 = max(int(round(np.interp(t0, stream.time, starts))), 0)
    chunk = signal[s0: s0 + n_out]
    out[:len(chunk)] = chunk
    return out, len(idx)


_LOWPASS = None


def downsample3(signal):
    """48 kHz -> 16 kHz: windowed-sinc low-pass (cut-off 7.2 kHz, 97 taps), then every 3rd sample."""
    global _LOWPASS
    if _LOWPASS is None:
        n = np.arange(97) - 48
        _LOWPASS = (0.3 * np.sinc(0.3 * n) * np.hamming(97)).astype(np.float32)
    return np.convolve(signal, _LOWPASS, mode="same")[::3].astype(np.float32)


def mel_filterbank(n_mels, n_fft, rate, f_min=50.0, f_max=None):
    f_max = f_max or rate / 2
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    edges = hz(np.linspace(mel(f_min), mel(f_max), n_mels + 2))
    freqs = np.linspace(0, rate / 2, n_fft // 2 + 1)
    bank = np.zeros((n_mels, len(freqs)), np.float32)
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        bank[m] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return bank


def log_mel(signal16k, n_mels=64, n_fft=400, hop=160):
    """(n_mels, 1 + len // hop) log-mel of a 16 kHz signal (25 ms Hann window, 10 ms hop)."""
    padded = np.pad(signal16k, (n_fft // 2, n_fft // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop] * np.hanning(n_fft)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log(mel_filterbank(n_mels, n_fft, MEL_RATE) @ power.T + 1e-6).astype(np.float32)


def audio_features(folder, streams, windows, settings):
    n = len(windows)
    frames = np.zeros((n, len(AUDIO)), np.int64)
    mels, stats = None, np.zeros((n, len(AUDIO), 2 * settings.n_mels), np.float32)
    for k, node in enumerate(AUDIO):
        if node not in streams:
            continue
        signal = decode_audio(folder, streams[node])
        for w, row in enumerate(windows):
            samples, frames[w, k] = window_audio(signal, streams[node], row["t_start"], row["t_end"])
            if not frames[w, k]:
                continue
            mel = log_mel(downsample3(samples), settings.n_mels)
            if mels is None:
                mels = np.zeros((n, len(AUDIO), settings.n_mels, mel.shape[1]), np.float16)
            mels[w, k] = mel
            stats[w, k] = np.concatenate([mel.mean(1), mel.std(1)])
    if mels is None:
        mels = np.zeros((n, len(AUDIO), settings.n_mels, 101), np.float16)
    return {"audio_mel": mels, "audio_stats": stats, "audio_frames": frames}


def audio_windows_16k(folder, streams, windows, node):
    """1 s mono 16 kHz signals of one node for every window (for the pretrained audio model)."""
    out = np.zeros((len(windows), MEL_RATE), np.float32)
    if node not in streams:
        return out
    signal = decode_audio(folder, streams[node])
    for w, row in enumerate(windows):
        samples, n = window_audio(signal, streams[node], row["t_start"], row["t_end"])
        if n:
            out[w] = downsample3(samples)[:MEL_RATE]
    return out


# Camera
def camera_frames(stream, windows):
    """(first, stop) camera-message indices of every window."""
    return np.array([(np.searchsorted(stream.time, r["t_start"]), np.searchsorted(stream.time, r["t_end"]))
                     for r in windows], np.int64)


# One recording
AVAILABILITY = ["camera_frames", "livox_avia_points", "livox_avia_fg_points", "livox_mid360_points",
                "livox_mid360_fg_points", "radar_points", "radar_fg_points", "radar_tracks"] + \
               [f"{a}_frames" for a in AUDIO]


def recording_features(folder, windows, settings=None, log=print):
    """All CPU features of one recording, as a dict of arrays (saved by `save_features`)."""
    settings = settings or FeatureSettings()
    meta, streams = open_recording(folder)
    out = {"window_id": np.array([w["window_id"] for w in windows]), "covered": coverage(streams, windows),
           "settings": np.array(str(asdict(settings)))}
    n = len(windows)
    avail = np.zeros((n, len(AVAILABILITY)), np.int64)
    if "camera" in streams:
        out["camera_frames"] = camera_frames(streams["camera"], windows)
        avail[:, 0] = out["camera_frames"][:, 1] - out["camera_frames"][:, 0]
    for cloud in CLOUDS:
        if cloud not in streams:
            log(f"  {cloud}: no stream")
            continue
        out.update(cloud_features(streams[cloud], windows, cloud, settings))
        col = AVAILABILITY.index(f"{cloud}_points")
        avail[:, col:col + 2] = out[f"{cloud}_counts"]
        log(f"  {cloud}: {int(out[f'{cloud}_background_voxels'])} background voxels, "
            f"median {np.median(out[f'{cloud}_counts'][:, 1]):.0f} foreground points per window")
    if "radar_tracks" in streams:
        avail[:, AVAILABILITY.index("radar_tracks")] = radar_track_counts(streams["radar_tracks"], windows)
    out.update(audio_features(folder, streams, windows, settings))
    avail[:, -len(AUDIO):] = out["audio_frames"]
    out["availability"] = avail
    out["availability_names"] = np.array(AVAILABILITY)
    out["complete_extraction"] = np.array(bool(meta.get("complete")))
    return out


def save_features(features, path):
    np.savez_compressed(path, **features)


def availability_summary(features):
    """Per sensor: fraction of covered windows with no data, and the quartiles of the counts."""
    avail, covered = features["availability"], features["covered"]
    rows = {}
    for j, name in enumerate(features["availability_names"]):
        v = avail[covered, j]
        rows[str(name)] = {"empty": round(float((v == 0).mean()), 3) if len(v) else None,
                           "q25_50_75": [float(q) for q in np.percentile(v, [25, 50, 75])] if len(v) else None}
    return rows
