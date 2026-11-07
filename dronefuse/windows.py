"""
Analysis windows and the train/val/test splits. These are built once and saved in splits/.

A window is 1 s long and is used only when every sensor is recording and the Leica tracker can
see the drone. Each recording is cut into ten time blocks, assigned T T V T S T T V T S (60/20/20),
so every split covers the start, middle and end of a flight. Windows keep a gap from blocks of a
different split, so neighbouring windows can't leak across splits.

protocol_sets() builds the evaluation protocols from these windows: closed-set, the two
cross-site directions (V1 -> V2/V3 and back), and leave-one-drone-type-out.
"""
import csv
import json
from pathlib import Path

import numpy as np

from .recordings import BY_NAME, DRONE_TYPES, RECORDINGS

WINDOW_S = 1.0
HOP_S = 0.5
N_BLOCKS = 10
BLOCK_PATTERN = "TTVTSTTVTS"
GAP_S = 1.0
SPLIT_NAMES = {"T": "train", "V": "val", "S": "test"}
WINDOW_FIELDS = ["window_id", "recording", "version", "drone_type", "t_start", "t_end", "block", "split",
                 "gt_messages"]


def usable_span(sensor_spans, truth_times):
    """Intersection of all sensor spans and the ground-truth span."""
    start = max([s for s, _ in sensor_spans.values()] + [float(truth_times[0])])
    end = min([e for _, e in sensor_spans.values()] + [float(truth_times[-1])])
    return start, end


def make_windows(recording, sensor_spans, truth_times, window_s=WINDOW_S, hop_s=HOP_S, n_blocks=N_BLOCKS,
                 pattern=BLOCK_PATTERN, gap_s=GAP_S):
    truth_times = np.sort(np.asarray(truth_times, dtype=np.float64))
    start, end = usable_span(sensor_spans, truth_times)
    if end - start < n_blocks * (window_s + 2 * gap_s):
        raise ValueError(f"{recording.name}: usable span {end - start:.1f} s is too short for {n_blocks} blocks")
    edges = np.linspace(start, end, n_blocks + 1)
    split_of = [SPLIT_NAMES[c] for c in pattern]
    rows = []
    t = start
    while t + window_s <= end + 1e-9:
        block = min(int(np.searchsorted(edges, t, side="right") - 1), n_blocks - 1)
        t_end = t + window_s
        inside = t_end <= edges[block + 1] + 1e-9
        left_ok = block == 0 or split_of[block - 1] == split_of[block] or t >= edges[block] + gap_s - 1e-9
        right_ok = (block == n_blocks - 1 or split_of[block + 1] == split_of[block]
                    or t_end <= edges[block + 1] - gap_s + 1e-9)
        n_truth = int(np.searchsorted(truth_times, t_end) - np.searchsorted(truth_times, t))
        if inside and left_ok and right_ok and n_truth > 0:
            rows.append({"window_id": f"{recording.name}_{len(rows):05d}", "recording": recording.name,
                         "version": recording.version, "drone_type": recording.drone_type,
                         "t_start": round(t, 6), "t_end": round(t_end, 6), "block": block,
                         "split": split_of[block], "gt_messages": n_truth})
        t += hop_s
    return rows


def save_windows(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WINDOW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_windows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["t_start"], row["t_end"] = float(row["t_start"]), float(row["t_end"])
        row["block"], row["gt_messages"] = int(row["block"]), int(row["gt_messages"])
    return rows


PROTOCOLS = ["closed", "v1_to_v2", "v2_to_v1"] + [f"loo_{t}" for t in DRONE_TYPES] + ["xrec_s1_to_s2", "xrec_s2_to_s1"]

# Car-park types flown twice. The cross-recording protocols train on one flight of each and test on the
# other, so a model can't score by recognising the recording. The Mavic 3E (one flight only) is the unknown.
FLIGHTS = {"s1": ["V2_m300s1", "V2_m30s1", "V2_p4s1"], "s2": ["V2_m300s2", "V2_m30s2", "V2_p4s2"]}
XREC_UNKNOWN = "V2_m3e"


def protocol_sets(rows, protocol):
    """{"known": [...], "unknown": [...], "train": rows, "val": rows, "test": rows} for one protocol."""
    if protocol == "closed":
        known = list(DRONE_TYPES)
        train = [r for r in rows if r["split"] == "train"]
        val = [r for r in rows if r["split"] == "val"]
        test = [r for r in rows if r["split"] == "test"]
    elif protocol in ("v1_to_v2", "v2_to_v1"):
        source, target = ("V1", "V2_V3") if protocol == "v1_to_v2" else ("V2_V3", "V1")
        known = sorted({r.drone_type for r in RECORDINGS if r.version == source})
        train = [r for r in rows if r["version"] == source and r["split"] == "train"]
        val = [r for r in rows if r["version"] == source and r["split"] == "val"]
        test = [r for r in rows if r["version"] == target]
    elif protocol.startswith("loo_"):
        held_out = protocol[4:]
        if held_out not in DRONE_TYPES:
            raise ValueError(f"unknown drone type {held_out!r}")
        known = [t for t in DRONE_TYPES if t != held_out]
        train = [r for r in rows if r["split"] == "train" and r["drone_type"] != held_out]
        val = [r for r in rows if r["split"] == "val" and r["drone_type"] != held_out]
        test = [r for r in rows if r["split"] == "test"]
    elif protocol in ("xrec_s1_to_s2", "xrec_s2_to_s1"):
        source, target = ("s1", "s2") if protocol == "xrec_s1_to_s2" else ("s2", "s1")
        known = sorted({BY_NAME[name].drone_type for name in FLIGHTS[source]})
        train = [r for r in rows if r["recording"] in FLIGHTS[source] and r["split"] == "train"]
        val = [r for r in rows if r["recording"] in FLIGHTS[source] and r["split"] == "val"]
        test = ([r for r in rows if r["recording"] in FLIGHTS[target]]
                + [r for r in rows if r["recording"] == XREC_UNKNOWN and r["split"] == "test"])
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    unknown = sorted({r["drone_type"] for r in test} - set(known))
    return {"known": known, "unknown": unknown, "train": train, "val": val, "test": test}


def protocol_table(rows):
    """Window counts per protocol and split (known / unknown), for the docs and the paper."""
    table = {}
    for protocol in PROTOCOLS:
        sets = protocol_sets(rows, protocol)
        table[protocol] = {"known": sets["known"], "unknown": sets["unknown"],
                           "train": len(sets["train"]), "val": len(sets["val"]),
                           "test_known": sum(r["drone_type"] in sets["known"] for r in sets["test"]),
                           "test_unknown": sum(r["drone_type"] not in sets["known"] for r in sets["test"])}
    return table


# Building the splits from the Hub (bag indexes + ground-truth bags, ~30 MB in total)
SPAN_TOPICS = ["/usb_cam/image_raw", "/livox_avia/lidar", "/livox_mid360/lidar", "/radar_enhanced_pcl",
               "/audio1/audio", "/audio2/audio", "/audio3/audio", "/audio4/audio"]


def recording_timing(repo_id, recording, files, session=None):
    """Sensor spans (from the sensor bag index) and ground-truth times (from the truth bag)."""
    from .extract import read_truth
    from .probe import bag_sources, open_bag
    from .rosbag_index import read_index
    sources = bag_sources(files)
    sensor_index = read_index(open_bag(repo_id, sources[recording.sensor_bag], session))
    spans = {}
    for conn in sensor_index.connections.values():
        if conn.topic in SPAN_TOPICS:
            times = [(c.start, c.end) for c in sensor_index.chunks if c.counts.get(conn.id)]
            spans[conn.topic] = (min(t[0] for t in times), max(t[1] for t in times))
    truth_reader = open_bag(repo_id, sources[recording.truth_bag], session)
    truth = read_truth(truth_reader, read_index(truth_reader))
    return spans, truth["relative"]["time"]


def build_splits(repo_id, files, out_dir, session=None, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, spans_out = [], {}
    for recording in RECORDINGS:
        spans, truth_times = recording_timing(repo_id, recording, files, session)
        windows = make_windows(recording, spans, truth_times)
        start, end = usable_span(spans, truth_times)
        spans_out[recording.name] = {"usable_start": start, "usable_end": end, "usable_s": round(end - start, 1),
                                     "truth_messages": int(len(truth_times)), "windows": len(windows),
                                     "sensor_spans": spans}
        log(f"{recording.name}: usable {end - start:.1f} s, {len(windows)} windows")
        rows += windows
    save_windows(rows, out_dir / "windows.csv")
    info = {"window_s": WINDOW_S, "hop_s": HOP_S, "n_blocks": N_BLOCKS, "block_pattern": BLOCK_PATTERN,
            "gap_s": GAP_S, "drone_types": DRONE_TYPES,
            "recordings": {r.name: {"version": r.version, "drone_type": r.drone_type, "model": r.model}
                           for r in RECORDINGS},
            "spans": spans_out, "protocols": protocol_table(rows)}
    with open(out_dir / "splits_info.json", "w") as f:
        json.dump(info, f, indent=1)
    return rows, info
