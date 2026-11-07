"""
Step 1: stream an MMAUD bag and keep a compact copy of every sensor, without storing the bag.

The bag is split into tasks of a few consecutive chunks. Each worker process downloads its byte
range in one request, decompresses it, decodes the topics we need and hands back small payloads
(JPEG camera frames, packed LiDAR/radar points, raw MP3 audio). The main process writes these to
the stream files in store.py.

Whether a message is kept depends only on its position in its topic, so the output is the same
no matter how the bag was cut up or which worker finished first.
"""
import io
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .remote import ConcatFile, HttpFile, LocalFile, hf_url
from .rosbag_index import (OP_CHUNK, _chunk_messages, _op, bz2_decoder, iter_records, read_chunk_range,
                           topic_table)
from .ros1 import Schema
from .store import StreamWriter, dtype_to_json, save_meta

STREAMS = {
    "/usb_cam/image_raw": ("camera", "image"),
    "/livox_avia/lidar": ("livox_avia", "livox"),
    "/livox_mid360/lidar": ("livox_mid360", "livox"),
    "/radar_enhanced_pcl": ("radar", "pointcloud"),
    "/radar_trk": ("radar_tracks", "pointcloud"),
    "/audio1/audio": ("audio1", "audio"),
    "/audio2/audio": ("audio2", "audio"),
    "/audio3/audio": ("audio3", "audio"),
    "/audio4/audio": ("audio4", "audio"),
}
LIDAR_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("reflectivity", "u1")])


@dataclass
class ExtractSettings:
    camera_every: int = 3              # 30 Hz -> 10 Hz
    lidar_every: int = 2               # 10 Hz -> 5 Hz
    camera_half: str = "left"          # "left", "right" or "both" (the bag image is a side-by-side stereo pair)
    camera_size: tuple = None          # (width, height) to resize to; None keeps the native 1280x960 half
    jpeg_quality: int = 92             # (the drone is a ~5-pixel speck in the fisheye sky view: no downscaling)
    chunks_per_task: int = 32
    bz2: str = "bz2"                   # "bz2" or "indexed_bzip2" (see choose_bz2_decoder)
    tasks_per_worker: int = 40         # a worker process is replaced by a fresh one after this many tasks (~2.7 GB)
    max_pool_restarts: int = 3         # new worker pools allowed per recording after a worker dies


# Sources (re-opened inside each worker process)
_READERS = {}


def open_source(spec):
    """spec = ("local", path) | ("local_parts", [paths]) | ("hf", repo_id, [(path, size), ...])."""
    key = repr(spec)
    if key not in _READERS:
        if spec[0] == "local":
            _READERS[key] = LocalFile(spec[1])
        elif spec[0] == "local_parts":
            _READERS[key] = ConcatFile([LocalFile(p) for p in spec[1]])
        elif spec[0] == "hf":
            import requests
            session = requests.Session()
            parts = [HttpFile(hf_url(spec[1], path), size=size, session=session) for path, size in spec[2]]
            _READERS[key] = parts[0] if len(parts) == 1 else ConcatFile(parts)
        else:
            raise ValueError(f"unknown source {spec!r}")
    return _READERS[key]


# Planning
def make_plan(index, settings):
    """{connection id: what to do with its messages} for the topics in STREAMS."""
    plan = {}
    for conn_id, conn in index.connections.items():
        if conn.topic not in STREAMS:
            continue
        stream, kind = STREAMS[conn.topic]
        every = settings.camera_every if kind == "image" else settings.lidar_every if kind == "livox" else 1
        plan[conn_id] = {"stream": stream, "kind": kind, "every": every, "topic": conn.topic,
                         "msgtype": conn.msgtype, "msgdef": conn.msgdef}
    counts = message_counts(index)
    for conn_id, entry in plan.items():
        if entry["kind"] == "image":           # one full stereo frame from the middle, to check the layout
            middle = counts.get(conn_id, 0) // 2
            entry["preview_ordinal"] = middle - middle % entry["every"]
    return plan


def message_counts(index):
    counts = {}
    for chunk in index.chunks:
        for conn_id, n in chunk.counts.items():
            counts[conn_id] = counts.get(conn_id, 0) + n
    return counts


def make_tasks(index, plan, chunks_per_task, max_chunks=None):
    """Consecutive chunk ranges, each with the position (ordinal) of its first message per topic."""
    n_chunks = len(index.chunks) if max_chunks is None else min(max_chunks, len(index.chunks))
    ordinals = {conn_id: 0 for conn_id in plan}
    tasks = []
    for first in range(0, n_chunks, chunks_per_task):
        last = min(first + chunks_per_task, n_chunks) - 1
        start = index.chunks[first].position
        stop = index.chunks[last + 1].position if last + 1 < len(index.chunks) else index.index_pos
        tasks.append({"first": first, "last": last, "start": start, "stop": stop, "ordinals": dict(ordinals)})
        for chunk in index.chunks[first:last + 1]:
            for conn_id in plan:
                ordinals[conn_id] += chunk.counts.get(conn_id, 0)
    return tasks


# The worker
_SCHEMAS = {}


def _schema(entry):
    key = (entry["msgtype"], entry["msgdef"])
    if key not in _SCHEMAS:
        _SCHEMAS[key] = Schema(entry["msgtype"], entry["msgdef"])
    return _SCHEMAS[key]


def _jpeg(array, size, quality):
    from PIL import Image
    image = Image.fromarray(np.ascontiguousarray(array))
    if size is not None and image.size != tuple(size):
        image = image.resize(tuple(size), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _image_pixels(message):
    height, width, encoding = message["height"], message["width"], message["encoding"]
    data = message["data"]
    channels = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(encoding)
    if channels is None:
        raise ValueError(f"image encoding {encoding!r} is not handled")
    pixels = data[:height * message["step"]].reshape(height, message["step"])[:, :width * channels]
    pixels = pixels.reshape(height, width, channels)
    if encoding == "bgr8":
        pixels = pixels[..., ::-1]
    return pixels[..., 0] if channels == 1 else pixels


def _half(pixels, half):
    width = pixels.shape[1]
    return pixels if half == "both" else pixels[:, :width // 2] if half == "left" else pixels[:, width // 2:]


def _decode(entry, raw, settings, stats, want_preview):
    """Return (header stamp or None, payload bytes, preview JPEG or None)."""
    kind = entry["kind"]
    if kind == "audio":
        (n,) = np.frombuffer(raw, dtype="<u4", count=1)
        stats["bytes_in"] = stats.get("bytes_in", 0) + int(n)
        return None, bytes(raw[4:4 + int(n)]), None
    message = _schema(entry).decode(raw)
    stamp = message["header"]["stamp"]
    if kind == "image":
        pixels = _image_pixels(message)
        stats["source"] = f"{message['encoding']} {message['width']}x{message['height']}"
        payload = _jpeg(_half(pixels, settings["camera_half"]), settings["camera_size"], settings["jpeg_quality"])
        preview = _jpeg(pixels, (pixels.shape[1] // 2, pixels.shape[0] // 2), 85) if want_preview else None
        return stamp, payload, preview
    if kind == "livox":
        points = message["points"]
        keep = (points["x"] != 0) | (points["y"] != 0) | (points["z"] != 0)
        out = np.empty(int(keep.sum()), dtype=LIDAR_DTYPE)
        for name in ("x", "y", "z", "reflectivity"):
            out[name] = points[name][keep]
        stats["declared_points"] = stats.get("declared_points", 0) + int(message["point_num"])
        stats["points_in_messages"] = stats.get("points_in_messages", 0) + len(points)
        stats["kept_points"] = stats.get("kept_points", 0) + len(out)
        return stamp, out.tobytes(), None
    if kind == "pointcloud":
        points, channels = message["points"], message["channels"]
        names = tuple(c["name"] for c in channels)
        if stats.setdefault("channel_names", list(names)) != list(names):
            stats["channel_names_changed"] = stats.get("channel_names_changed", 0) + 1
        table = np.full((len(points), 3 + len(channels)), np.nan, dtype=np.float32)
        table[:, 0], table[:, 1], table[:, 2] = points["x"], points["y"], points["z"]
        for j, channel in enumerate(channels):
            values = np.asarray(channel["values"], dtype=np.float32)
            n = min(len(values), len(points))
            table[:n, 3 + j] = values[:n]
        stats["points"] = stats.get("points", 0) + len(points)
        return stamp, table.tobytes(), None
    raise ValueError(f"unknown stream kind {kind!r}")


def process_task(spec, plan, task, settings):
    """Download, decompress and decode one chunk range. Runs in a worker process."""
    reader = open_source(spec)
    buffer = reader.read(task["start"], task["stop"] - task["start"])
    decompress = bz2_decoder(settings.get("bz2", "bz2"))
    ordinals = dict(task["ordinals"])
    results, stats, previews = {}, {}, {}
    for header, data, _ in iter_records(buffer):
        if _op(header) != OP_CHUNK:
            continue
        for conn_id, bag_time, raw in _chunk_messages(header["compression"].decode(), data, decompress):
            entry = plan.get(conn_id)
            if entry is None:
                continue
            ordinal = ordinals[conn_id]
            ordinals[conn_id] += 1
            if ordinal % entry["every"]:
                continue
            stream_stats = stats.setdefault(entry["stream"], {})
            want_preview = ordinal == entry.get("preview_ordinal")
            try:
                stamp, payload, preview = _decode(entry, raw, settings, stream_stats, want_preview)
            except Exception as error:
                stream_stats["decode_errors"] = stream_stats.get("decode_errors", 0) + 1
                stream_stats["last_error"] = f"{type(error).__name__}: {error}"[:200]
                continue
            results.setdefault(entry["stream"], []).append((bag_time, stamp, payload))
            if preview is not None:
                previews[entry["stream"]] = preview
    return {"results": results, "stats": stats, "previews": previews, "bytes": len(buffer),
            "chunks": task["last"] - task["first"] + 1}


def choose_bz2_decoder(reader, index, n_chunks=8):
    """
    Time the standard bz2 module and `indexed_bzip2` (if installed) on `n_chunks` real chunks from
    the middle of the bag; return the faster one that gives byte-identical output, with the speeds.
    """
    middle = len(index.chunks) // 2
    last = min(middle + n_chunks, len(index.chunks)) - 1
    stop = index.chunks[last + 1].position if last + 1 < len(index.chunks) else index.index_pos
    buffer = reader.read(index.chunks[middle].position, stop - index.chunks[middle].position)
    chunks = [data for header, data, _ in iter_records(buffer) if _op(header) == OP_CHUNK]
    megabytes = sum(map(len, chunks)) / 1e6
    speeds, reference, best = {}, None, "bz2"
    for name in ("bz2", "indexed_bzip2"):
        try:
            decompress = bz2_decoder(name)
            began = time.time()
            output = [decompress(data) for data in chunks]
            speeds[name] = round(megabytes / max(time.time() - began, 1e-9), 2)
        except Exception as error:
            speeds[name] = f"unavailable ({type(error).__name__})"
            continue
        if reference is None:
            reference = output
        elif output != reference:
            speeds[name] = "rejected: output differs from the bz2 module"
            continue
        if speeds[name] > speeds[best]:
            best = name
    return best, {"compressed_mb_per_s_per_core": speeds, "chosen": best}


# The main loop
def _merge_stats(total, part):
    for stream, values in part.items():
        target = total.setdefault(stream, {})
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                target[key] = target.get(key, 0) + value
            elif key == "last_error":
                target[key] = value
            else:
                target.setdefault(key, value)


def memory_in_use_gb():
    """Memory used by the processes of this container (not counting the file cache), in GB; None if unknown."""
    for path, key in (("/sys/fs/cgroup/memory.stat", "anon"), ("/sys/fs/cgroup/memory/memory.stat", "total_rss")):
        try:
            for line in Path(path).read_text().splitlines():
                name, value = line.split()
                if name == key:
                    return round(int(value) / 1e9, 1)
        except (OSError, ValueError):
            continue
    try:
        import psutil
        return round((psutil.virtual_memory().total - psutil.virtual_memory().available) / 1e9, 1)
    except ImportError:
        return None


def _worker_pool(workers, tasks_per_worker):
    """Worker processes started fresh ("spawn"), each replaced after `tasks_per_worker` tasks."""
    import multiprocessing
    import sys
    context = multiprocessing.get_context("spawn")
    if sys.version_info >= (3, 11) and tasks_per_worker:
        return ProcessPoolExecutor(max_workers=workers, mp_context=context, max_tasks_per_child=tasks_per_worker)
    return ProcessPoolExecutor(max_workers=workers, mp_context=context)


def extract_recording(spec, index, out_folder, settings=None, workers=4, deadline=None, max_chunks=None,
                      log=print, progress_every=0.05, worker=process_task):
    """
    Extract one sensor bag into `out_folder`. Returns the summary also saved in meta.json.
    `deadline` (time.time() value): stop starting new tasks after it (the recording is marked incomplete).
    `max_chunks`: only the first chunks (for quick tests).
    `worker`: the function run per task (tests replace it to simulate a crashing worker).
    """
    settings = settings or ExtractSettings()
    settings_dict = asdict(settings)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    plan = make_plan(index, settings)
    tasks = make_tasks(index, plan, settings.chunks_per_task, max_chunks)
    writers = {}
    for entry in plan.values():
        writers.setdefault(entry["stream"], StreamWriter(out_folder, entry["stream"]))

    stats, failures, fetched, chunks_done = {}, [], 0, 0
    total_chunks = sum(t["last"] - t["first"] + 1 for t in tasks)
    began, next_report, stopped_early = time.time(), progress_every, False

    def handle(result):
        nonlocal fetched, chunks_done, next_report
        for stream, rows in result["results"].items():
            writer = writers[stream]
            for bag_time, stamp, payload in rows:
                writer.append(bag_time, stamp, payload)
        for stream, jpeg in result["previews"].items():
            (out_folder / f"{stream}_preview_stereo.jpg").write_bytes(jpeg)
        _merge_stats(stats, result["stats"])
        fetched += result["bytes"]
        chunks_done += result["chunks"]
        if chunks_done / total_chunks >= next_report or chunks_done == total_chunks:
            elapsed = time.time() - began
            eta = elapsed / chunks_done * (total_chunks - chunks_done)
            log(f"  {chunks_done}/{total_chunks} chunks, {fetched / 1e9:.1f} GB read, "
                f"{fetched / 1e6 / max(elapsed, 1e-6):.0f} MB/s, {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min, "
                f"memory in use {memory_in_use_gb()} GB")
            next_report += progress_every

    def run_here(task_ids):
        nonlocal stopped_early
        for k in task_ids:
            if deadline is not None and time.time() > deadline:
                stopped_early = True
                return
            for attempt in (1, 2):
                try:
                    handle(worker(spec, plan, tasks[k], settings_dict))
                    break
                except Exception as error:
                    if attempt == 2:
                        failures.append({"chunks": [tasks[k]["first"], tasks[k]["last"]],
                                         "error": f"{type(error).__name__}: {error}"[:300]})

    if workers <= 1:
        run_here(range(len(tasks)))
    else:
        # The heavy work always runs in worker processes, never in the notebook's own process:
        # if the notebook process runs out of memory, Kaggle kills the kernel and the whole run is lost.
        # Each worker is replaced by a fresh process after `tasks_per_worker` tasks, so any memory a
        # worker accumulates is given back. If a worker dies anyway (out of memory), a new pool with one
        # worker fewer takes over the unfinished tasks.
        attempts, queue, pool_workers, restarts = {}, list(range(len(tasks))), workers, 0
        while queue:
            pending = {}
            try:
                with _worker_pool(pool_workers, settings.tasks_per_worker) as pool:
                    while queue or pending:
                        while queue and len(pending) < 2 * pool_workers:
                            if deadline is not None and time.time() > deadline:
                                stopped_early = True
                                queue.clear()
                                break
                            k = queue.pop(0)
                            attempts[k] = attempts.get(k, 0) + 1
                            pending[pool.submit(worker, spec, plan, tasks[k], settings_dict)] = k
                        if not pending:
                            break
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            try:
                                result = future.result()
                            except BrokenProcessPool:
                                raise
                            except Exception as error:
                                k = pending.pop(future)
                                if attempts[k] < 2:
                                    log(f"  task {k} failed ({type(error).__name__}: {error}); trying once more")
                                    queue.append(k)
                                else:
                                    failures.append({"chunks": [tasks[k]["first"], tasks[k]["last"]],
                                                     "error": f"{type(error).__name__}: {error}"[:300]})
                                continue
                            pending.pop(future)
                            handle(result)
            except BrokenProcessPool as error:
                queue = sorted(set(queue) | set(pending.values()))
                restarts += 1
                memory = memory_in_use_gb()
                if restarts > settings.max_pool_restarts:
                    log(f"  worker pool broke ({error}); memory in use {memory} GB; giving up on the last "
                        f"{len(queue)} tasks (the recording is marked incomplete)")
                    for k in queue:
                        failures.append({"chunks": [tasks[k]["first"], tasks[k]["last"]],
                                         "error": f"worker pool broke {restarts} times"})
                    break
                pool_workers = max(1, pool_workers - 1)
                log(f"  worker pool broke ({error}); memory in use {memory} GB; "
                    f"new pool with {pool_workers} workers for the {len(queue)} unfinished tasks")

    streams = {}
    for entry in plan.values():
        name = entry["stream"]
        if name in streams:
            continue
        messages, size = writers[name].close()
        info = {"topic": entry["topic"], "kind": entry["kind"], "every": entry["every"], "messages": messages,
                "mb": round(size / 1e6, 1)}
        info.update(stats.get(name, {}))
        if entry["kind"] == "image":
            info.update(payload="jpeg", half=settings.camera_half,
                        size=list(settings.camera_size) if settings.camera_size else "native")
        elif entry["kind"] == "livox":
            info.update(payload="points", dtype=dtype_to_json(LIDAR_DTYPE))
        elif entry["kind"] == "pointcloud":
            names = ["x", "y", "z"] + list(info.get("channel_names", []))
            info.update(payload="points", dtype=[[n, "<f4"] for n in names])
        elif entry["kind"] == "audio":
            info.update(payload="mp3 frames")
        streams[name] = info
    summary = {"chunks_total": len(index.chunks), "chunks_done": chunks_done, "chunks_failed": failures,
               "complete": chunks_done == len(index.chunks) and not failures,
               "stopped_by_time_budget": stopped_early, "gb_read": round(fetched / 1e9, 2),
               "minutes": round((time.time() - began) / 60, 1), "settings": settings_dict, "streams": streams,
               "topics_in_bag": topic_table(index)}
    return summary


# Ground truth
def read_truth(reader, index):
    """All Leica messages of a ground-truth bag: {"relative"/"absolute": {time, stamp, xyz[, quat]}}."""
    messages = read_chunk_range(reader, index, 0, len(index.chunks) - 1)
    schemas, out = {}, {}
    for conn_id, conn in index.connections.items():
        if conn.topic.startswith("/leica/"):
            schemas[conn_id] = (conn.topic.rsplit("/", 1)[1], Schema(conn.msgtype, conn.msgdef))
    for conn_id, bag_time, raw in messages:
        if conn_id not in schemas:
            continue
        name, schema = schemas[conn_id]
        message = schema.decode(raw)
        rows = out.setdefault(name, {"time": [], "stamp": [], "xyz": [], "quat": []})
        rows["time"].append(bag_time)
        rows["stamp"].append(message["header"]["stamp"])
        if "pose" in message:
            p, q = message["pose"]["position"], message["pose"]["orientation"]
            rows["xyz"].append((p["x"], p["y"], p["z"]))
            rows["quat"].append((q["x"], q["y"], q["z"], q["w"]))
        else:
            p = message["point"]
            rows["xyz"].append((p["x"], p["y"], p["z"]))
    result = {}
    for name, rows in out.items():
        order = np.argsort(rows["time"])
        result[name] = {key: np.asarray(values)[order] for key, values in rows.items() if len(values)}
    return result


def save_truth(truth, folder):
    arrays = {f"{name}_{key}": value for name, rows in truth.items() for key, value in rows.items()}
    np.savez(Path(folder) / "ground_truth.npz", **arrays)
    return {name: {"messages": len(rows["time"]), "start": float(rows["time"][0]), "end": float(rows["time"][-1])}
            for name, rows in truth.items()}


# One recording, end to end (used by the Step 1 notebooks)
def run_recording(repo_id, recording, sources, out_root, settings, workers, deadline=None, session=None, log=print):
    """Ground truth + sensor streams + meta.json + verification for one Recording."""
    from .probe import open_bag
    from .rosbag_index import read_index
    folder = Path(out_root) / recording.name
    folder.mkdir(parents=True, exist_ok=True)
    truth_reader = open_bag(repo_id, sources[recording.truth_bag], session)
    truth_info = save_truth(read_truth(truth_reader, read_index(truth_reader)), folder)
    sensor_parts = sources[recording.sensor_bag]
    index = read_index(open_bag(repo_id, sensor_parts, session))
    log(f"{recording.name}: {sum(s for _, s in sensor_parts) / 1e9:.1f} GB, {len(index.chunks)} chunks")
    summary = extract_recording(("hf", repo_id, sensor_parts), index, folder, settings, workers, deadline, log=log)
    meta = {"recording": asdict(recording), "repo_id": repo_id, "sensor_parts": sensor_parts,
            "ground_truth": truth_info}
    meta.update(summary)
    save_meta(folder, meta)
    meta["verify"] = verify_recording(folder)
    save_meta(folder, meta)
    return meta


def format_extract_report(metas, environment=None, bz2_choice=None):
    lines = []
    if environment:
        lines.append(f"MACHINE  CPUs {environment.get('cpu_count')} | RAM {environment.get('ram_total_gb')} GB | "
                     f"run type {environment.get('kaggle_run_type')}")
    if bz2_choice:
        lines.append(f"BZ2 DECODER  {bz2_choice}")
    for name, meta in metas.items():
        if meta is None or "streams" not in meta:
            lines.append(f"\n{name}: FAILED (see the traceback under its cell)")
            continue
        status = "complete" if meta["complete"] else (
            "STOPPED BY TIME BUDGET" if meta["stopped_by_time_budget"] else "INCOMPLETE")
        lines.append(f"\n{name}: {status} | {meta['chunks_done']}/{meta['chunks_total']} chunks | "
                     f"{meta['gb_read']} GB read in {meta['minutes']} min | failed ranges: {len(meta['chunks_failed'])}")
        for truth, info in meta.get("ground_truth", {}).items():
            lines.append(f"  ground truth {truth:9s} {info['messages']} messages")
        for stream, info in meta["streams"].items():
            check = meta.get("verify", {}).get(stream, {})
            extra = ""
            if info["kind"] == "livox" and info.get("points_in_messages"):
                extra = (f" | non-zero points {info['kept_points'] / info['points_in_messages']:.1%},"
                         f" {info['kept_points'] / max(info['messages'], 1):.0f}/scan")
            elif info["kind"] == "pointcloud":
                extra = f" | {info.get('points', 0) / max(info['messages'], 1):.1f} points/scan"
            if info.get("decode_errors"):
                extra += f" | DECODE ERRORS {info['decode_errors']}: {info.get('last_error')}"
            lines.append(f"  {stream:13s} {info['messages']:7d} msgs {check.get('rate_hz')} Hz "
                         f"{info['mb']:9.1f} MB{extra}")
    return "\n".join(lines)


# Checking a finished recording
def verify_recording(folder):
    """Re-open every stream and decode its first and last payload. Returns a dict of findings."""
    from PIL import Image

    from .store import StreamReader, load_meta
    meta = load_meta(folder)
    report = {}
    for name, info in meta["streams"].items():
        stream = StreamReader(folder, name, info)
        row = {"messages": len(stream)}
        if len(stream):
            row["start"], row["end"] = float(stream.time[0]), float(stream.time[-1])
            row["rate_hz"] = round((len(stream) - 1) / max(row["end"] - row["start"], 1e-9), 2)
            for which, i in (("first", 0), ("last", len(stream) - 1)):
                if info["payload"] == "jpeg":
                    row[which] = list(Image.open(io.BytesIO(stream.payload(i))).size)
                elif info["payload"] == "points":
                    points = stream.points(i)
                    row[which] = {"points": len(points)}
                else:
                    head = stream.payload(i)[:2]
                    row[which] = {"mp3_sync": len(head) == 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0}
        report[name] = row
    return report
