"""
Step 0 probe of MMAUD on Kaggle: what is in each recording, in what format, and how fast can
it be fetched. Every function returns plain dicts/lists so results can be saved as JSON.
"""
import re
from collections import defaultdict
from pathlib import Path

from .remote import ConcatFile, HttpFile, hf_url
from .rosbag_index import (bag_time_span, choose_sample_chunks, deserialize, read_index,
                           sample_messages, summarise_value, topic_table)

PART = re.compile(r"^(?P<base>.+\.bag)\.part(?P<n>\d+)$")


def bag_sources(files):
    """{bag path: [(part path, size), ...]} for every .bag in the inventory; byte parts are grouped in order."""
    bags = defaultdict(list)
    for path, size in files:
        match = PART.match(path)
        if match:
            bags[match["base"]].append((int(match["n"]), path, size))
        elif path.endswith(".bag"):
            bags[path].append((0, path, size))
    return {bag: [(p, s) for _, p, s in sorted(parts)] for bag, parts in sorted(bags.items())}


def open_bag(repo_id, parts, session=None):
    readers = [HttpFile(hf_url(repo_id, path), size=size, session=session) for path, size in parts]
    return readers[0] if len(readers) == 1 else ConcatFile(readers)


def probe_bag_index(repo_id, bag_path, parts, session=None):
    """Read one bag's index over HTTP. Returns (summary dict, index, reader)."""
    reader = open_bag(repo_id, parts, session)
    index = read_index(reader)
    start, end = bag_time_span(index)
    sizes = index.chunk_byte_sizes()
    summary = {
        "bag": bag_path,
        "size_gb": round(reader.size / 1e9, 3),
        "parts": len(parts),
        "chunks": len(index.chunks),
        "median_chunk_mb": round(sorted(sizes)[len(sizes) // 2] / 1e6, 2) if sizes else None,
        "largest_chunk_mb": round(max(sizes) / 1e6, 2) if sizes else None,
        "start_time": start,
        "seconds": round(end - start, 1) if start is not None else None,
        "index_mb_fetched": round(reader.bytes_fetched / 1e6, 2),
        "topics": topic_table(index),
    }
    return summary, index, reader


def probe_bag_messages(reader, index, n_spread=4, max_chunk_mb=64, max_total_mb=400):
    """
    Download a few chunks, decode the first message of every topic, and estimate each topic's
    total uncompressed size (mean message size x message count from the index).
    """
    chunk_ids = choose_sample_chunks(index, n_spread=n_spread, max_chunk_bytes=max_chunk_mb * 1_000_000)
    sampled = sample_messages(reader, index, chunk_ids, max_total_bytes=max_total_mb * 1_000_000)
    counts = defaultdict(int)
    for chunk in index.chunks:
        for conn_id, n in chunk.counts.items():
            counts[conn_id] += n
    topics = []
    for conn_id, conn in sorted(index.connections.items(), key=lambda kv: kv[1].topic):
        entry = sampled["connections"].get(conn_id)
        row = {"topic": conn.topic, "type": conn.msgtype, "messages": counts[conn_id]}
        if entry is None:
            row["sampled"] = 0
        else:
            mean_bytes = entry["bytes"] / entry["seen"]
            row.update(sampled=entry["seen"], mean_kb=round(mean_bytes / 1e3, 2),
                       estimated_total_gb=round(mean_bytes * counts[conn_id] / 1e9, 3))
            try:
                row["first_message"] = summarise_value(deserialize(entry["first_raw"], conn.msgtype, conn.msgdef))
            except Exception as error:
                row["first_message"] = f"could not decode: {type(error).__name__}: {error}"[:300]
        topics.append(row)
    ratio = (sampled["plain_message_bytes"] / sampled["compressed_bytes"]) if sampled["compressed_bytes"] else None
    return {"chunks_read": sampled["chunks_read"],
            "compressed_mb_read": round(sampled["compressed_bytes"] / 1e6, 1),
            "compression_ratio": round(ratio, 2) if ratio else None,
            "topics": topics}


def capture_audio(reader, index, out_dir, seconds=5.0, max_chunks=600, chunks_per_request=40):
    """
    Read consecutive chunks from the middle of the bag (`chunks_per_request` per HTTP request)
    until `seconds` of audio are covered, join each audio topic's payloads into one file, and
    try to decode it. MMAUD's audio_common_msgs/AudioData carries an encoded stream (MP3 in the
    bags checked), so the file is saved with the extension its first bytes indicate.
    """
    from .media import inspect_audio
    from .rosbag_index import read_chunk_range

    audio = {cid: c for cid, c in index.connections.items() if c.msgtype.endswith("AudioData")}
    if not audio:
        return {"error": "no AudioData topics in this bag"}
    payloads = {cid: bytearray() for cid in sorted(audio, key=lambda c: audio[c].topic)}
    first_time, last_time = {}, {}
    begin = len(index.chunks) // 2
    stop = min(begin + max_chunks, len(index.chunks))
    first = begin
    while first < stop:
        last = min(first + chunks_per_request, stop) - 1
        for conn_id, stamp, raw in read_chunk_range(reader, index, first, last):
            if conn_id in payloads:
                payloads[conn_id] += raw[4:]          # AudioData: uint8[] = uint32 length + bytes
                first_time.setdefault(conn_id, stamp)
                last_time[conn_id] = stamp
        first = last + 1
        if all(last_time.get(c, 0) - first_time.get(c, 0) >= seconds for c in audio):
            break
    results = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for conn_id, data in payloads.items():
        head = bytes(data[:4])
        is_mp3 = head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)
        is_wav = head == b"RIFF"
        suffix = ".mp3" if is_mp3 else ".wav" if is_wav else ".bin"
        path = out_dir / (audio[conn_id].topic.strip("/").replace("/", "_") + suffix)
        path.write_bytes(data)
        row = {"topic": audio[conn_id].topic, "file": path.name, "bytes": len(data), "head_hex": bytes(data[:8]).hex(),
               "bag_seconds": round(last_time.get(conn_id, 0) - first_time.get(conn_id, 0), 2),
               "kbit_per_s": round(len(data) * 8 / 1000 / max(last_time.get(conn_id, 0) - first_time.get(conn_id, 0), 1e-6), 1)}
        if suffix != ".bin":
            try:
                row["decoded"] = inspect_audio(path)
            except Exception as error:
                row["decoded"] = f"could not decode: {type(error).__name__}: {error}"[:200]
        results.append(row)
    return {"chunks_read": first - begin, "topics": results}


def sample_files(repo_id, files, folders, per_folder, scratch, thumbnail_dir=None):
    """Download the first `per_folder` files of each folder into `scratch` and describe them."""
    from huggingface_hub import hf_hub_download

    from .hf_inventory import files_in
    from .media import inspect_file, save_thumbnail

    results = []
    for folder in folders:
        listed = files_in(files, folder)
        if not listed:
            results.append({"folder": folder, "error": "folder not found in the inventory"})
            continue
        for path, size in listed[:per_folder]:
            local = hf_hub_download(repo_id, path, repo_type="dataset", local_dir=str(scratch))
            info = {"folder": folder, "file": path.rsplit("/", 1)[1], "files_in_folder": len(listed)}
            info.update(inspect_file(local))
            results.append(info)
            if thumbnail_dir is not None and info.get("kind") == "image":
                thumb = Path(thumbnail_dir) / (folder.replace("/", "__") + ".png")
                if not thumb.exists():
                    save_thumbnail(local, thumb)
            Path(local).unlink(missing_ok=True)
    return results


def format_report(summary):
    """Readable text report of everything the probe found (also saved as probe_report.txt)."""
    lines = []
    env = summary.get("environment") or {}
    lines.append("MACHINE")
    lines.append(f"  CPUs {env.get('cpu_count')} | RAM total {env.get('ram_total_gb')} GB, available "
                 f"{env.get('ram_available_gb')} GB, container limit {env.get('ram_container_limit_gb')} GB | "
                 f"GPU {env.get('gpu')} | run type {env.get('kaggle_run_type')} | internet {env.get('internet_status')}")
    for folder, disk in (env.get("disk") or {}).items():
        lines.append(f"  disk {folder:16s} total {disk['total_gb']:7.1f} GB, free {disk['free_gb']:7.1f} GB")
    lines.append(f"  scratch folder used: {summary.get('scratch')}")

    speed = summary.get("download_speed")
    if speed:
        lines.append(f"\nDOWNLOAD SPEED  {speed['mb_per_s']} MB/s ({speed['megabytes']} MB in {speed['seconds']} s)")

    inventory = summary.get("inventory_totals")
    if inventory:
        lines.append(f"\nHUB INVENTORY  {summary.get('inventory_files')} files, {summary.get('inventory_gb')} GB")
        for folder, n, gb in inventory[:40]:
            lines.append(f"  {gb:9.3f} GB  {n:6d} files  {folder}")

    bags = summary.get("bags") or []
    if bags:
        lines.append("\nRECORDINGS (from each bag's index; nothing else downloaded)")
        for bag in bags:
            if "error" in bag:
                lines.append(f"  {bag['bag']}: ERROR {bag['error']}")
                continue
            lines.append(f"  {bag['bag']}  {bag['size_gb']} GB, {bag['seconds']} s, {bag['chunks']} chunks "
                         f"(median {bag['median_chunk_mb']} MB, largest {bag['largest_chunk_mb']} MB)")
            for t in bag["topics"]:
                lines.append(f"      {t['topic']:42s} {t['type']:38s} {t['messages']:8d} msgs  {t['rate_hz']} Hz")

    for name, detail in (summary.get("bag_messages") or {}).items():
        lines.append(f"\nDECODED SAMPLE MESSAGES  {name}")
        if "error" in detail:
            lines.append(f"  ERROR {detail['error']}")
            continue
        lines.append(f"  {detail['chunks_read']} chunks, {detail['compressed_mb_read']} MB read, "
                     f"bz2 ratio {detail['compression_ratio']}")
        for t in detail["topics"]:
            size = f"{t.get('mean_kb')} kB/msg, ~{t.get('estimated_total_gb')} GB total" if t.get("sampled") else "not sampled"
            lines.append(f"  {t['topic']:42s} {size}")
            if "first_message" in t:
                lines.append(f"      {str(t['first_message'])[:400]}")

    for name, detail in (summary.get("audio_capture") or {}).items():
        lines.append(f"\nAUDIO CAPTURE  {name}")
        if "error" in detail:
            lines.append(f"  ERROR {detail['error']}")
            continue
        for t in detail["topics"]:
            lines.append(f"  {t['topic']:16s} {t['file']:22s} {t['bag_seconds']} s of bag time, {t['kbit_per_s']} kbit/s, "
                         f"head {t['head_hex']} -> {t.get('decoded')}")

    samples = summary.get("file_samples") or []
    if samples:
        lines.append("\nSAMPLE FILES")
        for s in samples:
            details = {k: v for k, v in s.items() if k not in ("folder", "file")}
            lines.append(f"  {s['folder']}/{s.get('file', '')}: {details}")

    steps = summary.get("steps") or []
    failed = [s for s in steps if s["status"] != "ok"]
    lines.append("\nSTEPS: " + ("all succeeded" if not failed else
                                "; ".join(f"{s['step']} -> {s['status']} ({s['message']})" for s in failed)))
    return "\n".join(lines)
