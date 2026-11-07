"""
DroneRFc-MM (Yu et al., Science Data Bank, doi:10.57760/sciencedb.j00173.00094, CC BY-SA 4.0):
PTZ camera, fisheye camera, RF (IQ), LiDAR, mmWave radar and a four-channel microphone array recording
six DJI drones on one afternoon (2026-04-01) at one site. Sessions A1, E1 and G1 were flown a second time
about 1.5 h later (A1-2, E1-2, G1-2), which allows a cross-recording test.

The release is 75.6 GB. A Kaggle notebook can save about 20 GB of output, so the files are packed into
parts of at most PART_GB: "core" (ground truth, labels, audio, radar, both cameras, code), then the LiDAR
and the RF archives. Files are downloaded straight into the Kaggle session with parallel range requests
(one connection gives ~0.3 MB/s, the server allows many), checked against the MD5 of the file list and
kept unchanged under their release paths.
"""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASET = "0af05173ce5d45528ebd707d67f3d641"
VERSION = "V1"
ROOT = f"/{VERSION}/DronRF-MM/"
TREE_API = "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"
DOWNLOAD = "https://china.scidb.cn/download?fileId="
AGENT = {"User-Agent": "dronefuse-research-client"}
PART_GB = 15.0
CHUNK = 32 << 20


def list_files(post=None):
    """[{id, path, size, md5}] of every file in the release, path relative to the dataset root."""
    if post is None:
        import requests
        session = requests.Session()
        session.headers.update(AGENT)
        post = lambda body: session.post(TREE_API, json=body, timeout=60).json()
    files = []

    def walk(path):
        last = 0
        while True:
            page = post({"dataSetId": DATASET, "version": VERSION, "path": path,
                         "lastIndex": last, "pageSize": 500}).get("data") or []
            for item in page:
                if item["dir"]:
                    walk(item["path"])
                else:
                    files.append({"id": item["id"], "path": item["path"].replace(ROOT, "", 1),
                                  "size": item["size"], "md5": item["md5"]})
            if len(page) < 500:
                return
            last = page[-1]["index"] + 1

    walk(f"/{VERSION}")
    return sorted(files, key=lambda f: f["path"])


def plan_parts(files, part_gb=PART_GB):
    """{part name: [files]}: "core" = everything except the LiDAR and RF raw data, then the LiDAR and
    the RF archives packed in path order into parts of at most `part_gb` (lidar_1, lidar_2, rf_1, ...)."""
    def kind(f):
        if f["path"].startswith("raw_data/lidar_raw/"):
            return "lidar"
        if f["path"].startswith("RF_raw/"):
            return "rf"
        return "core"

    parts = {"core": [f for f in files if kind(f) == "core"]}
    for name in ("lidar", "rf"):
        bins, size = [[]], 0
        for f in (f for f in files if kind(f) == name):
            if bins[-1] and size + f["size"] > part_gb * 1e9:
                bins.append([])
                size = 0
            bins[-1].append(f)
            size += f["size"]
        for i, b in enumerate(b for b in bins if b):
            parts[f"{name}_{i + 1}"] = b
    return parts


def _fetch_range(url, start, end, dest, retries=6):
    """Write bytes start..end (inclusive) of `url` to `dest`, retrying on dropped connections."""
    import requests
    want = end - start + 1
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers={**AGENT, "Range": f"bytes={start}-{end}"}, stream=True,
                              timeout=(30, 120)) as r:
                if r.status_code != 206:
                    raise IOError(f"HTTP {r.status_code} for a range request")
                with open(dest, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            if Path(dest).stat().st_size == want:
                return
            raise IOError(f"short chunk ({Path(dest).stat().st_size} of {want} bytes)")
        except Exception:
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(item, dest, workers=16, chunk=CHUNK, fetch=_fetch_range):
    """Download one file with `workers` parallel range requests into `dest`, resumable: finished chunks
    are kept in `<dest>.chunks/` until the file is assembled. Checks size and MD5. Returns the speed."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    began = time.time()
    size = item["size"]
    if size == 0:
        dest.write_bytes(b"")
    else:
        parts_dir = Path(str(dest) + ".chunks")
        parts_dir.mkdir(exist_ok=True)
        ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]

        def one(r):
            path = parts_dir / f"{r[0]:012d}"
            if not (path.exists() and path.stat().st_size == r[1] - r[0] + 1):
                fetch(DOWNLOAD + item["id"], r[0], r[1], path)
            return path

        with ThreadPoolExecutor(max_workers=workers) as pool:
            paths = list(pool.map(one, ranges))
        with open(dest, "wb") as out:
            for p in paths:
                out.write(p.read_bytes())
                p.unlink()
        parts_dir.rmdir()
    if dest.stat().st_size != size:
        raise IOError(f"{dest.name}: {dest.stat().st_size} bytes, expected {size}")
    if item.get("md5") and md5_of(dest) != item["md5"]:
        dest.unlink()
        raise IOError(f"{dest.name}: MD5 mismatch (file removed, run again)")
    seconds = max(time.time() - began, 1e-6)
    return {"bytes": size, "seconds": round(seconds, 1), "mb_per_s": round(size / 1e6 / seconds, 2)}


def speed_test(files, seconds_cap=60, workers=16, fetch=_fetch_range, scratch=None):
    """MB/s of `workers` parallel 8 MB range requests on the largest file (nothing is kept)."""
    import tempfile
    big = max(files, key=lambda f: f["size"])
    folder = Path(scratch or tempfile.mkdtemp())
    folder.mkdir(parents=True, exist_ok=True)
    step = min(8 << 20, big["size"] // workers)
    ranges = [(i * step, (i + 1) * step - 1) for i in range(workers)]
    began = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda r: fetch(DOWNLOAD + big["id"], r[0], r[1], folder / f"t{r[0]}"), ranges))
    seconds = time.time() - began
    for p in folder.glob("t*"):
        p.unlink()
    mb = workers * step / 1e6
    return {"workers": workers, "mb": round(mb, 1), "seconds": round(seconds, 1), "mb_per_s": round(mb / seconds, 2)}


def import_part(part, out_dir, files=None, done_dirs=(), budget_s=None, workers=16, log=print):
    """Download every file of `part` that is not yet in `out_dir` or in an attached earlier run
    (`done_dirs`) with the right size. Stops starting new files when less than ~20 min of the budget
    is left. Writes and returns a summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    began = time.time()
    files = files if files is not None else list_files()
    todo = plan_parts(files)[part]
    summary = {"part": part, "files": len(todo), "gb": round(sum(f["size"] for f in todo) / 1e9, 2),
               "done": {}, "skipped_existing": [], "not_reached": [], "failed": {}}
    for item in todo:
        target = out_dir / item["path"]
        earlier = [Path(d) / item["path"] for d in done_dirs]
        if any(p.exists() and p.stat().st_size == item["size"] for p in [target, *earlier]):
            summary["skipped_existing"].append(item["path"])
            continue
        if budget_s and time.time() - began > budget_s - 20 * 60:
            summary["not_reached"].append(item["path"])
            continue
        try:
            log(f"{item['path']}: {item['size'] / 1e6:.0f} MB")
            speed = download(item, target, workers=workers)
            summary["done"][item["path"]] = speed
            log(f"  ok, {speed['mb_per_s']} MB/s")
        except Exception as error:
            summary["failed"][item["path"]] = f"{type(error).__name__}: {error}"
            log(f"  FAILED: {summary['failed'][item['path']]}")
        (out_dir / "import_summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "import_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
