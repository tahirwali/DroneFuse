"""
TSMS-Drone (Han & Jung, Scientific Data 13:407, 2026; figshare doi:10.25452/figshare.plus.30027313,
CC BY 4.0): an FMCW radar, a CW radar and an RF receiver recording four drones and a corner reflector at
the same time, hovering 2-30 m away in 2 m steps, 500 measurements per target and distance.

The release is 393 GB of 7z archives, mostly raw FMCW range-Doppler maps. We keep only the processed
227 x 227 images that come with every measurement (range-Doppler map, Doppler spectrum, power spectral
density). This module downloads one archive at a time straight into the Kaggle session, checks it against
figshare's MD5, pulls out the images, packs them into a .tar and deletes the archive, so nothing has to
pass through a personal computer.
"""
import hashlib
import json
import re
import tarfile
import time
from pathlib import Path

ARTICLE = 30027313
API = f"https://api.figshare.com/v2/articles/{ARTICLE}"
AGENT = {"User-Agent": "dronefuse-research-client"}
SENSORS = ("FMCW", "CW", "RF")
TARGETS = ("Inspire 2", "Mavic 2 Pro", "Phantom 4 Pro", "Matrice 30", "Corner Reflector")
DISTANCES = tuple(range(2, 31, 2))


def archives(article=None):
    """The data archives of the release: [{id, name, size, md5, url, sensor, distance, target, tar}].
    FMCW comes as one archive per distance and target; CW and RF as one archive each."""
    if article is None:
        import requests
        article = requests.get(API, headers=AGENT, timeout=60).json()
    folders = article.get("folder_structure", {})
    out = []
    for f in article["files"]:
        if not f["name"].endswith(".7z"):
            continue
        folder = folders.get(str(f["id"]), "")
        item = {"id": f["id"], "name": f["name"], "size": f["size"], "md5": f.get("computed_md5"),
                "url": f["download_url"]}
        if "FMCW" in folder:
            distance = int(re.search(r"(\d+)m$", folder).group(1))
            target = f["name"][:-3]
            item.update(sensor="FMCW", distance=distance, target=target,
                        tar=f"FMCW_{distance:02d}m_{target.replace(' ', '')}.tar")
        else:
            sensor = "CW" if f["name"].startswith("CW") else "RF"
            item.update(sensor=sensor, distance=None, target=None, tar=None)
        out.append(item)
    return sorted(out, key=lambda a: (a["sensor"] != "CW", a["sensor"] != "RF", a["distance"] or 0, a["target"] or ""))


def download(url, dest, size=None, md5=None, retries=6, chunk=8 << 20, log=print):
    """Stream `url` to `dest`, resuming after a dropped connection, and check size and MD5.
    Always requests the figshare link: the S3 address it redirects to expires after 10 s."""
    import requests
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(AGENT)
    began = time.time()
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if size is not None and have >= size:
            break
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with session.get(url, headers=headers, stream=True, timeout=(30, 300), allow_redirects=True) as r:
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0                                  # the server ignored the range: start again
                with open(dest, "ab" if have else "wb") as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
            if size is None:
                break
        except Exception as error:
            if attempt == retries:
                raise
            log(f"  download interrupted ({type(error).__name__}: {error}); resuming, try {attempt + 1}/{retries}")
            time.sleep(10 * attempt)
    got = dest.stat().st_size
    if size is not None and got != size:
        raise IOError(f"{dest.name}: {got} bytes, expected {size}")
    if md5:
        digest = hashlib.md5()
        with open(dest, "rb") as f:
            for block in iter(lambda: f.read(16 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != md5:
            raise IOError(f"{dest.name}: MD5 mismatch")
    seconds = time.time() - began
    return {"bytes": got, "seconds": round(seconds, 1), "mb_per_s": round(got / 1e6 / max(seconds, 1e-6), 1)}


def image_members(names):
    """The .png members of an archive (the processed images; the raw .mat files are skipped)."""
    return [n for n in names if n.lower().endswith(".png") and "Image File" in n]


def pack_images(archive, item, out_dir, scratch):
    """Extract the images of one archive and pack them into .tar files in `out_dir`.
    FMCW: one tar for the archive. CW and RF: one tar per distance, holding all five targets.
    Inside a tar every image is <target>/<target>_<distance>m_<index>.png, as in the release."""
    import py7zr
    out_dir, scratch = Path(out_dir), Path(scratch)
    out_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive) as z:
        wanted = image_members(z.getnames())
        z.extract(path=scratch, targets=wanted)
    counts = {}
    if item["sensor"] == "FMCW":
        groups = {item["tar"]: [(scratch / n, f"{item['target']}/{Path(n).name}") for n in wanted]}
    else:
        groups = {}
        for n in wanted:
            distance, target = n.split("/")[0], n.split("/")[1]          # e.g. "10m/Inspire 2/Image File/x.png"
            tar = f"{item['sensor']}_{int(distance[:-1]):02d}m.tar"
            groups.setdefault(tar, []).append((scratch / n, f"{target}/{Path(n).name}"))
    for tar, files in groups.items():
        with tarfile.open(out_dir / tar, "w") as t:
            for path, arcname in sorted(files, key=lambda p: p[1]):
                t.add(path, arcname=arcname)
                path.unlink()
        counts[tar] = len(files)
    return counts


def import_release(out_dir, scratch, done_dirs=(), only=None, budget_s=None, log=print):
    """Download and pack every archive whose tars are not yet in `out_dir` or in one of `done_dirs`
    (an earlier part attached as input). `only`: e.g. {"CW", "RF"} or a set of FMCW distances in metres.
    Stops before an archive when less than ~25 min of the budget is left. Returns a summary."""
    out_dir, scratch = Path(out_dir), Path(scratch)
    out_dir.mkdir(parents=True, exist_ok=True)
    began = time.time()
    have = {p.name for d in [out_dir, *map(Path, done_dirs)] for p in Path(d).rglob("*.tar")}
    summary = {"done": {}, "skipped_existing": [], "not_reached": [], "failed": {}}
    for item in archives():
        key = item["sensor"] if item["sensor"] != "FMCW" else item["distance"]
        if only and key not in only and item["sensor"] not in only:
            continue
        expected = [item["tar"]] if item["tar"] else [f"{item['sensor']}_{d:02d}m.tar" for d in DISTANCES]
        label = item["tar"] or f"{item['sensor']} (all distances)"
        if all(t in have for t in expected):
            summary["skipped_existing"].append(label)
            continue
        if budget_s and time.time() - began > budget_s - 25 * 60:
            summary["not_reached"].append(label)
            continue
        archive = scratch / f"{item['id']}.7z"
        try:
            log(f"{label}: downloading {item['size'] / 1e9:.1f} GB")
            speed = download(item["url"], archive, item["size"], item["md5"], log=log)
            counts = pack_images(archive, item, out_dir, scratch / "x")
            summary["done"][label] = {"images": counts, **speed}
            log(f"  {sum(counts.values())} images in {len(counts)} tar(s); {speed['mb_per_s']} MB/s")
        except Exception as error:
            summary["failed"][label] = f"{type(error).__name__}: {error}"
            log(f"  FAILED: {summary['failed'][label]}")
        finally:
            archive.unlink(missing_ok=True)
        (out_dir / "import_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def read_tar(path):
    """{member name: PNG bytes} of one packed tar (for the feature step)."""
    with tarfile.open(path) as t:
        return {m.name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
