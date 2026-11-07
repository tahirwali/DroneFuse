"""
Offline checks for the DroneRFc-MM import: walking the file list, packing parts under the output limit,
and the parallel, resumable, MD5-checked download. A fake server stands in for Science Data Bank.
"""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronefuse import rfcmm

BLOB = bytes(range(256)) * 400          # 102,400 bytes of known content
MD5 = hashlib.md5(BLOB).hexdigest()


def fake_fetch(calls):
    def fetch(url, start, end, dest, retries=6):
        calls.append((start, end))
        Path(dest).write_bytes(BLOB[start:end + 1])
    return fetch


def test_list_files_walks_folders_and_pages():
    tree = {
        "/V1": [{"dir": True, "path": "/V1/DronRF-MM"}],
        "/V1/DronRF-MM": [{"dir": False, "id": "r", "path": "/V1/DronRF-MM/README.md", "size": 5, "md5": "m"},
                          {"dir": True, "path": "/V1/DronRF-MM/raw_data"}],
        "/V1/DronRF-MM/raw_data": [{"dir": False, "id": "g", "path": "/V1/DronRF-MM/raw_data/GT.csv",
                                    "size": 7, "md5": "n"}],
    }
    files = rfcmm.list_files(post=lambda body: {"data": tree.get(body["path"], [])})
    assert [f["path"] for f in files] == ["README.md", "raw_data/GT.csv"]
    assert files[0] == {"id": "r", "path": "README.md", "size": 5, "md5": "m"}


def test_parts_keep_core_together_and_split_big_raw_data():
    gb = int(1e9)
    files = [{"path": "README.md", "size": 1}, {"path": "raw_data/audio_raw/A1.dat", "size": gb},
             {"path": "raw_data/lidar_raw/A1.pcd", "size": 9 * gb}, {"path": "raw_data/lidar_raw/B1.pcd", "size": 9 * gb},
             {"path": "RF_raw/A1.rar", "size": 5 * gb}, {"path": "RF_raw/C1.rar", "size": 7 * gb}]
    parts = rfcmm.plan_parts(files, part_gb=15)
    assert list(parts) == ["core", "lidar_1", "lidar_2", "rf_1"]
    assert [f["path"] for f in parts["core"]] == ["README.md", "raw_data/audio_raw/A1.dat"]
    assert all(sum(f["size"] for f in p) <= 15 * gb for p in parts.values())


def test_download_assembles_chunks_checks_md5_and_resumes(tmp_path):
    item = {"id": "x", "path": "a/b.bin", "size": len(BLOB), "md5": MD5}
    calls = []
    dest = tmp_path / "a" / "b.bin"
    # a chunk left over from an interrupted run is reused, not fetched again
    chunks = Path(str(dest) + ".chunks")
    chunks.mkdir(parents=True)
    (chunks / f"{0:012d}").write_bytes(BLOB[:10000])
    speed = rfcmm.download(item, dest, workers=4, chunk=10000, fetch=fake_fetch(calls))
    assert dest.read_bytes() == BLOB and speed["bytes"] == len(BLOB)
    assert (0, 9999) not in calls and len(calls) == 10 and not chunks.exists()


def test_download_removes_a_file_with_the_wrong_md5(tmp_path):
    item = {"id": "x", "path": "b.bin", "size": len(BLOB), "md5": "0" * 32}
    with pytest.raises(IOError, match="MD5"):
        rfcmm.download(item, tmp_path / "b.bin", chunk=50000, fetch=fake_fetch([]))
    assert not (tmp_path / "b.bin").exists()


def test_import_part_skips_files_already_present_and_respects_the_budget(tmp_path, monkeypatch):
    files = [{"id": "1", "path": "README.md", "size": len(BLOB), "md5": MD5},
             {"id": "2", "path": "raw_data/GT_raw/A1.csv", "size": len(BLOB), "md5": MD5}]
    monkeypatch.setattr(rfcmm, "_fetch_range", None)
    got = []
    monkeypatch.setattr(rfcmm, "download", lambda item, target, workers=16: (
        got.append(item["path"]), Path(target).parent.mkdir(parents=True, exist_ok=True),
        Path(target).write_bytes(BLOB), {"mb_per_s": 1.0})[-1])
    earlier = tmp_path / "part1"
    (earlier / "raw_data/GT_raw").mkdir(parents=True)
    (earlier / "raw_data/GT_raw/A1.csv").write_bytes(BLOB)
    result = rfcmm.import_part("core", tmp_path / "out", files=files, done_dirs=[earlier], log=lambda *a: None)
    assert got == ["README.md"] and result["skipped_existing"] == ["raw_data/GT_raw/A1.csv"]
    late = rfcmm.import_part("core", tmp_path / "out2", files=files, budget_s=60, log=lambda *a: None)
    assert late["not_reached"] == ["README.md", "raw_data/GT_raw/A1.csv"]
