"""
Offline checks for the TSMS-Drone import: reading the figshare file list and packing the images of
small synthetic archives laid out like the release. No internet needed.
"""
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
py7zr = pytest.importorskip("py7zr")
from dronefuse import tsms

ARTICLE = {
    "files": [
        {"id": 1, "name": "CW Radar.7z", "size": 10, "computed_md5": "a", "download_url": "u1"},
        {"id": 2, "name": "Inspire 2.7z", "size": 10, "computed_md5": "b", "download_url": "u2"},
        {"id": 3, "name": "Example_CW_radar_processing.m", "size": 1, "download_url": "u3"},
        {"id": 4, "name": "RF Receiver.7z", "size": 10, "computed_md5": "c", "download_url": "u4"},
    ],
    "folder_structure": {"1": "TSMS-Drone/Dataset", "2": "TSMS-Drone/Dataset/FMCW Radar/28m",
                         "3": "TSMS-Drone/Sample Code/CW Radar", "4": "TSMS-Drone/Dataset"},
}


def test_archives_are_read_from_the_figshare_record():
    items = tsms.archives(ARTICLE)
    assert [i["sensor"] for i in items] == ["CW", "RF", "FMCW"]
    fmcw = items[2]
    assert fmcw["distance"] == 28 and fmcw["target"] == "Inspire 2" and fmcw["tar"] == "FMCW_28m_Inspire2.tar"


def _archive(path, members):
    with py7zr.SevenZipFile(path, "w") as z:
        for name in members:
            z.writestr(name.encode() if name.endswith(".png") else b"raw", name)


def test_fmcw_archive_becomes_one_tar_of_images(tmp_path):
    names = [f"Image File/Inspire 2_28m_{i:03d}.png" for i in range(1, 4)] + \
            [f"Raw File/Inspire 2_28m_{i:03d}.mat" for i in range(1, 4)]
    _archive(tmp_path / "a.7z", names)
    item = tsms.archives(ARTICLE)[2]
    counts = tsms.pack_images(tmp_path / "a.7z", item, tmp_path / "out", tmp_path / "x")
    assert counts == {"FMCW_28m_Inspire2.tar": 3}
    with tarfile.open(tmp_path / "out" / "FMCW_28m_Inspire2.tar") as t:
        assert t.getnames() == [f"Inspire 2/Inspire 2_28m_{i:03d}.png" for i in range(1, 4)]


def test_cw_archive_becomes_one_tar_per_distance(tmp_path):
    names = []
    for d in (2, 10):
        for target in ("Inspire 2", "Corner Reflector"):
            names += [f"{d}m/{target}/Image File/{target}_{d}m_001.png", f"{d}m/{target}/Raw File/{target}_{d}m_001.mat"]
    _archive(tmp_path / "cw.7z", names)
    counts = tsms.pack_images(tmp_path / "cw.7z", tsms.archives(ARTICLE)[0], tmp_path / "out", tmp_path / "x")
    assert counts == {"CW_02m.tar": 2, "CW_10m.tar": 2}
    images = tsms.read_tar(tmp_path / "out" / "CW_10m.tar")
    assert set(images) == {"Inspire 2/Inspire 2_10m_001.png", "Corner Reflector/Corner Reflector_10m_001.png"}
    assert images["Inspire 2/Inspire 2_10m_001.png"].startswith(b"10m/Inspire 2")


def test_import_release_skips_parts_already_done_and_respects_only(tmp_path, monkeypatch):
    names = [f"Image File/Inspire 2_28m_{i:03d}.png" for i in range(1, 3)]
    _archive(tmp_path / "fmcw.7z", names)
    items = tsms.archives(ARTICLE)
    monkeypatch.setattr(tsms, "archives", lambda article=None: items)
    fetched = []

    def fake_download(url, dest, size=None, md5=None, log=print):
        fetched.append(url)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes((tmp_path / "fmcw.7z").read_bytes())
        return {"bytes": 1, "seconds": 1.0, "mb_per_s": 1.0}

    monkeypatch.setattr(tsms, "download", fake_download)
    out = tmp_path / "out"
    result = tsms.import_release(out, tmp_path / "scratch", only={28}, log=lambda *a: None)
    assert fetched == ["u2"] and list(result["done"]) == ["FMCW_28m_Inspire2.tar"]
    assert (out / "FMCW_28m_Inspire2.tar").exists() and not list((tmp_path / "scratch").glob("*.7z"))
    again = tsms.import_release(tmp_path / "out2", tmp_path / "scratch", done_dirs=[out], only={28}, log=lambda *a: None)
    assert again["skipped_existing"] == ["FMCW_28m_Inspire2.tar"] and fetched == ["u2"]
    late = tsms.import_release(tmp_path / "out3", tmp_path / "scratch", only={28}, budget_s=60, log=lambda *a: None)
    assert late["not_reached"] == ["FMCW_28m_Inspire2.tar"]
