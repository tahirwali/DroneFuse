"""
On-disk format of the extracted dataset: one folder per recording, one "stream" per sensor.

A stream is two files:
  <name>.bin        the payloads appended one after another (JPEG bytes, packed point arrays, MP3 frames)
  <name>.index.npz  per message: bag time, header stamp, byte offset, byte length - sorted by bag time
plus an entry in the recording's meta.json (payload kind, numpy dtype of points, channel names...).
Later steps read a payload with one seek, or memory-map the whole .bin file.
"""
import json
from pathlib import Path

import numpy as np


class StreamWriter:
    def __init__(self, folder, name):
        self.folder = Path(folder)
        self.name = name
        self.file = open(self.folder / f"{name}.bin", "wb")
        self.offset = 0
        self.rows = []          # (bag time, header stamp, offset, length)

    def append(self, bag_time, stamp, payload):
        self.file.write(payload)
        self.rows.append((bag_time, stamp, self.offset, len(payload)))
        self.offset += len(payload)

    def close(self):
        self.file.close()
        rows = sorted(self.rows)
        np.savez(self.folder / f"{self.name}.index.npz",
                 time=np.array([r[0] for r in rows], dtype=np.float64),
                 stamp=np.array([np.nan if r[1] is None else r[1] for r in rows], dtype=np.float64),
                 offset=np.array([r[2] for r in rows], dtype=np.int64),
                 length=np.array([r[3] for r in rows], dtype=np.int64))
        return len(rows), self.offset


class StreamReader:
    def __init__(self, folder, name, meta=None):
        self.folder = Path(folder)
        self.name = name
        index = np.load(self.folder / f"{name}.index.npz")
        self.time, self.stamp = index["time"], index["stamp"]
        self.offset, self.length = index["offset"], index["length"]
        self.meta = meta if meta is not None else load_meta(folder)["streams"][name]
        self._dtype = np.dtype([tuple(f) for f in self.meta["dtype"]]) if self.meta.get("dtype") else None

    def __len__(self):
        return len(self.time)

    def payload(self, i):
        with open(self.folder / f"{self.name}.bin", "rb") as f:
            f.seek(int(self.offset[i]))
            return f.read(int(self.length[i]))

    def points(self, i):
        """Structured numpy array (for point-cloud streams)."""
        return np.frombuffer(self.payload(i), dtype=self._dtype)

    def between(self, t0, t1):
        """Indices of messages with bag time in [t0, t1)."""
        return np.arange(np.searchsorted(self.time, t0), np.searchsorted(self.time, t1))


def dtype_to_json(dtype):
    return [[name, dtype.fields[name][0].str] for name in dtype.names]


def save_meta(folder, meta):
    with open(Path(folder) / "meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)


def load_meta(folder):
    with open(Path(folder) / "meta.json") as f:
        return json.load(f)
