"""
Read byte ranges of files on the Hugging Face Hub (or on disk) without downloading them whole.

The Hub serves dataset files with HTTP range support, so a 20 GB ROS bag can be examined by
fetching only its header and index (a few MB at the end of the file) plus a few sample chunks.
A bag split into byte parts (MMAUD's m3e.bag.part001..003) is read as one virtual file.
"""
import time
from urllib.parse import quote

HF_ENDPOINT = "https://huggingface.co"


def hf_url(repo_id, path, revision="main"):
    return f"{HF_ENDPOINT}/datasets/{repo_id}/resolve/{revision}/{quote(path)}"


class LocalFile:
    def __init__(self, path):
        import os
        self.path = str(path)
        self.size = os.path.getsize(self.path)

    def read(self, offset, length):
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(length)


class HttpFile:
    """A remote file read by HTTP range requests, with retries."""

    def __init__(self, url, size=None, session=None, retries=4, timeout=(20, 180)):
        import requests
        self.url = url
        self.session = session or requests.Session()
        self.retries = retries
        self.timeout = timeout
        self.size = size if size is not None else self._remote_size()
        self.bytes_fetched = 0

    def _remote_size(self):
        response = self.session.get(self.url, headers={"Range": "bytes=0-0"}, timeout=self.timeout,
                                    allow_redirects=True)
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        return int(response.headers["Content-Length"])

    def read(self, offset, length):
        if length <= 0:
            return b""
        end = min(offset + length, self.size) - 1
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.get(self.url, headers={"Range": f"bytes={offset}-{end}"},
                                            timeout=self.timeout, allow_redirects=True, stream=True)
                response.raise_for_status()
                if response.status_code != 206 and offset > 0:
                    raise IOError("server ignored the byte range (status 200)")
                data = bytearray()
                for block in response.iter_content(1 << 20):
                    data += block
                    if len(data) >= end - offset + 1:
                        break
                response.close()
                data = bytes(data[:end - offset + 1])
                if len(data) != end - offset + 1:
                    raise IOError(f"short read: {len(data)} of {end - offset + 1} bytes")
                self.bytes_fetched += len(data)
                return data
            except Exception as error:
                if attempt == self.retries:
                    raise
                wait = 5 * attempt
                print(f"  range read failed ({error}); retry {attempt}/{self.retries - 1} in {wait} s")
                time.sleep(wait)


class ConcatFile:
    """Several files read as one, in order (for bags delivered as byte parts)."""

    def __init__(self, parts):
        self.parts = list(parts)
        self.starts = []
        position = 0
        for part in self.parts:
            self.starts.append(position)
            position += part.size
        self.size = position

    def read(self, offset, length):
        out = bytearray()
        end = min(offset + length, self.size)
        for part, start in zip(self.parts, self.starts):
            part_end = start + part.size
            if part_end <= offset or start >= end:
                continue
            local_start = max(offset, start) - start
            local_end = min(end, part_end) - start
            out += part.read(local_start, local_end - local_start)
        return bytes(out)

    @property
    def bytes_fetched(self):
        return sum(getattr(p, "bytes_fetched", 0) for p in self.parts)


def measure_download_speed(url, megabytes=300, offset=0, session=None):
    """Stream `megabytes` from `url`, discard them, and return MB/s."""
    import requests
    session = session or requests.Session()
    wanted = megabytes * 1_000_000
    began = time.time()
    received = 0
    with session.get(url, headers={"Range": f"bytes={offset}-{offset + wanted - 1}"},
                     stream=True, timeout=(20, 300), allow_redirects=True) as response:
        response.raise_for_status()
        for block in response.iter_content(4 << 20):
            received += len(block)
            if received >= wanted:
                break
    seconds = time.time() - began
    return {"megabytes": round(received / 1e6, 1), "seconds": round(seconds, 1),
            "mb_per_s": round(received / 1e6 / max(seconds, 1e-6), 1)}
