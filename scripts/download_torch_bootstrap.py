"""Download the pinned official wheel with ranges and verify its published SHA256."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import time
import urllib.request

URL = "https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl"
SIZE = 1039389795
SHA256 = "c301dc280458afd95450af794924c98fe07522dd148ff384739b810e3e3179f2"


def main():
    dest = Path.home() / ".cache/sm23-wheels/torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        with dest.open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == SHA256
        print("Verified existing wheel", dest, flush=True)
        return
    temporary = dest.with_suffix(".partial")
    fd = os.open(temporary, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, SIZE)
    block = 16 * 1024**2
    starts = list(range(0, SIZE, block))

    def download(start):
        end = min(start + block, SIZE) - 1
        for attempt in range(3):
            try:
                req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=40) as response:
                    assert response.status == 206
                    assert response.headers["Content-Range"] == f"bytes {start}-{end}/{SIZE}"
                    position = start
                    while data := response.read(1024**2):
                        assert position + len(data) <= end + 1
                        written = 0
                        while written < len(data):
                            written += os.pwrite(fd, data[written:], position + written)
                        position += len(data)
                    assert position == end + 1
                return start
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for n, _ in enumerate(pool.map(download, starts), 1):
                if n % 8 == 0 or n == len(starts):
                    print(f"Downloaded {n}/{len(starts)} ranges", flush=True)
    finally:
        os.close(fd)
    with temporary.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    assert actual == SHA256, (actual, SHA256)
    temporary.replace(dest)
    print("Official SHA256 verified:", dest, flush=True)


if __name__ == "__main__":
    main()
