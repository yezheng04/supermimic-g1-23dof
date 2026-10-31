#!/usr/bin/env python3
"""Download public Release assets and verify the tracked SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', action='store_true', help='Also fetch the model-only 80k bundle')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'release-assets.json').read_text())
    for item in manifest['assets']:
        if item['kind'] == 'model' and not args.models:
            continue
        target = (ROOT / item['path']).resolve()
        if not target.is_relative_to(ROOT) or Path(item['name']).name != item['name']:
            raise ValueError('Invalid asset destination')
        if target.is_file() and target.stat().st_size == item['bytes'] and digest(target) == item['sha256']:
            print('Verified:', item['path'])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.download-', dir=target.parent) as tmp:
            downloaded = Path(tmp) / item['name']
            url = (f'https://github.com/{manifest["repository"]}/releases/download/'
                   f'{quote(manifest["tag"], safe="")}/{quote(item["name"], safe="")}')
            request = Request(url, headers={'User-Agent': 'G1-in-Motion-asset-downloader'})
            with urlopen(request, timeout=120) as response, downloaded.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            if downloaded.stat().st_size != item['bytes'] or digest(downloaded) != item['sha256']:
                raise ValueError(f'Asset checksum mismatch: {item["name"]}')
            downloaded.replace(target)
        print('Downloaded and verified:', item['path'])


if __name__ == '__main__':
    main()
