#!/usr/bin/env python3
"""Preview the website locally with byte ranges for accurate video seeking."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        path = Path(self.translate_path(self.path))
        header = self.headers.get('Range', '')
        match = re.fullmatch(r'bytes=(\d+)-(\d*)', header)
        if not path.is_file() or not match:
            return super().send_head()
        size = path.stat().st_size
        start = int(match[1])
        end = min(int(match[2]) if match[2] else size - 1, size - 1)
        if start > end or start >= size:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        stream = path.open('rb')
        stream.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(self.remaining))
        self.end_headers()
        return stream

    def copyfile(self, source, output):
        if self.remaining is None:
            return super().copyfile(source, output)
        while self.remaining:
            chunk = source.read(min(65536, self.remaining))
            if not chunk:
                break
            output.write(chunk)
            self.remaining -= len(chunk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'website'
    with ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(root))) as server:
        print(f'Preview: http://127.0.0.1:{args.port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
