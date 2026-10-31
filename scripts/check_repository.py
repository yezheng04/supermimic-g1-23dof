#!/usr/bin/env python3
"""Check source publication boundaries and website/evidence consistency, without CUDA."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in filter(None, paths):
        p = ROOT / name
        assert p.is_file() and not p.is_symlink(), name
        assert p.stat().st_size < 20 * 1024 * 1024, f'Large source blob: {name}'
        assert not name.startswith(('data/', 'logs/', 'reports/', '.venv/', '.runtime/', 'wandb/')), name
        assert p.suffix not in ('.pt', '.pth', '.npz', '.mp4', '.pem', '.key'), name
        if p.suffix in ('.py', '.sh', '.md', '.json', '.html', '.js', '.css', '.yml', '.txt'):
            text = p.read_text()
            assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)', text), name
            assert '.'.join(('100', '87', '81', '6')) not in text, f'Private training host in {name}'
    evidence = json.loads((ROOT/'evidence/post100k/comparison.json').read_text())
    web = json.loads((ROOT/'website/data/checkpoints.json').read_text())
    assert len(web) == len(evidence) == 8
    for r, e in zip(web, evidence):
        assert r['iteration'] == e['iteration']
        for k, group in r['groups'].items():
            for field, value in group.items():
                assert value == e['groups'][k][field], (r['iteration'], k, field)
    manifest = json.loads((ROOT/'release-assets.json').read_text())
    assert len(manifest['assets']) >= 5
    assert sum(a['kind'] == 'model' for a in manifest['assets']) == 1
    for asset in manifest['assets']:
        assert len(asset['sha256']) == 64 and asset['bytes'] > 0
        assert (ROOT / asset['path']).resolve().is_relative_to(ROOT)
    showcase = json.loads((ROOT/'website/data/showcase.json').read_text())
    videos = {a['name'] for a in manifest['assets'] if a['kind'] == 'video'}
    assert videos == {'hero.mp4'} | {c['file'] for c in showcase['clips']}
    assert showcase['recording']['fps'] == 50
    for clip in showcase['clips']:
        assert (ROOT/'website/media'/clip['poster']).is_file(), clip['poster']
        assert clip['video_frames'] == clip['reference_frames']
        assert clip['steps'] <= clip['reference_frames'] - 1
    print(f'PASS: {sum(bool(p) for p in paths)} tracked source files; 8 evaluation rows; {len(manifest['assets'])} Release assets.')


if __name__ == '__main__':
    main()
