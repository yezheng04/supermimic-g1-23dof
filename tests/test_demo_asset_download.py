"""Runtime extraction must not replace source or escape the generated asset tree."""
import importlib.util
from pathlib import Path
import zipfile
import pytest

spec=importlib.util.spec_from_file_location('download_demo_assets',Path(__file__).parents[1]/'scripts/download_demo_assets.py')
download=importlib.util.module_from_spec(spec);spec.loader.exec_module(download)


def test_runtime_extracts_generated_files_without_touching_source(tmp_path,monkeypatch):
    monkeypatch.setattr(download,'ROOT',tmp_path)
    source=tmp_path/'website/demo/main.js';source.parent.mkdir(parents=True);source.write_text('source')
    archive=tmp_path/'runtime.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('assets/actor.onnx',b'model');z.writestr('vendor/module.js','runtime')
    download.unpack_browser(archive)
    assert source.read_text()=='source'
    assert (source.parent/'assets/actor.onnx').read_bytes()==b'model'


@pytest.mark.parametrize('name',['assets/../../escape','/assets/file','main.js','vendor/../main.js'])
def test_runtime_rejects_source_overwrite_and_traversal(tmp_path,monkeypatch,name):
    monkeypatch.setattr(download,'ROOT',tmp_path)
    archive=tmp_path/'runtime.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr(name,'bad')
    with pytest.raises(ValueError):download.unpack_browser(archive)


def test_runtime_rejects_symlinks(tmp_path,monkeypatch):
    monkeypatch.setattr(download,'ROOT',tmp_path);archive=tmp_path/'runtime.zip'
    entry=zipfile.ZipInfo('assets/link');entry.create_system=3;entry.external_attr=(0o120777<<16)
    with zipfile.ZipFile(archive,'w') as z:z.writestr(entry,'../../source')
    with pytest.raises(ValueError):download.unpack_browser(archive)
