import pytest
from supermimic.data.motion_convert import save_reference
from supermimic.data.synthetic import standing_raw


@pytest.fixture
def motion_file(tmp_path):
    path=tmp_path/"stand.npz"
    save_reference(standing_raw(),path,30.)
    return path
