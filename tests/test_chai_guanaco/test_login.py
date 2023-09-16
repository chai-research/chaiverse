import os

import pytest

from chai_guanaco.login_cli import auto_authenticate


@auto_authenticate
def dummy_function(submission_id, developer_key=None):
    return submission_id, developer_key


def test_auto_authenticate_wrapper_raises_when_not_logged_in_and_no_keys_passed():
    with pytest.raises(Exception, match="chai-guanaco login"):
        dummy_function("submission_id")


def test_auto_authenticate_wrapper_ignores_cached_key_when_devkey_is_passed(tmp_path):
    write_tmp_key_to_file(tmp_path, 'CR-12345')
    submission_id, developer_key = dummy_function("gpt-j-6b", "CR-2345")
    assert developer_key == 'CR-2345'
    assert submission_id == 'gpt-j-6b'


def test_auto_authenticate_wrapper_does_not_overwrite_cached_key_when_devkey_is_passed(tmp_path):
    write_tmp_key_to_file(tmp_path, 'CR-12345')
    dummy_function("gpt-j-6b", "CR-2345")
    developer_key_path = os.path.join(str(tmp_path / 'guanaco'), 'developer_key.json')
    with open(developer_key_path, 'r') as f:
        data = f.read()
    assert data == 'CR-12345'


def test_auto_authenticate_wrapper_loads_from_cached_key(tmp_path):
    write_tmp_key_to_file(tmp_path, 'CR-12345')
    submission_id, developer_key = dummy_function("gpt-j-6b")
    assert developer_key == 'CR-12345'
    assert submission_id == 'gpt-j-6b'


def write_tmp_key_to_file(tmp_path, key):
    mocked_dir = str(tmp_path / 'guanaco')
    os.makedirs(mocked_dir)
    os.environ['GUANACO_DATA_DIR'] = mocked_dir
    cached_key_path = os.path.join(mocked_dir, 'developer_key.json')
    write_key(cached_key_path, key)


def write_key(path, data):
    with open(path, 'w') as f:
        f.write(data)
