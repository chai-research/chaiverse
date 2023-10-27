import os
from mock import patch

import pytest

from chaiverse.dev import logging_utils


class dummy_mixin:
    pass


class dummy_class:

    @logging_utils.logging_manager()
    def __init__(self, a, b=123, c='abc', d=dummy_mixin()):
        pass

    @logging_utils.logging_manager(submit=False)
    def run(self):
        pass


@pytest.fixture(autouse="session")
def mock_post():
    with patch("logging_utils.requests.post") as func:
        yield func


@logging_utils.auto_authenticate
def dummy_function(info, developer_key=None):
    return info, developer_key


def test_logging_manager_submit_correct_payload(tmp_path, mock_post):
    write_tmp_key_to_file(tmp_path, 'CR-12345')
    dummy_class(321, c='cba')
    expected_submission = {
            'b': 123,
            'c': 'cba',
            'd': 'dummy_mixin',
            'args1': 'dummy_class',
            'args2': 321}
    headers = {"Authorization": "Bearer CR-12345"}
    expected_url = logging_utils.BASE_URL + logging_utils.CHAIVERSE_ANALYTIC_ENDPOINT
    mock_post.assert_called_once_with(
            url=expected_url,
            json=expected_submission,
            headers=headers)


def test_logging_manager_not_submit_payload(tmp_path, mock_post):
    write_tmp_key_to_file(tmp_path, 'CR-12345')
    cls = dummy_class(321)
    mock_post.call_count == 1
    cls.run()
    mock_post.call_count == 1


def test_auto_authenticate_wrapper_loads_from_cached_key(tmp_path):
    write_tmp_key_to_file(tmp_path, 'CR-12345')
    submission_id, developer_key = dummy_function('test123')
    assert developer_key == 'CR-12345'
    assert submission_id == 'test123'


def test_auto_authenticate_wrapper_loads_default_keys(tmp_path):
    write_tmp_key_to_file(tmp_path, 'CR-12345', file_name='wrong_name.json')
    submission_id, developer_key = dummy_function('test123')
    assert developer_key == logging_utils.DEFAULT_DEVELOPER_KEY
    assert submission_id == 'test123'


def write_tmp_key_to_file(tmp_path, key, file_name='developer_key.json'):
    mocked_dir = str(tmp_path / 'guanaco')
    os.makedirs(mocked_dir)
    os.environ['GUANACO_DATA_DIR'] = mocked_dir
    cached_key_path = os.path.join(mocked_dir, file_name)
    write_key(cached_key_path, key)


def write_key(path, data):
    with open(path, 'w') as f:
        f.write(data)
