import os
from mock import patch
import pytz
import pytest
from datetime import datetime

from chaiverse import logging_utils


class dummy_mixin:
    pass


class dummy_class:

    @logging_utils.logging_manager('test')
    def __init__(self, a, b=123, c='abc', d=dummy_mixin()):
        pass

    @logging_utils.logging_manager('run')
    def run(self, n_jobs=1):
        pass


@pytest.fixture(autouse="session")
def mock_post():
    with patch("chaiverse.logging_utils.requests.post") as func:
        yield func


@pytest.fixture
def mock_utc_now():
    fake_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
    with patch('chaiverse.logging_utils.get_utc_now', return_value=fake_time) as func:
        yield func


@logging_utils.auto_authenticate
def dummy_function(info, developer_key=None):
    return info, developer_key


def test_logging_manager_submit_correct_payload(tmp_path, mock_post, mock_utc_now):
    developer_key = 'CR-12345'
    write_tmp_key_to_file(tmp_path, developer_key)
    model = dummy_class(321, c='cba')
    expected_submission = {
            'developer_key': developer_key,
            'parameters': {
                'b': 123,
                'c': 'cba',
                'd': 'dummy_mixin',
                'args1': 'dummy_class',
                'args2': 321,
                'timestamp': "2023-01-01 00:00:00+00:00"},
            }
    headers = {"Authorization": f"Bearer {developer_key}"}
    expected_url = logging_utils.get_logging_endpoint('test')
    mock_post.assert_called_once_with(
            url=expected_url,
            json=expected_submission,
            headers=headers,
            timeout=5)

    model.run(n_jobs=10)
    expected_submission = {
            'developer_key': developer_key,
            'parameters': {
                'args1': 'dummy_class',
                'n_jobs': 10,
                'timestamp': "2023-01-01 00:00:00+00:00"},
            }
    expected_url = logging_utils.get_logging_endpoint('run')
    mock_post.assert_called_with(
            url=expected_url,
            json=expected_submission,
            headers=headers,
            timeout=5)


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
