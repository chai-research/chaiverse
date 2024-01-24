from datetime import datetime
import os
import pickle
import sys
import time

from freezegun import freeze_time
from mock import patch, Mock
import pytest
import pytz
import vcr

from chaiverse import utils


RESOURCE_DIR = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'resources')
MAX_RESPONSE_BODY_SIZE_FOR_URI_CHECKING_ONLY_VCR = 1024



def add(a, b):
    return a + b

@utils.cache
def cached_add(a, b):
    return a + b


@pytest.fixture(autouse=True)
def guanado_data_dir(tmpdir):
    with patch('chaiverse.utils.get_guanaco_data_dir_env') as get_data_dir:
        get_data_dir.return_value = str(tmpdir)
        yield get_data_dir


@patch('chaiverse.utils.requests')
def test_get_all_historical_submissions(requests):
    mock_response = Mock()
    requests.get.return_value = mock_response
    mock_response.status_code = 200
    mock_response.json.return_value = 'resp'
    result = utils.get_all_historical_submissions(developer_key='key')
    assert result == 'resp'
    expected_url = 'https://guanaco-submitter.chai-research.com/leaderboard'
    requests.get.assert_called_once_with(expected_url,headers={"developer_key": 'key'}, params=None)


@pytest.mark.parametrize("test_id, params, expected_uri", [
    (1, dict(start_date='from', end_date='to'), 'https://guanaco-submitter.chai-research.com/leaderboard?start_date=from&end_date=to'),
    (2, dict(start_date='from', end_date=None), 'https://guanaco-submitter.chai-research.com/leaderboard?start_date=from'),
    (3, dict(start_date=None, end_date='to'), 'https://guanaco-submitter.chai-research.com/leaderboard?end_date=to'),
    (4, None, 'https://guanaco-submitter.chai-research.com/leaderboard'),
])
def test_get_submissions(test_id, params, expected_uri):
    with vcr.use_cassette(os.path.join(RESOURCE_DIR, f'test_get_submissions_in_date_range{test_id}.yaml')) as cassette:
        utils.get_submissions(developer_key='key', params=params)
        assert len(cassette.requests) == 1
        assert cassette.requests[0].uri == expected_uri
        assert sys.getsizeof(cassette.responses[0]) < MAX_RESPONSE_BODY_SIZE_FOR_URI_CHECKING_ONLY_VCR


def test_cache_will_return_from_cache_or_not_based_on_regenerate_flag():
    mock_function = Mock()

    def my_func(a):
        return mock_function(a)

    def cached_my_func(a, regenerate=False):
        return utils.cache(my_func, regenerate)(a)

    mock_function.return_value = 1

    assert cached_my_func(1) == 1
    mock_function.return_value = 2
    assert cached_my_func(1) == 1
    assert cached_my_func(1, regenerate=True) == 2
    assert len(mock_function.mock_calls) == 2


def test_cache_will_not_return_from_cache_if_param_is_different():
    mock_function = Mock()

    def my_func(a):
        return mock_function(a)

    def cached_my_func(a, regenerate=False):
        return utils.cache(my_func, regenerate)(a)

    mock_function.return_value = 1

    assert cached_my_func(1) == 1
    mock_function.return_value = 2
    assert cached_my_func(2) == 2
    assert len(mock_function.mock_calls) == 2


@patch('chaiverse.utils.time')
@patch('os.path.getmtime')
def test_cache_will_auto_invalidate_after_set_time(getmtime_mock, time_mock):
    mock_function = Mock()

    def my_func(a):
        return mock_function(a)

    def cached_my_func(a, regenerate=False):
        return utils.cache(my_func, regenerate)(a)

    timestamp = 1704096000
    getmtime_mock.return_value = timestamp
    time_mock.return_value = timestamp
    mock_function.return_value = 1
    assert cached_my_func(1) == 1
    mock_function.return_value = 2
    assert cached_my_func(1) == 1

    time_mock.return_value = timestamp + 6*3600 -1
    assert cached_my_func(1) == 1

    time_mock.return_value = timestamp + 6*3600
    assert cached_my_func(1) == 2


def test_get_hex_digest():
    digest1 = utils.get_hexdigest('1')
    digest2 = utils.get_hexdigest('2')
    assert digest1 != digest2
    assert f'{digest1}' != f'{digest2}'


def load_from_file(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def dump_to_file(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def mock_time_past_hours(hours):
    return time.time() - hours * 3600


def test_get_localised_timestamp():
    timestamp = "2023-09-20T10:19:34.363369+00:00"
    timezone = pytz.timezone("US/Pacific")
    localised_timestamp = utils.get_localised_timestamp(timestamp, timezone)
    assert str(localised_timestamp) == "2023-09-20 03:19:34.363369-07:00"


def test_parse_log_entry():
    log = {
      "entry": "chaiml-llama-2-7b-chat-hf-v110-vllmizer: fast uploading tensorized tensors from /mnt/pvc/chaiml-llama-2-7b-chat-hf-v110",
      "level": "INFO",
      "line_number": 120,
      "path": "/home/alex/chai/guanaco/guanaco_services/src/guanaco_model_service/job_runner.py",
      "timestamp": "2023-09-20T10:19:34.363369+00:00"
    }
    timezone = pytz.timezone("US/Pacific")
    actual_parsed_log = utils.parse_log_entry(log, timezone)
    expected_parsed_log = "03:19:34:INFO:chaiml-llama-2-7b-chat-hf-v110-vllmizer: fast uploading tensorized tensors from /mnt/pvc/chaiml-llama-2-7b-chat-hf-v110"
    assert actual_parsed_log == expected_parsed_log


@pytest.mark.parametrize("worker_type", ['process', 'thread'])
class TestDistributeToWorkers:

    def test_distribute_to_workers_is_correct_with_one_arg_iterator_and_three_workers(self, worker_type):
        assert list(utils.distribute_to_workers(str, [1,2,3], max_workers=3, worker_type=worker_type)) == list('123')

    def test_distribute_to_workers_is_correct_with_one_arg_iterator_and_two_workers(self, worker_type):
        assert list(utils.distribute_to_workers(str, [1,2,3], max_workers=2, worker_type=worker_type)) == list('123')

    def test_distribute_to_workers_is_correct_with_one_arg_iterator_and_one_worker(self, worker_type):
        assert list(utils.distribute_to_workers(str, [1,2,3], max_workers=1, worker_type=worker_type)) == list('123')

    def test_distribute_to_workers_is_correct_with_one_arg_iterator_and_no_max_workers_specified(self, worker_type):
        assert list(utils.distribute_to_workers(str, [1,2,3])) == list('123')

    def test_distribute_to_workers_pool_is_correct_with_four_arg_iterators_and_four_workers(self, worker_type):
        assert list(utils.distribute_to_workers(max, [14, 23, 32, 41], [12, 25, 34, 42], [11, 22, 36, 43], [13, 24, 33, 47], max_workers=4, worker_type=worker_type)) == [14, 25, 36, 47]

    def test_distribute_to_workers_pool_is_correct_with_four_arg_iterators_and_one_worker(self, worker_type):
        assert list(utils.distribute_to_workers(max, [14, 23, 32, 41], [12, 25, 34, 42], [11, 22, 36, 43], [13, 24, 33, 47], max_workers=1, worker_type=worker_type)) == [14, 25, 36, 47]

    def test_distribute_to_workers_will_pass_kwargs_with_three_workers(self, worker_type):
        assert list(utils.distribute_to_workers(sorted, [[2,1,3], [5,3,4]], max_workers=3, worker_type=worker_type, reverse=True)) == [[3,2,1], [5,4,3]]
        assert list(utils.distribute_to_workers(sorted, [[2,1,3], [5,3,4]], max_workers=3, worker_type=worker_type, reverse=False)) == [[1,2,3], [3,4,5]]

    def test_distribute_to_workers_will_pass_kwargs_with_one_worker(self, worker_type):
        assert list(utils.distribute_to_workers(sorted, [[2,1,3], [5,3,4]], max_workers=1, worker_type=worker_type, reverse=True)) == [[3,2,1], [5,4,3]]
        assert list(utils.distribute_to_workers(sorted, [[2,1,3], [5,3,4]], max_workers=1, worker_type=worker_type, reverse=False)) == [[1,2,3], [3,4,5]]

