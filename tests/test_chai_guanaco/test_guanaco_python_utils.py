import os
import time
import pickle
from unittest.mock import patch

from chai_guanaco.utils import cache


@cache
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def test_cache_leaderboard_returns_from_cache(tmpdir):
    with patch('chai_guanaco.utils.guanaco_data_dir', return_value=tmpdir):
        assert add(1, 2) == 3
        cache_file = os.path.join(tmpdir, 'cache', 'add(a=1, b=2).pkl')
        assert load_from_file(cache_file) == 3

        # on second run, get_leaderboard loads from cache
        dump_to_file(cache_file, 4)
        assert add(1, 2) == 4

        # making sure function is re-ran with different input args
        assert add(2, 2) == 4
        cache_file = os.path.join(tmpdir, 'cache', 'add(a=2, b=2).pkl')
        assert load_from_file(cache_file) == 4


def test_cache_leaderboard_regenerates_cache_after_12_hours(tmpdir):
    with patch('chai_guanaco.utils.guanaco_data_dir', return_value=tmpdir), \
         patch('os.path.getmtime', return_value=mock_time_past_hours(13)):
        assert add(1, 2) == 3
        cache_file = os.path.join(tmpdir, 'cache', 'add(a=1, b=2).pkl')
        assert load_from_file(cache_file) == 3

        # on second run, get leaderboard does not read from cache as it is
        # after 12 hour
        dump_to_file(cache_file, 4)
        assert add(1, 2) == 3
        assert load_from_file(cache_file) == 3


def test_cache_leaderboard_can_regenerate(tmpdir):
    with patch('chai_guanaco.utils.guanaco_data_dir', return_value=tmpdir):
        assert cache(subtract)(1, 2) == -1
        cache_file = os.path.join(tmpdir, 'cache', 'subtract(a=1, b=2).pkl')
        dump_to_file(cache_file, 8)

        assert cache(subtract, regenerate=True)(1, 2) == -1
        assert load_from_file(cache_file) == -1


def load_from_file(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def dump_to_file(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def mock_time_past_hours(hours):
    return time.time() - hours * 3600
