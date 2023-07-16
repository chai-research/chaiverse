import os
import time
import pickle
from unittest.mock import patch

from chai_guanaco.utils import cache_leaderboard


@cache_leaderboard
def get_leaderboard(developer_key=None):
    return 'leaderboard'


def test_cache_leaderboard_returns_from_cache(tmpdir):
    with patch('chai_guanaco.utils.guanaco_data_dir', return_value=tmpdir):
        assert get_leaderboard() == 'leaderboard'
        cache_file = os.path.join(tmpdir, 'cache', 'leaderboard_cache.pkl')
        assert load_from_file(cache_file) == 'leaderboard'

        # on second run, get_leaderboard loads from cache
        dump_to_file(cache_file, 'leaderboard_changed')
        assert get_leaderboard() == 'leaderboard_changed'


def test_cache_leaderboard_regenerates_cache_after_12_hours(tmpdir):
    with patch('chai_guanaco.utils.guanaco_data_dir', return_value=tmpdir), \
         patch('os.path.getmtime', return_value=mock_time_past_hours(13)):
        assert get_leaderboard() == 'leaderboard'
        cache_file = os.path.join(tmpdir, 'cache', 'leaderboard_cache.pkl')
        assert load_from_file(cache_file) == 'leaderboard'

        # on second run, get leaderboard does not read from cache as it is
        # after 12 hour
        dump_to_file(cache_file, 'leaderboard_changed')
        assert get_leaderboard() == 'leaderboard'
        assert load_from_file(cache_file) == 'leaderboard'


def load_from_file(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def dump_to_file(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def mock_time_past_hours(hours):
    return time.time() - hours * 3600
