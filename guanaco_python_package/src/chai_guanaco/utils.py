import os
import pickle
import time

CACHE_UPDATE_WINDOW_HOURS = 12


def guanaco_data_dir():
    home_dir = os.path.expanduser("~")
    data_dir = os.environ.get('GUANACO_DATA_DIR', f'{home_dir}/.chai-guanaco')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def print_color(text, color):
    colors = {'blue': '\033[94m',
              'cyan': '\033[96m',
              'green': '\033[92m',
              'yellow': '\033[93m',
              'red': '\033[91m'}
    assert color in colors.keys()
    print(f'{colors[color]}{text}\033[0m')


def cache_leaderboard(get_leaderboard):
    def wrapper(developer_key=None):
        file_path = _get_cache_file_path()
        if _can_return_from_cache(file_path):
            result = _load_from_cache(file_path)
        else:
            result = get_leaderboard(developer_key=developer_key)
            _save_to_cache(file_path, result)
        return result
    return wrapper


def _get_cache_file_path():
    cache_dir = os.path.join(guanaco_data_dir(), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, 'leaderboard_cache.pkl')


def _can_return_from_cache(file_path):
    if not os.path.isfile(file_path):
        can_cache = False
    else:
        can_cache = _cache_file_within_time_window(file_path)
    return can_cache


def _cache_file_within_time_window(file_path):
    current_time = time.time()
    file_time = os.path.getmtime(file_path)
    cache_duration_hours = (current_time - file_time) // 3600
    can_cache = cache_duration_hours < CACHE_UPDATE_WINDOW_HOURS
    return can_cache


def _load_from_cache(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def _save_to_cache(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
