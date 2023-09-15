import os
import inspect
import pickle
from time import time

CACHE_UPDATE_HOURS = 6


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


def cache(func, regenerate=False):
    def wrapper(*args, **kwargs):
        file_path = _get_cache_file_path(func, args, kwargs)
        try:
            result = _load_from_cache(file_path)
            assert not regenerate
            # ensuring file is less than N hours old, otherwise regenerate
            assert (time() - os.path.getmtime(file_path)) < 3600 * CACHE_UPDATE_HOURS
        except (FileNotFoundError, AssertionError):
            result = func(*args, **kwargs)
            _save_to_cache(file_path, result)
        return result
    return wrapper


def _get_cache_file_path(func, args, kwargs):
    cache_dir = os.path.join(guanaco_data_dir(), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    fname = _func_call_as_string(func, args, kwargs)
    return os.path.join(cache_dir, f'{fname}.pkl')


def _load_from_cache(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def _save_to_cache(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def _func_call_as_string(func, args, kwargs):
    func_name = func.__name__
    param_names = list(inspect.signature(func).parameters.keys())
    arg_strs = [f'{name}={value!r}' for name, value in zip(param_names, args)]
    arg_strs += [f'{name}={value!r}' for name, value in kwargs.items()]
    return f"{func_name}({', '.join(arg_strs)})"
