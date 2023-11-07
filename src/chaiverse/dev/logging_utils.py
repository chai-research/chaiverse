from datetime import datetime
from functools import wraps
import functools
import inspect
import logging
import os
import pytz
import requests
import time


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_URL = "https://guanaco-training.chai-research.com"
DEFAULT_DEVELOPER_KEY = "CR_b590b336cb314e63ab8d83f189a73edc"


def auto_authenticate(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = _update_developer_key(func, args, kwargs)
        return func(*args, **kwargs)
    return wrapper


@auto_authenticate
def submit_logs(field, parameters, timeout=5, developer_key=None):
    endpoint = get_logging_endpoint(field)
    headers = {'Authorization': f"Bearer {developer_key}"}
    parameters['timestamp'] = str(get_utc_now())
    logging_request = {'developer_key': developer_key, 'parameters': parameters}
    response = requests.post(url=endpoint, json=logging_request, headers=headers, timeout=timeout)
    return response


def get_logging_endpoint(path):
    endpoint = f"{BASE_URL}/{path}/update"
    return endpoint


def get_utc_now():
    timestamp = int(time.time())
    timestamp = datetime.utcfromtimestamp(timestamp)
    timestamp = timestamp.replace(tzinfo=pytz.UTC)
    return timestamp


def _update_developer_key(func, args, kwargs):
    if 'developer_key' not in kwargs and _developer_key_not_in_args(func, args):
        developer_key = _get_developer_key()
        kwargs['developer_key'] = developer_key
    return args, kwargs


def _developer_key_not_in_args(func, args):
    func_args = inspect.signature(func).parameters
    positional_args = list(func_args.keys())[:len(args)]
    return 'developer_key' not in positional_args


def _get_developer_key():
    developer_key = DEFAULT_DEVELOPER_KEY
    cached_key_path = _get_cached_key_path()
    if os.path.exists(cached_key_path):
        developer_key = _get_cached_key()
    return developer_key


def _get_cached_key():
    cached_key_path = _get_cached_key_path()
    with open(cached_key_path, 'r') as f:
        return f.read()


def _get_cached_key_path():
    data_dir = _guanaco_data_dir()
    return os.path.join(data_dir, 'developer_key.json')


def _guanaco_data_dir():
    home_dir = os.path.expanduser("~")
    data_dir = os.environ.get('GUANACO_DATA_DIR', f'{home_dir}/.chai-guanaco')
    os.makedirs(os.path.join(data_dir, 'cache'), exist_ok=True)
    return data_dir


class logging_manager(object):
    def __init__(self, field):
        self.field = field

    def __call__(self, func):
        @wraps(func)
        def new_func(*args, **kwargs):
            params = self._format_function_args(func, *args, **kwargs)
            logger.info(params)
            try:
                submit_logs(field=self.field, parameters=params)
            except requests.RequestException:
                pass
            res = func(*args, **kwargs)
            return res
        return new_func

    def _format_function_args(self, func, *args, **kwargs):
        kws = self._get_default_kwargs(func)
        kws.update(kwargs)
        for num, arg in enumerate(args):
            kws[f'args{num+1}'] = arg
        for key, arg in kws.items():
            if hasattr(arg, '__dict__'):
                kws[key] = arg.__class__.__name__
        return kws

    def _get_default_kwargs(self, func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty}
