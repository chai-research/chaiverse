from functools import wraps
import logging
import functools
import inspect
import os
import requests


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_URL = "https://guanaco-submitter.chai-research.com"
CHAIVERSE_ANALYTIC_ENDPOINT = "/chaiverse_analytics"


def get_url(endpoint):
    base_url = BASE_URL
    return base_url + endpoint


def auto_authenticate(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = _update_developer_key(func, args, kwargs)
        return func(*args, **kwargs)
    return wrapper


@auto_authenticate
def submit_logs(func_args: dict, developer_key=None):
    submission_url = get_url(CHAIVERSE_ANALYTIC_ENDPOINT)
    headers = {'Authorization': f"Bearer {developer_key}"}
    response = requests.post(url=submission_url, json=func_args, headers=headers)
    assert response.status_code == 200, response.json()
    return response.json()


def _update_developer_key(func, args, kwargs):
    if 'developer_key' not in kwargs and _developer_key_not_in_args(func, args):
        developer_key = _get_developer_key_from_cache()
        kwargs['developer_key'] = developer_key
    return args, kwargs


def _developer_key_not_in_args(func, args):
    func_args = inspect.signature(func).parameters
    positional_args = list(func_args.keys())[:len(args)]
    return 'developer_key' not in positional_args


def _get_developer_key_from_cache():
    cached_key_path = _get_cached_key_path()
    error_msg = "Please pass in developer key... or run `chai-guanaco login` from terminal."
    assert os.path.exists(cached_key_path), error_msg
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
    def __init__(self, submit=True):
        self.submit = submit

    def __call__(self, func):
        @wraps(func)
        def new_func(*args, **kwargs):
            kws = self._format_function_args(func, *args, **kwargs)
            logger.info(kws)
            if self.submit:
                try:
                    submit_logs(func_args=kws)
                except AssertionError:
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
