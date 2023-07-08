import functools
import os
import inspect

import click

from chai_guanaco.utils import guanaco_data_dir


@click.group()
def cli():
    pass


@cli.command()
def login():
    return developer_login()


@cli.command()
def logout():
    cached_key_path = _get_cached_key_path()
    if os.path.exists(cached_key_path):
        os.remove(cached_key_path)
    print('Logged out!')


def developer_login():
    cached_key_path = _get_cached_key_path()
    text = f"""Welcome to Chai Guanaco 🚀!
By logging in, we will create a file under {cached_key_path}.
Please enter your developer key: """
    developer_key = input(text)
    with open(cached_key_path, 'w') as file:
        file.write(developer_key)


def auto_authenticate(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = _update_developer_key(func, args, kwargs)
        return func(*args, **kwargs)
    return wrapper


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
    data_dir = guanaco_data_dir()
    return os.path.join(data_dir, 'developer_key.json')


if __name__ == '__main__':
    cli()
