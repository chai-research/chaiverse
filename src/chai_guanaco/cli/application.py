from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path


class Application:
    def __init__(self, exit_func):
        self.__exit_func = exit_func
        self.__api_key = ''

    @property
    def api_key(self) -> str:
        return self.__api_key

    @cached_property
    def data_dir(self) -> Path:
        if data_dir := os.getenv('GUANACO_DATA_DIR'):
            return Path(data_dir)

        from platformdirs import user_data_dir

        return Path(user_data_dir('chai_guanaco', appauthor=False))

    @cached_property
    def credentials_file(self) -> Path:
        return self.data_dir / 'credentials.json'

    def set_api_key(self, api_key):
        self.__api_key = api_key

    def abort(self, text='', code=1):
        if text:
            import rich

            rich.console.Console().print(text, style='bold red')

        self.__exit_func(code)
