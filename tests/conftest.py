from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import pytest
from click.testing import CliRunner as __CliRunner


class CliRunner(__CliRunner):
    def __init__(self, command):
        super().__init__()
        self._command = command

    def __call__(self, *args, **kwargs):
        # Exceptions should always be handled
        kwargs.setdefault('catch_exceptions', False)

        return self.invoke(self._command, args, **kwargs)


@pytest.fixture(scope='session')
def chai():
    from chai_guanaco import cli

    return CliRunner(cli.chai_guanaco)


@pytest.fixture(scope='session', autouse=True)
def isolation() -> Generator[Path, None, None]:
    origin = os.getcwd()
    with TemporaryDirectory() as d:
        temp_dir = Path(d)

        data_dir = temp_dir / 'data'
        data_dir.mkdir()

        credentials_file = data_dir / 'credentials.json'
        credentials_file.write_text(json.dumps({'api_key': 'mock-api-key'}), encoding='utf-8')

        default_env_vars = {'CHAI_SELF_TESTING': 'true', 'GUANACO_DATA_DIR': str(data_dir)}
        os.environ.update(default_env_vars)

        os.chdir(str(temp_dir))
        try:
            yield temp_dir
        finally:
            os.chdir(origin)


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    path = Path(tmp_path, 'temp')
    path.mkdir()
    return path


@pytest.fixture(scope='session')
def helpers():
    # https://docs.pytest.org/en/latest/writing_plugins.html#assertion-rewriting
    pytest.register_assert_rewrite('tests.helpers.api')

    from .helpers import api

    return api
