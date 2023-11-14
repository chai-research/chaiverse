import os

import click
from click.testing import CliRunner
import pytest

from chai_guanaco.login_cli import auto_authenticate, login, logout


@click.group()
def cli():
    pass


@cli.command()
def print_inputed_auth():
    submission_id = input('')
    developer_key = input('')
    submission_id, developer_key = dummy_function(submission_id, developer_key)
    print(f"id={submission_id}, key={developer_key}")


@cli.command()
def print_cached_auth():
    submission_id = input('')
    submission_id, developer_key = dummy_function(submission_id)
    print(f"id={submission_id}, key={developer_key}")


@auto_authenticate
def dummy_function(submission_id, developer_key = None):
    return submission_id, developer_key


TEMP_TEST_DIR = 'tmp-test-dir'
KEY_PATH = f'{TEMP_TEST_DIR}/developer_key.json'
DEVELOPER_KEY = 'prepopulated-key'
RUNNER_ENVIRONMENT = { 'GUANACO_DATA_DIR': TEMP_TEST_DIR }


class TestLoginCli:
    @pytest.fixture(autouse = True)
    def runner(self):
        self.runner = CliRunner()
        with self.runner.isolated_filesystem():
            yield self.runner

    @pytest.fixture
    def key_file(self, runner):
        os.makedirs(TEMP_TEST_DIR)
        with open(KEY_PATH, 'w') as f:
            f.write(DEVELOPER_KEY)
        return KEY_PATH

    def test_chai_guanaco_login_from_terminal(self):
        self.runner.invoke(login, env = RUNNER_ENVIRONMENT, input = 'dummy-key\n')
        with open(KEY_PATH) as f:
            assert f.read() == 'dummy-key'

    def test_chai_guanaco_login_from_terminal_overrides_existing_key_file(self, key_file):
        self.runner.invoke(login, env = RUNNER_ENVIRONMENT, input = 'dummy-key\n')
        with open(KEY_PATH) as f:
            assert f.read() == 'dummy-key'

    def test_chai_guanaco_login_from_terminal_ignores_empty_key(self):
        result = self.runner.invoke(login, env = RUNNER_ENVIRONMENT, input = '\n')
        assert result.exit_code == 0

    def test_chai_guanaco_logout_terminal_removes_key(self, key_file):
        self.runner.invoke(logout, env = RUNNER_ENVIRONMENT)
        assert os.path.isdir(TEMP_TEST_DIR)
        assert not os.path.exists(KEY_PATH)

    def test_auto_authenticate_wrapper_raises_when_not_logged_in_and_no_keys_passed(self):
        result = self.runner.invoke(print_cached_auth, env = RUNNER_ENVIRONMENT, input = 'gpt-j-6b\n')
        assert result.exit_code == 1
        assert 'Please pass in developer key' in str(result.exception)

    def test_auto_authenticate_wrapper_ignores_cached_key_when_devkey_is_passed(self, key_file):
        result = self.runner.invoke(print_inputed_auth, env = RUNNER_ENVIRONMENT, input = 'gpt-j-6b\ndummy-key\n')
        assert 'id=gpt-j-6b, key=dummy-key' in result.stdout

    def test_auto_authenticate_wrapper_does_not_overwrite_cached_key_when_devkey_is_passed(self, key_file):
        result = self.runner.invoke(print_inputed_auth, env = RUNNER_ENVIRONMENT, input = 'gpt-j-6b\ndummy-key\n')
        assert 'id=gpt-j-6b, key=dummy-key' in result.stdout
        with open(KEY_PATH, 'r') as f:
            data = f.read()
            assert data == DEVELOPER_KEY

    def test_auto_authenticate_wrapper_loads_from_cached_key(self, key_file):
        result = self.runner.invoke(print_cached_auth, env = RUNNER_ENVIRONMENT, input = 'gpt-j-6b\n')
        assert f'id=gpt-j-6b, key={DEVELOPER_KEY}' in str(result.stdout)

if __name__ == '__main__':
    cli()
