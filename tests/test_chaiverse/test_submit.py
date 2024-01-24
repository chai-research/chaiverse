from mock import patch, Mock
import pytest

from chaiverse import submit, formatters, utils
from chaiverse import config

@pytest.fixture(autouse="session")
def mock_post():
    with patch("chaiverse.http_client.requests.post") as func:
        func.return_value.status_code = 200
        func.return_value.json.return_value = {"submission_id": "name_123456"}
        yield func


@pytest.fixture(autouse="session")
def mock_get():
    with patch("chaiverse.http_client.requests.get") as func:
        func.return_value.status_code = 200
        func.return_value.json.return_value = {'name_123456': {'status': 'pending'}}
        yield func


@pytest.fixture()
def mock_get_pending_to_success():
    responses = [{'status': 'pending'}] * 2 + [{'status': 'deployed'}]
    with patch("chaiverse.http_client.requests.get") as func:
        func.return_value.status_code = 200
        func.return_value.json.side_effect = responses
        yield func


@pytest.fixture()
def mock_submission():
    submission = {
        "model_repo": "ChaiML/test_model",
        "generation_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 40,
            "repetition_penalty": 1.0,
            "stopping_words": ["\n"],
        },
        "formatter": formatters.PygmalionFormatter().dict(),
    }
    return submission


@pytest.fixture()
def developer_key(tmpdir):
    keyfile = tmpdir.join('developer_key.json')
    keyfile.write('mock-key')
    with patch('chaiverse.login_cli._get_cached_key_path', Mock(return_value=f'{keyfile}')):
        yield keyfile


def test_model_submitter(mock_submission, mock_post, mock_get_pending_to_success, developer_key):
    model_submitter = submit.ModelSubmitter(developer_key="mock-key")
    model_submitter._sleep_time = 0
    model_submitter._get_request_interval = 1
    model_submitter_params = {
        "model_repo": "ChaiML/test_model",
        "generation_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 40,
            "repetition_penalty": 1.0,
            "stopping_words": ["\n"],
        },
        "formatter": formatters.PygmalionFormatter(),
    }
    with patch('builtins.input', return_value='accept'):
        submission_id = model_submitter.submit(model_submitter_params)
    headers = {"Authorization": "Bearer mock-key"}
    expected_url = utils.get_url(config.SUBMISSION_ENDPOINT)
    mock_post.assert_called_once_with(url=expected_url, headers=headers, json=mock_submission)
    assert mock_get_pending_to_success.call_count == 3
    assert submission_id == "name_123456"


@patch('builtins.input')
def test_model_submitter_fails_on_invalid_eula_agreement(text_input, developer_key):
    model_submitter = submit.ModelSubmitter(developer_key="mock-key")
    text_input.return_value = 'no'
    with pytest.raises(ValueError) as e:
        model_submitter.submit({})
    assert 'you must agree with our EULA' in str(e)


def test_client(mock_post, mock_submission):
    response = submit.submit_model(mock_submission, developer_key="mock-key")
    expected_submission_id = "name_123456"
    assert response == {"submission_id": expected_submission_id}


def test_submit_client_posts_with_correct_payload(mock_post, mock_submission):
    submit.submit_model(mock_submission, developer_key="mock-key")
    headers={"Authorization": "Bearer mock-key"}
    expected_url = utils.get_url(config.SUBMISSION_ENDPOINT)
    mock_post.assert_called_once_with(url=expected_url, json=mock_submission, headers=headers)


def test_submit_client_posts_raises_for_failed_post(mock_post, mock_submission):
    mock_post.return_value.status_code = 500
    mock_post.return_value.json.return_value = {'error': 'some error'}
    with pytest.raises(AssertionError) as e:
        submit.submit_model(mock_submission, developer_key="mock-key")
    msg = "some error"
    assert msg in str(e)


def test_submit_client_posts_returns_correct_subimission_id(mock_post, mock_submission):
    mock_post.return_value.json.return_value = {'submission_id': 'name_654321'}
    response = submit.submit_model(mock_submission, developer_key="mock-key")
    expected_submission_id = "name_654321"
    assert response == {"submission_id": expected_submission_id}


def test_get_model_info(mock_get):
    response = submit.get_model_info('name_123456', developer_key='key')
    expected = {'name_123456': {'status': 'pending'}}
    assert expected == response


def test_evaluate_mode_get_called_with_correct_url(mock_get):
    submit.evaluate_model('name_123456', developer_key='key')
    url = utils.get_url(config.EVALUATE_ENDPOINT, submission_id='name_123456')
    mock_get.assert_called_once_with(
        url=url,
        headers={"Authorization": "Bearer key"}
    )


def test_get_evalluate_model_raises_with_error(mock_get):
    mock_get.return_value.status_code = 500
    mock_get.return_value.json.return_value = {'error': 'some error'}
    with pytest.raises(AssertionError) as e:
        submit.evaluate_model('name_123456', developer_key='key')
    msg = "some error"
    assert msg in str(e)


def test_get_model_info_called_with_correct_url(mock_get):
    submit.get_model_info('name_123456', developer_key='key')
    url = utils.get_url(config.INFO_ENDPOINT, submission_id='name_123456')
    mock_get.assert_called_once_with(
        url=url,
        headers={"Authorization": "Bearer key"}
    )


def test_get_model_info_raises_with_error(mock_get):
    mock_get.return_value.status_code = 500
    mock_get.return_value.json.return_value = {'error': 'some error'}
    with pytest.raises(AssertionError) as e:
        submit.get_model_info('name_123456', developer_key='key')
    msg = "some error"
    assert msg in str(e)


def test_get_my_submissions(mock_get):
    mock_get.return_value.json.return_value = {'123': 'pending', '456': 'failed'}
    out = submit.get_my_submissions('dev_key')
    expected_url = utils.get_url(config.ALL_SUBMISSION_STATUS_ENDPOINT)
    mock_get.assert_called_once_with(
        url=expected_url,
        headers={"Authorization": "Bearer dev_key"}
    )
    assert out == {'123': 'pending', '456': 'failed'}


def test_get_my_submissions_raises(mock_get):
    mock_get.return_value.status_code = 500
    mock_get.return_value.json.return_value = {'error': 'bad dev key'}
    with pytest.raises(AssertionError) as e:
        submit.get_my_submissions('dev_key')
    msg = "bad dev key"
    assert msg in str(e)


def test_deactivate_model(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = ""
    submit.deactivate_model("test_model", developer_key="dev_key")
    expected_headers = {"Authorization": "Bearer dev_key"}
    url = utils.get_url(config.DEACTIVATE_ENDPOINT, submission_id = "test_model")
    mock_get.assert_called_once_with(url=url, headers=expected_headers)


def test_redeploy_model(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = ""
    submit.redeploy_model("test_model", developer_key="dev_key")
    expected_headers = {"Authorization": "Bearer dev_key"}
    url = utils.get_url(config.REDEPLOY_ENDPOINT, submission_id = "test_model")
    mock_get.assert_called_once_with(url=url, headers=expected_headers)


def test_teardown_model(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = ""
    submit.teardown_model("test_model", developer_key="dev_key")
    expected_headers = {"Authorization": "Bearer dev_key"}
    url = utils.get_url(config.TEARDOWN_ENDPOINT, submission_id="test_model")
    mock_get.assert_called_once_with(url=url, headers=expected_headers)
