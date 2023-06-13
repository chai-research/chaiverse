from mock import patch

import pytest

from chai_guanaco.submit import submit_model, get_model_info, SUBMISSION_URL, INFO_URL


@pytest.fixture(autouse="session")
def mock_post():
    with patch("chai_guanaco.submit.requests.post") as func:
        func.return_value.status_code = 200
        func.return_value.json.return_value = {"submission_id": "name_123456"}
        yield func


@pytest.fixture(autouse="session")
def mock_get():
    with patch("chai_guanaco.submit.requests.get") as func:
        func.return_value.status_code = 200
        func.return_value.json.return_value = {"name_123456": {'status': 'pending'}}
        yield func


@pytest.fixture()
def mock_submission():
    submission = {
        "model_repo": "ChaiML/test_model",
        "developer_uid": "name",
        "generation_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 40,
            "repetition_penalty": 1.0,
        },
        "formatter": "PygmalionFormatter",
    }
    return submission


def test_client(mock_post, mock_submission):
    response = submit_model(mock_submission, developer_key="mock-key")
    expected_submission_id = "name_123456"
    assert response == {"submission_id": expected_submission_id}


def test_submit_client_posts_with_correct_payload(mock_post, mock_submission):
    submit_model(mock_submission, developer_key="mock-key")
    headers={"developer_key": "mock-key"}
    mock_post.assert_called_once_with(url=SUBMISSION_URL, json=mock_submission, headers=headers)


def test_submit_client_posts_raises_for_failed_post(mock_post, mock_submission):
    mock_post.return_value.status_code = 500
    mock_post.return_value.json.return_value = {'error': 'some error'}
    with pytest.raises(AssertionError) as e:
        submit_model(mock_submission, developer_key="mock-key")
    msg = "some error"
    assert msg in str(e)


def test_submit_client_posts_returns_correct_subimission_id(mock_post, mock_submission):
    mock_post.return_value.json.return_value = {'submission_id': 'name_654321'}
    response = submit_model(mock_submission, developer_key="mock-key")
    expected_submission_id = "name_654321"
    assert response == {"submission_id": expected_submission_id}


def test_get_model_info(mock_get):
    response = get_model_info('name_123456', developer_key='key')
    expected = {'name_123456': {'status': 'pending'}}
    assert expected == response


def test_get_model_info_called_with_correct_url(mock_get):
    get_model_info('name_123456', developer_key='key')
    mock_get.assert_called_once_with(
        url=INFO_URL.format(submission_id='name_123456'),
        headers={"developer_key": "key"}
    )


def test_get_model_info_raises_with_error(mock_get):
    mock_get.return_value.status_code = 500
    mock_get.return_value.json.return_value = {'error': 'some error'}
    with pytest.raises(AssertionError) as e:
        get_model_info('name_123456', developer_key='key')
    msg = "some error"
    assert msg in str(e)
