from mock import patch

import pytest

from chai_guanaco.submit import submit_model, SUBMISSION_URL


@pytest.fixture(autouse="session")
def mock_post():
    with patch("chai_guanaco.submit.requests.post") as func:
        func.return_value.status_code = 200
        yield func


@pytest.fixture()
def mock_submission():
    submission = {
        "model_repo": "ChaiML/test_model",
        "developer_uid": "name",
        "developer_key": "mock-key",
        "generation_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 40,
            "repetition_penalty": 1.0,
        },
        "formatter": "PygmalionFormatter",
    }
    return submission


def test_submit_end_to_end(mock_post, mock_submission):
#    mock_post.return_value = {"submission_id": "name_123456"}
    response = submit_model(mock_submission)
    expected_submission_id = "name_123456"
    assert response == {"submission_id": expected_submission_id}


def test_submit_end_to_end_posts_with_correct_payload(mock_post, mock_submission):
    submit_model(mock_submission)
    mock_post.assert_called_once_with(url=SUBMISSION_URL, json=mock_submission)


def test_submit_end_to_end_posts_raises_for_failed_post(mock_post, mock_submission):
    mock_post.return_value.status_code = 500
    mock_post.return_value.json.return_value = {'error': 'some error'}
    with pytest.raises(AssertionError) as e:
        response = submit_model(mock_submission)
    msg = "some error"
    assert msg in str(e)
