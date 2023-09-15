from mock import patch

import pytest

from chai_guanaco_private import submit_blend


@pytest.fixture(autouse="session")
def mock_post():
    with patch("chai_guanaco.submit.requests.post") as func:
        func.return_value.status_code = 200
        func.return_value.json.return_value = {"submission_id": "name_123456"}
        yield func


@pytest.fixture()
def mock_submission():
    submission_ids = ['submission_id_1', 'submission_id_2', 'submission_id_3']
    return submission_ids


def test_client(mock_post, mock_submission):
    response = submit_blend.submit_blend(mock_submission, developer_key="mock-key")
    expected_submission_id = "name_123456"
    assert response == {"submission_id": expected_submission_id}


def test_submit_client_posts_with_correct_payload(mock_post, mock_submission):
    submit_blend.submit_blend(mock_submission, developer_key="mock-key")
    headers = {"Authorization": "Bearer mock-key"}
    expected_url = submit_blend.get_url(submit_blend.SUBMISSION_ENDPOINT)
    expected_payload = mock_submission
    mock_post.assert_called_once_with(url=expected_url, json=expected_payload, headers=headers)


def test_submit_client_posts_raises_for_failed_post(mock_post, mock_submission):
    mock_post.return_value.status_code = 500
    mock_post.return_value.json.return_value = {'error': 'some error'}
    with pytest.raises(AssertionError) as e:
        submit_blend.submit_blend(mock_submission, developer_key="mock-key")
    msg = "some error"
    assert msg in str(e)


def test_submit_client_posts_returns_correct_subimission_id(mock_post, mock_submission):
    mock_post.return_value.json.return_value = {'submission_id': 'name_654321'}
    response = submit_blend.submit_blend(mock_submission, developer_key="mock-key")
    expected_submission_id = "name_654321"
    assert response == {"submission_id": expected_submission_id}
