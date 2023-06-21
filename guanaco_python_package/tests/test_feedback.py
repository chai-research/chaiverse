from mock import patch

import pytest

from chai_guanaco import feedback


@pytest.fixture(autouse="session")
def mock_get():
    with patch("chai_guanaco.feedback.requests.get") as func:
        func.return_value.status_code = 200
        func.return_value.json.return_value = {"some": "feedback"}
        yield func


def test_get_feedback(mock_get):
    feedback.get_feedback(submission_id="test_model", developer_key="key")
    expected_headers = {"developer_key": "key"}
    expected_url = "https://guanaco-feedback.chai-research.com/feedback/test_model"
    mock_get.assert_called_once_with(expected_url, headers=expected_headers)


def test_get_feedback_raises_for_bad_request(mock_get):
    mock_get.return_value.status_code = 500
    mock_get.return_value.json.return_value = {"error": "some error"}
    with pytest.raises(AssertionError) as ex:
        feedback.get_feedback(submission_id="test_model", developer_key="key")
    assert "some error" in str(ex)
