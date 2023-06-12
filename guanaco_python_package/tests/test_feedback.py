from mock import patch

import pytest

from chai_guanaco import feedback


@pytest.fixture(autouse="session")
def mock_get():
    with patch("chai_guanaco.feedback.requests.get") as func:
        func.return_value.status_code = 200
        func.return_value.json.return_value = {"some": "feedback"}
        yield func


@pytest.fixture()
def mock_feedback_request():
    request = {
            "developer_uid": "user",
            "developer_key": "key",
            "model_name": "test_model"
    }
    return request


def test_get_feedback(mock_feedback_request, mock_get):
    feedback.get_feedback(mock_feedback_request)
    expected_headers = {"developer_uid": "user", "developer_key": "key"}
    expected_url = "https://guanaco-feedback.chai-research.com/feedback/test_model"
    mock_get.assert_called_once_with(expected_url, headers=expected_headers)


def test_get_feedback_raises_for_bad_request(mock_feedback_request, mock_get):
    mock_get.return_value.status_code = 500
    mock_get.return_value.json.return_value = {"error": "some error"}
    with pytest.raises(AssertionError) as ex:
        feedback.get_feedback(mock_feedback_request)
    assert "some error" in str(ex)
