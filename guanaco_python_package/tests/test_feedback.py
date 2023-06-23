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


def test_get_leaderboard(mock_get, mock_leaderboard):
    mock_get.return_value.json.return_value = mock_leaderboard
    output = feedback.get_leaderboard(developer_key="key")
    expected_headers = {"developer_key": "key"}
    expected_url = "https://guanaco-feedback.chai-research.com/feedback"
    mock_get.assert_called_once_with(expected_url, headers=expected_headers)
    expected_output = {
        "alekseykorshuk-pygmalion-6b-v0-lmg_1686855359": {
            "thumbs_down": 131,
            "thumbs_up": 594,
            "ratio": 0.82,
        },
        "vicuna-13b-reward-triton": {
            "thumbs_down": 70,
            "thumbs_up": 230,
            "ratio": 0.77,
        },
        "alekseykorshuk-pygmalion-6b-v1-eos_1686954501": {
            "thumbs_down": 187,
            "thumbs_up": 537,
            "ratio": 0.74,
        },
    }
    assert output == expected_output


@pytest.fixture
def mock_leaderboard():
    feedback = {
        "alekseykorshuk-pygmalion-6b-v0-lmg_1686855359": {
            "thumbs_down": 131,
            "thumbs_up": 594,
        },
        "alekseykorshuk-pygmalion-6b-v1-eos_1686954501": {
            "thumbs_down": 187,
            "thumbs_up": 537,
        },
        "vicuna-13b-reward-triton": {"thumbs_down": 70, "thumbs_up": 230},
    }
    return feedback
