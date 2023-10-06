from mock import patch, Mock
import os

import pandas as pd
import pytest
import vcr

from chai_guanaco import feedback


RESOURCE_DIR = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'resources')


@pytest.fixture()
def mock_get():
    with patch("chai_guanaco.feedback.requests.get") as func:
        func.return_value.status_code = 200
        func.return_value.json.return_value = {"some": "feedback"}
        yield func


def test_feedback_object(example_feedback):
    data = example_feedback
    user_feedback = feedback.Feedback(data)
    expected_cols = ['conversation_id', 'bot_id', 'user_id', 'conversation', 'thumbs_up', 'feedback', 'model_name', 'public']
    assert all(user_feedback.df.columns == expected_cols)
    expected_conversation = 'Bot: hello!\nUser: emmm hi?\nBot (deleted): I hate u!'
    assert all(user_feedback.df.conversation == expected_conversation)
    expected_thumbs_up = [False, True]
    assert all(user_feedback.df.thumbs_up == expected_thumbs_up)
    expected_feedback = ['he didnt like me', 'he liked me']
    assert all(user_feedback.df.feedback == expected_feedback)
    expected_bot_id = ['_bot_demo-123', '_bot_demo-234']
    assert all(user_feedback.df.bot_id == expected_bot_id)
    expected_user_id = ['user-id-123', 'user-id-1234']
    assert all(user_feedback.df.user_id == expected_user_id)


def test_get_feedback_with_cache(tmpdir):
    submission_id = "test_submission"
    developer_key = "test_key"

    mock_submissions = {submission_id: {"status": "not_deployed"}}
    mock_methods = {
        "get_all_historical_submissions": Mock(return_value=mock_submissions),
        "guanaco_data_dir": Mock(return_value=str(tmpdir)),
    }
    os.makedirs(os.path.join(tmpdir, 'cache'), exist_ok=True)

    with patch.multiple("chai_guanaco.utils", **mock_methods):
        with patch('chai_guanaco.feedback._get_latest_feedback', return_value='feedback'):
            result = feedback.get_feedback(submission_id, developer_key)
            expected_path = tmpdir / "cache" / f"{submission_id}.pkl"
            assert expected_path.exists()
            assert result == 'feedback'


def test_get_latest_feedback(mock_get):
    feedback._get_latest_feedback(submission_id="test_model", developer_key="key")
    expected_headers = {"developer_key": "key"}
    expected_url = "https://guanaco-feedback.chai-research.com/feedback/test_model"
    mock_get.assert_called_once_with(expected_url, headers=expected_headers)


def test_get_latest_feedback_raises_for_bad_request(mock_get):
    mock_get.return_value.status_code = 500
    mock_get.return_value.json.return_value = {"error": "some error"}
    with pytest.raises(AssertionError) as ex:
        feedback._get_latest_feedback(submission_id="test_model", developer_key="key")
    assert "some error" in str(ex)


@pytest.fixture
def example_feedback():
    messages = get_dummy_messages()
    cid1 = '_bot_demo-123_user-id-123_1687485384266_123'
    cid2 = '_bot_demo-234_user-id-1234_1687485384266_234'
    data1 = {
        'conversation_id': cid1,
        'messages': messages,
        'model_name': 'bot_demo',
        'text': 'he didnt like me',
        'thumbs_up': False,
        'user_id': 'incorrect_user_id',
    }
    data2 = {
        'conversation_id': cid2,
        'messages': messages,
        'model_name': 'bot_demo',
        'text': 'he liked me',
        'thumbs_up': True,
        'user_id': 'incorrect_user_id',
    }
    feedback = {cid1: data1, cid2: data2}
    out = {'feedback': feedback, 'thumbs_up': 20, 'thumbs_down': 10}
    return out


def get_dummy_messages():
    msg1 = {
        'content': 'hello!',
        'conversation_id': '_bot_demo-123_user-id-123_1687485384266_123',
        'deleted': False,
        'sender': {'name': 'Bot', 'uid': '_bot_demo-123'},
        'sent_date': '2021-03-21T20:09:44.266',
    }
    msg2 = {
        'content': 'emmm hi?',
        'conversation_id': '_bot_demo-123_user-id-1234_1687485384355_234',
        'deleted': False,
        'sender': {'name': 'User', 'uid': 'user-id-1234'},
        'sent_date': '2021-03-21T20:09:45.355',
    }
    msg3 = {
        'content': 'I hate u!',
        'conversation_id': '_bot_demo-123_user-id-12345_1687485387012_345',
        'deleted': True,
        'sender': {'name': 'Bot', 'uid': '_bot_demo-123'},
        'sent_date': '2021-03-21T20:10:45.355',
    }
    return [msg1, msg2, msg3]
