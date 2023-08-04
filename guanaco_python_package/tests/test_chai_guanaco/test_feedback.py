from mock import patch

import pytest

from chai_guanaco import feedback


@pytest.fixture(autouse="session")
def mock_get():
    with patch("chai_guanaco.feedback.requests.get") as func:
        func.return_value.status_code = 200
        func.return_value.json.return_value = {"some": "feedback"}
        yield func


def test_feedback_object(example_feedback):
    data = example_feedback
    user_feedback = feedback.Feedback(data)
    expected_cols = ['conversation_id', 'bot_id', 'user_id', 'conversation', 'thumbs_up', 'feedback', 'model_name']
    assert all(user_feedback.df.columns == expected_cols)
    expected_conversation = 'Bot: hello!\nUser: emmm hi?\nBot (deleted): I hate u!'
    assert all(user_feedback.df.conversation == expected_conversation)
    expected_thumbs_up = [False, True]
    assert all(user_feedback.df.thumbs_up == expected_thumbs_up)
    expected_feedback = ['he didnt like me', 'he liked me']
    assert all(user_feedback.df.feedback == expected_feedback)
    expected_bot_id = ['_bot_demo-123', '_bot_demo-234']
    assert all(user_feedback.df.bot_id == expected_bot_id)
    expected_user_id = ['user-id-123', 'user-id-123']
    assert all(user_feedback.df.user_id == expected_user_id)


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


@pytest.fixture
def example_feedback():
    messages = get_dummy_messages()
    cid1 = '_bot_demo-123_user-id-123_1687485384266_123'
    cid2 = '_bot_demo-234_user-id-123_1687485384266_234'
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
        'conversation_id': '_bot_demo-123_user-id-123_1687485384355_234',
        'deleted': False,
        'sender': {'name': 'User', 'uid': 'user-id-123'},
        'sent_date': '2021-03-21T20:09:45.355',
    }
    msg3 = {
        'content': 'I hate u!',
        'conversation_id': '_bot_demo-123_user-id-123_1687485387012_345',
        'deleted': True,
        'sender': {'name': 'Bot', 'uid': '_bot_demo-123'},
        'sent_date': '2021-03-21T20:10:45.355',
    }
    return [msg1, msg2, msg3]
