from mock import ANY, patch, Mock
import os

import pytest

from chaiverse import feedback


@pytest.fixture()
def mock_get():
    with patch("chaiverse.http_client.requests.get") as func:
        func.return_value.status_code = 200
        func.return_value.json.return_value = {"some": "feedback"}
        yield func


@patch('chaiverse.utils._load_from_cache')
def test_is_submission_updated_new_submission(load_cache_mock):
    load_cache_mock.side_effect = FileNotFoundError()
    assert feedback.is_submission_updated("file_not_found", 10)


@patch('chaiverse.utils._load_from_cache')
def test_is_submission_updated_cache_contains_no_value(load_cache_mock):
    load_cache_mock.return_value.raw_data = {}
    assert feedback.is_submission_updated("mock_sub_id", 1)


@patch('chaiverse.utils._load_from_cache')
def test_is_submission_updated_increase_total(load_cache_mock):
    load_cache_mock.return_value.raw_data = {'thumbs_up' : 10, 'thumbs_down' : 10}
    assert feedback.is_submission_updated("mock_sub_id", 21)


@patch('chaiverse.utils._load_from_cache')
def test_is_submission_updated_equal_total(load_cache_mock):
    load_cache_mock.return_value.raw_data = {'thumbs_up' : 10, 'thumbs_down' : 10}
    assert not feedback.is_submission_updated("mock_sub_id", 20)


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

    mock_methods = {
        "guanaco_data_dir": Mock(return_value=str(tmpdir)),
    }
    os.makedirs(os.path.join(tmpdir, 'cache'), exist_ok=True)

    request_mock = Mock()
    request_mock.get.return_value.status_code = 200
    request_mock.get.return_value.json.return_value = 'mock-feedback'

    with patch.multiple("chaiverse.utils", **mock_methods):
        with patch('chaiverse.http_client.requests', request_mock):
            result = feedback.get_feedback(submission_id, developer_key, reload=False)
            expected_path = tmpdir / "cache" / f"{submission_id}.pkl"
            assert expected_path.exists()
            assert result.raw_data == 'mock-feedback'


@patch("chaiverse.feedback.utils._save_to_cache")
def test_get_latest_feedback(save_to_cache_mock, mock_get):
    result = feedback._get_latest_feedback(submission_id="test_model", developer_key="key")
    expected_headers = {'Authorization': 'Bearer key'}
    expected_url = "https://guanaco-feedback.chai-research.com/feedback/test_model"
    mock_get.assert_called_once_with(url=expected_url, headers=expected_headers)
    save_to_cache_mock.assert_called_once_with(ANY, result)


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
