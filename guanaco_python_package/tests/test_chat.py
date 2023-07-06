import mock
import os
import json

from chai_guanaco.chat import Bot, BotConfig, SubmissionChatbot


@mock.patch('builtins.input')
@mock.patch('chai_guanaco.chat.requests')
def test_submission_chatbot(mock_requests, mock_input, tmpdir):
    mock_input.side_effect = ['hello', 'how are you?', 'exit']
    response = {'model_input': 'some_input', 'model_output': 'whatsup?'}
    mock_request = mock_requests.post.return_value
    mock_request.status_code = 200
    mock_request.json.return_value = response

    with mock.patch('chai_guanaco.chat.RESOURCE_DIR', str(tmpdir)):
        create_dummy_bot_config(str(tmpdir))
        chatbot = SubmissionChatbot('dummy_submission_id', 'CR-123')
        chatbot.chat('dummy_bot')

    mock_requests.post.assert_called_with(
        url="https://guanaco-submitter.chai-research.com/models/dummy_submission_id/chat",
        headers={"Authorization": "Bearer CR-123"},
        json={
            'memory': 'He is from planet Earth',
            'prompt': 'Just another human',
            'chat_history': [
                {'sender': 'Tom', 'message': 'Hi'},
                {'sender': 'user', 'message': 'hello'},
                {'sender': 'Tom', 'message': 'whatsup?'},
                {'sender': 'user', 'message': 'how are you?'},
            ],
            'bot_name': 'Tom',
            'user_name': 'You'
        }
    )



@mock.patch('chai_guanaco.chat.requests')
def test_chat(mock_request):
    submission_id = 'test-model'
    developer_key = 'CR-devkey'

    config_bot = BotConfig(
        memory='Bot memory',
        prompt='Bot prompt',
        first_message='this is the first message',
        bot_label='Bot name')

    bot = Bot(submission_id, developer_key, config_bot)

    output = {'model_input': 'some_input', 'model_output': 'how are you?'}
    response = mock_request.post.return_value
    response.status_code = 200
    response.json.return_value = output

    out = bot.response('hey!')
    assert out == output

    expected_payload = {
        "memory": 'Bot memory',
        "prompt": 'Bot prompt',
        "chat_history": [
            {"sender": "Bot name", "message": "this is the first message"},
            {"sender": "user", "message": "hey!"}
        ],
        "bot_name": 'Bot name',
        "user_name": "You",
    }
    expected_url = "https://guanaco-submitter.chai-research.com/models/test-model/chat"
    expected_headers = {"Authorization": "Bearer CR-devkey"}
    mock_request.post.assert_called_once_with(
        url=expected_url,
        json=expected_payload,
        headers=expected_headers)

    bot.response('I am fine')
    expected_payload = {
        "memory": 'Bot memory',
        "prompt": 'Bot prompt',
        "chat_history": [
            {"sender": "Bot name", "message": "this is the first message"},
            {"sender": "user", "message": "hey!"},
            {"sender": "Bot name", "message": "how are you?"},
            {"sender": "user", "message": "I am fine"},
        ],
        "bot_name": 'Bot name',
        "user_name": "You",
    }
    mock_request.post.assert_called_with(
        url=expected_url,
        json=expected_payload,
        headers=expected_headers)


def create_dummy_bot_config(save_dir):
    bot_config = {
            'memory': 'He is from planet Earth',
            'prompt': 'Just another human',
            'first_message': 'Hi',
            'bot_label': 'Tom'}
    save_path = os.path.join(save_dir, 'dummy_bot.json')
    with open(save_path, 'w') as f:
        json.dump(bot_config, f)
