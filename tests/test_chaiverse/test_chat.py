import os
import json
import mock

from chaiverse.chat import Bot, BotConfig, SubmissionChatbot, get_bot_names, get_bot_config, get_bot_response


@mock.patch('builtins.input')
@mock.patch('chaiverse.http_client.requests.post')
def test_submission_chatbot(mock_post, mock_input, tmpdir):
    mock_input.side_effect = ['hello', 'how are you?', 'exit']
    response = {'model_input': 'some_input', 'model_output': 'whatsup?'}
    mock_request = mock_post.return_value
    mock_request.status_code = 200
    mock_request.json.return_value = response

    with mock.patch('chaiverse.chat.RESOURCE_DIR', str(tmpdir)):
        create_dummy_bot_config(str(tmpdir))
        chatbot = SubmissionChatbot('dummy_submission_id', 'CR-123')
        chatbot.chat('dummy_bot')

    mock_post.assert_called_with(
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
        },
        timeout=20
    )



@mock.patch('chaiverse.http_client.requests.post')
def test_chat(mock_post):
    url = 'https://guanaco-submitter.chai-research.com/models/dummy_submission/chat'
    submission_id = "dummy_submission"
    developer_key = 'CR-devkey'

    config_bot = BotConfig(
        memory='Bot memory',
        prompt='Bot prompt',
        first_message='this is the first message',
        bot_label='Bot name')

    bot = Bot(submission_id, developer_key, config_bot)

    output = {'model_input': 'some_input', 'model_output': 'how are you?'}
    response = mock_post.return_value
    response.status_code = 200
    response.json.return_value = output

    out = bot.get_response('hey!')
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
    expected_headers = {"Authorization": "Bearer CR-devkey"}
    mock_post.assert_called_once_with(
        url=url,
        headers=expected_headers,
        json=expected_payload,
        timeout=20
    )

    bot.get_response('I am fine')
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
    mock_post.assert_called_with(
        url=url,
        json=expected_payload,
        headers=expected_headers,
        timeout=20
    )


def create_dummy_bot_config(save_dir):
    bot_config = {
            'memory': 'He is from planet Earth',
            'prompt': 'Just another human',
            'first_message': 'Hi',
            'bot_label': 'Tom'}
    save_path = os.path.join(save_dir, 'dummy_bot.json')
    with open(save_path, 'w') as f:
        json.dump(bot_config, f)


@mock.patch('chaiverse.chat.os.listdir')
def test_get_bot_names(listdir):
    listdir.return_value = ['x', 'a.json', 'b.json']
    result = get_bot_names()
    assert result == ['a', 'b']


def test_get_bot_config(tmpdir):
    mock_config = {
        'memory': 'mock-memory',
        'prompt': 'mock-prompt',
        'first_message': 'mock-message',
        'bot_label': 'mock-label'
    }
    with mock.patch('chaiverse.chat.RESOURCE_DIR', str(tmpdir)):
        (tmpdir / 'mock-bot.json').write_text(json.dumps(mock_config), 'UTF-8')
        result = get_bot_config('mock-bot')
        assert result.memory == 'mock-memory'
        assert result.prompt == 'mock-prompt'
        assert result.first_message == 'mock-message'
        assert result.bot_label == 'mock-label'


@mock.patch('chaiverse.http_client.requests.post')
def test_get_bot_response(mock_post):
    bot_config = mock.Mock()
    bot_config.memory = 'mock-memory'
    bot_config.prompt = 'mock-prompt'
    bot_config.bot_label = 'mock-label'
    bot_config.first_message = 'mock-first-message'
    http_response = mock.Mock()
    http_response.status_code = 200
    http_response.text = 'dummy-resp-text'
    http_response.json.return_value = {'model_output': 'mock-response'}
    mock_post.return_value = http_response

    result = get_bot_response(
        [('apple', 'use'), ('orange', 'bot'), ('pear', 'user')],
        'mock-submission-id',
        bot_config,
        'mock-key'
    )
    assert result == 'mock-response'
