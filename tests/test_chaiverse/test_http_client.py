import pytest
import requests
from unittest.mock import patch

import vcr
import os

from chaiverse.http_client import SubmitterClient, FeedbackClient
from chaiverse.login_cli import auto_authenticate


filtered_vcr = vcr.VCR(filter_headers = ["Authorization", "developer_key"])
current_dir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture()
def mock_post():
    with patch("chaiverse.http_client.requests.post") as func:
        func.return_value.status_code = 200
        func.return_value.json.return_value = {"submission_id": "name_123456"}
        yield func


@pytest.fixture()
def mock_submission():
    submission = {
        "model_repo": "ChaiML/test_model",
        "generation_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 40,
            "repetition_penalty": 1.0,
            "stopping_words": ["\n"],
        },
    }
    return submission


def test_submitter_client_gets_correct_authentication_header():
    developer_key = "CR_test"
    http_client = SubmitterClient(developer_key=developer_key)
    assert http_client.headers == {"Authorization": "Bearer CR_test"}


def test_feedback_client_gets_correct_authentication_header():
    developer_key = "CR_test"
    http_client = FeedbackClient(developer_key=developer_key)
    assert http_client.headers == {"Authorization": "Bearer CR_test"}


@filtered_vcr.use_cassette(os.path.join(current_dir, 'vcr_cassettes', 'test_submitter_client_get.yaml'))
def test_submitter_client_get(vcr):
    submission_id="huggyllama-llama-7b_v2"
    endpoint = "/models/{submission_id}"
    # Have to pass in mock developer key if cassette is already recorded, as we omit
    # developer key in cassette
    client_kwargs = {"developer_key": "CR_Mock"} if vcr else {}
    http_client = SubmitterClient(client_kwargs)
    response = http_client.get(endpoint, submission_id=submission_id)
    assert response['developer_uid'] == 'end_to_end_test'
    assert response['status'] == 'torndown'
    assert response['model_repo'] == 'huggyllama/llama-7b'
    assert response['reward_repo'] == 'ChaiML/reward_models_100_170000000_cp_498032'
    assert response['generation_params'] == {'temperature': 1.0,
         'top_p': 0.99,
         'top_k': 40,
         'presence_penalty': 0.0,
         'frequency_penalty': 0.0,
         'stopping_words': ['\n'],
         'max_input_tokens': 1024,
         'best_of': 4}
    assert response['timestamp'] == '2023-12-16T05:00:00+00:00'


@filtered_vcr.use_cassette(os.path.join(current_dir, 'vcr_cassettes','test_feedback_client_get.yaml'))
def test_feedback_client_get(vcr):
    submission_id="anhnv125-llama-op-v17-1_v26"
    endpoint = "/feedback/{submission_id}"
    # Have to pass in mock developer key if cassette is already recorded, as we omit
    # developer key in cassette
    client_kwargs = {"developer_key": "CR_Mock"} if vcr else {}
    http_client = FeedbackClient(client_kwargs)
    response = http_client.get(endpoint, submission_id=submission_id)
    assert list(response.keys()) == ['feedback', 'thumbs_down', 'thumbs_up']


def test_submitter_client_get_raises_with_bad_endpoint():
    developer_key = "CR_test"
    http_client = SubmitterClient(developer_key=developer_key)
    endpoint = "/models/bad_endpoint"
    with pytest.raises(AssertionError):
        response = http_client.get(endpoint)


def test_feedback_client_get_raises_with_bad_endpoint():
    developer_key = "CR_test"
    http_client = FeedbackClient(developer_key=developer_key)
    endpoint = "/bad_endpoint"
    with pytest.raises(AssertionError):
        response = http_client.get(endpoint)


def test_submitter_client_get_raises_with_bad_developer_key():
    developer_key = "CR_bad"
    submission_id="huggyllama/llama-7b"
    http_client = SubmitterClient(developer_key=developer_key)
    endpoint = "/models/{submission_id}"
    with pytest.raises(AssertionError):
        response = http_client.get(endpoint, submission_id=submission_id)


def test_feedback_client_get_raises_with_bad_developer_key():
    developer_key = "CR_bad"
    submission_id="huggyllama/llama-7b"
    http_client = FeedbackClient(developer_key=developer_key)
    endpoint = "/feedback/{submission_id}"
    with pytest.raises(AssertionError):
        response = http_client.get(endpoint, submission_id=submission_id)


def test_submitter_client_get_raises_with_bad_submission_id():
    developer_key = "CR_test"
    submission_id="chai-non-exist-submission"
    http_client = SubmitterClient(developer_key=developer_key)
    endpoint = "/models/{submission_id}"
    with pytest.raises(AssertionError):
        response = http_client.get(endpoint, submission_id=submission_id)


@filtered_vcr.use_cassette(os.path.join(current_dir, 'vcr_cassettes' ,'test_feedback_client_get_bad_submission_id.yaml'))
def test_feedback_client_empty_feedback_with_bad_submission_id(vcr):
    submission_id="chai-non-exist-submission"
    endpoint = "/feedback/{submission_id}"
    # Have to pass in mock developer key if cassette is already recorded, as we omit
    # developer key in cassette
    client_kwargs = {"developer_key": "CR_Mock"} if vcr else {}
    http_client = FeedbackClient(client_kwargs)
    response = http_client.get(endpoint, submission_id=submission_id)
    assert response == {'feedback': {}, 'thumbs_up': 0, 'thumbs_down': 0}


def test_mock_submitter_client_post(mock_post, mock_submission):
    developer_key = "CR_test"
    http_client = SubmitterClient(developer_key=developer_key)
    endpoint = "/models/submit"

    data = mock_submission
    response = http_client.post(endpoint=endpoint, data=data)

    assert response == {"submission_id": "name_123456"}


@filtered_vcr.use_cassette(os.path.join(current_dir, 'vcr_cassettes', 'test_submitter_client_post.yaml'))
def test_submitter_client_post(vcr):
    endpoint = "/models/{submission_id}/chat"
    submission_id = "anhnv125-llama-op-v17-1_v26"

    payload = {'memory': 'He is from planet Earth',
            'prompt': 'Just another human',
            'chat_history': [
                {'sender': 'Tom', 'message': 'Hi'},
                {'sender': 'user', 'message': 'hello'},
                {'sender': 'Tom', 'message': 'whatsup?'},
                {'sender': 'user', 'message': 'how are you?'},
            ],
            'bot_name': 'Tom',
            'user_name': 'You'}

    # Have to pass in mock developer key if cassette is already recorded, as we omit
    # developer key in cassette
    client_kwargs = {"developer_key": "CR_Mock"} if vcr else {}
    http_client = SubmitterClient(client_kwargs)
    response = http_client.post(endpoint=endpoint, data=payload, submission_id=submission_id, timeout=20)
    assert response['model_input'] == "### Instruction:\nAs the assistant, your task is to fully embody the given character, creating immersive, captivating narratives. Stay true to the character's personality and background, generating responses that not only reflect their core traits but are also accurate to their character. Your responses should evoke emotion, suspense, and anticipation in the user. The more detailed and descriptive your response, the more vivid the narrative becomes. Aim to create a fertile environment for ongoing interaction – introduce new elements, offer choices, or ask questions to invite the user to participate more fully in the conversation. This conversation is a dance, always continuing, always evolving.\nYour character: Tom.\nContext: He is from planet Earth\n### Input:\nJust another human\nTom: Hi\nYou: hello\nTom: whatsup?\nYou: how are you?\n### Response:\nTom:"
    assert type(response['model_output']) == str


def test_submitter_client_post_raises_with_bad_endpoint(mock_submission):
    developer_key = "CR_test"
    http_client = SubmitterClient(developer_key=developer_key)
    endpoint = "/models/bad_endpoint"
    data = mock_submission
    with pytest.raises(AssertionError):
        response = http_client.post(endpoint, data=data)


def test_submitter_client_post_raises_with_bad_developer_key(mock_submission):
    developer_key = "CR_bad"
    http_client = SubmitterClient(developer_key=developer_key)
    endpoint = "/models/submit"
    data = mock_submission
    with pytest.raises(AssertionError):
        response = http_client.post(endpoint, data=data)


