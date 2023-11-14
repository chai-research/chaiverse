import os
import unittest.mock as mock
from mock import patch

from freezegun import freeze_time
import pandas as pd
import pytest
import numpy as np
import vcr

import chai_guanaco as chai
from chai_guanaco import metrics

RESOURCE_DIR = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'resources')


@mock.patch('chai_guanaco.metrics.get_all_historical_submissions')
def test_developer_can_call_display_leaderboard_and_pass_in_developer_key_as_arg(get_submissions_mock):
    get_submissions_mock.side_effect = KeyError()
    with pytest.raises(KeyError):
        chai.display_leaderboard(max_workers=1, developer_key='bad-developer-key')
    get_submissions_mock.assert_called_with('bad-developer-key')


@mock.patch('chai_guanaco.metrics.get_all_historical_submissions')
def test_developer_can_call_get_leaderboard_and_pass_in_developer_key_as_arg(get_submissions_mock):
    get_submissions_mock.side_effect = KeyError()
    with pytest.raises(KeyError):
        chai.get_leaderboard(max_workers=1, developer_key='bad-developer-key')
    get_submissions_mock.assert_called_with('bad-developer-key')


@mock.patch('chai_guanaco.utils.get_all_historical_submissions')
def test_developer_can_call_get_submission_metrics_and_pass_in_developer_key_as_arg(get_submissions_mock):
    get_submissions_mock.side_effect = KeyError()
    with pytest.raises(KeyError):
        metrics.get_submission_metrics(submission_id='fake-submission-id', developer_key='bad-developer-key')
    get_submissions_mock.assert_called_with('bad-developer-key')


@mock.patch('chai_guanaco.feedback._submission_is_deployed')
@mock.patch('chai_guanaco.metrics.get_all_historical_submissions')
@mock.patch('chai_guanaco.utils.guanaco_data_dir')
@freeze_time('2023-07-28 00:00:00')
@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_get_leaderboard.yaml'))
def test_get_leaderboard(data_dir_mock, get_ids_mock, deployed_mock, tmpdir):
    deployed_mock.return_value = True
    data_dir_mock.return_value = str(tmpdir)
    get_ids_mock.return_value = historical_submisions()
    with patch("chai_guanaco.utils.get_all_historical_submissions", return_value={}):
        df = chai.get_leaderboard(max_workers=1, developer_key="key")
    expected_cols = [
        'submission_id',
        'thumbs_up_ratio',
        'user_writing_speed',
        'total_feedback_count',
        ]
    for col in expected_cols:
        assert col in df.columns, f'{col} not found in leaderboard'
    expected_data = [
        {
            'submission_id': 'alekseykorshuk-exp-sy_1690222960',
            'timestamp': '2023-07-24 18:22:40+00:00',
            'status': 'inactive',
            'model_repo': 'AlekseyKorshuk/exp-syn-friendly-cp475',
            'developer_uid': 'aleksey',
            'model_name': 'None',
            'thumbs_down': 42,
            'thumbs_up': 33,
            'mcl': 12.0,
            'thumbs_up_ratio': 0.45454545454545453,
            'thumbs_up_ratio_se': 0.043159749410503594,
            'repetition': 0.09309662846841879,
            'total_feedback_count': 33,
            'user_writing_speed': 2.258729,
            },
        {
            'submission_id': 'psiilu-funny-bunny-1-_1689922219',
            'timestamp': '2023-07-21 06:50:19+00:00',
            'status': 'inactive',
            'model_repo': 'psiilu/funny-bunny-1-1-1-1-1',
            'developer_uid': 'philipp',
            'model_name': 'None',
            'thumbs_down': 306,
            'thumbs_up': 187,
            'mcl': 9.241545893719806,
            'thumbs_up_ratio': 0.4057971014492754,
            'thumbs_up_ratio_se': 0.01675940260012055,
            'repetition': 0.26395981785675354,
            'total_feedback_count': 207,
            'user_writing_speed': 2.7696244,
            },
        {
            'submission_id': 'tehvenom-dolly-shygma_1690135695',
            'timestamp': '2023-07-23 18:08:15+00:00',
            'status': 'inactive',
            'model_repo': 'TehVenom/Dolly_Shygmalion-6b-Dev_V8P2',
            'developer_uid': 'tehvenom',
            'model_name': 'None',
            'thumbs_down': 76,
            'thumbs_up': 79,
            'mcl': 13.236363636363636,
            'thumbs_up_ratio': 0.4909090909090909,
            'thumbs_up_ratio_se': 0.033698849323782545,
            'repetition': 0.07319031123625354,
            'total_feedback_count': 55,
            'user_writing_speed': 3.1909,
            }
    ]
    pd.testing.assert_frame_equal(df, pd.DataFrame(expected_data))


@mock.patch('chai_guanaco.feedback._submission_is_deployed')
@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_get_submission_metrics.yaml'))
@freeze_time('2023-07-14 19:00:00')
def test_get_leaderboard_row(deployed_mock):
    deployed_mock.return_value = True
    with patch("chai_guanaco.utils.get_all_historical_submissions", return_value={}):
        results = metrics.get_leaderboard_row(('wizard-vicuna-13b-bo4', {'meta-data-key': 'meta-data-value'}), developer_key="key")
    expected_metrics = {
        'submission_id': 'wizard-vicuna-13b-bo4',
        'meta-data-key': 'meta-data-value',
        'mcl': pytest.approx(28.620229007633586),
        'thumbs_up_ratio': pytest.approx(0.7538167938931297),
        'thumbs_up_ratio_se': pytest.approx(0.008106970421151738),
        'repetition': pytest.approx(0.10992682598233437),
        'total_feedback_count': 524,
        'user_writing_speed': pytest.approx(2.2084751053670595),
    }
    assert results == expected_metrics


@mock.patch('chai_guanaco.feedback._submission_is_deployed')
@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_get_submission_metrics.yaml'))
@freeze_time('2023-07-14 19:00:00')
def test_get_submission_metrics(deployed_mock):
    deployed_mock.return_value = True
    with patch("chai_guanaco.utils.get_all_historical_submissions", return_value={}):
        results = metrics.get_submission_metrics('wizard-vicuna-13b-bo4', developer_key="key")
    expected_metrics = {
        'mcl': pytest.approx(28.620229007633586),
        'thumbs_up_ratio': pytest.approx(0.7538167938931297),
        'thumbs_up_ratio_se': pytest.approx(0.008106970421151738),
        'repetition': pytest.approx(0.10992682598233437),
        'total_feedback_count': 524,
        'user_writing_speed': pytest.approx(2.2084751053670595),
    }
    assert results == expected_metrics


def test_conversation_metrics():
    bot_sender_data = {'uid': '_bot_123'}
    user_sender_data = {'uid': 'XLQR6'}
    messages = [
        {'deleted': False, 'content': 'hi', 'sender': bot_sender_data},
        {'deleted': False, 'content': '123', 'sender': user_sender_data},
        {'deleted': True, 'content': 'bye', 'sender': bot_sender_data},
        {'deleted': True, 'content': 'bye~', 'sender': bot_sender_data},
        {'deleted': False, 'content': 'dont go!', 'sender': bot_sender_data},
        {'deleted': False, 'content': '123456', 'sender': user_sender_data},
        {'deleted': False, 'content': 'bye', 'sender': bot_sender_data},
    ]
    convo_metrics = metrics.ConversationMetrics(messages)
    assert convo_metrics.mcl == 5
    assert convo_metrics.repetition_score == 0.25


def test_conversation_metrics_profile_conversation():
    bot_sender_data = {'uid': '_bot_123'}
    user_sender_data = {'uid': 'XLQR6'}
    messages = [
        {'deleted': False, 'content': 'hi', 'sender': bot_sender_data, 'sent_date': '2023-09-01T12:00:00'},
        {'deleted': False, 'content': '123', 'sender': user_sender_data, 'sent_date': '2023-09-01T12:00:30'},
        {'deleted': True, 'content': 'bye', 'sender': bot_sender_data, 'sent_date': '2023-09-01T12:00:45'},
        {'deleted': True, 'content': 'bye~', 'sender': bot_sender_data, 'sent_date': '2023-09-01T12:00:55'},
        {'deleted': False, 'content': 'dont go!', 'sender': bot_sender_data, 'sent_date': '2023-09-01T12:01:05'},
        {'deleted': False, 'content': '123456', 'sender': user_sender_data, 'sent_date': '2023-09-01T12:01:25'},
        {'deleted': False, 'content': 'bye', 'sender': bot_sender_data, 'sent_date': '2023-09-01T12:01:30'},

    ]
    convo_metrics = metrics.ConversationMetrics(messages)
    out = convo_metrics.get_conversation_profile()
    expected = pd.DataFrame([
        {'duration': 30., 'bot_num_characters': 2, 'user_num_characters': 3},
        {'duration': 20., 'bot_num_characters': 8, 'user_num_characters': 6},
    ])
    assert out.equals(expected)


def test_summarise_convo_profile():
    writing_speed = 12
    reading_speed = 53
    thinking_time = 21
    df = pd.DataFrame({
        'bot_num_characters': [23, 32, 45, 9, 6, 67],
        'user_num_characters': [12, 89, 34, 68, 54, 90]})
    df['duration'] = df['bot_num_characters'] / reading_speed + df['user_num_characters'] / writing_speed + thinking_time
    out = metrics.summarise_conversation_profile(df)
    assert np.isclose(out['writing_speed'], writing_speed)
    assert np.isclose(out['reading_speed'], reading_speed)
    assert np.isclose(out['thinking_time'], thinking_time)


def test_summarise_convo_profile_returns_nan_when_not_enough_data_points():
    writing_speed = 12
    reading_speed = 53
    thinking_time = 21
    df = pd.DataFrame({
        'bot_num_characters': [23],
        'user_num_characters': [12]})
    df['duration'] = df['bot_num_characters'] / reading_speed + df['user_num_characters'] / writing_speed + thinking_time
    out = metrics.summarise_conversation_profile(df)
    assert np.isnan(out['writing_speed'])
    assert np.isnan(out['reading_speed'])
    assert np.isnan(out['thinking_time'])


def test_print_formatted_leaderboard():
    data = {
        'submission_id': ['tom_1689542168', 'tom_1689404889', 'val_1689051887', 'zl_1689542168'],
        'total_feedback_count': [151, 160, 290, 101],
        'mcl': [1.0, 2.0, 3.0, 4.0],
        'retry_score': [.5, .6, .7, .8],
        'thumbs_up_ratio': [0.1, 0.5, 0.8, 0.2],
        'user_writing_speed': [1.25, 3.2, 1.2, 1.09],
        'model_name': ['psutil', 'htop', 'watch', 'gunzip'],
        'developer_uid': ['tom', 'tom', 'val', 'zl'],
        'timestamp': ['2023-07-24 18:22:40+00:00'] * 3 + ['2023-07-24T18:22:40+00:00'],
        'model_repo': ['psutil', 'psutil', 'watch', 'gunzip']
    }
    all_metrics_df = pd.DataFrame(data)

    df = metrics._print_formatted_leaderboard(all_metrics_df, detailed=True)

    assert len(df) == 3
    expected_columns = [
            'submission_id',
            'total_feedback_count',
            'mcl',
            'retry_score',
            'thumbs_up_ratio',
            'user_writing_speed',
            'model_name',
            'developer_uid',
            'model_repo',
            'date',
            'overall_score',
            'overall_rank'
        ]
    assert list(df.columns) == expected_columns
    assert pd.api.types.is_integer_dtype(df['overall_rank'])


def test_get_repetition_score_is_one_if_all_responses_are_the_same():
    responses = ['Hi', 'Hi', 'hi']
    score = metrics.get_repetition_score(responses)
    assert score == 1.


def test_get_repetition_score_is_zero_if_all_responses_are_different():
    responses = ['Hi', 'Hey', 'How are you?']
    score = metrics.get_repetition_score(responses)
    assert score == 0.


def test_get_repetition_score_ignores_repetition():
    responses = ['Hi !', '...Hi', 'hi']
    score = metrics.get_repetition_score(responses)
    assert score == 1.


def test_get_repetition_score_handels_corrupt_responses():
    responses = ['! !', '...', '.']
    score = metrics.get_repetition_score(responses)
    assert score == 1.


def test_get_repetition_score_handels_semi_corrupt_responses():
    responses = ['Heya', '...', '...']
    score = metrics.get_repetition_score(responses)
    assert score == 0.5


def test_get_repetition_score():
    bad_responses = ['Hi! I am Tom', 'Hey! I am Val', 'Hi, im tOM']
    good_responses = ['Hey! I am Tom', 'Hello there, I am Val', 'Byee~~~']
    bad_score = metrics.get_repetition_score(bad_responses)
    good_score = metrics.get_repetition_score(good_responses)
    assert bad_score > good_score


def historical_submisions():
    data = {
       "alekseykorshuk-exp-sy_1690222960": {
          "timestamp": "2023-07-24 18:22:40+00:00",
          "status": "inactive",
          "model_repo": "AlekseyKorshuk/exp-syn-friendly-cp475",
          "developer_uid": "aleksey",
          "model_name": "None",
          "thumbs_down": 42,
          "thumbs_up": 33
       },
       "psiilu-funny-bunny-1-_1689922219": {
          "timestamp": "2023-07-21 06:50:19+00:00",
          "status": "inactive",
          "model_repo": "psiilu/funny-bunny-1-1-1-1-1",
          "developer_uid": "philipp",
          "model_name": "None",
          "thumbs_down": 306,
          "thumbs_up": 187
       },
       "tehvenom-dolly-shygma_1690135695": {
          "timestamp": "2023-07-23 18:08:15+00:00",
          "status": "inactive",
          "model_repo": "TehVenom/Dolly_Shygmalion-6b-Dev_V8P2",
          "developer_uid": "tehvenom",
          "model_name": "None",
          "thumbs_down": 76,
          "thumbs_up": 79
       }
    }
    return data
