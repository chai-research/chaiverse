import os
import unittest.mock as mock
from mock import patch

import time_machine
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


@mock.patch('chai_guanaco.metrics.get_feedback')
def test_developer_can_call_get_submission_metrics_and_pass_in_developer_key_as_arg(feedback_mock):
    feedback_mock.side_effect = KeyError()
    with pytest.raises(KeyError):
        metrics.get_submission_metrics(submission_id='fake-submission-id', developer_key='bad-developer-key')
    feedback_mock.assert_called_with('fake-submission-id', 'bad-developer-key', reload=True)


@mock.patch('chai_guanaco.metrics.get_all_historical_submissions')
@mock.patch('chai_guanaco.utils.guanaco_data_dir')
@time_machine.travel('2023-07-28 00:00:00')
@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_get_raw_leaderboard.yaml'))
def test_get_raw_leaderboard(data_dir_mock, get_ids_mock, tmpdir):
    data_dir_mock.return_value = str(tmpdir)
    get_ids_mock.return_value = historical_submisions()
    df = chai.metrics.get_raw_leaderboard(max_workers=1, developer_key="key")
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
            'status': 'deployed',
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
            'reward_repo': None,
            'is_custom_reward': None,
            },
        {
            'submission_id': 'psiilu-funny-bunny-1-_1689922219',
            'timestamp': '2023-07-21 06:50:19+00:00',
            'status': 'deployed',
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
            'reward_repo': 'mock-custom-reward-repo',
            'is_custom_reward': True,
            },
        {
            'submission_id': 'tehvenom-dolly-shygma_1690135695',
            'timestamp': '2023-07-23 18:08:15+00:00',
            'status': 'deployed',
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
            'reward_repo': 'mock-default-reward-repo',
            'is_custom_reward': False,
            }
    ]
    pd.testing.assert_frame_equal(df, pd.DataFrame(expected_data))


@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_get_submission_metrics.yaml'))
@time_machine.travel('2023-07-14 19:00:00')
def test_get_leaderboard_row():
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


@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_get_submission_metrics.yaml'))
@time_machine.travel('2023-07-14 19:00:00')
def test_get_submission_metrics():
    results = metrics.get_submission_metrics('wizard-vicuna-13b-bo4', developer_key="key", reload=True)
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

    df = metrics._get_processed_leaderboard(all_metrics_df, detailed=True)

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
            'timestamp',
            'model_repo',
            'is_custom_reward',
            'reward_repo',
            'repetition',
            'overall_rank',
            'safety_score',
            'overall_score',
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


def test_get_processed_leaderboard_will_default_model_name_to_submission_id_if_not_existed_for_backwards_compatibility():
    df = make_unique_submissions(2)
    df.update({'submission_id': ['mock-submission-1', 'mock-submission-2']})
    assert list(metrics._get_processed_leaderboard(df, True)['model_name']) == ['mock-submission-1', 'mock-submission-2']


def test_get_processed_leaderboard_will_default_model_name_to_submission_id_if_none_for_backwards_compatibility():
    df = make_unique_submissions(2)
    df['model_name'] = [None, 'mock-model-name']
    df.update({
        'submission_id': ['mock-submission-1', 'mock-submission-2'],
    })
    assert list(metrics._get_processed_leaderboard(df, True)['model_name']) == ['mock-submission-1', 'mock-model-name']


def test_get_processed_leaderboard_will_default_is_custom_reward_to_false_if_not_present():
    df = make_unique_submissions(1)
    result = metrics._get_processed_leaderboard(df, True)
    assert result['is_custom_reward'][0] == False


def test_get_processed_leaderboard_will_convert_is_custom_reward_to_boolean():
    df = make_unique_submissions(3)
    df['is_custom_reward'] = [None, True, False]
    assert list(metrics._get_processed_leaderboard(df, True)['is_custom_reward']) == [False, True, False]


def tet_get_processed_leaderboard_will_default_reward_repo_to_none():
    df = make_unique_submissions(1)
    df.update({'submission_id': 'mock-submission-id'})
    assert metrics._get_processed_leaderboard(df, True)['reward_repo'][0] == None


def test_get_processed_leaderboard_will_remove_duplicate_submission_of_lower_rank_if_model_repo_and_reward_repo_are_the_same_if_not_in_detailed_mode():
    df = make_unique_submissions(2)
    df.update({
        'submission_id': ['submission-1', 'submission-2'],
        'model_repo': ['mock-model-repo', 'mock-model-repo'],
        'reward_repo': ['mock-reward-repo', 'mock-reward-repo'],
        'thumbs_up_ratio': [0.8, 0.9]
    })
    result = metrics._get_processed_leaderboard(df, False)
    assert list(result['submission_id']) == ['submission-2']


def test_get_processed_leaderboard_will_sort_by_rank_for_same_reward_repo_but_different_model_repo_if_not_in_detailed_mode():
    df = make_unique_submissions(2)
    df.update({
        'submission_id': ['submission-1', 'submission-2'],
        'reward_repo': ['mock-reward-repo', 'mock-reward-repo'],
        'thumbs_up_ratio': [0.8, 0.9]
    })
    result = metrics._get_processed_leaderboard(df, False)
    assert list(result['submission_id']) == ['submission-2', 'submission-1']


def test_get_processed_leaderboard_will_sort_by_rank_for_same_model_repo_but_different_reward_repo_if_not_in_detailed_mode():
    df = make_unique_submissions(2)
    df.update({
        'submission_id': ['submission-1', 'submission-2'],
        'model_repo': ['mock-model-repo', 'mock-model-repo'],
        'thumbs_up_ratio': [0.8, 0.9]
    })
    result = metrics._get_processed_leaderboard(df, False)
    assert list(result['submission_id']) == ['submission-2', 'submission-1']


def test_get_processed_leaderboard_will_sort_but_will_not_remove_duplicate_by_rank_if_in_detail_mode():
    df = make_unique_submissions(2)
    df.update({
        'submission_id': ['submission-1', 'submission-2'],
        'model_repo': ['mock-model-repo', 'mock-model-repo'],
        'reward_repo': ['mock-reward-repo', 'mock-reward-repo'],
        'thumbs_up_ratio': [0.8, 0.9]
    })
    result = metrics._get_processed_leaderboard(df, True)
    assert list(result['submission_id']) == ['submission-2', 'submission-1']


def test_get_processed_leaderboard_will_contain_up_to_one_submission_of_one_dev_id_if_not_in_detailed_mode():
    df = make_unique_submissions(2)
    df.update({
        'submission_id': ['submission-1', 'submission-2'],
        'developer_uid': ['dev-uid-1', 'dev-uid-1'],
        'thumbs_up_ratio': [0.8, 0.9]
    })
    result = metrics._get_processed_leaderboard(df, False)
    assert list(result['submission_id']) == ['submission-2']


def test_get_processed_leaderboard_will_not_limit_submissions_of_one_dev_id_if_in_detailed_mode():
    df = make_unique_submissions(2)
    df.update({
        'submission_id': ['submission-1', 'submission-2'],
        'developer_uid': ['dev-uid-1', 'dev-uid-1'],
        'thumbs_up_ratio': [0.8, 0.9]
    })
    result = metrics._get_processed_leaderboard(df, True)
    assert list(result['submission_id']) == ['submission-2', 'submission-1']


def test_get_processed_leaderboard_will_remove_submissions_with_few_feedback():
    df = make_unique_submissions(3)
    df.update({
        'submission_id': ['submission-1', 'submission-2', 'submission-3'],
        'model_repo': ['mock-model-repo-1', 'mock-model-repo-2', 'mock-model-repo-3'],
        'total_feedback_count': [149, 150, 151]
    })
    result = metrics._get_processed_leaderboard(df, True)
    assert list(result['submission_id']) == ['submission-2', 'submission-3']


@pytest.mark.parametrize(
        "thumbs_up_ratios, user_writing_speeds, overall_scores, overall_ranks, winning_model", [
        ([0.9, 0.8], [50, 100], [1.0, 2.0], [1,2], 'model1'),
        ([0.9, 0.8], [100, 50], [1.5, 1.5], [1,1], 'model1'),
        ([0.8, 0.9], [100, 50], [1.0, 2.0], [1,2], 'model2') ])
def test_get_procssed_leaderboard_will_set_overall_score_and_overall_rank_correctly(
        thumbs_up_ratios, user_writing_speeds, overall_scores, overall_ranks, winning_model):
    df = make_unique_submissions(2)
    df.update({
        'submission_id': ['submission-1', 'submission-2'],
        'model_repo': ['model1', 'model2'],
        'reward_repo': ['mock-default-repo', 'mock-default-repo'],
        'total_feedback_count': [1000, 1000],
        'thumbs_up_ratio': thumbs_up_ratios,
        'user_writing_speed': user_writing_speeds,
    })
    assert len(df) == 2
    result = metrics._get_processed_leaderboard(df, True)
    assert list(result['overall_score']) == overall_scores
    assert list(result['overall_rank']) == overall_ranks
    assert result['model_repo'][0] == winning_model

@patch('chai_guanaco.metrics.get_all_historical_submissions')
@patch('chai_guanaco.metrics.get_leaderboard')
def test_get_sorted_available_models(get_leaderboard, get_submissions):
    get_submissions.return_value = {
        'model1': {'status': 'inactive'},
        'model2': {'status': 'deployed'},
        'model3': {'status': 'deployed'},
        'model4': {'status': 'deployed'}
    }
    get_leaderboard.return_value = {
        "submission_id": ['model4', 'model2']
    }
    result = metrics.get_sorted_available_models('mock-key')
    assert result == ['model4', 'model2', 'model3']
    get_submissions.assert_called_once_with(developer_key='mock-key')
    get_leaderboard.assert_called_once_with(regenerate=False, developer_key='mock-key')

def historical_submisions():
    data = {
       "alekseykorshuk-exp-sy_1690222960": {
          "timestamp": "2023-07-24 18:22:40+00:00",
          "status": "deployed",
          "model_repo": "AlekseyKorshuk/exp-syn-friendly-cp475",
          "developer_uid": "aleksey",
          "model_name": "None",
          "thumbs_down": 42,
          "thumbs_up": 33
       },
       "psiilu-funny-bunny-1-_1689922219": {
          "timestamp": "2023-07-21 06:50:19+00:00",
          "status": "deployed",
          "model_repo": "psiilu/funny-bunny-1-1-1-1-1",
          "developer_uid": "philipp",
          "reward_repo": "mock-custom-reward-repo",
          "is_custom_reward": True,
          "model_name": "None",
          "thumbs_down": 306,
          "thumbs_up": 187
       },
       "tehvenom-dolly-shygma_1690135695": {
          "timestamp": "2023-07-23 18:08:15+00:00",
          "status": "deployed",
          "model_repo": "TehVenom/Dolly_Shygmalion-6b-Dev_V8P2",
          "reward_repo": "mock-default-reward-repo",
          "is_custom_reward": False,
          "developer_uid": "tehvenom",
          "model_name": "None",
          "thumbs_down": 76,
          "thumbs_up": 79
       },
    }
    return data


def make_unique_submissions(count):
    df = pd.DataFrame(range(count))
    df['total_feedback_count'] = 1000
    df['thumbs_up_ratio'] = 0.5
    df['user_writing_speed'] = 50
    _fill_unique_ids(df, 'submission_id', prefix='mock-submission-id')
    _fill_unique_ids(df, 'model_repo', prefix='mock-model-repo')
    _fill_unique_ids(df, 'reward_repo', prefix='mock-reward-repo')
    _fill_unique_ids(df, 'developer_uid', prefix='mock-dev-id')
    return df


def _fill_unique_ids(df, field_name, prefix):
    df[field_name] = ["{prefix}-{i}".format(prefix=prefix, i=i) for i in range(len(df))]

