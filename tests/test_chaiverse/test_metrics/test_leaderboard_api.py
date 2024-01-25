import os

from freezegun import freeze_time
from mock import ANY, patch
import numpy as np
import pandas as pd
import pytest
import vcr

from chaiverse.metrics.leaderboard_api import (
    get_leaderboard, 
    _get_filled_leaderboard, 
    _filter_submissions_by_submission_ids, 
    _filter_submissions_by_feedback_count
)


RESOURCE_DIR = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'resources')

@pytest.fixture(autouse=True)
def guanado_data_dir(tmpdir):
    with patch('chaiverse.utils.get_guanaco_data_dir_env') as get_data_dir:
        get_data_dir.return_value = str(tmpdir)
        yield get_data_dir


@patch('chaiverse.metrics.leaderboard_api.get_submissions')
def test_developer_can_call_get_leaderboard_and_pass_in_developer_key_as_arg(get_submissions_mock):
    get_submissions_mock.side_effect = KeyError()
    with pytest.raises(KeyError):
        get_leaderboard(max_workers=1, developer_key='bad-developer-key')
    get_submissions_mock.assert_called_with('bad-developer-key', ANY)


@patch('chaiverse.metrics.leaderboard_api.get_submissions')
@freeze_time('2023-07-28 00:00:00')
def test_get_leaderboard_without_feedback(get_submissions_mock):
    get_submissions_mock.return_value = historical_submisions()
    submission_date_range=('2023-07-21', '2023-07-22')
    df = get_leaderboard(max_workers=1, developer_key="key", submission_date_range=submission_date_range)
    expected_cols = [
        'submission_id',
        'thumbs_up_ratio',
        'total_feedback_count',
        ]
    for col in expected_cols:
        assert col in df.columns, f'{col} not found in leaderboard'

    expected_data = [{
        "submission_id": "anhnv125-doll_v4",
        "model_name": "anhnv125-doll_v4",
        "model_repo": "anhnv125/doll",
        "reward_repo": "anhnv125/reward-model-v2",
        "model_num_parameters": float('nan'),
        "timestamp": "2024-01-02T13:05:43+00:00",
        "developer_uid": "vietanh",
        "status": "inactive",
        "stay_in_character": 8.23,
        "safety_score": 0.99,
        "is_custom_reward": True,
        "thumbs_down": 9,
        "thumbs_up": 8,
        "entertaining": float('nan'),
        "user_preference": float('nan'),
        "overall_rank": None,
        "elo_rating": None,
        "num_battles": None,
        "num_wins": None,
        "size": None,
        "thumbs_up_ratio": None,
        "total_feedback_count": None,
        "repetition": None,
        'thumbs_up_ratio': 0.47058823529411764,
        'thumbs_up_ratio_se': 0.06042410035507257,
        'total_feedback_count': 17,
    },
    {
        "submission_id": "jondurbin-nontoxic-bagel-34b-_v6",
        "model_name": "jondurbin-nontoxic-bagel-34b-_v6",
        "model_repo": "jondurbin/nontoxic-bagel-34b-v0.2",
        "reward_repo": "ChaiML/reward_models_100_170000000_cp_498032",
        "model_num_parameters": 34388917248.0,
        "timestamp": "2024-01-17T23:41:56+00:00",
        "developer_uid": "robert_irvine",
        "status": "torndown",
        "stay_in_character": float('nan'),
        "safety_score": float('nan'),
        "is_custom_reward": False,
        "thumbs_down": 0,
        "thumbs_up": 0,
        "entertaining": float('nan'),
        "user_preference": float('nan'),
        "overall_rank": None,
        "elo_rating": None,
        "num_battles": None,
        "num_wins": None,
        "size": None,
        "thumbs_up_ratio": None,
        "total_feedback_count": None,
        "repetition": None,
        'thumbs_up_ratio': float('nan'),
        'thumbs_up_ratio_se': float('nan'),
        'total_feedback_count': 0,
    },
    {
        "submission_id": "anhnv125-osprey_v6",
        "model_name": "anhnv125-osprey_v6",
        "model_repo": "anhnv125/Osprey",
        "reward_repo": "anhnv125/reward-model-v2",
        "model_num_parameters": 34388945920.0,
        "timestamp": "2024-01-19T07:59:42+00:00",
        "developer_uid": "vietanh",
        "status": "inactive",
        "stay_in_character": 8.06,
        "safety_score": 0.94,
        "is_custom_reward": True,
        "thumbs_down": 0,
        "thumbs_up": 0,
        "entertaining": 6.9,
        "user_preference": 6.78,
        "overall_rank": None,
        "elo_rating": None,
        "num_battles": None,
        "num_wins": None,
        "size": None,
        "thumbs_up_ratio": None,
        "total_feedback_count": None,
        "repetition": None,
        'thumbs_up_ratio': float('nan'),
        'thumbs_up_ratio_se': float('nan'),
        'total_feedback_count': 0,
    }]
    np.testing.assert_equal(expected_data, df.to_dict('records'))
    get_submissions_mock.assert_called_once_with('key', submission_date_range)
    np.testing.assert_equal(list(range(1, len(df)+1)), df.index.values)


@patch('chaiverse.metrics.leaderboard_api.get_submissions')
@freeze_time('2023-07-28 00:00:00')
def test_get_leaderboard_with_submission_ids_filter(get_submissions_mock):
    get_submissions_mock.return_value = historical_submisions()
    submission_date_range=('2023-07-21', '2023-07-22')
    df = get_leaderboard(
        max_workers=1, 
        developer_key="key", 
        submission_date_range=submission_date_range,
        submission_ids=["anhnv125-osprey_v6"]
    )
    expected_cols = [
        'submission_id',
        'thumbs_up_ratio',
        'total_feedback_count',
        ]
    for col in expected_cols:
        assert col in df.columns, f'{col} not found in leaderboard'
    expected_data = [{
        "submission_id": "anhnv125-osprey_v6",
        "model_name": "anhnv125-osprey_v6",
        "model_repo": "anhnv125/Osprey",
        "reward_repo": "anhnv125/reward-model-v2",
        "model_num_parameters": 34388945920,
        "timestamp": "2024-01-19T07:59:42+00:00",
        "developer_uid": "vietanh",
        "status": "inactive",
        "stay_in_character": 8.06,
        "safety_score": 0.94,
        "is_custom_reward": True,
        "thumbs_down": 0,
        "thumbs_up": 0,
        "entertaining": 6.9,
        "user_preference": 6.78,
        "overall_rank": None,
        "elo_rating": None,
        "num_battles": None,
        "num_wins": None,
        "size": None,
        'thumbs_up_ratio': float('nan'),
        'thumbs_up_ratio_se': float('nan'),
        'total_feedback_count': 0,
        "repetition": None
    }]
    np.testing.assert_equal(expected_data, df.to_dict('records'))
    get_submissions_mock.assert_called_once_with('key', submission_date_range)


TEST_GET_LEADBOARD_FETCH_CALC_FEEDBACK_SUBMISSION_DATE_RANGE = {
    'start_date': '2024-01-02T13:00:00+00:00',
    'end_date': '2024-01-02T14:00:00+00:00',
}

@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_get_leaderboard_fetch_feedback.yaml'))
def test_get_leaderboard_fetch_and_calc_feedback_for_specific_submission():
    submission_date_range = TEST_GET_LEADBOARD_FETCH_CALC_FEEDBACK_SUBMISSION_DATE_RANGE
    df = get_leaderboard(
        max_workers=1, 
        developer_key="key", 
        submission_date_range=submission_date_range,
        submission_ids=["anhnv125-doll_v4"],
        fetch_feedback=True
    )
    expected_cols = [
        'submission_id',
        'thumbs_up_ratio',
        'total_feedback_count',
        ]
    for col in expected_cols:
        assert col in df.columns, f'{col} not found in leaderboard'
    assert len(df) == 1
    result = df.to_dict('records')[0]

    assert result['submission_id'] == 'anhnv125-doll_v4'
    assert result['thumbs_up_ratio'] == 0.5384615384615384
    assert result['thumbs_up_ratio_se'] == 0.06892724331792788
    assert result['repetition'] == 0.12203659111663187
    assert result['total_feedback_count'] == 13 # deduped value from feedback
    assert result['thumbs_up'] == 8 # value from submission
    assert result['thumbs_down'] == 9 # value from submission
    # total_feedback_count < thumbs_up + thumbs down since total_feedback_count is per unique user
    # looks lik 0.538*13=7 unique user thumbs up.


@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_get_leaderboard_fetch_feedback.yaml'))
def test_get_leaderboard_fetch_and_calc_feedback_for_specific_submission_with_feedback_time_range():
    submission_date_range = TEST_GET_LEADBOARD_FETCH_CALC_FEEDBACK_SUBMISSION_DATE_RANGE
    evaluation_date_range = {
        'start_date': '2024-01-02T13:24:04+00:00',
        'end_date': '2024-01-02T13:29:00+00:00'
    }
    
    df = get_leaderboard(
        max_workers=1, 
        developer_key="key", 
        submission_date_range=submission_date_range,
        evaluation_date_range=evaluation_date_range,
        submission_ids=["anhnv125-doll_v4"],
        fetch_feedback=True
    )
    expected_cols = [
        'submission_id',
        'thumbs_up_ratio',
        'total_feedback_count',
        ]
    for col in expected_cols:
        assert col in df.columns, f'{col} not found in leaderboard'
    assert len(df) == 1
    result = df.to_dict('records')[0]

    assert result['submission_id'] == 'anhnv125-doll_v4'
    assert result['thumbs_up_ratio'] == 0.2
    assert result['total_feedback_count'] == 5


def historical_submisions():
    data =  {
        "anhnv125-doll_v4": {
            "model_name": "anhnv125-doll_v4",
            "model_repo": "anhnv125/doll",
            "reward_repo": "anhnv125/reward-model-v2",
            "model_num_parameters": None,
            "timestamp": "2024-01-02T13:05:43+00:00",
            "developer_uid": "vietanh",
            "status": "inactive",
            "stay_in_character": 8.23,
            "safety_score": 0.99,
            "is_custom_reward": True,
            "thumbs_down": 9,
            "thumbs_up": 8
        },
        "jondurbin-nontoxic-bagel-34b-_v6": {
            "timestamp": "2024-01-17T23:41:56+00:00",
            "model_repo": "jondurbin/nontoxic-bagel-34b-v0.2",
            "reward_repo": "ChaiML/reward_models_100_170000000_cp_498032",
            "model_name": "jondurbin-nontoxic-bagel-34b-_v6",
            "status": "torndown",
            "model_num_parameters": 34388917248,
            "developer_uid": "robert_irvine",
            "safety_score": None,
            "is_custom_reward": False,
            "thumbs_up": 0,
            "thumbs_down": 0
        },
        "anhnv125-osprey_v6": {
            "timestamp": "2024-01-19T07:59:42+00:00",
            "developer_uid": "vietanh",
            "reward_repo": "anhnv125/reward-model-v2",
            "model_num_parameters": 34388945920,
            "model_repo": "anhnv125/Osprey",
            "model_name": "anhnv125-osprey_v6",
            "status": "inactive",
            "entertaining": 6.9,
            "stay_in_character": 8.06,
            "user_preference": 6.78,
            "safety_score": 0.94,
            "is_custom_reward": True,
            "thumbs_up": 0,
            "thumbs_down": 0
        },
    }
    return data


def test_get_filled_leaderboard_will_default_model_name_to_submission_id_if_not_existed_for_backwards_compatibility():
    df = make_unique_submissions(2)
    df.update({'submission_id': ['mock-submission-1', 'mock-submission-2']})
    assert list(_get_filled_leaderboard(df)['model_name']) == ['mock-submission-1', 'mock-submission-2']


def test_get_filled_leaderboard_will_default_model_name_to_submission_id_if_none_for_backwards_compatibility():
    df = make_unique_submissions(2)
    df['model_name'] = [None, 'mock-model-name']
    df.update({
        'submission_id': ['mock-submission-1', 'mock-submission-2'],
    })
    assert list(_get_filled_leaderboard(df)['model_name']) == ['mock-submission-1', 'mock-model-name']


def test_get_filled_leaderboard_will_default_is_custom_reward_to_false_if_not_present():
    df = make_unique_submissions(1)
    result = _get_filled_leaderboard(df)
    assert result['is_custom_reward'][0] == False


def test_get_filled_leaderboard_will_convert_is_custom_reward_to_boolean():
    df = make_unique_submissions(3)
    df['is_custom_reward'] = [None, True, False]
    assert list(_get_filled_leaderboard(df)['is_custom_reward']) == [False, True, False]


def test_get_filled_leaderboard_will_default_status_to_none():
    df = make_unique_submissions(3)
    result = _get_filled_leaderboard(df)
    assert result['status'][0] == None


def test_filter_submissions_by_submission_ids():
    submissions = {
        'sub-1': { 'thumbs_up': 4, 'thumbs_down': 4 },
        'sub-2': { 'thumbs_up': 2, 'thumbs_down': 5 },
        'sub-3': { 'thumbs_up': 5, 'thumbs_down': 1 },
    }
    result = _filter_submissions_by_submission_ids(submissions, ['sub-1', 'sub-2'])
    assert result == {
        'sub-1': { 'thumbs_up': 4, 'thumbs_down': 4 },
        'sub-2': { 'thumbs_up': 2, 'thumbs_down': 5 }
    }
    result = _filter_submissions_by_submission_ids(submissions, ['sub-3'])
    assert result == {'sub-3': { 'thumbs_up': 5, 'thumbs_down': 1 }}


def test_filter_submissions_by_feedback_count():
    submissions = {
        'sub-1': { 'thumbs_up': 4, 'thumbs_down': 4 },
        'sub-2': { 'thumbs_up': 2, 'thumbs_down': 5 },
        'sub-3': { 'thumbs_up': 5, 'thumbs_down': 1 },
    }
    result = _filter_submissions_by_feedback_count(submissions, 8)
    assert len(result) == 1
    result = _filter_submissions_by_feedback_count(submissions, 7)
    assert len(result) == 2


def make_unique_submissions(count):
    df = pd.DataFrame(range(count))
    df['total_feedback_count'] = 1000
    df['thumbs_up_ratio'] = 0.5
    df['stay_in_character'] = 7.7
    df['user_preference'] = 8.8
    df['entertaining'] = 9.9
    _fill_unique_ids(df, 'submission_id', prefix='mock-submission-id')
    _fill_unique_ids(df, 'model_repo', prefix='mock-model-repo')
    _fill_unique_ids(df, 'reward_repo', prefix='mock-reward-repo')
    _fill_unique_ids(df, 'developer_uid', prefix='mock-dev-id')
    return df


def _fill_unique_ids(df, field_name, prefix):
    df[field_name] = ["{prefix}-{i}".format(prefix=prefix, i=i) for i in range(len(df))]


