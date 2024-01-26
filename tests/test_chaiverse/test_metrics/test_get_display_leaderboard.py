from datetime import datetime

from mock import patch
import pandas as pd
import pytest


from chaiverse.metrics.get_display_leaderboard import (
    _add_individual_rank,
    _add_overall_rank,
    _get_isoformatted_timestamp,
    _get_model_size,
    _get_ranked_leaderboard,
    _get_deduped_leaderboard,
    _sort,
)


@pytest.fixture(autouse=True)
def guanado_data_dir(tmpdir):
    with patch('chaiverse.utils.get_guanaco_data_dir_env') as get_data_dir:
        get_data_dir.return_value = str(tmpdir)
        yield get_data_dir


def test_get_ranked_leaderboard_will_sort_by_rank_for_same_reward_repo_but_different_model_repo_if_not_in_detailed_mode():
    df = make_unique_submissions(2)
    df.update({
        'submission_id': ['submission-1', 'submission-2'],
        'reward_repo': ['mock-reward-repo', 'mock-reward-repo'],
        'stay_in_character': [8, 9]
    })
    result = _get_ranked_leaderboard(df, sort_params=dict(by='overall_rank'))
    assert list(result['submission_id']) == ['submission-2', 'submission-1']


def test_get_ranked_leaderboard_will_sort_by_rank_for_same_model_repo_but_different_reward_repo_if_not_in_detailed_mode():
    df = make_unique_submissions(2)
    df.update({
        'submission_id': ['submission-1', 'submission-2'],
        'model_repo': ['mock-model-repo', 'mock-model-repo'],
        'stay_in_character': [8, 9]
    })
    result = _get_ranked_leaderboard(df, sort_params=dict(by='overall_rank'))
    assert list(result['submission_id']) == ['submission-2', 'submission-1']


def test_get_ranked_leaderboard_will_sort_by_rank_for_same_model_repo_and_same_reward_repo_but_different_developer_uid_if_not_in_detailed_mode():
    df = make_unique_submissions(4)
    df.update({
        'developer_uid': ['developer_uid-2', 'developer_uid-1', 'developer_uid-2', 'developer_uid-3'],
        'submission_id': ['submission-2-bad', 'submission-1', 'submission-2-top2', 'submission-3-top1'],
        'model_repo': ['mock-model-repo-bad', 'random-mock-model-repo', 'mock-model-repo', 'mock-model-repo'],
        'reward_repo': ['mock-reward-repo-bad', 'random-mock-reward-repo', 'mock-reward-repo', 'mock-reward-repo'],
        'stay_in_character': [1, 7, 8, 9]
    })
    result = _get_ranked_leaderboard(df, sort_params=dict(by='overall_rank'))
    result = _get_deduped_leaderboard(result)
    assert list(result['submission_id']) == ['submission-3-top1', 'submission-2-top2', 'submission-1']


def test_get_isoformatted_timestamp():
    result = _get_isoformatted_timestamp('2024-01-01')
    assert result == datetime(2024, 1, 1, 0, 0)


@pytest.mark.parametrize(
    "num_parameters, expected_size", [
    ([34388917248, '34']),
    ([19994362880, '20']),
    ([13015864320, '13']),
    ([10731524096, '11']),
    ([7241732096, '7']),
    ([float('nan'), 'n/a']),
    ([None, 'n/a']),
])
def test_get_model_size(num_parameters, expected_size):
    result = _get_model_size(num_parameters)
    assert expected_size == result


@pytest.mark.parametrize(
        "value, expected_rank", [
        ([0.9, 0.8], [1.0, 2.0]),
        ([0.8, 0.8], [1.5, 1.5]),
        ([0.8, 0.9], [2.0, 1.0]),
        ([float('nan'), 0.9], [2.0, 1.0]),
        ([0.8, float('nan')], [1.0, 2.0])])
def test_add_individual_rank_descending(value, expected_rank):
    df = pd.DataFrame({
        'value': value,
    })
    result = _add_individual_rank(df, value_column='value', rank_column='rank', ascending=False)
    assert list(result['rank']) == expected_rank


@pytest.mark.parametrize(
        "value, expected_rank", [
        ([0.9, 0.8], [2.0, 1.0]),
        ([0.8, 0.8], [1.5, 1.5]),
        ([0.8, 0.9], [1.0, 2.0]),
        ([float('nan'), 0.9], [2.0, 1.0]),
        ([0.8, float('nan')], [1.0, 2.0])])
def test_add_individual_rank_ascending(value, expected_rank):
    df = pd.DataFrame({
        'value': value,
    })
    result = _add_individual_rank(df, value_column='value', rank_column='rank', ascending=True)
    assert list(result['rank']) == expected_rank


def test_add_overall_rank():
    df = pd.DataFrame({
        "rank1": [float('nan'), 3.0, 4.0, 1.5, 1.5],
        "rank2": [float('nan'), 4.0, 2.5, 2.5, 1.0 ],
    })
    expected = pd.DataFrame({
        "overall_score": [float('nan'), 7/2, 6.5/2, 4/2, 2.5/2],
        "overall_rank": [5.0, 4.0, 3.0, 2.0, 1.0],
    })
    result = _add_overall_rank(df, ['rank1', 'rank2'])
    pd.testing.assert_frame_equal(result[['overall_score', 'overall_rank']], expected)



def test_sort_ascending():
    df = pd.DataFrame({
        "overall-score": [float('nan'), 10, 25, 40, 25],
        "overall_rank": [float('nan'), 4, 2.5, 1, 2.5]
    })
    expected = pd.DataFrame({
        "overall-score": [40, 25, 25, 10, float('nan')],
        "overall_rank": [1, 2.5, 2.5, 4.0, float('nan')]
    })
    result = _sort(df, sort_params=dict(by='overall_rank', ascending=True))
    pd.testing.assert_frame_equal(result, expected)


def test_sort_descending():
    df = pd.DataFrame({
        "overall-score": [float('nan'), 10, 25, 40, 25],
        "overall_rank": [float('nan'), 4, 2.5, 1, 2.5]
    })
    expected = pd.DataFrame({
        "overall-score": [10, 25, 25, 40, float('nan')],
        "overall_rank": [4.0, 2.5, 2.5, 1, float('nan')]
    })
    result = _sort(df, sort_params=dict(by='overall_rank', ascending=False))
    pd.testing.assert_frame_equal(result, expected)


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


