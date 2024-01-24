import os

from mock import ANY, mock, patch
import numpy as np
import pytest
import vcr

import chaiverse as chai


RESOURCE_DIR = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'resources')


@pytest.fixture(autouse=True)
def guanado_data_dir(tmpdir):
    with patch('chaiverse.utils.get_guanaco_data_dir_env') as get_data_dir:
        get_data_dir.return_value = str(tmpdir)
        yield get_data_dir


@mock.patch('chaiverse.metrics.get_leaderboard.get_submissions')
def test_developer_can_call_display_leaderboard_and_pass_in_developer_key_as_arg(get_submissions_mock):
    get_submissions_mock.side_effect = KeyError()
    with pytest.raises(KeyError):
        chai.display_leaderboard(max_workers=1, developer_key='bad-developer-key')
    get_submissions_mock.assert_called_with('bad-developer-key', ANY)


@mock.patch('chaiverse.metrics.display_leaderboard.display_competition_leaderboard')
def test_display_leaderboard_will_call_display_competition_leaderboard(display_competition_leaderboard_mock):
    display_competition_leaderboard_mock.return_value = 'leaderboard-data'
    result = chai.display_leaderboard(max_workers=1)
    assert result == 'leaderboard-data'
    display_competition_leaderboard_mock.assert_called_with(
        competition={
            'id': 'Default',
            'type': 'default',
            'submission_start_date': None,
            'submission_end_date': None,
        }, 
        detailed=False, 
        regenerate=False, 
        developer_key=ANY,
        max_workers=ANY
    )


@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_display_competition_leaderboard_does_not_regress_for_round_robin_competition.yaml'))
def test_display_competition_leaderboard_does_not_regress_for_round_robin_competition(guanado_data_dir):
    competition = {
        "id": 'test_closed_submission_round_robin_competition',
        "type": "submission_closed_feedback_round_robin",
        "submission_start_date": "2024-01-02T13:05",
        "submission_end_date": "2024-01-02T13:06",
        "feedback_sampling": {
            "round_robin": {
                "percentage": 100,
            },
    },
    "submissions": [
        "anhnv125-doll_v4"
        ]
    }
    result = chai.display_competition_leaderboard(max_workers=1, competition=competition)
    expected = [{
        "submission_id": "anhnv125-doll_v4",
        "timestamp": "2024-01-02T13:05:43+00:00",
        "model_name": "anhnv125-doll_v4",
        "model_repo": "anhnv125/doll",
        "developer_uid": "vietanh",
        "reward_repo": "anhnv125/reward-model-v2",
        "model_num_parameters": None,
        "status": "inactive",
        "stay_in_character": 8.23,
        "safety_score": 0.99,
        "is_custom_reward": True,
        "thumbs_down": 9,
        "thumbs_up": 8,
        "mcl": 15.076923076923077,
        "thumbs_up_ratio": 0.5384615384615384,
        "thumbs_up_ratio_se": 0.06892724331792788,
        "repetition": 0.12203659111663187,
        "total_feedback_count": 13,
        "user_preference": None,
        "entertaining": None,
        "overall_rank": None,
        "elo_ratings": None,
        "num_battles": None,
        "num_wins": None,
        "size": None
    }]
    np.testing.assert_equal(expected, result.to_dict('records'))


@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_display_competition_leaderboard_does_not_regress_for_default_competition.yaml'))
def test_display_competition_leaderboard_does_not_regress_for_default_competition(guanado_data_dir):
    competition = {
        "id": 'test_default_competition',
        "type": "default",
        "submission_start_date": "2024-01-02T13:05",
        "submission_end_date": "2024-01-02T13:06",
        "feedback_sampling": {
            "round_robin": {
                "percentage": 100,
            },
    },
    "submissions": [
        "anhnv125-doll_v4"
        ]
    }
    result = chai.display_competition_leaderboard(max_workers=1, competition=competition)
    expected = [{
        "submission_id": "anhnv125-doll_v4",
        "reward_repo": "anhnv125/reward-model-v2",
        "timestamp": "2024-01-02T13:05:43+00:00",
        "developer_uid": "vietanh",
        "model_name": "anhnv125-doll_v4",
        "model_repo": "anhnv125/doll",
        "model_num_parameters": None,
        "status": "inactive",
        "stay_in_character": 8.23,
        "safety_score": 0.99,
        "is_custom_reward": True,
        "thumbs_down": 9,
        "thumbs_up": 8,
        "user_preference": None,
        "entertaining": None,
        "overall_rank": None,
        "elo_ratings": None,
        "num_battles": None,
        "num_wins": None,
        "size": None,
        "thumbs_up_ratio": None,
        "total_feedback_count": None,
        "repetition": None
    }]
    np.testing.assert_equal(expected, result.to_dict('records'))


