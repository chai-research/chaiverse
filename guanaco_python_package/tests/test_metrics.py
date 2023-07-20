import os
import unittest.mock as mock

import vcr
import pandas as pd
from freezegun import freeze_time

import chai_guanaco as chai
from chai_guanaco import metrics

RESOURCE_DIR = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'resources')


@mock.patch('chai_guanaco.metrics.get_all_historical_submissions')
@mock.patch('chai_guanaco.metrics._filter_old_submissions')
@mock.patch('chai_guanaco.utils.guanaco_data_dir')
@freeze_time('2023-07-14 19:00:00')
@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_get_leaderboard.yaml'))
def test_get_leaderboard(data_dir_mock, get_ids_mock, tmpdir):
    data_dir_mock.return_value = str(tmpdir)
    get_ids_mock.return_value = [
            'wizard-vicuna-13b-bo4',
            'psiilu-funny-bunny-1-1-1-1-1_1688985871',
            'pygmalionai-pygmalion-6b_1688673204'
    ]
    df = chai.get_leaderboard()
    expected_cols = [
        'submission_id',
        'thumbs_up_ratio',
        'mcl',
        'user_response_length'
        ]
    for col in expected_cols:
        assert col in df.columns, f'{col} not found in leaderboard'
    expected_data = [{
        'submission_id': 'wizard-vicuna-13b-bo4',
        'thumbs_up_ratio': 0.7464789,
        'mcl': 31.5070,
        'user_response_length': 33.5366,
        'total_feedback_count': 71
     },
     {
        'submission_id': 'psiilu-funny-bunny-1-1-1-1-1_1688985871',
        'thumbs_up_ratio': 0.61538,
        'mcl': 11.282051282051283,
        'user_response_length': 25.6599,
        'total_feedback_count': 39
      },
     {
        'submission_id': 'pygmalionai-pygmalion-6b_1688673204',
        'thumbs_up_ratio': 0.60227,
        'mcl': 17.3977,
        'user_response_length': 38.105,
        'total_feedback_count': 88
        }
     ]
    pd.testing.assert_frame_equal(df, pd.DataFrame(expected_data))


@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_get_submission_metrics.yaml'))
@freeze_time('2023-07-14 19:00:00')
def test_get_submission_metrics():
    results = metrics.get_submission_metrics('wizard-vicuna-13b-bo4')
    expected_metrics = {
            'thumbs_up_ratio': 0.7464788732394366,
            'mcl': 31.507042253521128,
            'user_response_length': 33.53655529171172,
            'total_feedback_count': 71
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
    assert convo_metrics.user_response_length == 4.5


def test_print_formatted_leaderboard():
    data = {
        'submission_id': ['tom_1689542168', 'tom_1689404889', 'val_1689051887', 'zl_1689542168'],
        'total_feedback_count': [10, 60, 100, 51],
        'mcl': [1.0, 2.0, 3.0, 4.0],
        'user_response_length': [500, 600, 700, 800],
        'thumbs_up_ratio': [0.1, 0.5, 0.8, 0.2],
    }
    all_metrics_df = pd.DataFrame(data)

    df = metrics._print_formatted_leaderboard(all_metrics_df)

    assert len(df) == 3
    expected_columns = [
            'submission_id', 'mcl', 'user_response_length',
            'thumbs_up_ratio', 'engagement_score', 'overall_rank'
        ]
    assert list(df.columns) == expected_columns
    assert pd.api.types.is_integer_dtype(df['overall_rank'])
    pd.testing.assert_frame_equal(all_metrics_df, pd.DataFrame(data))
