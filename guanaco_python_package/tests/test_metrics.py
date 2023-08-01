import os
import unittest.mock as mock

import vcr
import pandas as pd
from freezegun import freeze_time

import chai_guanaco as chai
from chai_guanaco import metrics

RESOURCE_DIR = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'resources')


@mock.patch('chai_guanaco.metrics.get_all_historical_submissions')
@mock.patch('chai_guanaco.utils.guanaco_data_dir')
@freeze_time('2023-07-28 00:00:00')
@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_get_leaderboard.yaml'))
def test_get_leaderboard(data_dir_mock, get_ids_mock, tmpdir):
    data_dir_mock.return_value = str(tmpdir)
    get_ids_mock.return_value = historical_submisions()
    df = chai.get_leaderboard()
    expected_cols = [
        'submission_id',
        'thumbs_up_ratio',
        'mcl',
        'user_response_length'
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
                'thumbs_up_ratio': 0.45454545454545453,
                'mcl': 12.0,
                'user_response_length': 75.0,
                'total_feedback_count': 33
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
                'thumbs_up_ratio': 0.41148325358851673,
                'mcl': 9.191387559808613,
                'user_response_length': 84.0,
                'total_feedback_count': 209
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
                'thumbs_up_ratio': 0.48214285714285715,
                'mcl': 13.053571428571429,
                'user_response_length': 98.5,
                'total_feedback_count': 56
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
            'user_response_length': 354.,
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
    assert convo_metrics.user_response_length == 9


def test_print_formatted_leaderboard():
    data = {
        'submission_id': ['tom_1689542168', 'tom_1689404889', 'val_1689051887', 'zl_1689542168'],
        'total_feedback_count': [10, 60, 100, 51],
        'mcl': [1.0, 2.0, 3.0, 4.0],
        'user_response_length': [500, 600, 700, 800],
        'thumbs_up_ratio': [0.1, 0.5, 0.8, 0.2],
        'model_name': ['psutil', 'htop', 'watch', 'gunzip'],
        'developer_uid': ['tom', 'tom', 'val', 'zl'],
        'timestamp': ['2023-07-24 18:22:40+00:00'] * 4,
        'model_repo': ['psutil', 'psutil', 'watch', 'gunzip']
    }
    all_metrics_df = pd.DataFrame(data)

    df = metrics._print_formatted_leaderboard(all_metrics_df, detailed=True)

    assert len(df) == 3
    expected_columns = [
            'submission_id',
            'total_feedback_count',
            'mcl',
            'user_response_length',
            'thumbs_up_ratio',
            'model_name',
            'developer_uid',
            'model_repo',
            'date',
            'overall_rank'
        ]
    assert list(df.columns) == expected_columns
    assert pd.api.types.is_integer_dtype(df['overall_rank'])


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
