from datetime import datetime
import requests

import numpy as np
import pandas as pd
from tqdm import tqdm

from chai_guanaco.feedback import get_feedback
from chai_guanaco.login_cli import auto_authenticate
from chai_guanaco.submit import get_url
from chai_guanaco.utils import print_color, cache


LEADERBOARD_ENDPOINT = "/leaderboard"
FEEDBACK_CUTOFF_DAYS = 7
MINIMUM_FEEDBACK_NUMBER_TO_DISPLAY = 50
LEADERBOARD_DISPLAY_COLS = [
    'developer_uid',
    'model_name',
    'thumbs_up_ratio',
    'user_response_length',
    'overall_rank',
]


@auto_authenticate
def display_leaderboard(developer_key=None, regenerate=False, detailed=False):
    df = cache(get_leaderboard, regenerate)(developer_key)
    _print_formatted_leaderboard(df, detailed)
    return df


@auto_authenticate
def get_leaderboard(developer_key=None):
    submission_data = get_all_historical_submissions(developer_key)
    leaderboard = []
    for submission_id, meta_data in tqdm(submission_data.items(), 'Getting Metrics'):
        metrics = get_submission_metrics(submission_id, developer_key)
        meta_data.update(metrics)
        leaderboard.append({'submission_id': submission_id, **meta_data})
    return pd.DataFrame(leaderboard)


@auto_authenticate
def get_submission_metrics(submission_id, developer_key):
    feedback = get_feedback(submission_id, developer_key)
    feedback_metrics = FeedbackMetrics(feedback.raw_data)

    metrics = {
        'thumbs_up_ratio': feedback_metrics.thumbs_up_ratio,
        'mcl': feedback_metrics.mcl,
        'user_response_length': feedback_metrics.user_response_length,
        'total_feedback_count': feedback_metrics.total_feedback_count
    }
    return metrics


def get_all_historical_submissions(developer_key):
    headers = {
        "developer_key": developer_key,
    }
    url = get_url(LEADERBOARD_ENDPOINT)
    resp = requests.get(url, headers=headers)
    assert resp.status_code == 200, resp.json()
    return resp.json()


class FeedbackMetrics():
    def __init__(self, feedback_data):
        feedbacks = _filter_feedbacks(feedback_data['feedback'])
        self.feedbacks = feedbacks.values()

    @property
    def convo_metrics(self):
        return [ConversationMetrics(feedback['messages']) for feedback in self.feedbacks]

    @property
    def thumbs_up_ratio(self):
        is_thumbs_up = [feedback['thumbs_up'] for feedback in self.feedbacks]
        thumbs_up = sum(is_thumbs_up)
        thumbs_up_ratio = 0 if not thumbs_up else thumbs_up / len(is_thumbs_up)
        return thumbs_up_ratio

    @property
    def total_feedback_count(self):
        return len(self.feedbacks)

    @property
    def mcl(self):
        return np.mean([m.mcl for m in self.convo_metrics])

    @property
    def user_response_length(self):
        return np.median([m.user_response_length for m in self.convo_metrics])


class ConversationMetrics():
    def __init__(self, messages):
        self.messages = messages

    @property
    def mcl(self):
        return len([m for m in self.messages if not m['deleted']])

    @property
    def user_response_length(self):
        response_length = [len(m['content']) for m in self.messages if self._is_from_user(m)]
        return np.sum(response_length)

    def _is_from_user(self, message):
        return '_bot' not in message['sender']['uid']


def _print_formatted_leaderboard(raw_df, detailed):
    df = _get_processed_leaderboard(raw_df)
    if not detailed:
        df = _get_df_with_unique_hf_repo(df)
        df = df[LEADERBOARD_DISPLAY_COLS].copy()
    _pprint_leaderboard(df, '💎 Grand Prize Contenders:', 'overall_rank', detailed, ascending=True)
    _pprint_leaderboard(df, '😎 Engagement Prize Contenders:', 'user_response_length', detailed, ascending=False)
    _pprint_leaderboard(df, '👍 Thumbs Up Prize Contenders:', 'thumbs_up_ratio', detailed, ascending=False)
    return df


def _get_processed_leaderboard(df):
    # maintain backwards compatibility with model_name field
    df['model_name'] = df['model_name'].fillna(df['submission_id'])
    df = _format_leaderboard_date(df)
    df = _filter_submissions_with_few_feedback(df)
    df = _add_overall_rank(df)
    return df


def _format_leaderboard_date(df):
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df.drop(['timestamp'], axis=1, inplace=True)
    return df


def _get_df_with_unique_hf_repo(df):
    df = df.sort_values(['total_feedback_count'], ascending=False)
    df = df.drop_duplicates('model_repo', keep='first')
    return df


def _filter_submissions_with_few_feedback(df):
    filtered_df = df[df.total_feedback_count > MINIMUM_FEEDBACK_NUMBER_TO_DISPLAY]
    return filtered_df


def _add_overall_rank(df):
    response_len, thumbs_up = df['user_response_length'], df['thumbs_up_ratio']
    response_len_rank = response_len.rank(ascending=False)
    thumbs_up_rank = thumbs_up.rank(ascending=False)
    overall_score = (response_len_rank + thumbs_up_rank) / 2
    df['overall_rank'] = overall_score.rank().astype(int)
    df = df.sort_values('overall_rank').reset_index(drop=True)
    return df


def _get_df_with_unique_dev_id(df):
    out = df.drop_duplicates('developer_uid', keep='first').reset_index(drop=True)
    return out


def _pprint_leaderboard(df, title, sort_by, detailed=False, ascending=True):
    print_color(f'\n{title}', 'red')
    df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    df = df if detailed else _get_df_with_unique_dev_id(df)
    print(df.round(3).head(15))


def _filter_feedbacks(feedbacks):
    out = {k: v for k, v in feedbacks.items() if _is_n_days_within_current_date(k)}
    return out


def _is_n_days_within_current_date(convo_id):
    timestamp = _get_timestamp_from_convo_id(convo_id)
    current_utc = datetime.utcnow()
    delta = current_utc - timestamp
    return delta.days <= FEEDBACK_CUTOFF_DAYS


def _get_timestamp_from_convo_id(convo_id):
    timestamp = int(convo_id.split('_')[-1])
    timestamp = datetime.utcfromtimestamp(timestamp)
    return timestamp
