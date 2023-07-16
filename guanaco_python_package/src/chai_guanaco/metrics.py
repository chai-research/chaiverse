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


@auto_authenticate
def display_leaderboard(developer_key=None):
    df = get_leaderboard(developer_key)
    _print_formatted_leaderboard(df)
    return df


@cache
@auto_authenticate
def get_leaderboard(developer_key=None):
    submission_ids = _get_leaderboard_submission_ids(developer_key)
    leaderboard = []
    for submission_id in tqdm(submission_ids, 'Getting Metrics'):
        metrics = get_submission_metrics(submission_id, developer_key)
        leaderboard.append({'submission_id': submission_id, **metrics})
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
        return np.mean([m.user_response_length for m in self.convo_metrics])


class ConversationMetrics():
    def __init__(self, messages):
        self.messages = messages

    @property
    def mcl(self):
        return len([m for m in self.messages if not m['deleted']])

    @property
    def user_response_length(self):
        response_length = [len(m['content']) for m in self.messages if self._is_from_user(m)]
        return np.mean(response_length)

    def _is_from_user(self, message):
        return '_bot' not in message['sender']['uid']


def _get_leaderboard_submission_ids(developer_key):
    headers = {
        "developer_key": developer_key,
    }
    url = get_url(LEADERBOARD_ENDPOINT)
    resp = requests.get(url, headers=headers)
    assert resp.status_code == 200, resp.json()
    return resp.json().keys()


def _print_formatted_leaderboard(df):
    df = _get_filtered_leaderboard(df)
    df['engagement_score'] = _get_engagement_score(df.mcl, df.user_response_length)
    df['overall_rank'] = _get_overall_rank(df.engagement_score, df.thumbs_up_ratio)
    df = df.sort_values('overall_rank').reset_index(drop=True)
    _print_grand_prize(df)
    _print_engagement_prize(df)
    _print_thumbs_up_prize(df)
    return df


def _get_engagement_score(mcl, user_response_length):
    return mcl * user_response_length


def _get_filtered_leaderboard(df):
    filtered_df = df[df.total_feedback_count > MINIMUM_FEEDBACK_NUMBER_TO_DISPLAY]
    filtered_df = filtered_df.drop(['total_feedback_count'], axis=1)
    return filtered_df


def _get_overall_rank(engagement_score, thumbs_up_ratio):
    engagement_rank = engagement_score.rank(ascending=False)
    thumbs_up_rank = thumbs_up_ratio.rank(ascending=False)
    overall_score = (engagement_rank + thumbs_up_rank) / 2
    overall_rank = overall_score.rank().astype(int)
    return overall_rank


def _print_grand_prize(df):
    print_color('\n💎 Grand Prize Contenders:', 'red')
    df = df.sort_values('overall_rank').reset_index(drop=True)
    print(df.round(3).head(15))


def _print_thumbs_up_prize(df):
    print_color('\n👍 Thumbs Up Prize Contenders:', 'red')
    df = df.sort_values('thumbs_up_ratio', ascending=False).reset_index(drop=True)
    print(df.round(3).head(15))


def _print_engagement_prize(df):
    print_color('\n😎 Engagement Prize Contenders:', 'red')
    df = df.sort_values('engagement_score', ascending=False).reset_index(drop=True)
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
