import requests
from datetime import datetime
import string

import numpy as np
import pandas as pd
from tqdm import tqdm

from chai_guanaco.feedback import get_feedback
from chai_guanaco.login_cli import auto_authenticate
from chai_guanaco.submit import get_url
from chai_guanaco.utils import print_color, cache


LEADERBOARD_ENDPOINT = "/leaderboard"
PUBLIC_LEADERBOARD_MINIMUM_FEEDBACK_COUNT = 100
LEADERBOARD_DISPLAY_COLS = [
    'developer_uid',
    'model_name',
    'thumbs_up_ratio',
    'user_engagement',
    'retry_score',
    'repetition',
    'total_feedback_count',
    'overall_rank',
]


@auto_authenticate
def display_leaderboard(
        developer_key=None,
        regenerate=False,
        detailed=False,
        ):
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
    metrics = {}
    if len(feedback_metrics.convo_metrics) > 0:
        metrics = {
            'mcl': feedback_metrics.mcl,
            'thumbs_up_ratio': feedback_metrics.thumbs_up_ratio,
            'thumbs_up_ratio_se': feedback_metrics.thumbs_up_ratio_se,
            'retry_score': feedback_metrics.retry_score,
            'repetition': feedback_metrics.repetition_score,
            'user_engagement': feedback_metrics.mean_user_engagement,
            'user_engagement_se': feedback_metrics.user_engagement_se,
            'total_feedback_count': feedback_metrics.total_feedback_count,
        }
    return metrics


def get_all_historical_submissions(developer_key):
    headers = {"developer_key": developer_key}
    url = get_url(LEADERBOARD_ENDPOINT)
    resp = requests.get(url, headers=headers)
    assert resp.status_code == 200, resp.json()
    return resp.json()


class FeedbackMetrics():
    def __init__(self, feedback_data):
        feedbacks = feedback_data['feedback']
        self.feedbacks = feedbacks.values()

    @property
    def convo_metrics(self):
        return [ConversationMetrics(feedback['messages']) for feedback in self.feedbacks]

    @property
    def thumbs_up_ratio(self):
        is_thumbs_up = [feedback['thumbs_up'] for feedback in self.feedbacks]
        thumbs_up = sum(is_thumbs_up)
        thumbs_up_ratio = np.nan if not thumbs_up else thumbs_up / len(is_thumbs_up)
        return thumbs_up_ratio

    @property
    def thumbs_up_ratio_se(self):
        num = self.thumbs_up_ratio * (1 - self.thumbs_up_ratio)
        denom = self.total_feedback_count**0.5
        se = np.nan if self.total_feedback_count < 2 else num / denom
        return se

    @property
    def total_feedback_count(self):
        return len(self.feedbacks)

    @property
    def mcl(self):
        return np.mean([m.mcl for m in self.convo_metrics])

    @property
    def mean_user_engagement(self):
        # taking geometric mean over user response length
        log_engagement = [np.log(m.user_engagement+1) for m in self.convo_metrics]
        geometric_mean = np.exp(np.mean(log_engagement))
        return geometric_mean

    @property
    def user_engagement_se(self):
        log_engagement = [np.log(m.user_engagement+1) for m in self.convo_metrics]
        log_mean = np.mean(log_engagement)
        log_se = np.std(log_engagement) / len(log_engagement)**0.5
        mean = np.exp(log_mean)
        se = mean * log_se
        return se

    @property
    def retry_score(self):
        total_retries = sum([m.num_retries for m in self.convo_metrics])
        total_bot_responses = sum([m.num_model_responses for m in self.convo_metrics])
        return 0 if (total_bot_responses) == 0 else total_retries / total_bot_responses

    @property
    def repetition_score(self):
        scores = np.array([m.repetition_score for m in self.convo_metrics])
        is_public = np.array([feedback.get('public', True) for feedback in self.feedbacks])
        return np.nanmean(scores[is_public])


class ConversationMetrics():
    def __init__(self, messages):
        self.messages = messages

    @property
    def mcl(self):
        return len([m for m in self.messages if not m['deleted']])

    @property
    def num_retries(self):
        return len([m for m in self.messages if m['deleted']])

    @property
    def num_model_responses(self):
        user_messages = [m for m in self.messages if self._is_from_user(m)]
        return len(self.messages) - len(user_messages)

    @property
    def user_engagement(self):
        response_length = [len(m['content']) for m in self.messages if self._is_from_user(m)]
        return np.sum(response_length)

    @property
    def repetition_score(self):
        responses = [m['content'] for m in self.messages if not self._is_from_user(m)]
        score = np.nan if len(responses) < 2 else get_repetition_score(responses)
        return score

    def _is_from_user(self, message):
        return '_bot' not in message['sender']['uid']


def get_repetition_score(responses):
    # average jaccard similarities over unigrams
    list_of_tokens = _tokenize_responses(responses)
    pairs = zip(list_of_tokens[:-1], list_of_tokens[1:])
    similarities = [_get_jaccard_similarity(set1, set2) for set1, set2 in pairs]
    return np.mean(similarities)


def _get_jaccard_similarity(set1, set2):
    intersection_len = len(set1.intersection(set2))
    union_len = len(set1.union(set2))
    return intersection_len / union_len


def _tokenize_responses(responses):
    return [set(_remove_punctuation(text).split()) for text in responses]


def _remove_punctuation(text):
    translation_table = str.maketrans('', '', string.punctuation)
    cleaned_text = text.translate(translation_table)
    if len(cleaned_text.split()) == 0:
        cleaned_text = '...'
    return cleaned_text.lower()


def _print_formatted_leaderboard(raw_df, detailed):
    df = _get_processed_leaderboard(raw_df)
    if not detailed:
        df = _get_df_with_unique_hf_repo(df)
        df = df[LEADERBOARD_DISPLAY_COLS].copy()
    _pprint_leaderboard(df, '💎 Leaderboard:', 'overall_rank', detailed, ascending=True)
    return df


def _get_processed_leaderboard(df):
    # maintain backwards compatibility with model_name field
    df['model_name'] = df['model_name'].fillna(df['submission_id'])
    df = _format_leaderboard_date(df)
    df = _filter_submissions_with_few_feedback(df)
    df = df.reset_index(drop=True)
    df = _add_overall_rank(df)
    return df


def _format_leaderboard_date(df):
    df['timestamp'] = df.apply(lambda x: datetime.fromisoformat(x['timestamp']), axis=1)
    df['date'] = df['timestamp'].dt.date
    df.drop(['timestamp'], axis=1, inplace=True)
    return df


def _get_df_with_unique_hf_repo(df):
    df = df.sort_values(['overall_rank'], ascending=True)
    df = df.drop_duplicates('model_repo', keep='first')
    return df


def _filter_submissions_with_few_feedback(df):
    filtered_df = df[df.total_feedback_count >= PUBLIC_LEADERBOARD_MINIMUM_FEEDBACK_COUNT]
    return filtered_df


def _add_overall_rank(df):
    response_len = df['user_engagement']
    thumbs_up = df['thumbs_up_ratio']
    retry_score = df['retry_score']
    response_len_rank = response_len.rank(ascending=False)
    thumbs_up_rank = thumbs_up.rank(ascending=False)
    retry_rate_rank = retry_score.rank(ascending=True)
    overall_score = (response_len_rank + thumbs_up_rank + retry_rate_rank) / 3
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
    print(df.round(3).head(30))
