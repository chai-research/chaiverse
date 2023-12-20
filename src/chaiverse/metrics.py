from collections import defaultdict
from datetime import datetime
import itertools
import os
import string
from time import time
import warnings

import numpy as np
import pandas as pd

from chaiverse.feedback import get_feedback
from chaiverse.login_cli import auto_authenticate
from chaiverse.utils import print_color, cache, get_all_historical_submissions, distribute_to_workers


DEFAULT_MAX_WORKERS = max(1, min(20, os.cpu_count() - 3))
PUBLIC_LEADERBOARD_MINIMUM_FEEDBACK_COUNT = 150
LEADERBOARD_DISPLAY_COLS = [
    'developer_uid',
    'model_name',
    'is_custom_reward',
    'submission_id',
    'thumbs_up_ratio',
    'stay_in_character',
    'total_feedback_count',
    'overall_rank',
    'safety_score',
    'repetition',
]


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)


warnings.filterwarnings('ignore', 'Mean of empty slice')


def display_leaderboard(
        developer_key=None,
        regenerate=False,
        detailed=False,
        max_workers=DEFAULT_MAX_WORKERS
        ):
    df = get_leaderboard(
        developer_key=developer_key,
        regenerate=regenerate,
        detailed=detailed,
        max_workers=max_workers
        )
    df = _get_formatted_leaderboard(df, detailed)
    _pprint_leaderboard(df)


def get_leaderboard(
        developer_key=None,
        regenerate=False,
        detailed=False,
        max_workers=DEFAULT_MAX_WORKERS
        ):
    df = cache(get_raw_leaderboard, regenerate)(max_workers=max_workers, developer_key=developer_key)
    df = _get_processed_leaderboard(df, detailed)
    return df


@auto_authenticate
def get_raw_leaderboard(max_workers=DEFAULT_MAX_WORKERS, developer_key=None):
    submissions = get_all_historical_submissions(developer_key)
    leaderboard = distribute_to_workers(
        get_leaderboard_row,
        submissions.items(),
        developer_key=developer_key,
        max_workers=max_workers
    )
    return pd.DataFrame(leaderboard)


def get_leaderboard_row(submission_item, developer_key=None):
    submission_id, meta_data = submission_item
    is_deployed = meta_data.get('status') == 'deployed'
    metrics = get_submission_metrics(submission_id, developer_key, reload=is_deployed)
    return {'submission_id': submission_id, **meta_data, **metrics}


@auto_authenticate
def get_submission_metrics(submission_id, developer_key, reload=True):
    feedback = get_feedback(submission_id, developer_key, reload=reload)
    metrics = calc_metrics(feedback)
    return metrics


def calc_metrics(feedback):
    feedback_metrics = FeedbackMetrics(feedback.raw_data)
    metrics = {}
    if len(feedback_metrics.convo_metrics) > 0:
        metrics = {
            'mcl': feedback_metrics.mcl,
            'thumbs_up_ratio': feedback_metrics.thumbs_up_ratio,
            'thumbs_up_ratio_se': feedback_metrics.thumbs_up_ratio_se,
            'repetition': feedback_metrics.repetition_score,
            'total_feedback_count': feedback_metrics.total_feedback_count,
        }
    return metrics


class FeedbackMetrics():
    def __init__(self, feedback_data):
        feedbacks = feedback_data['feedback'].values()
        self.feedbacks = self._filter_duplicated_uid_feedbacks(feedbacks)

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
    def repetition_score(self):
        scores = np.array([m.repetition_score for m in self.convo_metrics])
        is_public = np.array([feedback.get('public', True) for feedback in self.feedbacks])
        return np.nanmean(scores[is_public])

    def _filter_duplicated_uid_feedbacks(self, feedbacks):
        user_feedbacks = defaultdict(list)
        for feedback in feedbacks:
            user_id = feedback["conversation_id"].split("_")[3]
            user_feedbacks[user_id].append(feedback)
        feedbacks = [metrics[0] for _, metrics in user_feedbacks.items()]
        return feedbacks


class ConversationMetrics():
    def __init__(self, messages):
        self.messages = messages

    @property
    def mcl(self):
        return len([m for m in self.messages if not m['deleted']])

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


def _get_processed_leaderboard(df, detailed):
    # maintain backwards compatibility with model_name field
    _fill_default_value(df, 'model_name', df['submission_id'])
    _fill_default_value(df, 'is_custom_reward', False)
    _fill_default_value(df, 'reward_repo', None)
    for col in LEADERBOARD_DISPLAY_COLS:
        _fill_default_value(df, col, None)

    df = _filter_submissions_with_few_feedback(df)
    df = _add_individual_rank(df, value_column='thumbs_up_ratio', rank_column='thumbs_up_rank', ascending=False)
    df = _add_individual_rank(df, value_column='stay_in_character', rank_column='stay_in_character_rank', ascending=False)
    df = _add_overall_rank(df, rank_columns=['thumbs_up_rank', 'stay_in_character_rank'])
    df = _sort_by_overall_rank(df)
    df = df if detailed else _get_submissions_with_unique_model(df)
    df = df if detailed else _get_submissions_with_unique_dev_id(df)
    return df


def _fill_default_value(df, field, default_value):
    if field not in df:
        df[field] = None
    if default_value is not None:
        df[field] = df[field].fillna(default_value)


def _get_formatted_leaderboard(df, detailed):
    df['timestamp'] = df.apply(lambda x: datetime.fromisoformat(x['timestamp']), axis=1)
    df['date'] = df['timestamp'].dt.date
    df.drop(['timestamp'], axis=1, inplace=True)
    df['is_custom_reward'] = df['is_custom_reward'].replace({
        True: '✅',
        False: '❌'
    })
    if not detailed:
        df = df[LEADERBOARD_DISPLAY_COLS]
    df = df.reset_index(drop=True)
    return df


def _get_submissions_with_unique_model(df):
    df = df.drop_duplicates(subset=['model_repo', 'reward_repo'], keep='first')
    return df


def _filter_submissions_with_few_feedback(df):
    filtered_df = df[df.total_feedback_count >= PUBLIC_LEADERBOARD_MINIMUM_FEEDBACK_COUNT]
    return filtered_df


def _add_individual_rank(df, value_column, rank_column, ascending=True):
    df[rank_column] = df[value_column].rank(ascending=ascending, na_option='bottom')
    return df


def _add_overall_rank(df, rank_columns):
    ranks = [df[column] for column in rank_columns]
    overall_score = np.mean(ranks, axis=0)
    df.loc[:, 'overall_score'] = overall_score
    df.loc[:, 'overall_rank'] = df.overall_score.rank(na_option='bottom')
    return df


def _sort_by_overall_rank(df):
    df = df.sort_values('overall_rank', ascending=True, na_position='last').reset_index(drop=True)
    return df


def _get_submissions_with_unique_dev_id(df):
    out = df.drop_duplicates('developer_uid', keep='first')
    return out


def _pprint_leaderboard(df):
    print_color(f'\n💎 Leaderboard:', 'red')
    print(df.round(3).head(30))


def get_sorted_available_models(developer_key):
    models = get_all_historical_submissions(developer_key=developer_key)
    available_models = [k for k, v in models.items() if v['status'] == 'deployed']
    sorted_available_models = sorted(available_models)
    return sorted_available_models
