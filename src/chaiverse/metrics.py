from collections import defaultdict
from datetime import datetime
import itertools
import os
import string
from time import time
import warnings

import numpy as np
import pandas as pd

from chaiverse.competition import get_competitions
from chaiverse.feedback import get_feedback, is_submission_updated
from chaiverse.login_cli import auto_authenticate
from chaiverse.utils import print_color, cache, get_all_historical_submissions, distribute_to_workers

DEFAULT_MAX_WORKERS = max(1, min(20, os.cpu_count() - 3))
PUBLIC_LEADERBOARD_MINIMUM_FEEDBACK_COUNT = 1
LEADERBOARD_DISPLAY_COLS = [
    'developer_uid',
    'model_name',
    'submission_id',
    'is_custom_reward',
    'stay_in_character',
    'user_preference',
    'entertaining',
    'overall_rank',
    'repetition',
    'safety_score',
    'thumbs_up_ratio',
    'total_feedback_count',
    'status'
]

COMPETITON_LEADERBOARD_DISPLAY_COLS = [
    'developer_uid',
    'model_name',
    'thumbs_up_ratio',
    'overall_rank',
    'total_feedback_count',
    'repetition',
    'stay_in_character',
    'user_preference',
    'entertaining',
    'safety_score',
    'is_custom_reward',
    'submission_id',
    'model_parameter_size',
]

MODEL_EVAL_SCORE_COLS = ['stay_in_character', 'user_preference', 'entertaining']

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)
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
        max_workers=max_workers
        )

    display_df = df.copy()
    display_df = get_display_leaderboard(display_df, detailed)
    _pprint_leaderboard(display_df, title='Leaderboard')

    return df


def display_competition_leaderboard(
        developer_key=None,
        max_workers=DEFAULT_MAX_WORKERS
        ):
    competition = get_competitions()[-1]
    df = get_competition_leaderboard(
        competition,
        developer_key=developer_key,
        max_workers=max_workers
        )

    display_df = df.copy()
    display_df = get_display_competition_leaderboard(display_df)
    competition_id = competition.get('id')
    _pprint_leaderboard(display_df, f'{competition_id} Leaderboard')

    return df


def get_leaderboard(
        developer_key=None,
        regenerate=False,
        max_workers=DEFAULT_MAX_WORKERS
        ):
    df = cache(get_raw_leaderboard, regenerate)(max_workers=max_workers, developer_key=developer_key)
    df = _get_filled_leaderboard(df)
    return df


def get_competition_leaderboard(
        competition,
        developer_key=None,
        max_workers=DEFAULT_MAX_WORKERS
        ):
    time_range_in_sec = competition['start_time'], competition['end_time']
    submission_ids = competition['submissions']
    df = get_raw_leaderboard(max_workers=max_workers, developer_key=developer_key, time_range_in_sec=time_range_in_sec, submission_ids=submission_ids)
    df = _get_filled_leaderboard(df)
    df = _get_ranked_leaderboard(df, sort_column='thumbs_up_rank')
    df = _get_formatted_leaderboard(df)
    return df


def get_display_leaderboard(df, detailed):
    df = _get_ranked_leaderboard(df, 'overall_rank')
    df = df if detailed else _get_deduped_leaderboard(df)
    df = _get_formatted_leaderboard(df)
    df = df if detailed else df[LEADERBOARD_DISPLAY_COLS]
    return df


def get_display_competition_leaderboard(df):
    df = _get_deduped_leaderboard(df)
    df = df[COMPETITON_LEADERBOARD_DISPLAY_COLS]
    return df


@auto_authenticate
def get_raw_leaderboard(max_workers=DEFAULT_MAX_WORKERS, developer_key=None, time_range_in_sec=None, submission_ids=None):
    submissions = get_all_historical_submissions(developer_key)
    submissions = _filter_submissions(submissions, submission_ids) if submission_ids else submissions
    leaderboard = distribute_to_workers(
        get_leaderboard_row,
        submissions.items(),
        developer_key=developer_key,
        time_range_in_sec=time_range_in_sec,
        max_workers=max_workers
    )
    return pd.DataFrame(leaderboard)


def _filter_submissions(submissions, submission_ids):
    filtered_submissions = {
        submission_id: data
        for submission_id, data in submissions.items()
        if submission_id in submission_ids
    }
    return filtered_submissions


def get_leaderboard_row(submission_item, developer_key=None, time_range_in_sec=None):
    submission_id, meta_data = submission_item
    server_feedback_no = int(meta_data["thumbs_up"]) + int(meta_data["thumbs_down"])
    is_updated = is_submission_updated(submission_id, server_feedback_no)
    metrics = get_submission_metrics(submission_id, developer_key, reload=is_updated, time_range_in_sec=time_range_in_sec)
    return {'submission_id': submission_id, **meta_data, **metrics}


@auto_authenticate
def get_submission_metrics(submission_id, developer_key, reload=True, time_range_in_sec=None):
    feedback = get_feedback(submission_id, developer_key, reload=reload)
    feedback_metrics = FeedbackMetrics(feedback.raw_data)
    feedback_metrics.filter_for_timestamp_range(time_range_in_sec)
    feedback_metrics.filter_duplicated_uid()
    metrics = calc_metrics(feedback_metrics)
    return metrics


def calc_metrics(feedback_metrics):
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
        feedback_dict = feedback_data['feedback']
        feedback_dict = _insert_server_timestamp(feedback_dict)
        self.feedbacks = list(feedback_dict.values())

    def filter_duplicated_uid(self):
        self.feedbacks = _filter_duplicated_uid_feedbacks(self.feedbacks)

    def filter_for_timestamp_range(self, time_range_in_sec=None):
        (begin, end) = time_range_in_sec if time_range_in_sec else (0, float('inf'))
        self.feedbacks = [
            feedback for feedback in self.feedbacks
            if begin < feedback['server_timestamp'] < end 
        ]

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
        breakpoint()
        return np.nanmean(scores[is_public])


def _insert_server_timestamp(feedback_dict):
    for feedback_id, feedback in feedback_dict.items():
        feedback['server_timestamp'] = int(feedback_id.split('_')[-1])
    return feedback_dict


def _filter_duplicated_uid_feedbacks(feedbacks):
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


def _get_filled_leaderboard(df):
    # maintain backwards compatibility with model_name field
    _fill_default_value(df, 'model_name', df['submission_id'])
    _fill_default_value(df, 'is_custom_reward', False)
    _fill_default_value(df, 'reward_repo', None)
    for col in LEADERBOARD_DISPLAY_COLS:
        _fill_default_value(df, col, None)
    return df


def _get_ranked_leaderboard(df, sort_column):
    df = _filter_submissions_with_few_feedback(df)
    df = _add_individual_rank(df, value_column='thumbs_up_ratio', rank_column='thumbs_up_rank', ascending=False)
    rank_columns = []
    for score_column in MODEL_EVAL_SCORE_COLS:
        rank_column = f'{score_column}_rank'
        rank_columns.append(rank_column)
        df = _add_individual_rank(df, value_column=score_column, rank_column=rank_column, ascending=False)
    df = _add_overall_rank(df, rank_columns=rank_columns)
    df = _sort_by_rank(df, sort_column=sort_column)
    return df


def _get_deduped_leaderboard(df):
    df = _get_submissions_with_unique_model(df)
    df = _get_submissions_with_unique_dev_id(df)
    return df


def _fill_default_value(df, field, default_value):
    if field not in df:
        df[field] = None
    if default_value is not None:
        df[field] = df[field].fillna(default_value)


def _get_formatted_leaderboard(df):
    df['timestamp'] = df.apply(lambda x: datetime.fromisoformat(x['timestamp']), axis=1)
    df['date'] = df['timestamp'].dt.date
    df.drop(['timestamp'], axis=1, inplace=True)
    df['is_custom_reward'] = df['is_custom_reward'].replace({
        True: '✅',
        False: '❌'
    })
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


def _sort_by_rank(df, sort_column):
    df = df.sort_values(sort_column, ascending=True, na_position='last').reset_index(drop=True)
    return df


def _get_submissions_with_unique_dev_id(df):
    out = df.drop_duplicates('developer_uid', keep='first')
    return out


def _pprint_leaderboard(df, title):
    print_color(f'\n💎 {title}:', 'red')
    print(df.round(3).head(30))


def get_sorted_available_models(developer_key):
    models = get_all_historical_submissions(developer_key=developer_key)
    available_models = [k for k, v in models.items() if v['status'] == 'deployed']
    sorted_available_models = sorted(available_models)
    return sorted_available_models
