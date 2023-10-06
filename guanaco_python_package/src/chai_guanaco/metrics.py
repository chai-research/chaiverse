from collections import defaultdict
from datetime import datetime
import string

import numpy as np
import pandas as pd
from tqdm import tqdm

from chai_guanaco.feedback import get_feedback
from chai_guanaco.login_cli import auto_authenticate
from chai_guanaco.utils import print_color, cache, get_all_historical_submissions


PUBLIC_LEADERBOARD_MINIMUM_FEEDBACK_COUNT = 100
LEADERBOARD_DISPLAY_COLS = [
    'developer_uid',
    'model_name',
    'submission_id',
    'thumbs_up_ratio',
    'user_writing_speed',
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
            'repetition': feedback_metrics.repetition_score,
            'total_feedback_count': feedback_metrics.total_feedback_count,
            'user_writing_speed': feedback_metrics.user_writing_speed,
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

    @property
    def user_writing_speed(self):
        df = pd.concat([convo.get_conversation_profile() for convo in self.convo_metrics])
        # remove outliers
        for column in df.columns:
            ix = df[column] < np.percentile(df[column], 95)
            df = df[ix]
        summary = summarise_conversation_profile(df)
        return summary['writing_speed']

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

    def get_conversation_profile(self):
        messages = sorted(self.messages, key=lambda x: x['sent_date'])
        data = {'duration': [], 'bot_num_characters': [], 'user_num_characters': []}
        for i, m in enumerate(messages):
            if self._is_from_user(m) and i > 0:
                sent_time = datetime.fromisoformat(m['sent_date'])
                received_time = datetime.fromisoformat(messages[i-1]['sent_date'])
                delta = sent_time - received_time
                data['duration'].append(delta.total_seconds())
                data['bot_num_characters'].append(len(messages[i-1]['content'].strip()))
                data['user_num_characters'].append(len(m['content'].strip()))
        return pd.DataFrame(data)

    def _is_from_user(self, message):
        return '_bot' not in message['sender']['uid']


def summarise_conversation_profile(convo_profile: pd.DataFrame):
    y = convo_profile['duration'].values
    X = convo_profile[['user_num_characters', 'bot_num_characters']].values
    X = np.concatenate([X, np.ones(len(X)).reshape(-1, 1)], axis=1)
    cov = np.dot(X.T, X)
    if np.isclose(np.linalg.det(cov), 0):
        inv_cov = cov * np.nan
    else:
        inv_cov = np.linalg.inv(cov)
    beta = np.dot(inv_cov, np.dot(X.T, y))
    result = {
        'writing_speed': 1 / beta[0],
        'reading_speed': 1 / beta[1],
        'thinking_time': beta[2]
    }
    return result


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
    thumbs_up_rank = df['thumbs_up_ratio'].rank(ascending=False)
    writing_speed_rank = df['user_writing_speed'].rank(ascending=True)
    df['overall_score'] = np.mean([writing_speed_rank, thumbs_up_rank], axis=0)
    df['overall_rank'] = df.overall_score.rank().astype(int)
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
