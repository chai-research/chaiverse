__all__ = ["get_leaderboard"]


import numpy as np
import pandas as pd

from chaiverse.lib import binomial_tools
from chaiverse.utils import get_submissions, distribute_to_workers
from chaiverse.metrics.feedback_metrics import FeedbackMetrics
from chaiverse import constants, feedback


def get_leaderboard(
        developer_key=None,
        max_workers=constants.DEFAULT_MAX_WORKERS,
        submission_date_range=None,
        evaluation_date_range=None,
        submission_ids=None,
        fetch_feedback=False,
        ):
    submissions = get_submissions(developer_key, submission_date_range)
    submissions = _filter_submissions_by_submission_ids(submissions, submission_ids) if submission_ids != None else submissions
    submissions = _filter_submissions_by_feedback_count(submissions, constants.PUBLIC_LEADERBOARD_MINIMUM_FEEDBACK_COUNT)
    df = distribute_to_workers(
        get_leaderboard_row,
        submissions.items(),
        developer_key=developer_key,
        evaluation_date_range=evaluation_date_range,
        max_workers=max_workers,
        fetch_feedback=fetch_feedback
    )
    df = pd.DataFrame(df)
    if len(df):
        df = _get_filled_leaderboard(df)
        df.index = np.arange(1, len(df)+1)
    return df


def get_leaderboard_row(submission_item, developer_key=None, evaluation_date_range=None, fetch_feedback=False):
    submission_id, submission_data = submission_item
    submission_feedback_total = submission_data['thumbs_up'] + submission_data['thumbs_down']
    is_updated = feedback.is_submission_updated(submission_id, submission_feedback_total)

    feedback_metrics = {
        'thumbs_up_ratio': binomial_tools.get_ratio(submission_data['thumbs_up'], submission_data['thumbs_down']),
        'thumbs_up_ratio_se': binomial_tools.get_ratio_se(submission_data['thumbs_up'], submission_data['thumbs_down']),
        'total_feedback_count': submission_data['thumbs_up'] + submission_data['thumbs_down'],
    }
    if fetch_feedback:
        feedback_metrics = get_submission_metrics(
            submission_id, 
            developer_key, 
            reload=is_updated, 
            evaluation_date_range=evaluation_date_range
        )
    return {'submission_id': submission_id, **submission_data, **feedback_metrics}


def _get_filled_leaderboard(df):
    # maintain backwards compatibility with model_name field
    _fill_default_value(df, 'model_name', df['submission_id'])
    _fill_default_value(df, 'is_custom_reward', False)
    for col in _get_filled_columns():
        _fill_default_value(df, col, None)
    return df


def _get_filled_columns():
    columns = []
    for competition_type in constants.COMPETITION_TYPE_CONFIGURATION.keys():
        new_columns = constants.COMPETITION_TYPE_CONFIGURATION[competition_type]['output_columns']
        columns.extend(new_columns)
    return columns


def _fill_default_value(df, field, default_value):
    if field not in df:
        df[field] = None
    if default_value is not None:
        df[field] = df[field].fillna(default_value)


def _filter_submissions_by_submission_ids(submissions, submission_ids):
    filtered_submissions = {
        submission_id: data
        for submission_id, data in submissions.items()
        if submission_id in submission_ids
    }
    return filtered_submissions


def _filter_submissions_by_feedback_count(submissions, min_feedback_count):
    submissions = {
        submission_id: submission_data for submission_id, submission_data in submissions.items()
        if submission_data['thumbs_up'] + submission_data['thumbs_down'] >= min_feedback_count
    }
    return submissions


def get_submission_metrics(submission_id, developer_key, reload=True, evaluation_date_range=None):
    feedback_data = feedback.get_feedback(submission_id, developer_key, reload=reload)
    feedback_metrics = FeedbackMetrics(feedback_data.raw_data)
    feedback_metrics.filter_for_date_range(evaluation_date_range)
    feedback_metrics.filter_duplicated_uid()
    metrics = feedback_metrics.calc_metrics()
    return metrics
