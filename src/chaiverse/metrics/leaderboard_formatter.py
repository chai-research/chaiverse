__all__ = ["format_leaderboard"]


from datetime import datetime

import numpy as np

from chaiverse import constants


def format_leaderboard(df, detailed, competition_type):
    competition_configuration = constants.COMPETITION_TYPE_CONFIGURATION[competition_type]
    sort_params = competition_configuration['sort_params']
    output_columns = competition_configuration['output_columns']

    df = df if detailed else _get_ranked_leaderboard(df, sort_params)
    df = df if detailed else _get_deduped_leaderboard(df)
    df = _get_formatted_leaderboard(df)
    df = df if detailed else df[output_columns]
    df.index = np.arange(1, len(df)+1)
    return df


def _get_ranked_leaderboard(df, sort_params):
    df = _add_individual_rank(df, value_column='thumbs_up_ratio', rank_column='thumbs_up_rank', ascending=False)
    rank_columns = []
    for score_column in constants.MODEL_EVAL_SCORE_COLS:
        rank_column = f'{score_column}_rank'
        rank_columns.append(rank_column)
        df = _add_individual_rank(df, value_column=score_column, rank_column=rank_column, ascending=False)
    df = _add_overall_rank(df, rank_columns=rank_columns)
    df = _sort(df, sort_params)
    return df


def _get_deduped_leaderboard(df):
    df = _get_submissions_with_unique_model(df)
    df = _get_submissions_with_unique_dev_id(df)
    return df


def _get_formatted_leaderboard(df):
    df['timestamp'] = df.timestamp.apply(_get_isoformatted_timestamp)
    df['size'] = df.model_num_parameters.apply(_get_model_size)
    df['date'] = df['timestamp'].dt.date
    df.drop(['timestamp'], axis=1, inplace=True)
    df['is_custom_reward'] = df['is_custom_reward'].replace({
        True: '✅',
        False: '❌'
    })
    df = df.reset_index(drop=True)
    return df


def _get_isoformatted_timestamp(timestamp):
    formatted = datetime.fromisoformat(timestamp)
    return formatted


def _get_model_size(num_parameters):
    size = 'n/a'
    if num_parameters is not None and not np.isnan(num_parameters):
        size = f'{int(round(num_parameters/1e9,0))}'
    return size


def _get_submissions_with_unique_model(df):
    df = df.drop_duplicates(subset=['model_repo', 'reward_repo'], keep='first')
    return df


def _get_submissions_with_unique_dev_id(df):
    out = df.drop_duplicates('developer_uid', keep='first')
    return out


def _add_individual_rank(df, value_column, rank_column, ascending=True):
    df[rank_column] = df[value_column].rank(ascending=ascending, na_option='bottom')
    return df


def _add_overall_rank(df, rank_columns):
    ranks = [df[column] for column in rank_columns]
    overall_score = np.mean(ranks, axis=0)
    df.loc[:, 'overall_score'] = overall_score
    df.loc[:, 'overall_rank'] = df.overall_score.rank(na_option='bottom')
    return df


def _sort(df, sort_params):
    df = df.sort_values(**sort_params, na_position='last').reset_index(drop=True)
    return df
