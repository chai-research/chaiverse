__all__ = ["display_leaderboard", "display_competition_leaderboard"]


import warnings

import pandas as pd
from tabulate import tabulate

from chaiverse.competition import get_competitions
from chaiverse import constants
from chaiverse.metrics.leaderboard_formatter import format_leaderboard
from chaiverse.metrics.leaderboard_api import get_leaderboard
from chaiverse.utils import print_color, cache

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 500)
pd.set_option("display.colheader_justify","center")

warnings.filterwarnings('ignore', 'Mean of empty slice')


def display_leaderboard(
    developer_key=None,
    regenerate=False,
    detailed=False,
    max_workers=constants.DEFAULT_MAX_WORKERS,
):
    default_competition = {
        'id': 'Default',
        'type': 'default',
        'submission_start_date': None,
        'submission_end_date': None,
        'leaderboard_should_use_feedback': False
    }

    df = display_competition_leaderboard(
        competition=default_competition,
        detailed=detailed,
        regenerate=regenerate,
        developer_key=developer_key,
        max_workers=max_workers,
    )
    return df


def display_competition_leaderboard(
    competition=None,
    detailed=False,
    regenerate=False, 
    developer_key=None,
    max_workers=constants.DEFAULT_MAX_WORKERS
):
    competition = competition if competition else get_competitions()[-1]
    competition_type = competition.get('type') or 'submission_closed_feedback_round_robin'
    fetch_feedback = competition.get('leaderboard_should_use_feedback', False)

    submission_date_range = competition.get('submission_date_range')
    evaluation_date_range = competition.get('evaluation_date_range')
    submission_ids = competition.get('submissions')
    competition_id = competition.get('id')
    display_title = f'{competition_id} Leaderboard'

    df = cache(get_leaderboard, regenerate)(
        developer_key=developer_key,
        max_workers=max_workers,
        submission_date_range=submission_date_range,
        evaluation_date_range=evaluation_date_range,
        submission_ids=submission_ids,
        fetch_feedback=fetch_feedback
        )

    if len(df) > 0:
        display_df = df.copy()
        display_df = format_leaderboard(
            display_df, 
            detailed=detailed, 
            competition_type=competition_type
        )
        _pprint_leaderboard(display_df, display_title)
    else:
        print('No eligible submissions found!')
    return df


def _pprint_leaderboard(df, title):
    print_color(f'\n💎 {title}:', 'red')
    print(tabulate(df.round(3).head(30), headers=df.columns, numalign='decimal'))

