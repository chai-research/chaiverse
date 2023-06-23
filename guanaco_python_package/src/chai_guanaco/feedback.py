import os
import requests

import pandas as pd

BASE_URL = "https://guanaco-feedback.chai-research.com"
FEEDBACK_ENDPOINT = "/feedback/{submission_id}"

FEEDBACK_URL = BASE_URL + FEEDBACK_ENDPOINT


def get_feedback(submission_id: str, developer_key: str):
    headers = {
        "developer_key": developer_key,
    }
    url = FEEDBACK_URL.format(submission_id=submission_id)
    resp = requests.get(url, headers=headers)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def display_leaderboard(developer_key: str):
    leaderboard = get_leaderboard(developer_key)
    df = pd.DataFrame(leaderboard).T
    df.reset_index(inplace=True, drop=False)
    df.index += 1
    print(df)


def get_leaderboard(developer_key: str):
    headers = {
        "developer_key": developer_key,
    }
    url = BASE_URL + "/feedback"
    resp = requests.get(url, headers=headers)
    assert resp.status_code == 200, resp.json()
    leaderboard = resp.json()
    leaderboard = _add_ratios_to_leaderboard(leaderboard)
    leaderboard = _sort_leaderboard_by_ratio(leaderboard)
    return leaderboard


def _add_ratios_to_leaderboard(leaderboard):
    for model, metrics in leaderboard.items():
        assert "thumbs_up" in metrics.keys()
        assert "thumbs_down" in metrics.keys()
        total_feedbacks = metrics["thumbs_up"] + metrics["thumbs_down"]
        ratio = metrics["thumbs_up"] / total_feedbacks
        metrics["ratio"] = round(ratio, 2)
    return leaderboard


def _sort_leaderboard_by_ratio(leaderboard):
    sort_key = lambda x: x[1]["ratio"]
    sorted_leaderboard = sorted(leaderboard.items(), key=sort_key, reverse=True)
    sorted_leaderboard = dict(sorted_leaderboard)
    return sorted_leaderboard
