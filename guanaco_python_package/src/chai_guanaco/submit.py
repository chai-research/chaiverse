import requests

import pandas as pd


BASE_URL = "https://guanaco-submitter.chai-research.com"
SUBMISSION_ENDPOINT = "/models/submit"
ALL_SUBMISSION_STATUS_ENDPOINT = "/models/"
INFO_ENDPOINT = "/models/{submission_id}"
DEACTIVATE_ENDPOINT = "/models/{submission_id}/deactivate"
LEADERBOARD_ENDPOINT = "/leaderboard"

SUBMISSION_URL = BASE_URL + SUBMISSION_ENDPOINT
MY_SUBMISSIONS_STATUS_URL = BASE_URL + ALL_SUBMISSION_STATUS_ENDPOINT
INFO_URL = BASE_URL + INFO_ENDPOINT
DEACTIVATE_URL = BASE_URL + DEACTIVATE_ENDPOINT
LEADERBOARD_URL = BASE_URL + LEADERBOARD_ENDPOINT


def submit_model(model_submission: dict, developer_key: str):
    """
        Submits a model to the Guanaco service and exposes it to beta-testers on the Chai app.
        developer_key: str
        model_submission: dict
            model_repo: str - HuggingFace repo
            generation_params: dict
                temperature: float
                top_p: float
                top_k: int
                repetition_penalty: float
    """
    submission_url = get_submission_url()
    headers = {'Authorization': f"Bearer {developer_key}"}
    response = requests.post(url=submission_url, json=model_submission, headers=headers)
    assert response.status_code == 200, response.json()
    return response.json()


def get_model_info(submission_id, developer_key):
    url = get_info_url().format(submission_id=submission_id)
    headers = {'Authorization': f"Bearer {developer_key}"}
    response = requests.get(url=url, headers=headers)
    assert response.status_code == 200, response.json()
    return response.json()


def get_my_submissions(developer_key):
    url = get_my_submissions_url()
    headers = {'Authorization': f"Bearer {developer_key}"}
    response = requests.get(url=url, headers=headers)
    assert response.status_code == 200, response.json()
    return response.json()


def deactivate_model(submission_id, developer_key):
    url = get_deactivate_url().format(submission_id=submission_id)
    headers = {'Authorization': f"Bearer {developer_key}"}
    response = requests.get(url=url, headers=headers)
    assert response.status_code == 200, response.json()
    print(response.json())
    return response.json()


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
    resp = requests.get(LEADERBOARD_URL, headers=headers)
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


def get_my_submissions_url():
    return MY_SUBMISSIONS_STATUS_URL


def get_submission_url():
    return SUBMISSION_URL


def get_info_url():
    return INFO_URL


def get_deactivate_url():
    return DEACTIVATE_URL
