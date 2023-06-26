import os
import requests


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
