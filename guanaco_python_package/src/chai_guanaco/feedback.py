import os
import requests

#BASE_URL = "https://guanaco-feedback.chai-research.com"
BASE_URL = "http://localhost:8080"
FEEDBACK_ENDPOINT = "/feedback/{model_name}"

FEEDBACK_URL = BASE_URL + FEEDBACK_ENDPOINT


FEEDBACK_ENDPOINT = "/feedback/{model_name}"

FEEDBACK_URL = BASE_URL + FEEDBACK_ENDPOINT


def get_feedback(feedback_request):
    headers = {
        "developer_uid": feedback_request["developer_uid"],
        "developer_key": feedback_request["developer_key"],
    }
    url = FEEDBACK_URL.format(model_name=feedback_request["model_name"])
    resp = requests.get(url, headers=headers)
    assert resp.status_code == 200, resp.json()
    return resp
