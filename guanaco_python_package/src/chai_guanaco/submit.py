import requests
import os

BASE_URL = "http://guanaco.chai-research.com"
SUBMISSION_ENDPOINT = "/models/submit"

SUBMISSION_URL = os.path.join(BASE_URL, SUBMISSION_ENDPOINT)

def submit_model(model_submission):
    submission_url = get_submission_url()
    response = requests.post(submission_url, json=model_submission)
    assert response.status_code == 200, response.json()["error"]
    return response.json()

def get_submission_url():
    return SUBMISSION_URL
