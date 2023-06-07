import requests

SUBMISSION_URL = "http://submit.chai-research.com"

def submit_model(model_submission):
    response = requests.post(url=SUBMISSION_URL, json=model_submission)
    assert response.status_code == 200, response.json()["error"]
    return response.json()
