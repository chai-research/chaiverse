import requests
import os


BASE_URL = "https://guanaco-submitter.chai-research.com"
SUBMISSION_ENDPOINT = "/models/submit"
ALL_SUBMISSION_STATUS_ENDPOINT = "/models/"
INFO_ENDPOINT = "/models/{submission_id}"

SUBMISSION_URL = BASE_URL + SUBMISSION_ENDPOINT
MY_SUBMISSIONS_STATUS_URL = BASE_URL + ALL_SUBMISSION_STATUS_ENDPOINT
INFO_URL = BASE_URL + INFO_ENDPOINT


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


def get_my_submissions_url():
    return MY_SUBMISSIONS_STATUS_URL


def get_submission_url():
    return SUBMISSION_URL


def get_info_url():
    return INFO_URL
