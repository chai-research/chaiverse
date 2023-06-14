import requests
import os


BASE_URL = "https://guanaco-submitter.chai-research.com"
SUBMISSION_ENDPOINT = "/models/submit"
INFO_ENDPOINT = "/models/{submission_id}"

SUBMISSION_URL = BASE_URL + SUBMISSION_ENDPOINT
INFO_URL = BASE_URL + INFO_ENDPOINT


def submit_model(model_submission: dict, developer_key: str):
    """
        Submits a model to the Guanaco service and exposes it to beta-testers on the Chai app.
        developer_key: str
        model_submission: dict
            model_repo: str - HuggingFace repo
            developer_uid: str
            generation_params: dict
                temperature: float
                top_p: float
                top_k: int
                repetition_penalty: float
            formatter: str - "PygmalionFormatter"
    """
    submission_url = get_submission_url()
    headers = {'developer_key': developer_key}
    response = requests.post(url=submission_url, json=model_submission, headers=headers)
    assert response.status_code == 200, response.json()
    return response.json()


def get_model_info(submission_id, developer_key):
    url = get_info_url().format(submission_id=submission_id)
    headers = {'developer_key': developer_key}
    response = requests.get(url=url, headers=headers)
    assert response.status_code == 200, response.json()
    return response.json()


def get_submission_url():
    return SUBMISSION_URL


def get_info_url():
    return INFO_URL
