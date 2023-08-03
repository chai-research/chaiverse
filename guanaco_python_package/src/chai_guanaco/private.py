import requests

from chai_guanaco.login_cli import auto_authenticate


BASE_URL = "https://guanaco-submitter.chai-research.com"
SUBMISSION_ENDPOINT = "/models/submit_blend"


def get_url(endpoint):
    base_url = BASE_URL
    return base_url + endpoint


@auto_authenticate
def submit_blend(submission_ids, developer_key):
    submission_url = get_url(SUBMISSION_ENDPOINT)
    headers = {'Authorization': f"Bearer {developer_key}"}
    response = requests.post(url=submission_url, json=submission_ids, headers=headers)
    assert response.status_code == 200, response.json()
    return response.json()
