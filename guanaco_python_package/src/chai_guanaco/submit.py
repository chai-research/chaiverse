import itertools
import requests
import sys
import time

import pandas as pd

from chai_guanaco.utils import print_color


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


class ModelSubmitter:
    """
    Submits a model to the Guanaco service and exposes it to beta-testers on the Chai app.

    Attributes
    --------------
    developer_key : str

    Methods
    --------------
    submit(submission_params)
    Submits the model to the Guanaco service.

    Example usage:
    --------------
    submitter = ModelSubmitter(developer_key)
    submitter.submit(submission_params)
    """
    def __init__(self, developer_key):
        self.developer_key = developer_key
        self._animation = self._spinner_animation_generator()
        self._progress = 0
        self._sleep_time = 0.5
        self._get_request_interval = int(10 / self._sleep_time)

    def submit(self, submission_params):
        """
        Submits the model to the Guanaco service and wait for the deployment to finish.

        submission_params: dict
            model_repo: str - HuggingFace repo
            generation_params: dict
                temperature: float
                top_p: float
                top_k: int
                repetition_penalty: float
        """
        submission_id = self._get_submission_id(submission_params)
        self._print_submission_header(submission_id)
        status = self._wait_for_model_submission(submission_id)
        self._print_submission_result(status)
        self._progress = 0
        return submission_id

    def _get_submission_id(self, submission_params):
        response = submit_model(submission_params, self.developer_key)
        return response.get('submission_id')

    def _wait_for_model_submission(self, submission_id):
        status = 'pending'
        while status not in {'deployed', 'failed', 'inactive'}:
            status = self._get_submission_status(submission_id)
            self._display_animation(status)
            time.sleep(self._sleep_time)
        return status

    def _get_submission_status(self, submission_id):
        self._progress += 1
        status = 'pending'
        if self._progress % self._get_request_interval == 0:
            model_info = get_model_info(submission_id, self.developer_key)
            status = model_info.get('status')
        return status

    def _spinner_animation_generator(self):
        animations = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
        return itertools.cycle(animations)

    def _display_animation(self, status):
        sys.stdout.write(f" \r{next(self._animation)} {status}...")
        sys.stdout.flush()

    def _print_submission_header(self, submission_id):
        print_color(f'\nModel Submission ID: {submission_id}', 'green')
        print("Your model is being deployed to Chai Guanaco, please wait for approximately 10 minutes...")

    def _print_submission_result(self, status):
        success = status == 'deployed'
        text_success = 'Model successfully deployed!'
        text_failed = 'Model deployment failed, please seek help on our Discord channel'
        text = text_success if success else text_failed
        color = 'green' if success else 'red'
        print_color(f'\n{text}', color)


def submit_model(model_submission: dict, developer_key: str):
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
