import itertools
import requests
import time

from chai_guanaco.utils import print_color
from chai_guanaco.login_cli import auto_authenticate


BASE_URL = "https://guanaco-submitter.chai-research.com"
SUBMISSION_ENDPOINT = "/models/submit"
ALL_SUBMISSION_STATUS_ENDPOINT = "/models/"
INFO_ENDPOINT = "/models/{submission_id}"
DEACTIVATE_ENDPOINT = "/models/{submission_id}/deactivate"


def get_url(endpoint):
    base_url = BASE_URL
    return base_url + endpoint


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
    @auto_authenticate
    def __init__(self, developer_key=None):
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
            formatter (optional): PromptFormatter
            model_name (optional): str - custom alias for your model
        """
        submission_params = self._preprocess_submission(submission_params)
        submission_id = self._get_submission_id(submission_params)
        self._print_submission_header(submission_id)
        status = self._wait_for_model_submission(submission_id)
        self._print_submission_result(status)
        self._progress = 0
        return submission_id

    def _get_submission_id(self, submission_params):
        response = submit_model(submission_params, self.developer_key)
        return response.get('submission_id')

    def _preprocess_submission(self, submission_params):
        if submission_params.get("formatter", None):
            submission_params = submission_params.copy()
            submission_params["formatter"] = submission_params["formatter"].dict()
        return submission_params

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
        print(f" {next(self._animation)} {status}...", end='\r')

    def _print_submission_header(self, submission_id):
        print_color(f'\nModel Submission ID: {submission_id}', 'green')
        print("Your model is being deployed to Chai Guanaco, please wait for approximately 10 minutes...")

    def _print_submission_result(self, status):
        success = status == 'deployed'
        text_success = 'Model successfully deployed!'
        text_failed = 'Model deployment failed, please seek help on our Discord channel'
        text = text_success if success else text_failed
        color = 'green' if success else 'red'
        print('\n')
        print_color(f'\n{text}', color)


@auto_authenticate
def submit_model(model_submission: dict, developer_key=None):
    submission_url = get_url(SUBMISSION_ENDPOINT)
    headers = {'Authorization': f"Bearer {developer_key}"}
    response = requests.post(url=submission_url, json=model_submission, headers=headers)
    assert response.status_code == 200, response.json()
    return response.json()


@auto_authenticate
def get_model_info(submission_id, developer_key=None):
    url = get_url(INFO_ENDPOINT)
    url = url.format(submission_id=submission_id)
    headers = {'Authorization': f"Bearer {developer_key}"}
    response = requests.get(url=url, headers=headers)
    assert response.status_code == 200, response.json()
    return response.json()


@auto_authenticate
def get_my_submissions(developer_key=None):
    url = get_url(ALL_SUBMISSION_STATUS_ENDPOINT)
    headers = {'Authorization': f"Bearer {developer_key}"}
    response = requests.get(url=url, headers=headers)
    assert response.status_code == 200, response.json()
    return response.json()


@auto_authenticate
def deactivate_model(submission_id, developer_key=None):
    url = get_url(DEACTIVATE_ENDPOINT)
    url = url.format(submission_id=submission_id)
    headers = {'Authorization': f"Bearer {developer_key}"}
    response = requests.get(url=url, headers=headers)
    assert response.status_code == 200, response.json()
    print(response.json())
    return response.json()
