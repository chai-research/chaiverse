import requests

from chaiverse.login_cli import auto_authenticate
from chaiverse.config import BASE_SUBMITTER_URL, BASE_FEEDBACK_URL
from chaiverse.utils import get_url


class _ChaiverseHTTPClient():
    def __init__(self, developer_key=None, hostname=None):
        self.developer_key = developer_key
        self.hostname = hostname

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.developer_key}"}

    def get(self, endpoint, submission_id=None, **kwargs):
        url = get_url(endpoint, hostname=self.hostname, submission_id=submission_id)
        response = self._request(requests.get, url=url, **kwargs)
        return response

    def post(self, endpoint, data, submission_id=None, **kwargs):
        url = get_url(endpoint, hostname=self.hostname, submission_id=submission_id)
        response = self._request(requests.post, url=url, json=data, **kwargs)
        return response

    def _request(self, func, url, **kwargs):
        response = func(url=url, headers=self.headers, **kwargs)
        assert response.status_code == 200, response.json()
        return response.json()


@auto_authenticate
class SubmitterClient(_ChaiverseHTTPClient):
    def __init__(self,
            developer_key=None,
            hostname=BASE_SUBMITTER_URL):
        super().__init__(developer_key, hostname)


@auto_authenticate
class FeedbackClient(_ChaiverseHTTPClient):
    def __init__(self,
            developer_key=None,
            hostname=BASE_FEEDBACK_URL):
        super().__init__(developer_key, hostname)

