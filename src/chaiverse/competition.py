import requests

from chaiverse.utils import get_url


COMPETITIONS_ENDPOINT = '/competitions'


def get_competitions():
    url = get_url(COMPETITIONS_ENDPOINT)
    response = requests.get(url)
    assert response.ok, response.json()
    return response.json()

