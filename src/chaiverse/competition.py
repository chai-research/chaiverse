import requests

import pandas as pd

from chaiverse.login_cli import auto_authenticate
from chaiverse.submit import get_model_info, redeploy_model
from chaiverse.utils import get_all_historical_submissions, get_url
from chaiverse import config


def get_competitions():
    url = get_url(config.COMPETITIONS_ENDPOINT)
    response = requests.get(url)
    assert response.ok, response.json()
    return response.json()

