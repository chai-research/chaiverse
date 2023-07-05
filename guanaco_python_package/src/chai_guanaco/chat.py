from dataclasses import dataclass
import os
import json
import requests


BASE_URL = "https://guanaco-submitter.chai-research.com"

REPO_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BOT_CONFIG = os.path.join(REPO_PATH, '..', '..', 'resources', 'bot_config', 'vampire_queen.json')


@dataclass
class BotConfig:
    memory: str
    prompt: str
    first_message: str
    bot_label: str

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class Bot:

    def __init__(
            self,
            submission_id,
            developer_key,
            bot_config=None):
        self.submission_id = submission_id
        self.developer_key = developer_key
        self.bot_config = bot_config or BotConfig.from_json(DEFAULT_BOT_CONFIG)
        self._chat_history = self._init_chat_history()

    def response(self, user_input):
        self._update_chat_history(user_input, 'user')
        response = self._get_response()
        self._update_chat_history(response['model_output'], 'bot')
        return response

    def _get_response(self):
        payload = {
            "memory": self.bot_config.memory,
            "prompt": self.bot_config.prompt,
            "chat_history": self._chat_history[:],
            "bot_name": self.bot_config.bot_label,
            "user_name": "You"
        }
        headers = {"Authorization": f"Bearer {self.developer_key}"}
        response = requests.post(url=self._url, json=payload, headers=headers)
        assert response.status_code == 200, response.text
        return response.json()

    @property
    def _url(self):
        endpoint = f'/submissions/{self.submission_id}/chat'
        return BASE_URL + endpoint

    def _update_chat_history(self, message, sender):
        message = f'{sender}: {message}'
        self._chat_history.append(message)

    def _init_chat_history(self):
        return [f'bot: {self.bot_config.first_message}']
