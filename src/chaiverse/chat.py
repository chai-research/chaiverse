from dataclasses import dataclass
import json
import os
from pathlib import Path

from chaiverse.login_cli import auto_authenticate
from chaiverse.utils import print_color
from chaiverse.http_client import SubmitterClient
from chaiverse.config import BASE_SUBMITTER_URL, CHAT_ENDPOINT


REPO_PATH = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(REPO_PATH, 'resources', 'bot_config')


class SubmissionChatbot():
    """
    Chat with a model that has been submitted and deployed to Chaiverse

    Attributes
    --------------
    submission_id : str
    developer_key : str

    Methods
    --------------
    chat(bot_config_file_name, show_model_input=False)
    Launch a chat session with the bot profile specified in bot_config_file_name

    show_available_bots()
    Displays the list of bots that are available for chat that can be passed into chat()

    Example usage:
    --------------
    chatbot = ChatWithSubmission(submission_id)
    chatbot.chat('vampire_queen')
    """
    @auto_authenticate
    def __init__(self, submission_id, developer_key=None):
        self.submission_id = submission_id
        self.developer_key = developer_key

    def chat(self, bot_config_file_name, show_model_input=False):
        """
        Launch a chat session with the bot profile specified in bot_config_file_name

        bot_config_file_name: str - name of the bot profile to chat with
        show_model_input: bool - whether to display the raw model input to your language model
        """
        bot_config = self._get_bot_config(bot_config_file_name)
        bot = Bot(self.submission_id, self.developer_key, bot_config)
        self._print_greetings_header(bot_config)
        user_input = input("You: ")
        while user_input != 'exit':
            response = bot.get_response(user_input)
            self._print_bot_response(bot_config, response, show_model_input)
            user_input = input("You: ")

    @staticmethod
    def show_available_bots():
        avaliable_bots = get_available_bots()
        print_color('Available Bots:', 'yellow')
        print(avaliable_bots)

    @staticmethod
    def _get_bot_config(config_file_name):
        bot_config_path = os.path.join(RESOURCE_DIR, f'{config_file_name}.json')
        return BotConfig.from_json(bot_config_path)

    def _print_greetings_header(self, bot_config):
        print_color(f'\nChatting with {bot_config.bot_label}, reply "exit" to quit.\n', 'yellow')
        print_color(f'{bot_config.bot_label}: {bot_config.first_message}', 'green')

    def _print_bot_response(self, bot_config, response, show_model_input):
        bot_response = response['model_output']
        model_input = response['model_input']
        if show_model_input:
            print_color('<Begin model input>', 'yellow')
            print_color(model_input, 'cyan')
            print_color('<End model input>', 'yellow')
        print_color(f"{bot_config.bot_label}: {bot_response}", 'green')


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
            bot_config,
            endpoint = CHAT_ENDPOINT
    ):
        self.submission_id = submission_id
        self.developer_key = developer_key
        self.bot_config = bot_config
        self.endpoint = endpoint
        self._chat_history = self._init_chat_history()

    def get_response(self, user_input):
        response = self._get_raw_response(user_input)
        model_output = response['model_output']
        self._update_chat_history(model_output, self.bot_config.bot_label)
        return response

    def _get_raw_response(self, user_input):
        self._update_chat_history(user_input, 'user')
        payload = {
            "memory": self.bot_config.memory,
            "prompt": self.bot_config.prompt,
            "chat_history": self._chat_history[:],
            "bot_name": self.bot_config.bot_label,
            "user_name": "You"
        }
        http_client = SubmitterClient(self.developer_key)
        response = http_client.post(endpoint=self.endpoint, submission_id=self.submission_id, timeout=20, data=payload)
        return response

    def _update_chat_history(self, message, sender):
        message = {"sender": sender, "message": message}
        self._chat_history.append(message)

    def _init_chat_history(self):
        return [{"sender": self.bot_config.bot_label, "message": self.bot_config.first_message}]


def get_available_bots():
    bots = os.listdir(RESOURCE_DIR)
    avaliable_bots = '\n'.join([bot.replace('.json', '') for bot in bots if bot.endswith('.json')])
    return avaliable_bots


def get_bot_names():
    files = os.listdir(RESOURCE_DIR)
    bot_names = [file.replace('.json', '') for file in files if file.endswith('.json')]
    return bot_names


def get_bot_config(bot_name):
    config_file_name = f'{bot_name}.json'
    bot_config_path = Path(RESOURCE_DIR) / config_file_name
    return BotConfig.from_json(bot_config_path.absolute())


def get_bot_response(messages, submission_id, bot_config, developer_key):
    chai_bot = Bot(submission_id, developer_key, bot_config)
    for content, sender in messages[:-1]:
        chai_bot._update_chat_history(content, sender)
    content, sender = messages[-1]
    response = chai_bot.get_response(content)["model_output"]
    return response
