import unittest
from chaiverse.data_processors.chatml_processor import ConvoFormatter, ChatMLConvoProcessor
from chaiverse.data_processors.chatml_processor import (
    DEFAULT_USER_LABELS,
    SYSTEM_LABEL,
    VICUNA_SYSTEM_PROMPT,
)
from transformers import AutoTokenizer



class TestChatMLProcessor(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")

    def test_get_label(self):
        formatter = ConvoFormatter(self.tokenizer, 1024)
        convo = [
            {"role": "System", "content": "Good bot!"},
            {"role": "Bot", "content": "Hello User!"},
            {"role": "User", "content": "Hello Bot!"},
        ]
        user_label = formatter._user_label(convo)
        self.assertEqual(user_label, "User")
        bot_label = formatter._bot_label(convo)
        self.assertEqual(bot_label, "Bot")

    def test_format_pygmalion(self):
        processor = ChatMLConvoProcessor(self.tokenizer, "pygmalion")
        convo = [
            {"role": "System", "content": "Good bot!"},
            {"role": "Bot", "content": "Hello User!"},
            {"role": "User", "content": "Hello Bot!"},
        ]
        formatted = processor._format_convo(convo)
        expected_result = [
            {"prompt": "Bot's Persona: Good bot!\n###\n<START>\n", "mask": 0},
            {"prompt": "Bot: Hello User!<|endoftext|>\n", "mask": 1},
            {"prompt": "User: Hello Bot!\n", "mask": 0},
        ]
        self.assertEqual(formatted, expected_result)

    def test_format_vicuna(self):
        processor = ChatMLConvoProcessor(self.tokenizer, "vicuna")
        convo = [
            {"role": "User", "content": "Hello Bot!"},
            {"role": "Bot", "content": "I'm a bot."},
        ]
        formatted = processor._format_convo(convo)
        expected_result = [
            {"prompt": "USER: Hello Bot!\n", "mask": 0},
            {"prompt": "ASSISTANT: I'm a bot.<|endoftext|>\n", "mask": 1},
        ]
        self.assertEqual(formatted, expected_result)


    def test_get_tokenized_convo(self):
        processor = ChatMLConvoProcessor(self.tokenizer, "vicuna")
        convo = [
            {"role": "User", "content": "Hello Bot!"},
            {"role": "Bot", "content": "I'm a bot."},
        ]
        convo = {"conversation": convo}
        tokenized = processor.get_tokenized_convo(convo)
        expected_result = {
            "input_ids": [29904, 25, 18435, 18579, 0, 198, 10705, 8808, 8643, 25, 314, 1101, 257, 10214, 13, 50256, 198],
            "label_mask": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        self.assertEqual(tokenized, expected_result)


if __name__ == "__main__":
    unittest.main()

