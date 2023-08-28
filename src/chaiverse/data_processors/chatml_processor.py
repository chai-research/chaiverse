from typing import List

DEFAULT_USER_LABELS = ["User", "Me", "You"]
SYSTEM_LABEL = "System"
VICUNA_SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

class ConvoFormatter:
    def __init__(self, tokenizer, max_length):
        self.user_candidates = DEFAULT_USER_LABELS
        self.max_length = max_length
        self.eos_token = tokenizer.eos_token

    def _bot_label(self, convo):
        return self._get_label(convo, False)

    def _user_label(self, convo):
        return self._get_label(convo, True)

    def format_conversation(self, conversation):
        bot_label = self._bot_label(conversation)
        user_label = self._user_label(conversation)
        formatted_turns = []
        if conversation[0]["role"] == SYSTEM_LABEL:
            prompt = self._get_system_prompt(bot_label, conversation[0]["content"])
            start_idx = 1
            formatted_turns.append({"prompt": prompt, "mask": 0})
        else:
            start_idx = 0
        for turn in conversation[start_idx:]:
            prompt = turn["content"]
            if turn["role"] == user_label:
                prompt = self._get_user_prompt(user_label, prompt)
            else:
                prompt = self._get_bot_prompt(bot_label, prompt)
            formatted_turns.append(
                {"prompt": prompt, "mask": int(bot_label == turn["role"])}
            )
        return formatted_turns

    def _get_label(self, convo, is_user):
        role = None
        for turn in convo:
            if is_user:
                check = turn["role"] in self.user_candidates
            else:
                check = turn["role"] not in self.user_candidates + [SYSTEM_LABEL]
            if check:
                role = turn["role"]
                break
        if not role:
            role = convo[-1]["role"] if is_user else convo[-2]["role"]
        return role

    def _get_system_prompt(self, bot_label, content):
        raise NotImplementedError

    def _get_user_prompt(self, content):
        raise NotImplementedError

    def _get_bot_prompt(self, content):
        raise NotImplementedError


class PygmalionFormatter(ConvoFormatter):
    def _get_system_prompt(self, bot_label, content):
        return f"{bot_label}'s Persona: {content}\n###\n<START>\n"

    def _get_user_prompt(self, user_label, content):
        return f"{user_label}: {content}\n"

    def _get_bot_prompt(self, bot_label, content):
        return f"{bot_label}: {content}{self.eos_token}\n"


class VicunaFormatter(ConvoFormatter):
    def _get_system_prompt(self, bot_label, content):
        return VICUNA_SYSTEM_PROMPT

    def _get_user_prompt(self, user_label, content):
        return f"USER: {content}\n"

    def _get_bot_prompt(self, bot_label, content):
        return f"ASSISTANT: {content}{self.eos_token}\n"


class ChatMLConvoProcessor:
    def __init__(self, tokenizer, formatter, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        accepted_formatters = ["vicuna", "pygmalion"]
        assert (
            formatter in accepted_formatters
        ), f"Formatter must be in {accepted_formatters}"
        if formatter == "pygmalion":
            self.formatter = PygmalionFormatter(self.tokenizer, self.max_length)
        elif formatter == "vicuna":
            self.formatter = VicunaFormatter(self.tokenizer, self.max_length)

    def get_tokenized_convo(self, convo):
        convo = convo["conversation"]
        if len(convo) < 2:
            formatted_convo = []
        else:
            formatted_convo = self._format_convo(convo)
        if not formatted_convo:
            return {"input_ids": [], "label_mask": [], "attention_mask": []}
        return self._tokenize_formatted_convo(formatted_convo)

    def _format_convo(self, convo):
        formatted_data = self.formatter.format_conversation(convo)
        return formatted_data

    def _tokenize_formatted_convo(self, messages):
        prompt_texts = [turn["prompt"] for turn in messages]
        encoded_texts = self.tokenizer.batch_encode_plus(prompt_texts)["input_ids"]
        input_ids, label_mask, attention_mask = self._build_tensors(
            messages, encoded_texts
        )
        return {
            "input_ids": input_ids,
            "label_mask": label_mask,
            "attention_mask": attention_mask,
        }

    def _truncate_tensors(self, input_ids, label_mask, attention_mask):
        input_ids = input_ids[-self.max_length :]
        label_mask = label_mask[-self.max_length :]
        attention_mask = attention_mask[-self.max_length :]
        return input_ids, label_mask, attention_mask

    def _build_tensors(self, messages, encoded_texts):
        input_ids = []
        label_mask = []
        attention_mask = []
        for turn, tokens in zip(messages, encoded_texts):
            input_ids.extend(tokens)
            mask = [int(turn["mask"])] * len(tokens)
            label_mask.extend(mask)
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(label_mask) == len(attention_mask)
        if len(input_ids) > self.max_length:
            input_ids, label_mask, attention_mask = self._truncate_tensors(
                input_ids, label_mask, attention_mask
            )
        return input_ids, label_mask, attention_mask
