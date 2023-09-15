from dataclasses import dataclass

from pydantic import BaseModel, validator

class PromptFormatter(BaseModel):
    memory_template: str
    prompt_template: str
    bot_template: str
    user_template: str
    response_template: str

    @validator("memory_template")
    def validate_memory(cls, memory_template):
        if "{memory}" not in memory_template:
            raise ValueError("Formatter's memory_template must contain '{memory}'!")
        return memory_template

    @validator("prompt_template")
    def validate_formatter(cls, prompt_template):
        if "{prompt}" not in prompt_template:
            raise ValueError("Formatter's prompt_template must contain '{prompt}'!")
        return prompt_template

    @validator("bot_template")
    def validate_bot(cls, bot_template):
        if "{message}" not in bot_template:
            raise ValueError("Formatter's bot_template must contain '{message}'!")
        return bot_template

    @validator("user_template")
    def validate_user(cls, user_template):
        if "{message}" not in user_template:
            raise ValueError("Formatter's user_template must contain '{message}'!")
        return user_template


class PygmalionFormatter(PromptFormatter):
    memory_template = "{bot_name}'s Persona: {memory}\n####\n"
    prompt_template = "{prompt}\n<START>\n"
    bot_template = "{bot_name}: {message}\n"
    user_template = "{user_name}: {message}\n"
    response_template = "{bot_name}:"


class VicunaFormatter(PromptFormatter):
    memory_template = "### Instruction:\n{memory}\n"
    prompt_template = "### Input:\n{prompt}\n"
    bot_template = "{bot_name}: {message}\n"
    user_template = "{user_name}: {message}\n"
    response_template = "### Response:\n{bot_name}:"
