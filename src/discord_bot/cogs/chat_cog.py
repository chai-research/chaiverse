import logging

import discord
from discord.enums import ChannelType, MessageType
from discord.ext import commands
from discord_bot import config

from chaiverse import chat as chai_chat, metrics as chai_metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


OPTION_BATCH_SIZE = 20


class ChatCog(commands.Cog):

    @commands.command(description='Starts conversation with the bot, served by deployed submission.')
    async def chat(self, ctx):
        await _start_chat(ctx)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        channel = message.channel
        if _is_chatting_thread(channel) and _is_chatting_message(message):
            await _reply_chat_message(message)


def _is_chatting_thread(thread):
    if thread.type not in [ChannelType.public_thread, ChannelType.private_thread]:
        return False
    if thread.archived or thread.locked:
        return False
    if not thread.name.startswith('Chat with '):
        return False
    if len(thread.name.split(' ')) < 5:
        return False
    return True


def _is_chatting_message(message):
    if message.type != MessageType.default:
        return False
    if message.is_system():
        return False
    if message.author.bot:
        return False
    return True


async def _start_chat(ctx):
    thread = await ctx.channel.create_thread(name="Initializing", type=discord.ChannelType.public_thread)
    submission_id = await _choose_submission(ctx, thread)
    bot_name = await _choose_bot(ctx, thread)
    await _prepare_chat(thread, submission_id, bot_name)


async def _choose_submission(ctx, thread):
    async with thread.typing():
        available_models = chai_metrics.get_sorted_available_models(developer_key=config.DEVELOPER_KEY)
    await _prompt_options(thread, available_models, "Please select a model by typing its number:")
    submission_id = await _choose_option(ctx, thread, available_models)
    return submission_id


async def _choose_bot(ctx, thread):
    available_bots = chai_chat.get_bot_names()
    await _prompt_options(thread, available_bots, "Please select a bot by typing its number:")
    bot_name = await _choose_option(ctx, thread, available_bots)
    return bot_name


async def _prepare_chat(thread, submission_id, bot_name):
    await thread.purge()
    new_thread_name = f"Chat with {bot_name} by {submission_id}"
    await thread.edit(name=new_thread_name)
    bot_config = chai_chat.get_bot_config(bot_name)
    await thread.send(f"{bot_config.bot_label}: {bot_config.first_message}")


async def _prompt_options(thread, options, prompt_message):
    await thread.send(prompt_message)
    for batch_start in range(0, len(options), OPTION_BATCH_SIZE):
        batch_end = min(batch_start + OPTION_BATCH_SIZE, len(options))
        options_batch = options[batch_start:batch_end]
        options_text = _get_options_text(options_batch, batch_start)
        await thread.send(options_text)


async def _choose_option(ctx, thread, options):
    def is_message_from_thread(message):
        return message.channel == thread

    while True:
        choice_msg = await ctx.bot.wait_for('message', check=is_message_from_thread)
        if choice_msg.content.isdigit() and 0 < int(choice_msg.content) <= len(options):
            return options[int(choice_msg.content) - 1]
        await thread.send("Please write a number from the provided list.")


def _get_options_text(options_subset, start_idx=0):
    options = [f"{index + 1 + start_idx}. {option}" for index, option in enumerate(options_subset)]
    return "\n".join(options)


async def _reply_chat_message(message):
    async with message.channel.typing():
        thread = message.channel
        bot_name, submission_id = _parse_thread_name(thread.name)
        bot_config = chai_chat.get_bot_config(bot_name)
        messages = [
            _build_bot_message(message, bot_config.bot_label) 
            async for message in thread.history(limit=100)
            if not message.is_system()
        ]
        messages = messages[::-1]
        response = chai_chat.get_bot_response(messages, submission_id, bot_config, developer_key=config.DEVELOPER_KEY)
        await message.reply(f"{bot_config.bot_label}: {response}")


def _parse_thread_name(thread_name):
    parts = thread_name.split(' ')
    bot_name = parts[2]
    submission_id = parts[4]
    return bot_name, submission_id


def _build_bot_message(message, bot_label):
    content = message.content
    sender = "user"
    if message.author.id == config.APPLICATION_ID:
        _, content = content.split(": ", 1)
        sender = bot_label
    return content, sender

