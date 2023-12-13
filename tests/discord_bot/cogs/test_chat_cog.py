from contextlib import asynccontextmanager
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from discord.enums import ChannelType, MessageType
from discord_bot.cogs import ChatCog


@asynccontextmanager
async def mock_typing():
    yield


MOCK_BOT_CONFIG = Mock()
MOCK_BOT_CONFIG.bot_label = '[bot]'
MOCK_BOT_CONFIG.first_message = 'mock-first-message'


@pytest.fixture(autouse=True)
def chai_metrics():
    with patch('discord_bot.cogs.chat_cog.chai_metrics') as mock_chai_metrics:
        mock_chai_metrics.get_sorted_available_models.return_value = ['model1', 'model2', 'model3']
        yield mock_chai_metrics


@pytest.fixture(autouse=True)
def bot():
    bot = AsyncMock()
    yield bot


@pytest.fixture(autouse=True)
def chai_chat():
    with patch('discord_bot.cogs.chat_cog.chai_chat') as mock_chai_chat:
        mock_chai_chat.get_bot_config.return_value = MOCK_BOT_CONFIG
        mock_chai_chat.get_bot_names.return_value = ['bot1', 'bot2', 'bot3']
        mock_chai_chat.get_bot_response.return_value = 'mock-reply'
        yield mock_chai_chat


@pytest.fixture(autouse=True)
def config():
    with patch('discord_bot.cogs.chat_cog.config') as mock_config:
        mock_config.APPLICATION_ID = 'mock-app-id'
        mock_config.DEVELOPER_KEY = 'mock-dev-key'
        mock_config.BOT_TOKEN = 'mock-bot-token'
        yield mock_config


@pytest.mark.asyncio
async def test_slash_chat_can_start_chat_thread_with_model1_bot1(bot):
    ctx = AsyncMock()
    ctx.channel.create_thread.return_value.typing = MagicMock(mock_typing)
    thread = ctx.channel.create_thread.return_value

    choice_msg = Mock()
    choice_msg.content = '1'
    bot.wait_for.return_value = choice_msg

    cog = ChatCog(bot)
    await cog.chat.callback(cog, ctx)

    assert ctx.channel.create_thread.await_count == 1
    assert thread.typing.call_count == 1
    assert 'select a model' in thread.send.call_args_list[0].args[0]
    assert '1. model1\n2. model2\n3. model3' == thread.send.call_args_list[1].args[0]
    assert 'select a bot' in thread.send.call_args_list[2].args[0]
    assert '1. bot1\n2. bot2\n3. bot3' in thread.send.call_args_list[3].args[0]
    assert thread.purge.call_count == 1
    thread.edit.assert_awaited_with(name='Chat with bot1 by model1')
    assert '[bot]: mock-first-message' in thread.send.call_args_list[4].args[0]


@pytest.mark.asyncio
async def test_slash_chat_can_start_chat_thread_with_model2_bot2(bot):
    ctx = AsyncMock()
    ctx.channel.create_thread.return_value.typing = MagicMock(mock_typing)
    thread = ctx.channel.create_thread.return_value

    choice_msg = Mock()
    choice_msg.content = '2'
    bot.wait_for.return_value = choice_msg

    cog = ChatCog(bot)
    await cog.chat.callback(cog, ctx)

    assert ctx.channel.create_thread.await_count == 1
    assert thread.typing.call_count == 1
    assert 'select a model' in thread.send.call_args_list[0].args[0]
    assert '1. model1\n2. model2\n3. model3' == thread.send.call_args_list[1].args[0]
    assert 'select a bot' in thread.send.call_args_list[2].args[0]
    assert '1. bot1\n2. bot2\n3. bot3' in thread.send.call_args_list[3].args[0]
    assert thread.purge.call_count == 1
    thread.edit.assert_awaited_with(name='Chat with bot2 by model2')
    assert '[bot]: mock-first-message' in thread.send.call_args_list[4].args[0]


@pytest.mark.asyncio
@patch('discord_bot.cogs.chat_cog.OPTION_BATCH_SIZE', 2)
async def test_slash_chat_can_send_options_in_batches(bot):
    ctx = AsyncMock()
    ctx.channel.create_thread.return_value.typing = MagicMock(mock_typing)
    thread = ctx.channel.create_thread.return_value

    choice_msg = Mock()
    choice_msg.content = '2'
    bot.wait_for.return_value = choice_msg

    cog = ChatCog(bot)
    await cog.chat.callback(cog, ctx)

    assert ctx.channel.create_thread.await_count == 1
    assert thread.typing.call_count == 1
    assert 'select a model' in thread.send.call_args_list[0].args[0]
    assert '1. model1\n2. model2' == thread.send.call_args_list[1].args[0]
    assert '3. model3' == thread.send.call_args_list[2].args[0]
    assert 'select a bot' in thread.send.call_args_list[3].args[0]
    assert '1. bot1\n2. bot2' in thread.send.call_args_list[4].args[0]
    assert '3. bot3' in thread.send.call_args_list[5].args[0]
    assert thread.purge.call_count == 1
    thread.edit.assert_awaited_with(name='Chat with bot2 by model2')
    assert '[bot]: mock-first-message' in thread.send.call_args_list[6].args[0]


@pytest.mark.asyncio
async def test_bot_will_reply_on_chatting_message_received_in_chatting_thread(chai_chat, bot):
    ctx = Mock()
    thread = MagicMock()
    thread.send = AsyncMock()
    thread.typing = MagicMock(mock_typing)
    thread.type = ChannelType.public_thread
    thread.archived = False
    thread.locked = False
    thread.name = 'Chat with bot1 by model2'
    message1 = Mock()
    message2 = Mock()
    message3 = Mock()
    message1.is_system.return_value = True
    message2.author.id = 'mock-app-id'
    message2.content = 'bot-label: mock-msg2'
    message2.is_system.return_value = False
    message3.author.id = 'mock-user-id'
    message3.content = 'mock-msg3'
    message3.is_system.return_value = False
    async def async_history(limit):
        yield message3
        yield message2
        yield message1
    thread.history = async_history

    message = Mock()
    message.type = MessageType.default
    message.is_system.return_value = False
    message.author = Mock()
    message.author.bot = False
    message.channel = thread
    message.reply = AsyncMock()

    cog = ChatCog(bot)
    await cog.on_message(message)

    assert thread.typing.call_count == 1
    message.reply.assert_awaited_with('[bot]: mock-reply')
    chai_chat.get_bot_config.assert_called_with('bot1')
    chai_chat.get_bot_response.assert_called_once()
    chai_chat.get_bot_response.assert_called_once_with(
        [('mock-msg2', '[bot]'), ('mock-msg3', 'user')],
        'model2',
        MOCK_BOT_CONFIG,
        developer_key="mock-dev-key"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thread_type, thread_archived, thread_locked, thread_name, expected_await_count", [
        (ChannelType.public_thread, False, False, 'Chat with bot1 by model2', 1),
        (ChannelType.private_thread, False, False, 'Chat with bot1 by model2', 1),
        (ChannelType.voice, False, False, 'Chat with bot1 by model2', 0),
        (ChannelType.public_thread, True, False, 'Chat with bot1 by model2', 0),
        (ChannelType.public_thread, False, True, 'Chat with bot1 by model2', 0),
        (ChannelType.public_thread, False, False, 'Initializing', 0),
    ]
)
@patch('discord_bot.cogs.chat_cog._reply_chat_message')
async def test_bot_only_reply_in_chatting_thread(reply_chat_message, thread_type, thread_archived, thread_locked, thread_name, expected_await_count, bot):
    thread = MagicMock()
    thread.send = AsyncMock()
    thread.typing = MagicMock(mock_typing)
    thread.type = thread_type
    thread.archived = thread_archived
    thread.locked = thread_locked
    thread.name = thread_name
    
    message = Mock()
    message.type = MessageType.default
    message.is_system.return_value = False
    message.author.bot = False
    message.channel = thread
    message.reply = AsyncMock()
    
    cog = ChatCog(bot)
    await cog.on_message(message)
    assert expected_await_count == reply_chat_message.await_count


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_type, message_is_system, message_author_is_bot, expected_await_count", [
        (MessageType.default, False, False, 1),
        (MessageType.call, False, False, 0),
        (MessageType.default, True, False, 0),
        (MessageType.default, False, True, 0),
    ]
)
@patch('discord_bot.cogs.chat_cog._reply_chat_message')
async def test_bot_only_reply_chatting_message(reply_chat_message, message_type, message_is_system, message_author_is_bot, expected_await_count, bot):
    thread = MagicMock()
    thread.send = AsyncMock()
    thread.typing = MagicMock(mock_typing)
    thread.type = ChannelType.public_thread
    thread.archived = False
    thread.locked = False
    thread.name = 'Chat with bot1 by model2'
    
    message = Mock()
    message.type = message_type
    message.is_system.return_value = message_is_system
    message.author.bot = message_author_is_bot
    message.channel = thread
    message.reply = AsyncMock()
    
    cog = ChatCog(bot)
    await cog.on_message(message)
    assert expected_await_count == reply_chat_message.await_count