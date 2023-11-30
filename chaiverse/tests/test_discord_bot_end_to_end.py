import asyncio
import os
import pytest
import pytest_asyncio
from unittest import mock

import discord
import discord.ext.test as dpytest
from discord_bot import discord_bot_client
import vcr


RESOURCE_DIR = os.path.join(os.path.abspath(os.path.join(__file__, '..')), 'resources')


@pytest_asyncio.fixture
async def bot():
    async with discord_bot_client() as bot:
        dpytest.configure(bot)
        bot.ws.socket = mock.Mock() # dpytest neglected to provide a socket mock for its websocket mock
        yield bot
        await dpytest.empty_queue()


@pytest.mark.asyncio
@vcr.use_cassette(os.path.join(RESOURCE_DIR, 'test_new_user_joined_will_be_greeted_with_the_same_token.yaml'))
async def test_new_user_joined_will_be_greeted_with_the_same_token(bot):
    await dpytest.member_join()
    await dpytest.run_all_events()
    assert dpytest.verify().message().contains().content('CR_ce85dc06c97a4da79d0a9480fbb5257e')
    await dpytest.member_join()
    await dpytest.run_all_events()
    assert dpytest.verify().message().contains().content('CR_ce85dc06c97a4da79d0a9480fbb5257e')


@pytest.mark.asyncio
async def test_slash_info_has_not_regressed(bot):
    guild = bot.guilds[0]
    channel = guild.text_channels[0]
    await dpytest.message('/info', channel)
    await dpytest.run_all_events()
    assert dpytest.verify().message().contains().content('commands you need to know')
